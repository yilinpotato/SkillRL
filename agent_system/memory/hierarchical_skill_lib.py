# Copyright 2025 CoSkill.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Hierarchical Skill Lib (层次化 Skill Lib) —— CoSkill 闭环组件 ②，调度枢纽。

在现有 SkillsOnlyMemory（扁平技能库 + 模板/embedding 检索）之上，增加 L0/L1/L2
冷热分层与生命周期管理，作为云端与端侧之间的"多级语义缓存"：

  L0 极热缓冲层：新提炼的 Env/Task-specific skills，以 Context 文本热下发。
  L1 温数据演化层：多周期稳定、高通用、高成功率的技能（本期仍以加权 Context 下发，
                   编译轻量 LoRA 见 docs §12 TODO）。
  L2 极冷标记层：长期多环境验证的 Global skills，进入待固化队列（Skill2param）。

兼容性（docs §10）：
  - 无 lifecycle 字段的旧技能默认视为 L0、未固化，照常注入。
  - enable_hierarchy=False 时退化为原 SkillsOnlyMemory 行为（不分层、不过滤）。
  - retrieve()/format_for_prompt() 完全复用父类；热路径仅在父类结果上做 layer 过滤。
"""

from typing import Any, Dict, List, Optional, Tuple

from .skills_only_memory import SkillsOnlyMemory


class HierarchicalSkillLib(SkillsOnlyMemory):

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        enable_hierarchy: bool = True,
        stable_cycles_l1: int = 3,
        stable_cycles_l2: int = 5,
        success_l1: float = 0.7,
        demote_threshold: float = 0.3,
        min_calls: int = 20,
        min_task_types_l2: int = 3,
    ):
        super().__init__(
            skills_json_path=skills_json_path,
            retrieval_mode=retrieval_mode,
            embedding_model_path=embedding_model_path,
            task_specific_top_k=task_specific_top_k,
        )
        self.enable_hierarchy = enable_hierarchy
        self.stable_cycles_l1 = stable_cycles_l1
        self.stable_cycles_l2 = stable_cycles_l2
        self.success_l1 = success_l1
        self.demote_threshold = demote_threshold
        self.min_calls = min_calls
        self.min_task_types_l2 = min_task_types_l2
        self.cycle = 0

        # 为所有已有技能补默认 lifecycle（缺省安全）
        for s in self._iter_all_skills():
            self._ensure_lifecycle(s)

    # ------------------------------------------------------------------ #
    # lifecycle 工具                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_lifecycle(skill: Dict) -> Dict:
        lc = skill.get("lifecycle")
        if lc is None:
            lc = {}
            skill["lifecycle"] = lc
        lc.setdefault("layer", "L0")
        lc.setdefault("created_cycle", 0)
        lc.setdefault("last_modified_cycle", 0)
        lc.setdefault("stable_cycles", 0)
        lc.setdefault("call_count", 0)
        lc.setdefault("success_when_used", 0)
        lc.setdefault("success_rate", None)
        lc.setdefault("internalized", False)
        lc.setdefault("task_types_seen", [])
        return lc

    def _iter_all_skills(self):
        for s in self.skills.get("general_skills", []):
            yield s
        for skills in self.skills.get("task_specific_skills", {}).values():
            for s in skills:
                yield s

    def _skill_by_id(self, skill_id: str) -> Optional[Dict]:
        for s in self._iter_all_skills():
            if s.get("skill_id") == skill_id:
                return s
        return None

    # ------------------------------------------------------------------ #
    # 接收云端补丁                                                         #
    # ------------------------------------------------------------------ #

    def ingest_patches(self, patches: List[Dict]) -> int:
        """接收云端 SkillPatch，按 scope 落到 L0 并登记 lifecycle。返回新增数。"""
        added = 0
        existing = self._get_all_skill_ids()
        for p in patches:
            sid = p.get("skill_id")
            if not sid or sid in existing:
                print(f"[HierSkillLib] skip duplicate/empty patch: {sid}")
                continue
            lc = self._ensure_lifecycle(p)
            lc["layer"] = "L0"
            lc["created_cycle"] = self.cycle
            lc["last_modified_cycle"] = self.cycle

            scope = p.get("scope", "general")
            if scope == "task_specific" and p.get("task_type"):
                cat = p["task_type"]
                self.skills.setdefault("task_specific_skills", {}).setdefault(cat, []).append(p)
            else:
                self.skills.setdefault("general_skills", []).append(p)
            existing.add(sid)
            added += 1
            print(f"[HierSkillLib] ingested L0 patch: {sid} - {p.get('title', 'N/A')}")
        if added > 0:
            self._skill_embeddings_cache = None  # 失效 embedding 缓存
        return added

    # ------------------------------------------------------------------ #
    # 使用统计（共享归因，docs §11.4）                                      #
    # ------------------------------------------------------------------ #

    def record_usage(self, skill_ids: List[str], success: bool,
                     task_type: Optional[str] = None) -> None:
        """共享归因：episode 注入的所有技能 call_count+1，成功则 success_when_used+1。"""
        for sid in set(skill_ids or []):
            s = self._skill_by_id(sid)
            if s is None:
                continue
            lc = self._ensure_lifecycle(s)
            lc["call_count"] += 1
            if success:
                lc["success_when_used"] += 1
            cc = lc["call_count"]
            lc["success_rate"] = (lc["success_when_used"] / cc) if cc else None
            if task_type and task_type not in lc["task_types_seen"]:
                lc["task_types_seen"].append(task_type)

    # ------------------------------------------------------------------ #
    # 生命周期推进：晋升 / 降级                                             #
    # ------------------------------------------------------------------ #

    def advance_lifecycle(self, modified_ids: Optional[List[str]] = None) -> Dict:
        """每个更新周期调用一次。modified_ids: 本周期被云端修改/新增的技能。

        返回本周期的迁移事件摘要。
        """
        self.cycle += 1
        modified = set(modified_ids or [])
        events = {"to_l1": [], "to_l2": [], "demoted": [], "deprecated": []}

        for s in self._iter_all_skills():
            lc = self._ensure_lifecycle(s)
            sid = s.get("skill_id")

            # 稳定度：本周期未被修改 → stable_cycles+1，否则清零
            if sid in modified:
                lc["stable_cycles"] = 0
                lc["last_modified_cycle"] = self.cycle
            else:
                lc["stable_cycles"] += 1

            sr = lc["success_rate"]
            cc = lc["call_count"]
            layer = lc["layer"]

            # 降级 / 淘汰：低成功率且样本充分
            if sr is not None and cc >= self.min_calls and sr < self.demote_threshold:
                if layer == "L2":
                    lc["layer"] = "L1"; events["demoted"].append(sid)
                elif layer == "L1":
                    lc["layer"] = "L0"; events["demoted"].append(sid)
                else:  # L0 持续低效 → 弃用
                    lc["internalized"] = False
                    lc["deprecated"] = True
                    events["deprecated"].append(sid)
                continue

            if lc.get("deprecated"):
                continue

            # 晋升 L0 → L1
            if (layer == "L0"
                    and lc["stable_cycles"] >= self.stable_cycles_l1
                    and sr is not None and sr >= self.success_l1
                    and cc >= self.min_calls):
                lc["layer"] = "L1"; events["to_l1"].append(sid)
            # 晋升 L1 → L2（跨多 task_type 且长期稳定）
            elif (layer == "L1"
                    and lc["stable_cycles"] >= self.stable_cycles_l2
                    and len(lc["task_types_seen"]) >= self.min_task_types_l2):
                lc["layer"] = "L2"; events["to_l2"].append(sid)

        nonempty = {k: v for k, v in events.items() if v}
        if nonempty:
            print(f"[HierSkillLib] cycle {self.cycle} lifecycle events: {nonempty}")
        return events

    # ------------------------------------------------------------------ #
    # 调度接口                                                             #
    # ------------------------------------------------------------------ #

    def retrieve(self, task_description: str, top_k: int = 6, **kwargs) -> Dict[str, Any]:
        """热路径检索：复用父类，再剔除已固化(internalized)与弃用(deprecated)技能。

        为避免"先取 top_k 再过滤"把结果数量打到 top_k 以下，这里向父类**过量请求**
        （top_k + 已固化/弃用技能数），过滤后再截断回 top_k。
        """
        if not self.enable_hierarchy:
            return super().retrieve(task_description, top_k=top_k, **kwargs)

        def keep(s: Dict) -> bool:
            lc = s.get("lifecycle") or {}
            return not lc.get("internalized", False) and not lc.get("deprecated", False)

        # 统计被过滤技能数，决定过量请求的额度
        n_filtered = sum(1 for s in self._iter_all_skills() if not keep(s))
        ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
        result = super().retrieve(
            task_description, top_k=top_k + n_filtered, **kwargs
        )

        gen = [s for s in result.get("general_skills", []) if keep(s)][:top_k]
        task = [s for s in result.get("task_specific_skills", []) if keep(s)][:ts_top_k]
        result["general_skills"] = gen
        result["task_specific_skills"] = task
        # 记录本次注入的技能 id，供端侧 record_usage 回写
        result["injected_skill_ids"] = (
            [s.get("skill_id") for s in gen] + [s.get("skill_id") for s in task]
        )
        return result

    def get_cold_skills(self) -> List[Dict]:
        """返回 L2 待固化队列（未固化的 Global skills），供 Skill2param 内化。"""
        return [s for s in self._iter_all_skills()
                if (s.get("lifecycle") or {}).get("layer") == "L2"
                and not (s.get("lifecycle") or {}).get("internalized", False)]

    def has_cold_skills(self) -> bool:
        return len(self.get_cold_skills()) > 0

    def mark_internalized(self, skill_ids: List[str]) -> None:
        """固化完成：标记 internalized=True，热路径不再注入。"""
        for sid in skill_ids:
            s = self._skill_by_id(sid)
            if s is not None:
                self._ensure_lifecycle(s)["internalized"] = True

    def layer_counts(self) -> Dict[str, int]:
        counts = {"L0": 0, "L1": 0, "L2": 0, "internalized": 0, "deprecated": 0}
        for s in self._iter_all_skills():
            lc = s.get("lifecycle") or {}
            counts[lc.get("layer", "L0")] = counts.get(lc.get("layer", "L0"), 0) + 1
            if lc.get("internalized"):
                counts["internalized"] += 1
            if lc.get("deprecated"):
                counts["deprecated"] += 1
        return counts
