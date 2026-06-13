# Copyright 2025 CoSkill.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Cloud Analyzer (云端分析器) —— CoSkill 闭环组件 ①。

承担高认知负荷的离线归纳：吃轨迹池上传的 CompressedBatch（同时含成功 + 失败），
做**正负对比蒸馏 (Contrastive Distillation)**，输出结构化 SkillPatch（§2.3）：
  - trigger      触发条件（环境条件）
  - action_flow  核心动作流
  - avoid        绝对规避清单
  - principle / when_to_apply  兼容旧 SkillsOnlyMemory.format_for_prompt 的拼接字段

后端复用现有 SkillUpdater 的 DeepSeek / Azure 客户端逻辑。本期云端模型为
DeepSeek V4 Flash（DEEPSEEK_MODEL=deepseek-v4-flash）。

与旧 SkillUpdater.analyze_failures 的区别：
  1. 同时利用成功样本（对比"做对了什么"）与前缀树分叉点（决策分歧点）。
  2. 输出三段式结构化补丁，而非仅 principle。
  3. 接口以 CompressedBatch 为输入，天然对接 TracesPool.export_batch()。
"""

import json
import os
import re
from typing import Any, Dict, List, Optional


class CloudAnalyzer:
    """正负对比蒸馏。后端经 SKILL_UPDATER_BACKEND 选择（deepseek / azure）。"""

    def __init__(
        self,
        max_new_skills_per_update: int = 3,
        max_completion_tokens: int = 4096,
        output_dir: Optional[str] = None,
    ):
        from openai import AzureOpenAI, OpenAI  # 延迟导入，避免无依赖时整包不可用

        backend = os.environ.get("SKILL_UPDATER_BACKEND", "deepseek").lower()
        if backend == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            if not api_key or not endpoint:
                raise EnvironmentError(
                    "CloudAnalyzer (azure) requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
                )
            self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint,
                                      api_version=api_version)
            self.model = os.environ.get("AZURE_OPENAI_MODEL", "o3")
        else:  # deepseek
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise EnvironmentError("CloudAnalyzer (deepseek) requires DEEPSEEK_API_KEY.")
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            )
            # 本期默认 DeepSeek V4 Flash
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

        self.max_completion_tokens = max_completion_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.output_dir = None
        if output_dir:
            self.output_dir = os.path.join(output_dir, "cloud_io")
            os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 公共接口                                                             #
    # ------------------------------------------------------------------ #

    def contrastive_distill(
        self,
        compressed_batch: Dict[str, Any],
        current_skills: Dict[str, Any],
    ) -> List[Dict]:
        """对 CompressedBatch 做正负对比蒸馏，返回结构化 SkillPatch 列表。

        Args:
            compressed_batch: TracesPool.export_batch() 的输出（§2.2）。
            current_skills:   当前技能库 dict（用于去重提示 + 计算下一个 dyn_ ID）。
        """
        success = compressed_batch.get("success_samples", []) or []
        failure = compressed_batch.get("failure_samples", []) or []
        if not success and not failure:
            return []

        next_dyn_idx = self._next_dyn_index(current_skills)
        prompt = self._build_contrastive_prompt(
            compressed_batch, current_skills, next_dyn_idx
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            content = response.choices[0].message.content
            raw_skills = self._parse_patches(content)

            if hasattr(response, "usage") and response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            patches = self._normalize_patches(
                raw_skills, next_dyn_idx, compressed_batch
            )

            self.update_history.append({
                "batch_id": compressed_batch.get("batch_id"),
                "trigger_reason": compressed_batch.get("trigger_reason"),
                "n_success": len(success),
                "n_failure": len(failure),
                "num_patches": len(patches),
                "skill_ids": [p.get("skill_id") for p in patches],
            })

            patches = patches[: self.max_new_skills_per_update]
            if self.output_dir is not None:
                self._dump_patches(compressed_batch, patches)
            return patches

        except Exception as e:
            print(f"[CloudAnalyzer] Error calling {self.model}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Prompt 构造（正负对比）                                              #
    # ------------------------------------------------------------------ #

    def _build_contrastive_prompt(
        self,
        batch: Dict[str, Any],
        current_skills: Dict[str, Any],
        next_dyn_idx: int,
    ) -> str:
        task_type = batch.get("task_type", "unknown")
        succ_txt = self._format_difftraces(batch.get("success_samples", []), limit=5)
        fail_txt = self._format_difftraces(batch.get("failure_samples", []), limit=6)
        fork_txt = self._format_forks(batch.get("prefix_tree", {}))

        existing_titles = [s.get("title", "") for s in current_skills.get("general_skills", [])]
        for tt, skills in current_skills.get("task_specific_skills", {}).items():
            for s in skills:
                existing_titles.append(f"[{tt}] {s.get('title', '')}")

        example_ids = ", ".join(
            f'"dyn_{next_dyn_idx + j:03d}"' for j in range(self.max_new_skills_per_update)
        )

        return f"""You are an expert at distilling embodied-agent experience into reusable skills.

You are given COMPRESSED trajectories for task_type="{task_type}". Each step shows the action
taken and the OBSERVATION DELTA (only what changed in the environment: +added / -removed lines).

SUCCESSFUL TRAJECTORIES (what to do):
{succ_txt}

FAILED TRAJECTORIES (what went wrong):
{fail_txt}

DECISION FORKS (where successful and failed runs diverged after a shared prefix):
{fork_txt}

TASK — Contrastive Analysis:
1. Compare success vs failure: identify what successful runs did right and the exact step where
   failed runs went off track (use the decision forks).
2. Abstract the (success - failure) difference into 1-{self.max_new_skills_per_update} NEW, generalizable skills.
3. Avoid duplicating these existing skills: {existing_titles}

WRITING STYLE — keep skills SHORT and SIMPLE so a small 4B model can follow them.
Match the format of these hand-written seed skills exactly:
  {{"skill_id": "gen_001", "title": "Systematic Exploration", "principle": "Search every plausible surface or container exactly once before revisiting; prioritize unopened or unseen locations.", "when_to_apply": "Anytime the goal object count is not yet met and unexplored locations remain."}}

Return ONLY a JSON array. Each skill MUST have EXACTLY these fields (no extra fields):
  - "skill_id":      one of {example_ids}
  - "title":         3-5 word title
  - "scope":         "general" or "task_specific"
  - "task_type":     "{task_type}" (or "" if general)
  - "principle":     ONE or TWO plain sentences stating the rule. Keep it under 30 words. No JSON, no lists.
  - "when_to_apply": ONE short sentence naming the situation that triggers this skill. Under 20 words.

Example:
[{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Open Before Search", "scope": "task_specific", "task_type": "{task_type}", "principle": "If the target may be inside a closed container, open each closed container before searching its contents.", "when_to_apply": "When the goal object is not visible and closed containers remain."}}]

Return ONLY the JSON array, no other text."""

    def _format_difftraces(self, traces: List[Dict], limit: int) -> str:
        if not traces:
            return "(none)"
        out = []
        for i, tr in enumerate(traces[:limit]):
            lines = [f"\nTrajectory {i + 1} [{tr.get('outcome', '?')}] task: {tr.get('task', '')}"]
            for s in tr.get("steps", [])[:12]:
                obs_text = s.get('obs_delta', '')
                # obs_is_full=True 表示这是完整观测原文(短观测不差分), 否则是 +/- 增量。
                label = "obs" if s.get('obs_is_full') else "delta"
                lines.append(f"  action: {s.get('action', '')}  | {label}: {obs_text[:400]}")
            if tr.get("dropped_loops"):
                lines.append(f"  (dropped {tr['dropped_loops']} looping actions)")
            out.append("\n".join(lines))
        return "\n".join(out)

    def _format_forks(self, prefix_tree: Dict, max_forks: int = 6) -> str:
        """从前缀树中找出分叉节点（children>1），展示各分支的成功/失败计数。"""
        forks: List[str] = []

        def walk(node, path):
            if len(forks) >= max_forks:
                return
            children = node.get("children", {})
            if len(children) > 1:
                branch_desc = "; ".join(
                    f"'{a}' (succ={c['n_success']},fail={c['n_failure']})"
                    for a, c in list(children.items())[:5]
                )
                forks.append(f"After [{' -> '.join(path) if path else 'start'}]: {branch_desc}")
            for a, c in children.items():
                walk(c, path + [a])

        walk(prefix_tree, [])
        return "\n".join(forks) if forks else "(no clear divergence point)"

    # ------------------------------------------------------------------ #
    # 解析 + 规范化                                                        #
    # ------------------------------------------------------------------ #

    def _parse_patches(self, response: str) -> List[Dict]:
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return [s for s in data if isinstance(s, dict) and "title" in s]
        except json.JSONDecodeError as e:
            print(f"[CloudAnalyzer] JSON parse error: {e}")
        return []

    def _normalize_patches(
        self,
        skills: List[Dict],
        start_idx: int,
        batch: Dict[str, Any],
    ) -> List[Dict]:
        """重分配 dyn_ ID、补全兼容字段（principle/when_to_apply）、附 evidence。"""
        n_succ = len(batch.get("success_samples", []))
        n_fail = len(batch.get("failure_samples", []))
        out: List[Dict] = []
        for i, s in enumerate(skills):
            patch = dict(s)
            patch["skill_id"] = f"dyn_{start_idx + i:03d}"
            action_flow = patch.get("action_flow") or []
            avoid = patch.get("avoid") or []
            trigger = patch.get("trigger", "")

            # 新格式直接给 principle/when_to_apply（与初始 gen_* 种子技能一致）。
            # 兼容旧格式：若模型仍回 trigger+action_flow，则拼接降级为 principle。
            if not patch.get("principle"):
                flow_str = "; ".join(action_flow) if isinstance(action_flow, list) else str(action_flow)
                patch["principle"] = (f"{trigger}. Steps: {flow_str}" if flow_str else trigger) or patch.get("title", "")
            if not patch.get("when_to_apply"):
                patch["when_to_apply"] = trigger
            # 丢弃冗长的结构化字段，保持与初始技能相同的精简 4 字段风格，
            # 避免注入 prompt 时塞给 4B 小模型过长难读的内容。
            patch.pop("action_flow", None)
            patch.pop("avoid", None)
            patch.pop("trigger", None)
            patch.setdefault("scope", "general")
            patch["evidence"] = {"from_success": n_succ, "from_failure": n_fail}
            out.append(patch)
        return out

    def _dump_patches(self, batch: Dict, patches: List[Dict]) -> None:
        try:
            bid = batch.get("batch_id", "unknown")[:8]
            path = os.path.join(self.output_dir, f"patches_{bid}.json")
            with open(path, "w") as f:
                json.dump({"batch_id": batch.get("batch_id"), "patches": patches},
                          f, ensure_ascii=False, indent=2)
            print(f"[CloudAnalyzer] wrote {len(patches)} patches → {path}")
        except Exception as e:
            print(f"[CloudAnalyzer] patch dump failed: {e}")

    # ------------------------------------------------------------------ #
    # ID 管理（复用旧逻辑）                                                 #
    # ------------------------------------------------------------------ #

    def _next_dyn_index(self, current_skills: Dict) -> int:
        max_idx = 0
        pattern = re.compile(r"^dyn_(\d+)$")
        for skill in current_skills.get("general_skills", []):
            m = pattern.match(skill.get("skill_id", ""))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        for skills in current_skills.get("task_specific_skills", {}).values():
            for skill in skills:
                m = pattern.match(skill.get("skill_id", ""))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {"total_updates": 0, "total_patches": 0}
        return {
            "total_updates": len(self.update_history),
            "total_patches": sum(h["num_patches"] for h in self.update_history),
            "all_skill_ids": [sid for h in self.update_history for sid in h["skill_ids"]],
            "large_model_prompt_tokens": self.total_prompt_tokens,
            "large_model_completion_tokens": self.total_completion_tokens,
            "large_model_total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
