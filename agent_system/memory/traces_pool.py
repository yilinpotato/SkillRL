# Copyright 2025 CoSkill.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Traces Pool (轨迹记录池) —— CoSkill 闭环的"心脏起搏器"。

后台静默收集端侧 SLM 产生的所有 RawTrace（成功 + 失败），并在收集时即时做
**流式压缩**，大幅降低后续云端分析的输入成本：

  1. 状态差分法 (_diff_compress)：只记录相邻 observation 的环境变化增量 obs_delta，
     去掉每步重复的环境全文。
  2. 死循环过滤 (_filter_loops)：检测连续重复 action 且无 reward / 无 obs_delta 变化的
     无意义动作并丢弃。
  3. 前缀树合并 (_merge_prefix_tree)：把同 task_type 多条轨迹按动作序列建前缀树，
     合并公共起始动作，分叉点即"决策分歧点"，供云端对比归因。

并通过**异步表现/质量水位线**决定何时唤醒云端：
  - 质量/容量水位线 (capacity_watermark)：累计压缩后高质量可分析轨迹 token 数达阈值 → 打包。
  - 表现水位线 (perf_watermark)：某 task_type 近期失败率超阈值 → 立即打包。
  - 趋势水位线：某 task_type 近期成功率停滞或下降 → 打包，让云端生成/修补 skill tree。

数据契约见 docs/CoSkill_架构设计.md §2。所有产物落盘到 OUTPUT_DIR/traces_pool/。
"""

import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

# ALFWorld/interactive-env actions embed a specific object-instance index
# ("go to cabinet 3", "go to cabinet 7", ...). Merging the prefix tree on the
# literal action string means two episodes that make the SAME semantic
# decision (check a cabinet next) almost never share a tree edge, because
# they happened to sample different instance numbers. Collapsing instance
# indices to "#" lets the tree merge on the decision itself.
_INSTANCE_INDEX_RE = re.compile(r"\b\d+\b")
_WEBSHOP_ACTION_RE = re.compile(
    r"^\s*(search|click)\s*\[(.*)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_action_for_merge(action: str) -> str:
    """Normalize a merge key without erasing WebShop option semantics.

    ALFWorld numbers identify interchangeable object instances. WebShop
    numbers can instead be a model, size, quantity, price, or option value, so
    replacing them with ``#`` would falsely merge different purchase choices.
    """
    raw = (action or "").strip()
    webshop = _WEBSHOP_ACTION_RE.fullmatch(raw)
    if webshop:
        verb = webshop.group(1).lower()
        payload = re.sub(r"\s+", " ", webshop.group(2)).strip()
        return f"{verb}[{payload}]"
    return _INSTANCE_INDEX_RE.sub("#", raw)


# ALFWorld/TextWorld's very first observation of every episode wraps the one
# piece of real information (the receptacle list) in a fixed banner, and ends
# with a restatement of the task that is already stored verbatim as the
# trace's own ``task`` field. In a real ablation batch this framing text made
# up ~21% of ALL observation characters despite carrying zero decision
# content. Both phrases are exact literal TextWorld/ALFRED strings, so
# stripping them is a no-op (and therefore harmless) for any other
# environment's text.
_ALFWORLD_BOILERPLATE_PREFIXES = (
    "-= Welcome to TextWorld, ALFRED! =-\n\n",
    "You are in the middle of a room. Looking quickly around you, you see ",
)
_ALFWORLD_TASK_SUFFIX_RE = re.compile(r"\n\nYour task is to:.*$", re.DOTALL)


def _strip_known_boilerplate(obs: str) -> str:
    out = obs
    for phrase in _ALFWORLD_BOILERPLATE_PREFIXES:
        out = out.replace(phrase, "")
    return _ALFWORLD_TASK_SUFFIX_RE.sub("", out)


def _approx_tokens(text: str) -> int:
    """粗略 token 估计：~4 字符/token，无需 tokenizer 依赖。"""
    if not text:
        return 0
    return max(1, len(text) // 4)


def longest_common_action_prefix(traces: List[dict]) -> List[str]:
    """轨迹集合的**共识前缀**：所有轨迹一致的最长起始动作序列。

    语义（端云通信协议 §4.1.1）：多条轨迹共享的起始段 = 端侧【已掌握的共识】，
    分歧从第一个不一致处开始。只在**存在成功轨迹**时才认定共识（否则可能把
    "所有轨迹一起犯的同一个错"误当共识），且要求该前缀确实是某条成功轨迹的起始。
    返回共识动作列表（可能为空）。
    """
    seqs = [[(s.get("action") or "") for s in (t.get("steps") or [])] for t in traces]
    seqs = [s for s in seqs if s]
    if not seqs:
        return []
    lcp: List[str] = []
    for tup in zip(*seqs):
        a = tup[0]
        if a and all(x == a for x in tup):
            lcp.append(a)
        else:
            break
    return lcp


class TracesPool:
    """收集 + 流式压缩 + 双轨水位线触发。进程内单例即可起步。"""

    def __init__(
        self,
        capacity_watermark: int = 50_000,
        perf_watermark: float = 0.6,
        min_samples: int = 8,
        loop_threshold: int = 3,
        recent_window: int = 20,
        output_dir: Optional[str] = None,
        max_keep_per_type: int = 200,
        stagnation_delta: float = 0.05,
        decline_delta: float = 0.05,
        stagnation_success_ceiling: float = 0.95,
        enable_loop_filter: bool = True,
        enable_obs_delta: bool = True,
        enable_prefix_tree: bool = True,
        enable_consensus_prefix: bool = True,
        cloud_evidence_mode: str = "tree_only",
    ):
        """
        Args:
            capacity_watermark:  容量轨阈值（压缩后估算 token 数）。
            perf_watermark:      表现轨阈值（近期失败率，超过则触发）。
            min_samples:         触发表现轨所需的最小近期样本数。
            loop_threshold:      死循环判定：同一 action 连续无效重复次数。
            recent_window:       计算近期失败率的滑动窗口大小（按 task_type）。
            output_dir:          落盘根目录，实际写入 output_dir/traces_pool/。
            max_keep_per_type:   每个 task_type 在池中保留的最大压缩轨迹数（防爆内存）。
            stagnation_delta:    前后半窗口成功率差小于该值视为停滞。
            decline_delta:       后半窗口成功率比前半窗口下降超过该值视为下降。
            stagnation_success_ceiling:
                                  成功率已接近满分时不因“停滞”触发，避免无意义云端调用。
            cloud_evidence_mode:  ``tree_only`` 仅向云端投影自包含轨迹树；
                                  ``flat`` 仅用于明确关闭压缩的消融。
        """
        self.capacity_watermark = capacity_watermark
        self.perf_watermark = perf_watermark
        self.min_samples = min_samples
        self.loop_threshold = loop_threshold
        self.recent_window = recent_window
        self.max_keep_per_type = max_keep_per_type
        self.stagnation_delta = stagnation_delta
        self.decline_delta = decline_delta
        self.stagnation_success_ceiling = stagnation_success_ceiling
        # These switches are intentionally independent.  They are primarily
        # used by fixed-trajectory ablations; their defaults preserve the
        # production compression behaviour.
        self.enable_loop_filter = bool(enable_loop_filter)
        self.enable_obs_delta = bool(enable_obs_delta)
        self.enable_prefix_tree = bool(enable_prefix_tree)
        self.enable_consensus_prefix = bool(enable_consensus_prefix)
        self.cloud_evidence_mode = str(cloud_evidence_mode or "").strip().lower()
        if self.cloud_evidence_mode not in {"tree_only", "flat"}:
            raise ValueError("cloud_evidence_mode must be 'tree_only' or 'flat'")
        if self.cloud_evidence_mode == "tree_only" and not self.enable_prefix_tree:
            raise ValueError(
                "cloud_evidence_mode='tree_only' requires enable_prefix_tree=True"
            )

        # 压缩后轨迹按 task_type + outcome 分桶
        self._success: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_keep_per_type))
        self._failure: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_keep_per_type))
        # 近期 outcome 滑窗（按 task_type），用于表现水位线
        self._recent_outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=recent_window))

        self._token_count = 0          # 自上次导出以来累计的压缩 token 数
        self._n_added = 0              # 自上次导出以来收集的轨迹数
        self._total_added = 0          # 历史累计（仅统计）
        self._total_dropped_loops = 0  # 历史累计被丢弃的死循环动作数

        self.output_dir = None
        if output_dir:
            self.output_dir = os.path.join(output_dir, "traces_pool")
            os.makedirs(self.output_dir, exist_ok=True)
            # 原始轨迹追加日志（便于人工查看，不参与触发计算）
            self._raw_log_path = os.path.join(self.output_dir, "raw_traces.jsonl")
        else:
            self._raw_log_path = None

    # ------------------------------------------------------------------ #
    # 收集                                                                 #
    # ------------------------------------------------------------------ #

    def add_trace(self, raw_trace: dict) -> None:
        """收集一条 RawTrace 并即时压缩；更新水位线计数。

        raw_trace schema 见 §2.1：
          {traj_uid, task, task_type, outcome, episode_reward, steps[], meta{}}
          steps[i] = {step, observation, action, reward}
        """
        task_type = raw_trace.get("task_type", "unknown")
        outcome = raw_trace.get("outcome")
        if outcome not in ("success", "failure"):
            # 容错：用 episode_reward 推断
            outcome = "success" if raw_trace.get("episode_reward", 0) > 0 else "failure"

        steps = raw_trace.get("steps", []) or []
        raw_steps = list(steps)
        if self.enable_loop_filter:
            cleaned, dropped = self._filter_loops(raw_steps)
        else:
            cleaned, dropped = raw_steps, 0
        if self.enable_obs_delta:
            encoded_steps = self._diff_compress(cleaned)
        else:
            # Do not emit an ``obs_delta`` field in the all-off payload: this
            # makes it mechanically auditable that the cloud saw raw
            # observations rather than a delta representation.
            encoded_steps = self._full_observation_steps(cleaned)

        stage_tokens = {
            "raw": self._steps_tokens(raw_steps, observation_key="observation"),
            "loop_filtered": self._steps_tokens(cleaned, observation_key="observation"),
            "encoded": self._steps_tokens(
                encoded_steps,
                observation_key="obs_delta" if self.enable_obs_delta else "observation",
            ),
        }

        diff_trace = {
            "traj_uid": raw_trace.get("traj_uid", str(uuid.uuid4())),
            "task": raw_trace.get("task", ""),
            # task_type 随压缩轨迹一并保留：export_batch 默认合并所有类型
            # (task_type="ALL"),下游 playbook 进化需按 task_type 分组成功/失败样本。
            "task_type": task_type,
            "outcome": outcome,
            "episode_reward": raw_trace.get("episode_reward", 0),
            # WebShop exposes a graded terminal score even though CoSkill uses
            # strict score==1.0 for success.  Preserve it so the cloud can tell
            # a near match from a completely wrong purchase.
            "task_score": raw_trace.get(
                "task_score", (raw_trace.get("meta") or {}).get("task_score")
            ),
            "steps": encoded_steps,
            "dropped_loops": dropped,
            "skill_ids_used": (raw_trace.get("meta") or {}).get("skill_ids_used", []),
            "compression_trace_stats": {
                "raw_steps": len(raw_steps),
                "loop_filtered_steps": len(cleaned),
                "encoded_steps": len(encoded_steps),
                "tokens": stage_tokens,
                "chars": {
                    "raw": self._steps_chars(raw_steps, observation_key="observation"),
                    "loop_filtered": self._steps_chars(cleaned, observation_key="observation"),
                    "encoded": self._steps_chars(
                        encoded_steps,
                        observation_key="obs_delta" if self.enable_obs_delta else "observation",
                    ),
                },
            },
        }

        bucket = self._success if outcome == "success" else self._failure
        bucket[task_type].append(diff_trace)
        self._recent_outcomes[task_type].append(outcome)

        # 水位线计数：用压缩后的 token 估计
        tok = stage_tokens["encoded"]
        self._token_count += tok
        self._n_added += 1
        self._total_added += 1
        self._total_dropped_loops += dropped

        if self._raw_log_path is not None:
            self._append_raw_log(raw_trace)

    def _append_raw_log(self, raw_trace: dict) -> None:
        try:
            with open(self._raw_log_path, "a") as f:
                f.write(json.dumps(raw_trace, ensure_ascii=False) + "\n")
        except Exception as e:  # 落盘失败不应影响训练
            print(f"[TracesPool] raw log write failed: {e}")

    # ------------------------------------------------------------------ #
    # 流式压缩                                                             #
    # ------------------------------------------------------------------ #

    def _filter_loops(self, steps: List[dict]) -> Tuple[List[dict], int]:
        """死循环过滤：同一 action 连续重复且无 reward 增益、无 observation 变化时丢弃。

        返回 (cleaned_steps, dropped_count)。
        """
        if not steps:
            return [], 0
        cleaned: List[dict] = []
        dropped = 0
        run_action = None
        run_len = 0
        prev_obs = None
        for s in steps:
            action = (s.get("action") or "").strip()
            obs = s.get("observation") or ""
            reward = s.get("reward", 0) or 0
            obs_changed = (obs != prev_obs)
            if action == run_action and not obs_changed and reward <= 0:
                run_len += 1
                if run_len >= self.loop_threshold:
                    dropped += 1
                    prev_obs = obs
                    continue  # 丢弃这一步
            else:
                run_action = action
                run_len = 1
            cleaned.append(s)
            prev_obs = obs
        return cleaned, dropped

    def _diff_compress(self, steps: List[dict]) -> List[dict]:
        """状态压缩：把 observation 的重复/增量部分替换成更短的引用。

        按序尝试两级降本：

          1) **精确复现**：这条轨迹里这个 observation 之前逐字出现过——重新
             造访同一个 receptacle/物体，或原地重复一个无效动作，都会产生一
             模一样的 observation 原文。真实消融数据里约 37% 的 step 属于这一
             类（含"和上一步完全相同"这种最简单的子情形）。命中时直接回指
             "和执行 X 之后看到的一样"，信息零损失，且比逐行差分适用范围更广
             （不要求任何行级重叠）。
          2) **逐行差分**：仅当观测够长、且差分结果比原文短得多时才用。这条
             路径对 ALFWorld 这种每步几乎都是单行文本的环境基本不会触发（单
             行一旦变化，"整行替换"式差分不会比原文短），但保留它是因为它对
             多行 observation 的环境（如 WebShop 商品页，行与行之间常有大段
             不变的样板文字）依然有效，不应该被 ALFWorld 的特例移除。

          都不划算时才回退到完整观测原文（并做 ALFWorld 固定套话剥离，见
          ``_strip_known_boilerplate``，对其它环境的文本是无害的 no-op）。
        """
        # 仅当完整观测超过该长度才考虑逐行差分（短观测直接存原文/精确复现引用）。
        min_len_to_diff = getattr(self, "diff_min_obs_chars", 400)
        # 且差分结果需至少比原文短这个比例，才认为划算（否则存原文）。
        min_savings_ratio = getattr(self, "diff_min_savings", 0.5)
        diff_steps: List[dict] = []
        prev_lines: List[str] = []
        prev_action = "(episode start)"
        seen_obs: Dict[str, str] = {}  # 逐字 observation -> 首次出现前那一步的 action
        for s in steps:
            obs = _strip_known_boilerplate(s.get("observation") or "")
            obs_clean = obs.strip()
            cur_lines = [ln.strip() for ln in obs.splitlines() if ln.strip()]
            cur_action = (s.get("action") or "").strip()

            # 收集所有候选压缩表示，取其中最短者；只有当它确实比原文短时才
            # 采用（对很短的 observation，"(same as after 'X')" 或
            # "(no change)" 这类占位文本本身可能比原文还长，必须显式兜底，
            # 不能假设"命中了某种压缩路径"就一定划算）。
            candidates: List[str] = []
            anchor_action = seen_obs.get(obs_clean) if obs_clean else None
            if anchor_action is not None:
                candidates.append(f"(same as after '{anchor_action}')")
            line_delta = self._line_delta(prev_lines, cur_lines)
            if line_delta == "(no change)":
                candidates.append(line_delta)
            elif (len(obs_clean) >= min_len_to_diff
                    and len(line_delta) <= len(obs_clean) * (1.0 - min_savings_ratio)):
                candidates.append(line_delta)

            best = min(candidates, key=len) if candidates else None
            if best is not None and len(best) < len(obs_clean):
                delta, worth_diff = best, True
            else:
                delta, worth_diff = obs_clean, False

            diff_steps.append({
                "action": cur_action,
                # 划算则存引用/增量，否则存完整观测（云端直接读原文）。
                "obs_delta": delta if worth_diff else obs_clean,
                "obs_is_full": not worth_diff,
                "reward": s.get("reward", 0) or 0,
            })
            # 只有真正存了完整原文的那一步才能被后面的精确复现引用——如果
            # 允许指向一个自己也是引用/差分的节点，读者要顺着链条一路找到
            # 底才能还原真实文本，等于埋了个悬空指针。只记录首次出现即可，
            # 无需在每次复现时更新成"最近一次"，语义更简单也不会累积误差。
            if obs_clean and not worth_diff and obs_clean not in seen_obs:
                seen_obs[obs_clean] = prev_action
            prev_lines = cur_lines
            prev_action = cur_action
        return diff_steps

    @staticmethod
    def _full_observation_steps(steps: List[dict]) -> List[dict]:
        """Return an uncompressed cloud representation with raw observations."""
        return [{
            "action": (s.get("action") or "").strip(),
            "observation": (s.get("observation") or "").strip(),
            "reward": s.get("reward", 0) or 0,
        } for s in steps]

    @staticmethod
    def _steps_chars(steps: List[dict], observation_key: str) -> int:
        return sum(len(str(s.get(observation_key, "") or "")) +
                   len(str(s.get("action", "") or "")) for s in steps)

    @classmethod
    def _steps_tokens(cls, steps: List[dict], observation_key: str) -> int:
        return sum(_approx_tokens(s.get(observation_key, "")) +
                   _approx_tokens(s.get("action", "")) for s in steps)

    @staticmethod
    def _line_delta(prev_lines: List[str], cur_lines: List[str]) -> str:
        """逐行集合差分，输出 '+新增 / -消失'。首步全部视为新增。"""
        prev_set = set(prev_lines)
        cur_set = set(cur_lines)
        added = [ln for ln in cur_lines if ln not in prev_set]
        removed = [ln for ln in prev_lines if ln not in cur_set]
        parts = [f"+{ln}" for ln in added] + [f"-{ln}" for ln in removed]
        # 若完全无变化，保留一个占位，避免云端误判
        return " | ".join(parts) if parts else "(no change)"

    def _merge_prefix_tree(self, traces: List[dict]) -> dict:
        """前缀树合并：把多条轨迹的 action 序列合并成树，记录每节点 outcome 计数。

        合并 key 是 **归一化后** 的 action（instance 编号折叠为 "#"），因为
        字面 action（如 "go to cabinet 3" / "go to cabinet 7"）本来就带着具体
        实例编号——按字面合并会让"检查一个 cabinet"这个语义决策在不同 episode
        里几乎从不合并（除非两条轨迹恰好抽到同一个编号），分叉点也就退化成
        "哪个具体编号"而不是"去 cabinet 还是 drawer"这种真正的决策分歧。归一
        化后同一决策下的不同实例才会汇入同一节点，分叉点才是有意义的决策点。

        节点 schema:
          {"action": str（归一化后，用于合并与展示）, "count": int,
           "n_success": int, "n_failure": int, "n_variants": int（合并了多少
           个不同的具体实例 action）, "example_actions": [具体实例样例，至多3个],
           "children": {normalized_action: node}}
        分叉点（children > 1）即决策分歧点，供云端对比归因。
        """
        root = {"action": "<root>", "count": 0, "n_success": 0,
                "n_failure": 0, "children": {}, "_variants": set()}
        for tr in traces:
            node = root
            outcome = tr.get("outcome", "failure")
            for step in tr.get("steps", []):
                raw_action = step.get("action", "") or ""
                key = _normalize_action_for_merge(raw_action)
                child = node["children"].get(key)
                if child is None:
                    child = {"action": key, "count": 0, "n_success": 0,
                             "n_failure": 0, "children": {}, "_variants": set()}
                    node["children"][key] = child
                child["count"] += 1
                child["_variants"].add(raw_action)
                if outcome == "success":
                    child["n_success"] += 1
                else:
                    child["n_failure"] += 1
                node = child
        self._finalize_variants(root)
        return root

    @staticmethod
    def _tree_evidence_codec(root: dict, traces: List[dict]) -> dict:
        """Build the compact cloud representation of a trajectory trie.

        The historical nested ``prefix_tree`` was an *additional* JSON object:
        flat trajectories remained in the batch and the tree repeated actions,
        counts, variants, and child-map keys.  This codec is the replacement
        sent to cloud prompt builders.  Version 3 interns normalized actions in
        a shared vocabulary because a large trie can contain hundreds of nodes
        but only a few dozen distinct action types.  Each node therefore stores
        an action id rather than repeating the action text, and each rollout
        stores only a path of node ids plus observation deltas and sparse
        non-zero rewards.  Task texts and exact per-task consensus prefixes are
        interned/summarized inside the codec so the cloud renderer never needs
        the local flat step arrays.  ``count`` is derivable as
        ``success + failure`` and instance variants are not useful to a skill
        author, so neither is serialized.
        """
        actions: List[str] = []
        action_ids: Dict[str, int] = {}
        tasks: List[str] = []
        task_ids: Dict[str, int] = {}
        nodes: List[List[object]] = []  # [parent_id, action_id, succ, fail]
        edge_ids: Dict[Tuple[int, str], int] = {}

        def intern_action(action: str) -> int:
            action_id = action_ids.get(action)
            if action_id is None:
                action_id = len(actions) + 1
                action_ids[action] = action_id
                actions.append(action)
            return action_id

        def intern_task(task: str) -> int:
            task_id = task_ids.get(task)
            if task_id is None:
                task_id = len(tasks) + 1
                task_ids[task] = task_id
                tasks.append(task)
            return task_id

        def visit(node: dict, parent_id: int) -> None:
            for action, child in (node.get("children") or {}).items():
                node_id = len(nodes) + 1
                edge_ids[(parent_id, action)] = node_id
                nodes.append([
                    parent_id,
                    intern_action(action),
                    int(child.get("n_success", 0) or 0),
                    int(child.get("n_failure", 0) or 0),
                ])
                visit(child, node_id)

        visit(root, 0)
        records = []
        for trace in traces:
            parent_id = 0
            path: List[int] = []
            for step in trace.get("steps", []) or []:
                action = _normalize_action_for_merge(step.get("action", "") or "")
                node_id = edge_ids.get((parent_id, action))
                if node_id is None:
                    # This cannot happen for a tree constructed from these
                    # traces, but retaining an explicit sentinel makes a
                    # malformed external trace auditable rather than silent.
                    break
                path.append(node_id)
                parent_id = node_id
            encoded_steps = []
            for step in trace.get("steps", []) or []:
                if "obs_delta" in step:
                    encoded_steps.append([
                        step.get("obs_delta", ""),
                        int(bool(step.get("obs_is_full", False))),
                    ])
                else:
                    encoded_steps.append([step.get("observation", ""), 1])
            steps = trace.get("steps", []) or []
            step_numbers = [
                int(step.get("step", index))
                for index, step in enumerate(steps, start=1)
            ]
            nonzero_rewards = [
                [step_numbers[index], step.get("reward", 0)]
                for index, step in enumerate(steps)
                if step.get("reward", 0)
            ]
            record = {
                "u": trace.get("traj_uid", ""),
                "t": trace.get("task_type", "unknown"),
                "o": "S" if trace.get("outcome") == "success" else "F",
                "g": intern_task(str(trace.get("task", "") or "")),
                "q": path,
                "x": encoded_steps,
            }
            if step_numbers != list(range(1, len(step_numbers) + 1)):
                record["k"] = step_numbers
            if nonzero_rewards:
                record["w"] = nonzero_rewards
            if trace.get("task_score") is not None:
                record["r"] = trace.get("task_score")
            if trace.get("dropped_loops"):
                record["d"] = int(trace.get("dropped_loops", 0) or 0)
            records.append(record)
        success_by_type: Dict[str, List[dict]] = defaultdict(list)
        for trace in traces:
            if trace.get("outcome") == "success":
                success_by_type[str(trace.get("task_type", "unknown"))].append(trace)
        consensus_by_type = {
            task_type: [
                _normalize_action_for_merge(action) for action in prefix
            ]
            for task_type, samples in success_by_type.items()
            if (prefix := longest_common_action_prefix(samples))
        }
        return {
            "version": 3,
            "mode": "tree_only",
            "actions": actions,
            "tasks": tasks,
            "nodes": nodes,
            "records": records,
            "consensus_by_type": consensus_by_type,
        }

    @staticmethod
    def _finalize_variants(node: dict) -> None:
        """Replace the transient ``_variants`` merge-time set with a JSON-safe
        summary (count + a few concrete examples) so callers can still ground
        a normalized branch label without re-serializing an unbounded set."""
        variants = sorted(node.pop("_variants", set()))
        node["n_variants"] = len(variants)
        node["example_actions"] = variants[:3]
        for child in node.get("children", {}).values():
            TracesPool._finalize_variants(child)

    # ------------------------------------------------------------------ #
    # 触发                                                                 #
    # ------------------------------------------------------------------ #

    def recent_failure_rate(self, task_type: str) -> float:
        outcomes = self._recent_outcomes.get(task_type)
        if not outcomes:
            return 0.0
        n_fail = sum(1 for o in outcomes if o == "failure")
        return n_fail / len(outcomes)

    def recent_success_rate(self, task_type: str) -> float:
        outcomes = self._recent_outcomes.get(task_type)
        if not outcomes:
            return 0.0
        n_success = sum(1 for o in outcomes if o == "success")
        return n_success / len(outcomes)

    def recent_success_trend(self, task_type: str) -> Tuple[Optional[str], float, float]:
        """Return (trend_reason, previous_half_sr, recent_half_sr).

        A trend is only reported when the per-task recent window has enough
        samples to split into two comparable halves. ``success_decline`` means
        the second half is materially worse than the first; ``success_stagnation``
        means success is not improving and remains below the near-perfect ceiling.
        """
        outcomes = list(self._recent_outcomes.get(task_type, []))
        if len(outcomes) < max(self.min_samples, 4):
            return None, 0.0, 0.0
        half = len(outcomes) // 2
        if half == 0 or len(outcomes) - half == 0:
            return None, 0.0, 0.0

        prev = outcomes[:half]
        recent = outcomes[half:]
        prev_sr = sum(1 for o in prev if o == "success") / len(prev)
        recent_sr = sum(1 for o in recent if o == "success") / len(recent)
        delta = recent_sr - prev_sr
        if delta <= -self.decline_delta:
            return "success_decline", prev_sr, recent_sr
        if (abs(delta) <= self.stagnation_delta
                and recent_sr < self.stagnation_success_ceiling):
            return "success_stagnation", prev_sr, recent_sr
        return None, prev_sr, recent_sr

    def has_min_samples(self, task_type: Optional[str] = None) -> bool:
        if task_type is not None:
            return len(self._recent_outcomes.get(task_type, [])) >= self.min_samples
        return self._n_added >= self.min_samples

    def should_trigger(self) -> Tuple[bool, Optional[str]]:
        """返回 (是否触发, 原因)。容量轨优先级低于表现/趋势轨。"""
        # 表现轨：任一 task_type 近期失败率超阈值且样本充分
        for task_type in self._recent_outcomes:
            if (self.has_min_samples(task_type)
                    and self.recent_failure_rate(task_type) >= self.perf_watermark):
                return True, "performance_watermark"
            trend_reason, _, _ = self.recent_success_trend(task_type)
            if trend_reason:
                return True, trend_reason
        # 容量轨
        if self._token_count >= self.capacity_watermark:
            return True, "capacity_watermark"
        return False, None

    # ------------------------------------------------------------------ #
    # 导出                                                                 #
    # ------------------------------------------------------------------ #

    def export_batch(self, task_type: Optional[str] = None,
                     trigger_reason: str = "manual") -> dict:
        """导出 CompressedBatch（§2.2）。task_type=None 时合并所有类型。

        导出后清空已导出的桶与水位线计数（近期 outcome 滑窗保留，避免触发抖动）。
        """
        if task_type is not None:
            types = [task_type]
        else:
            types = sorted(set(self._success) | set(self._failure))

        success_samples: List[dict] = []
        failure_samples: List[dict] = []
        for t in types:
            success_samples.extend(list(self._success.get(t, [])))
            failure_samples.extend(list(self._failure.get(t, [])))

        all_samples = success_samples + failure_samples
        n_succ, n_fail = len(success_samples), len(failure_samples)
        total = n_succ + n_fail
        # 共识前缀：成功轨迹一致的最长起始动作段（端侧已掌握，不需重教）。
        consensus = (longest_common_action_prefix(success_samples)
                     if self.enable_consensus_prefix else [])
        prefix_tree = self._merge_prefix_tree(all_samples) if self.enable_prefix_tree else None
        batch = {
            "batch_id": str(uuid.uuid4()),
            "trigger_reason": trigger_reason,
            "task_type": task_type or "ALL",
            "success_samples": success_samples,
            "failure_samples": failure_samples,
            "stats": {
                "n_success": n_succ,
                "n_failure": n_fail,
                "avg_success_rate": (n_succ / total) if total else 0.0,
                "dropped_loops_total": self._total_dropped_loops,
                "consensus_len": len(consensus),
            },
            "compression": {
                "enable_loop_filter": self.enable_loop_filter,
                "enable_obs_delta": self.enable_obs_delta,
                "enable_prefix_tree": self.enable_prefix_tree,
                "enable_consensus_prefix": self.enable_consensus_prefix,
                "cloud_evidence_mode": self.cloud_evidence_mode,
                "accounting": "chars_div_4",
                "trace_stage_totals": self._batch_trace_stage_totals(all_samples),
                "consensus_prefix": {
                    "steps": len(consensus),
                    "chars": sum(len(action) for action in consensus),
                    "tokens": sum(_approx_tokens(action) for action in consensus),
                },
            },
        }
        if self.enable_consensus_prefix:
            batch["consensus_prefix"] = consensus
        if self.enable_prefix_tree:
            # ``tree_evidence`` is the cloud-facing replacement encoding.  Do
            # not attach the old nested tree as a second payload: it was the
            # reason the previous "compression" arm could be larger than the
            # flat arm before any API request was made.
            batch["tree_evidence"] = self._tree_evidence_codec(prefix_tree, all_samples)
            batch["compression"]["prefix_tree"] = self._prefix_tree_stats(
                prefix_tree, self._batch_trace_stage_totals(all_samples)["encoded"]["steps"]
            )

        # 清空已导出桶
        for t in types:
            self._success.pop(t, None)
            self._failure.pop(t, None)
        self._token_count = 0
        self._n_added = 0

        if self.output_dir is not None:
            self._dump_batch(batch)
        return batch

    @staticmethod
    def project_cloud_batch(batch: dict) -> dict:
        """Return the exact batch allowed to cross the cloud boundary.

        Production CoSkill uses ``tree_only``: flat step arrays remain in the
        local ``traces_pool`` artifact for audit/resume, while CloudAnalyzer
        receives only lightweight rollout indexes plus the self-contained
        trajectory-tree codec.  This prevents prompt builders from
        accidentally serializing both representations or silently falling
        back to the flat trajectories.

        Compression-off ablations explicitly use ``flat`` and retain the
        historical full batch unchanged.
        """
        compression = batch.get("compression") or {}
        mode = str(compression.get("cloud_evidence_mode") or "flat")
        if mode == "flat":
            return batch
        if mode != "tree_only":
            raise ValueError(f"unsupported cloud evidence mode: {mode!r}")
        tree_evidence = batch.get("tree_evidence")
        if not isinstance(tree_evidence, dict) or not tree_evidence.get("records"):
            raise ValueError(
                "tree_only cloud evidence requires a non-empty tree_evidence codec"
            )

        def metadata_only(trace: dict) -> dict:
            return {
                "traj_uid": trace.get("traj_uid", ""),
                "task_type": trace.get("task_type", "unknown"),
                "outcome": trace.get("outcome", "failure"),
            }

        projected = dict(batch)
        projected["success_samples"] = [
            metadata_only(trace) for trace in (batch.get("success_samples") or [])
        ]
        projected["failure_samples"] = [
            metadata_only(trace) for trace in (batch.get("failure_samples") or [])
        ]
        projected["consensus_prefix"] = [
            _normalize_action_for_merge(action)
            for action in (batch.get("consensus_prefix") or [])
        ]
        projected["cloud_projection"] = {
            "mode": "tree_only",
            "flat_steps_uploaded": False,
            "codec_version": int(tree_evidence.get("version", 0) or 0),
        }
        return projected

    @staticmethod
    def _batch_trace_stage_totals(traces: List[dict]) -> dict:
        totals = {
            "raw": {"steps": 0, "chars": 0, "tokens": 0},
            "loop_filtered": {"steps": 0, "chars": 0, "tokens": 0},
            "encoded": {"steps": 0, "chars": 0, "tokens": 0},
        }
        for trace in traces:
            stat = trace.get("compression_trace_stats") or {}
            steps = {
                "raw": stat.get("raw_steps", 0),
                "loop_filtered": stat.get("loop_filtered_steps", 0),
                "encoded": stat.get("encoded_steps", 0),
            }
            for stage in totals:
                totals[stage]["steps"] += int(steps[stage] or 0)
                totals[stage]["chars"] += int((stat.get("chars") or {}).get(stage, 0) or 0)
                totals[stage]["tokens"] += int((stat.get("tokens") or {}).get(stage, 0) or 0)
        return totals

    @staticmethod
    def _prefix_tree_stats(root: Optional[dict], total_steps: int) -> dict:
        """Compact structural accounting for the prefix-tree compression arm."""
        if not root:
            return {"node_count": 0, "edge_count": 0, "merged_step_ratio": 0.0,
                    "chars": 0, "tokens": 0}
        nodes = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend((node.get("children") or {}).values())
        # Exclude synthetic root from semantic nodes/edges.
        semantic_nodes = max(0, len(nodes) - 1)
        text = json.dumps(root, ensure_ascii=False, sort_keys=True)
        return {
            "node_count": semantic_nodes,
            "edge_count": semantic_nodes,
            "merged_step_ratio": round(1.0 - semantic_nodes / max(total_steps, 1), 6),
            "chars": len(text), "tokens": _approx_tokens(text),
        }

    def _dump_batch(self, batch: dict) -> None:
        try:
            ts = int(time.time())
            path = os.path.join(self.output_dir, f"batch_{ts}_{batch['batch_id'][:8]}.json")
            with open(path, "w") as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)
            print(f"[TracesPool] exported CompressedBatch → {path} "
                  f"(succ={batch['stats']['n_success']}, fail={batch['stats']['n_failure']})")
        except Exception as e:
            print(f"[TracesPool] batch dump failed: {e}")

    # ------------------------------------------------------------------ #
    # 统计                                                                 #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "total_added": self._total_added,
            "pending_added": self._n_added,
            "pending_tokens": self._token_count,
            "total_dropped_loops": self._total_dropped_loops,
            "cloud_evidence_mode": self.cloud_evidence_mode,
            "task_types": sorted(set(self._success) | set(self._failure)),
            "recent_success_rate": {
                tt: self.recent_success_rate(tt)
                for tt in sorted(self._recent_outcomes)
            },
            "recent_failure_rate": {
                tt: self.recent_failure_rate(tt)
                for tt in sorted(self._recent_outcomes)
            },
        }
