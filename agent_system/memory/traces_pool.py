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
import time
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


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
        cleaned, dropped = self._filter_loops(steps)
        diff_steps = self._diff_compress(cleaned)

        diff_trace = {
            "traj_uid": raw_trace.get("traj_uid", str(uuid.uuid4())),
            "task": raw_trace.get("task", ""),
            # task_type 随压缩轨迹一并保留：export_batch 默认合并所有类型
            # (task_type="ALL"),下游 playbook 进化需按 task_type 分组成功/失败样本。
            "task_type": task_type,
            "outcome": outcome,
            "steps": diff_steps,
            "dropped_loops": dropped,
            "skill_ids_used": (raw_trace.get("meta") or {}).get("skill_ids_used", []),
        }

        bucket = self._success if outcome == "success" else self._failure
        bucket[task_type].append(diff_trace)
        self._recent_outcomes[task_type].append(outcome)

        # 水位线计数：用压缩后的 token 估计
        tok = sum(_approx_tokens(s.get("obs_delta", "")) + _approx_tokens(s.get("action", ""))
                  for s in diff_steps)
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
        """状态压缩：把相邻 observation 的增量写入 obs_delta。

        但差分只在"划算"时才做——观测够长且差分能省下足够多内容时才用 +/-
        增量，否则直接保留完整观测原文。短观测做差分反而让云端大模型读起来更
        费劲（只看到零散的 +/- 行而非完整场景），得不偿失。
        """
        # 仅当完整观测超过该长度才考虑差分（短观测直接存原文）。
        min_len_to_diff = getattr(self, "diff_min_obs_chars", 400)
        # 且差分结果需至少比原文短这个比例，才认为划算（否则存原文）。
        min_savings_ratio = getattr(self, "diff_min_savings", 0.5)
        diff_steps: List[dict] = []
        prev_lines: List[str] = []
        for s in steps:
            obs = s.get("observation") or ""
            cur_lines = [ln.strip() for ln in obs.splitlines() if ln.strip()]
            delta = self._line_delta(prev_lines, cur_lines)
            obs_clean = obs.strip()
            # 是否值得差分：观测够长 + 差分确实省了 >= min_savings_ratio。
            worth_diff = (
                len(obs_clean) >= min_len_to_diff
                and delta != "(no change)"
                and len(delta) <= len(obs_clean) * (1.0 - min_savings_ratio)
            )
            diff_steps.append({
                "action": (s.get("action") or "").strip(),
                # 划算则存增量，否则存完整观测（云端直接读原文）。
                "obs_delta": delta if worth_diff else obs_clean,
                "obs_is_full": not worth_diff,
                "reward": s.get("reward", 0) or 0,
            })
            prev_lines = cur_lines
        return diff_steps

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

        节点 schema:
          {"action": str, "count": int, "n_success": int, "n_failure": int,
           "children": {action: node}}
        分叉点（children > 1）即决策分歧点，供云端对比归因。
        """
        root = {"action": "<root>", "count": 0, "n_success": 0,
                "n_failure": 0, "children": {}}
        for tr in traces:
            node = root
            outcome = tr.get("outcome", "failure")
            for step in tr.get("steps", []):
                action = step.get("action", "")
                child = node["children"].get(action)
                if child is None:
                    child = {"action": action, "count": 0, "n_success": 0,
                             "n_failure": 0, "children": {}}
                    node["children"][action] = child
                child["count"] += 1
                if outcome == "success":
                    child["n_success"] += 1
                else:
                    child["n_failure"] += 1
                node = child
        return root

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
        consensus = longest_common_action_prefix(success_samples)
        batch = {
            "batch_id": str(uuid.uuid4()),
            "trigger_reason": trigger_reason,
            "task_type": task_type or "ALL",
            "success_samples": success_samples,
            "failure_samples": failure_samples,
            "consensus_prefix": consensus,
            "stats": {
                "n_success": n_succ,
                "n_failure": n_fail,
                "avg_success_rate": (n_succ / total) if total else 0.0,
                "dropped_loops_total": self._total_dropped_loops,
                "consensus_len": len(consensus),
            },
            "prefix_tree": self._merge_prefix_tree(all_samples),
        }

        # 清空已导出桶
        for t in types:
            self._success.pop(t, None)
            self._failure.pop(t, None)
        self._token_count = 0
        self._n_added = 0

        if self.output_dir is not None:
            self._dump_batch(batch)
        return batch

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
