







""

import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple







_INSTANCE_INDEX_RE = re.compile(r"\b\d+\b")
_WEBSHOP_ACTION_RE = re.compile(
    r"^\s*(search|click)\s*\[(.*)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_action_for_merge(action: str) -> str:
    ""
    raw = (action or "").strip()
    webshop = _WEBSHOP_ACTION_RE.fullmatch(raw)
    if webshop:
        verb = webshop.group(1).lower()
        payload = re.sub(r"\s+", " ", webshop.group(2)).strip()
        return f"{verb}[{payload}]"
    return _INSTANCE_INDEX_RE.sub("#", raw)










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
    ""
    if not text:
        return 0
    return max(1, len(text) // 4)


def longest_common_action_prefix(traces: List[dict]) -> List[str]:
    ""
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
    ""

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
        ""
        self.capacity_watermark = capacity_watermark
        self.perf_watermark = perf_watermark
        self.min_samples = min_samples
        self.loop_threshold = loop_threshold
        self.recent_window = recent_window
        self.max_keep_per_type = max_keep_per_type
        self.stagnation_delta = stagnation_delta
        self.decline_delta = decline_delta
        self.stagnation_success_ceiling = stagnation_success_ceiling



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


        self._success: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_keep_per_type))
        self._failure: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_keep_per_type))

        self._recent_outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=recent_window))

        self._token_count = 0
        self._n_added = 0
        self._total_added = 0
        self._total_dropped_loops = 0

        self.output_dir = None
        if output_dir:
            self.output_dir = os.path.join(output_dir, "traces_pool")
            os.makedirs(self.output_dir, exist_ok=True)

            self._raw_log_path = os.path.join(self.output_dir, "raw_traces.jsonl")
        else:
            self._raw_log_path = None





    def add_trace(self, raw_trace: dict) -> None:
        ""
        task_type = raw_trace.get("task_type", "unknown")
        outcome = raw_trace.get("outcome")
        if outcome not in ("success", "failure"):

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


            "task_type": task_type,
            "outcome": outcome,
            "episode_reward": raw_trace.get("episode_reward", 0),



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
        except Exception as e:
            print(f"[TracesPool] raw log write failed: {e}")





    def _filter_loops(self, steps: List[dict]) -> Tuple[List[dict], int]:
        ""
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
                    continue
            else:
                run_action = action
                run_len = 1
            cleaned.append(s)
            prev_obs = obs
        return cleaned, dropped

    def _diff_compress(self, steps: List[dict]) -> List[dict]:
        ""

        min_len_to_diff = getattr(self, "diff_min_obs_chars", 400)

        min_savings_ratio = getattr(self, "diff_min_savings", 0.5)
        diff_steps: List[dict] = []
        prev_lines: List[str] = []
        prev_action = "(episode start)"
        seen_obs: Dict[str, str] = {}
        for s in steps:
            obs = _strip_known_boilerplate(s.get("observation") or "")
            obs_clean = obs.strip()
            cur_lines = [ln.strip() for ln in obs.splitlines() if ln.strip()]
            cur_action = (s.get("action") or "").strip()





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

                "obs_delta": delta if worth_diff else obs_clean,
                "obs_is_full": not worth_diff,
                "reward": s.get("reward", 0) or 0,
            })




            if obs_clean and not worth_diff and obs_clean not in seen_obs:
                seen_obs[obs_clean] = prev_action
            prev_lines = cur_lines
            prev_action = cur_action
        return diff_steps

    @staticmethod
    def _full_observation_steps(steps: List[dict]) -> List[dict]:
        ""
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
        ""
        prev_set = set(prev_lines)
        cur_set = set(cur_lines)
        added = [ln for ln in cur_lines if ln not in prev_set]
        removed = [ln for ln in prev_lines if ln not in cur_set]
        parts = [f"+{ln}" for ln in added] + [f"-{ln}" for ln in removed]

        return " | ".join(parts) if parts else "(no change)"

    def _merge_prefix_tree(self, traces: List[dict]) -> dict:
        ""
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
        ""
        actions: List[str] = []
        action_ids: Dict[str, int] = {}
        tasks: List[str] = []
        task_ids: Dict[str, int] = {}
        nodes: List[List[object]] = []
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
        ""
        variants = sorted(node.pop("_variants", set()))
        node["n_variants"] = len(variants)
        node["example_actions"] = variants[:3]
        for child in node.get("children", {}).values():
            TracesPool._finalize_variants(child)





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
        ""
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
        ""

        for task_type in self._recent_outcomes:
            if (self.has_min_samples(task_type)
                    and self.recent_failure_rate(task_type) >= self.perf_watermark):
                return True, "performance_watermark"
            trend_reason, _, _ = self.recent_success_trend(task_type)
            if trend_reason:
                return True, trend_reason

        if self._token_count >= self.capacity_watermark:
            return True, "capacity_watermark"
        return False, None





    def export_batch(self, task_type: Optional[str] = None,
                     trigger_reason: str = "manual") -> dict:
        ""
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




            batch["tree_evidence"] = self._tree_evidence_codec(prefix_tree, all_samples)
            batch["compression"]["prefix_tree"] = self._prefix_tree_stats(
                prefix_tree, self._batch_trace_stage_totals(all_samples)["encoded"]["steps"]
            )


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
        ""
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
        ""
        if not root:
            return {"node_count": 0, "edge_count": 0, "merged_step_ratio": 0.0,
                    "chars": 0, "tokens": 0}
        nodes = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend((node.get("children") or {}).values())

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
