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
        environment_name: str = "generic",
    ):
        from openai import AzureOpenAI, OpenAI  # 延迟导入，避免无依赖时整包不可用

        backend = os.environ.get("SKILL_UPDATER_BACKEND", "deepseek").lower()
        if backend == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            if not api_key or not endpoint:
                raise OSError("CloudAnalyzer (azure) requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
            self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
            self.model = os.environ.get("AZURE_OPENAI_MODEL", "o3")
        else:  # deepseek
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise OSError("CloudAnalyzer (deepseek) requires DEEPSEEK_API_KEY.")
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            )
            # 本期默认 DeepSeek V4 Flash
            self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

        self.max_completion_tokens = max_completion_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.environment_name = str(environment_name or "generic")
        # Normal production runs retain the legacy small prompt budget unless
        # an experiment explicitly opts in.  The trajectory-compression
        # ablation uses 10x so the cloud model sees near-complete evidence.
        try:
            self.evidence_budget_multiplier = max(1, int(os.environ.get("COSKILL_CLOUD_EVIDENCE_MULTIPLIER", "1")))
        except ValueError as exc:
            raise ValueError("COSKILL_CLOUD_EVIDENCE_MULTIPLIER must be a positive integer") from exc
        self.evidence_render_limits = {
            "multiplier": self.evidence_budget_multiplier,
            "contrastive_success_examples": 5 * self.evidence_budget_multiplier,
            "contrastive_failure_examples": 6 * self.evidence_budget_multiplier,
            "diagnose_success_examples": 3 * self.evidence_budget_multiplier,
            "diagnose_failure_examples": 6 * self.evidence_budget_multiplier,
            "tree_success_examples": 4 * self.evidence_budget_multiplier,
            "tree_failure_examples": 5 * self.evidence_budget_multiplier,
            "steps_per_trace": 12 * self.evidence_budget_multiplier,
            "observation_chars_per_step": 400 * self.evidence_budget_multiplier,
            "decision_forks": 6 * self.evidence_budget_multiplier,
            "branches_per_fork": 5 * self.evidence_budget_multiplier,
        }
        self.update_history: List[Dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # DeepSeek exposes the input split used for billing.  Keep it separate
        # from ``total_prompt_tokens``: prompt tokens are a usage measure,
        # whereas cache hit/miss tokens are the only safe inputs for a cache
        # aware cost estimate.
        self.total_prompt_cache_hit_tokens = 0
        self.total_prompt_cache_miss_tokens = 0
        # Per-task_type cloud token breakdown. Only evolve_playbook is cleanly
        # attributable to a single task_type (it's already called once per
        # task_type); contrastive_distill/diagnose_failures each mix every
        # task_type in one call and are tracked separately under *_mixed
        # instead of being force-split across task_types.
        self.total_prompt_tokens_by_task_type: Dict[str, int] = {}
        self.total_completion_tokens_by_task_type: Dict[str, int] = {}
        self.total_prompt_tokens_mixed = 0
        self.total_completion_tokens_mixed = 0
        # Provider responses occasionally omit ``usage``.  A missing usage
        # object is not a zero-token cloud call; retain it explicitly so cost
        # reports never present an unreported call as free.
        self.usage_reported_calls = 0
        self.usage_missing_calls = 0
        self.usage_missing_calls_by_task_type: Dict[str, int] = {}
        self.usage_missing_calls_mixed = 0
        self.cache_usage_reported_calls = 0
        self.cache_usage_missing_calls = 0
        # Skill-tree 进化 / 失败诊断的可观测计数（并入 get_update_summary）。
        self.playbook_history: List[Dict] = []
        self.n_diagnose_calls = 0
        self.n_evolve_calls = 0
        # Append-only metadata for audit/reporting.  Raw text remains in the
        # caller's artifact directory rather than being silently discarded.
        self.call_audit: List[Dict[str, Any]] = []

        self.output_dir = None
        self.playbook_io_dir = None
        if output_dir:
            self.output_dir = os.path.join(output_dir, "cloud_io")
            os.makedirs(self.output_dir, exist_ok=True)
            # Skill-tree 进化 / 失败诊断的产物与 skill 补丁同属"云端产物"，统一落在
            # cloud_io/ 下（不再单独开 playbook_io/ 目录）——playbook_io_dir 这个
            # 属性名保留，只是指向同一个目录，避免改动所有引用点。
            self.playbook_io_dir = self.output_dir

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
        prompt = self._build_contrastive_prompt(compressed_batch, current_skills, next_dyn_idx)
        if self.output_dir is not None:
            self._dump_text(self.output_dir, f"distill_prompt_{compressed_batch.get('batch_id', 'x')[:8]}.txt", prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            content = response.choices[0].message.content
            self._record_call("contrastive_distill", prompt, content, getattr(response, "usage", None))
            if self.output_dir is not None:
                self._dump_text(self.output_dir, f"distill_response_{compressed_batch.get('batch_id', 'x')[:8]}.txt", content)
            raw_skills = self._parse_patches(content)

            if hasattr(response, "usage") and response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                # Mixes every task_type in one call - not attributable to a
                # single subtask, so it's tracked as an honest "mixed" bucket.
                self.total_prompt_tokens_mixed += response.usage.prompt_tokens
                self.total_completion_tokens_mixed += response.usage.completion_tokens

            patches = self._normalize_patches(raw_skills, next_dyn_idx, compressed_batch)

            self.update_history.append(
                {
                    "batch_id": compressed_batch.get("batch_id"),
                    "trigger_reason": compressed_batch.get("trigger_reason"),
                    "n_success": len(success),
                    "n_failure": len(failure),
                    "num_patches": len(patches),
                    "skill_ids": [p.get("skill_id") for p in patches],
                }
            )

            patches = patches[: self.max_new_skills_per_update]
            if self.output_dir is not None:
                self._dump_patches(compressed_batch, patches)
            return patches

        except Exception as e:
            print(f"[CloudAnalyzer] Error calling {self.model}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # 失败原因分析（每周期一次批量诊断）                                    #
    # ------------------------------------------------------------------ #

    def diagnose_failures(
        self,
        compressed_batch: Dict[str, Any],
        *,
        task_type: Optional[str] = None,
        max_success_examples: Optional[int] = None,
        max_failure_examples: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """一次 LLM 调用，诊断本批**所有失败轨迹**的错误原因。

        以同类**成功 rollout**为参照、环境成功判据为正确性锚（不读取 oracle 或
        ground-truth 文件），给每条失败轨迹产出结构化诊断，按 task_type 分组返回。产物喂给
        :meth:`evolve_playbook`（``skill_tree_gap`` + ``patch_location`` 指出本可由
        skill tree 避免的错误、以及补丁该加在 skill tree 的哪个位置）。

        Returns:
            ``{task_type: [diagnosis, ...]}``；无失败样本时返回 ``{}``。
        """
        max_success_examples = self.evidence_render_limits["diagnose_success_examples"] if max_success_examples is None else max_success_examples
        max_failure_examples = self.evidence_render_limits["diagnose_failure_examples"] if max_failure_examples is None else max_failure_examples
        failures = compressed_batch.get("failure_samples", []) or []
        if not failures:
            return {}

        successes = compressed_batch.get("success_samples", []) or []
        fail_by_type = self._group_by_task_type(failures)
        succ_by_type = self._group_by_task_type(successes)

        prompt = self._build_diagnose_prompt(
            fail_by_type,
            succ_by_type,
            compressed_batch.get("tree_evidence"),
            max_success_examples=max_success_examples,
            max_failure_examples=max_failure_examples,
        )
        # 落盘发给云端大模型的原始 prompt（诊断产物本身已落盘，这里额外存"看到的输入"，
        # 便于核对共识折叠/分叉点是否真的按预期传给了大模型）。
        if self.playbook_io_dir is not None:
            self._dump_text(self.playbook_io_dir, f"diagnose_prompt_{compressed_batch.get('batch_id', 'x')[:8]}.txt", prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            content = response.choices[0].message.content
            self._record_call("diagnose_failures", prompt, content, getattr(response, "usage", None), task_type=task_type)
            if self.playbook_io_dir is not None:
                self._dump_text(self.playbook_io_dir, f"diagnose_response_{compressed_batch.get('batch_id', 'x')[:8]}.txt", content)
            if hasattr(response, "usage") and response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                if task_type:
                    self.total_prompt_tokens_by_task_type[task_type] = self.total_prompt_tokens_by_task_type.get(task_type, 0) + response.usage.prompt_tokens
                    self.total_completion_tokens_by_task_type[task_type] = self.total_completion_tokens_by_task_type.get(task_type, 0) + response.usage.completion_tokens
                else:
                    # Legacy callers can still submit a mixed batch.
                    self.total_prompt_tokens_mixed += response.usage.prompt_tokens
                    self.total_completion_tokens_mixed += response.usage.completion_tokens
            self.n_diagnose_calls += 1

            diags = self._parse_json_array(content)
            grouped: Dict[str, List[Dict]] = {}
            for d in diags:
                if not isinstance(d, dict):
                    continue
                tt = d.get("task_type") or "unknown"
                grouped.setdefault(tt, []).append(d)

            if self.playbook_io_dir is not None:
                self._dump_json(
                    self.playbook_io_dir,
                    f"diagnoses_{compressed_batch.get('batch_id', 'x')[:8]}.json",
                    {"batch_id": compressed_batch.get("batch_id"), "diagnoses": grouped},
                )
            return grouped
        except Exception as e:
            print(f"[CloudAnalyzer] diagnose_failures error ({self.model}): {e}")
            return {}

    def _build_diagnose_prompt(
        self,
        fail_by_type: Dict[str, List[Dict]],
        succ_by_type: Dict[str, List[Dict]],
        tree_evidence: Optional[Dict],
        *,
        max_success_examples: Optional[int] = None,
        max_failure_examples: Optional[int] = None,
    ) -> str:
        from .traces_pool import longest_common_action_prefix

        max_success_examples = self.evidence_render_limits["diagnose_success_examples"] if max_success_examples is None else max_success_examples
        max_failure_examples = self.evidence_render_limits["diagnose_failure_examples"] if max_failure_examples is None else max_failure_examples
        sections = []
        for tt, fails in fail_by_type.items():
            # 给每条失败轨迹一个稳定 ref: "<task_type>#<i>"
            fail_labeled = []
            for i, tr in enumerate(fails[:max_failure_examples]):
                tr = dict(tr)
                tr["_ref"] = f"{tt}#{i}"
                fail_labeled.append(tr)
            # 共识前缀（该类型成功轨迹一致的起始段）：端侧已掌握，折叠不重发。
            consensus = longest_common_action_prefix(succ_by_type.get(tt, []))
            succ_txt = self._format_difftraces(succ_by_type.get(tt, []), limit=max_success_examples, consensus=consensus, tree_evidence=tree_evidence)
            fail_txt = self._format_difftraces_reflabeled(fail_labeled, consensus=consensus, tree_evidence=tree_evidence)
            con_line = f"CONSENSUS PREFIX (the edge already masters these opening steps — the failure is NOT here, look at the DIVERGENCE after it): {' → '.join(consensus)}\n" if consensus else ""
            sections.append(f"=== task_type: {tt} ===\n{con_line}SUCCESSFUL ROLLOUT REFERENCE (observed successful behaviour):\n{succ_txt}\n\nFAILED trajectories to diagnose (each tagged [ref=...]):\n{fail_txt}")
        forks = self._format_forks(tree_evidence=tree_evidence)

        domain_context = self._domain_context()
        return f"""Role: You are an expert failure-analysis agent for sequential decision-making agent
tasks. The current environment is {self.environment_name}. Reason from the trajectories and the
environment contract below; do not import assumptions from a different benchmark.

ENVIRONMENT CONTRACT:
{domain_context}

Goal: For each FAILED trajectory, diagnose WHY it failed. Use successful rollouts of the same
task_type as observed references of successful behaviour, and the environment's success criterion
as the correctness anchor. They are not oracle demonstrations or ground-truth action plans. Each
step shows the action taken and the OBSERVATION DELTA
(+added / -removed lines).

{chr(10).join(sections)}

DECISION FORKS (where successful and failed runs diverged after a shared prefix):
{forks}

For EACH failed trajectory, identify the single causal failure reason, how it could be avoided, and
WHERE a corrective patch belongs in the agent's skill tree. The skill tree is a markdown TREE whose
heading depth is its hierarchy (a deeper heading refines its parent). So point the patch at a location
by naming the section heading, and say whether the fix belongs directly in it or as a deeper
refinement nested under it — not just "somewhere in the skill tree".
Return ONLY a JSON array. One object per failed trajectory, EXACTLY these fields:
  - "traj_ref":      the [ref=...] tag of the failed trajectory
  - "task_type":     its task_type
  - "failure_type":  one of "wrong_target" | "wrong_order" | "inefficient_exploration" | "loop" |
                     "premature_stop" | "invalid_action" | "misread_state" | "gave_up" | "other"
  - "root_cause":    ONE sentence, the causal reason it failed.
  - "evidence":      the step / observation that proves it (quote briefly).
  - "corrective_rule": ONE short imperative rule that would have prevented it.
  - "skill_tree_gap": what is missing/weak in the skill tree that let this error through.
  - "patch_location": WHERE the fix belongs — name the target markdown heading, and whether it should
                     be a new/deeper subsection ('##'/'###') under it. Use "new top-level section" if
                     no fitting heading exists yet.
  - "confidence":    a number 0.0-1.0.
Return ONLY the JSON array, no other text."""

    def _format_difftraces_reflabeled(self, traces: List[Dict], consensus: Optional[List[str]] = None, tree_evidence: Optional[Dict] = None) -> str:
        """同 _format_difftraces，但用每条轨迹的 _ref 作标签（供诊断引用）；折叠共识前缀。"""
        if not traces:
            return "(none)"
        if tree_evidence:
            return self._format_tree_coded_traces(traces, consensus=consensus, tree_evidence=tree_evidence, reflabeled=True)
        out = []
        for tr in traces:
            steps = tr.get("steps", [])
            fold = self._fold_count(steps, consensus)
            score = tr.get("task_score")
            score_text = f" task_score={score}" if score is not None else ""
            lines = [f"\n[ref={tr.get('_ref', '?')}]{score_text} task: {tr.get('task', '')}"]
            if fold:
                lines.append(f"  [consensus prefix ✓: {' → '.join(consensus[:fold])}]  (folded)")
            lines.extend(self._format_causal_transitions(steps, fold=fold))
            if tr.get("dropped_loops"):
                lines.append(f"  (dropped {tr['dropped_loops']} looping actions)")
            out.append("\n".join(lines))
        return "\n".join(out)

    # ------------------------------------------------------------------ #
    # Skill-tree 进化（大模型从零生成 + 层次化细化）                        #
    # ------------------------------------------------------------------ #

    def evolve_playbook(
        self,
        task_type: str,
        current_playbook: Optional[str],
        success_traces: List[Dict],
        failure_traces: List[Dict],
        diagnoses: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        target_depth: Optional[int] = None,
        repair_candidate: Optional[str] = None,
        repair_feedback: Optional[Dict[str, Any]] = None,
        tree_evidence: Optional[Dict] = None,
        max_success_examples: Optional[int] = None,
        max_failure_examples: Optional[int] = None,
        max_tree_nodes: Optional[int] = None,
        max_tree_chars: Optional[int] = None,
        preserve_parent_tree: bool = False,
        render_full_trajectories: bool = False,
    ) -> Optional[Dict]:
        """生成 / 细化该 agent 唯一的 skill tree（大模型从零撰写，层次化推进）。

        大模型先**批判**小模型是否看懂/用对当前 skill tree（结合失败诊断的
        ``skill_tree_gap``），再决定 keep / refine / rewrite，并在需要时细化执行
        指南、操作步骤或按需加 few-shot（层级 outline→detailed→fewshot）。

        ``history`` 保留在函数签名中仅为兼容旧调用；新 prompt 不再查看 previous
        versions，也不再要求模型比较前几版样式。

        Returns:
            ``{action, level, skill_tree, critique, changelog}``；``action="keep"``
            表示无需改动。LLM 出错或无内容时返回 ``None``（调用方保留旧版本）。
        """
        max_success_examples = self._evidence_limit("tree_success_examples", 4) if max_success_examples is None else max_success_examples
        max_failure_examples = self._evidence_limit("tree_failure_examples", 5) if max_failure_examples is None else max_failure_examples
        if not success_traces and not failure_traces:
            return None

        prompt = self._build_evolve_prompt(
            task_type,
            current_playbook,
            success_traces,
            failure_traces,
            diagnoses or [],
            history or [],
            target_depth=target_depth,
            repair_candidate=repair_candidate,
            repair_feedback=repair_feedback,
            tree_evidence=tree_evidence,
            max_success_examples=max_success_examples,
            max_failure_examples=max_failure_examples,
            max_tree_nodes=max_tree_nodes,
            max_tree_chars=max_tree_chars,
            preserve_parent_tree=preserve_parent_tree,
            render_full_trajectories=render_full_trajectories,
        )
        # 落盘发给云端大模型的原始 prompt（call 计数器区分同一 task_type 的多轮进化）。
        if self.playbook_io_dir is not None:
            self._dump_text(self.playbook_io_dir, f"evolve_skill_tree_{task_type}_call{self.n_evolve_calls:03d}.txt", prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            content = response.choices[0].message.content
            self._record_call("evolve_playbook", prompt, content, getattr(response, "usage", None), task_type=task_type)
            if self.playbook_io_dir is not None:
                self._dump_text(self.playbook_io_dir, f"evolve_skill_tree_response_{task_type}_call{self.n_evolve_calls:03d}.txt", content)
            if hasattr(response, "usage") and response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                # Cleanly attributable: this call is already scoped to task_type.
                self.total_prompt_tokens_by_task_type[task_type] = self.total_prompt_tokens_by_task_type.get(task_type, 0) + response.usage.prompt_tokens
                self.total_completion_tokens_by_task_type[task_type] = self.total_completion_tokens_by_task_type.get(task_type, 0) + response.usage.completion_tokens
            self.n_evolve_calls += 1

            obj = self._parse_json_object(content)
            if not obj:
                return None
            action = (obj.get("action") or "").lower().strip()
            tree_text = (obj.get("skill_tree") or obj.get("playbook") or "").strip()
            if action not in ("keep", "refine", "rewrite"):
                # 有内容默认视为 refine/rewrite，无内容视为 keep。
                action = "refine" if tree_text else "keep"
            from .traces_pool import longest_common_action_prefix

            consensus = longest_common_action_prefix(success_traces)
            result = {
                "action": action,
                "level": obj.get("level") or "outline",
                "skill_tree": tree_text,
                "critique": obj.get("critique") or "",
                "changelog": obj.get("changelog") or "",
                # V4 uses these auditable fields to reject unsupported
                # deepening rather than accepting a structurally valid but
                # evidence-free tree.
                "new_node_grounding": obj.get("new_node_grounding") or [],
                "unsupported_claims": obj.get("unsupported_claims") or [],
                # 观测用：本轮用于折叠的共识前缀 + 诊断原文，供节点树 debug 落盘引用。
                "consensus_prefix": consensus,
                "n_success": len(success_traces),
                "n_failure": len(failure_traces),
            }
            if target_depth is not None:
                result.update(
                    {
                        "target_depth": int(target_depth),
                        **self._validate_tree_depth(tree_text, int(target_depth), max_nodes=max_tree_nodes, max_chars=max_tree_chars),
                    }
                )
            self.playbook_history.append(
                {
                    "task_type": task_type,
                    "action": result["action"],
                    "level": result["level"],
                    "had_current": bool(current_playbook),
                }
            )
            return result
        except Exception as e:
            print(f"[CloudAnalyzer] evolve_playbook error ({self.model}, {task_type}): {e}")
            return None

    def _build_evolve_prompt(
        self,
        task_type: str,
        current_playbook: Optional[str],
        success_traces: List[Dict],
        failure_traces: List[Dict],
        diagnoses: List[Dict],
        history: Optional[List[Dict]] = None,
        target_depth: Optional[int] = None,
        repair_candidate: Optional[str] = None,
        repair_feedback: Optional[Dict[str, Any]] = None,
        tree_evidence: Optional[Dict] = None,
        max_success_examples: Optional[int] = None,
        max_failure_examples: Optional[int] = None,
        max_tree_nodes: Optional[int] = None,
        max_tree_chars: Optional[int] = None,
        preserve_parent_tree: bool = False,
        render_full_trajectories: bool = False,
    ) -> str:
        from .traces_pool import longest_common_action_prefix

        max_success_examples = self._evidence_limit("tree_success_examples", 4) if max_success_examples is None else max_success_examples
        max_failure_examples = self._evidence_limit("tree_failure_examples", 5) if max_failure_examples is None else max_failure_examples
        cur = (current_playbook or "").strip() or "(none — no skill tree yet for this goal family; write the FIRST version from scratch)"
        # A production update may fold a consensus prefix because the online
        # agent already mastered it.  A representation ablation must instead
        # expose every selected transition: otherwise a terminal action can
        # disappear while the prefix is still labelled as a success.
        consensus = [] if render_full_trajectories else longest_common_action_prefix(success_traces)
        succ_txt = self._format_difftraces(success_traces, limit=max_success_examples, consensus=consensus, tree_evidence=tree_evidence)
        fail_txt = self._format_difftraces(failure_traces, limit=max_failure_examples, consensus=consensus, tree_evidence=tree_evidence)
        diag_txt = self._format_diagnoses(diagnoses)
        con_txt = " → ".join(consensus) if consensus else "(none — no shared successful opening yet)"
        if render_full_trajectories:
            consensus_section = """
TRAJECTORY RENDERING: Every selected trajectory is rendered independently from its initial
pre-action state through its recorded terminal transition. No action prefix is folded, and no
opening action is presumed mastered merely because it appears in successful traces.
"""
        else:
            consensus_section = f"""
CONSENSUS PREFIX — the opening steps the agent ALREADY performs reliably (folded out of the traces
below): {con_txt}
The agent has MASTERED these steps. Do NOT spend skill-tree depth re-teaching them — keep that part of
the tree shallow (or drop redundant detail about it). Focus the skill tree on the DIVERGENCE that
follows, where success and failure split.
"""

        full_evidence_section = ""
        if render_full_trajectories:
            full_evidence_section = f"""

COMPLETE-EVIDENCE PROTOCOL (mandatory): The {len(success_traces)} successful and
{len(failure_traces)} failed selected trajectories below are the complete evidence set for this
task family. No consensus prefix is folded. Read every displayed transition through its terminal
reward. A missing post-observation is explicitly marked as unrecorded; NEVER infer its contents.
Absence of evidence is not evidence for a hidden environment mechanic. If the traces do not support
a proposed rule, omit it and list it under unsupported_claims instead of filling the gap with common
sense or a plausible-looking action sequence.
"""

        progressive_section = ""
        if preserve_parent_tree and current_playbook:
            progressive_section = f"""

MONOTONIC PROGRESSIVE-DEPTH PROTOCOL (mandatory): The CURRENT SKILL TREE is the accepted L{max(0, int(target_depth or 1) - 1)} parent. Produce its L{int(target_depth or 1)}
extension by INSERTING new evidence-grounded lines only.
- Preserve every non-empty parent line VERBATIM and in the same order.
- Never delete, rewrite, reorder, weaken, or contradict a parent rule.
- Attach every new deepest heading beneath a heading that already exists at the parent's deepest
  level; do not create a separate new root-to-leaf chain to satisfy the requested depth.
- Insert only headings at the newly requested deepest level and their body lines. Do not add new
  shallower headings or ungrounded body text elsewhere in the accepted parent.
- New depth must add conditional assistance, recovery guidance, or a finer evidence-backed decision;
  it must not merely rename or repeat a parent.
- If evidence cannot justify a deeper rule, put that proposal in unsupported_claims. Do not invent it.
"""
        grounding_schema = ""
        if preserve_parent_tree or render_full_trajectories:
            grounding_schema = """
  - "new_node_grounding": an array with one item for EVERY newly inserted heading at the requested
                 deepest level. Each item has exactly
                 {"heading_path": "Parent > Child", "evidence": [
                 {"traj_ref": "the exact traj_uid shown above", "step": 1}],
                 "supported_claim": "the precise rule supported by those transitions"}.
                 Evidence references must name displayed trajectories and real displayed step numbers.
  - "unsupported_claims": an array of mechanics or rules considered but not established by the
                 displayed evidence and explicit environment contract; use [] when none.
"""

        depth_constraint = ""
        depth_execution_override = ""
        if target_depth is not None:
            budget_terms = []
            if max_tree_nodes is not None:
                budget_terms.append(f"at most {int(max_tree_nodes)} semantic headings")
            if max_tree_chars is not None:
                budget_terms.append(f"at most {int(max_tree_chars)} characters")
            budget_constraint = " Also keep the complete tree to " + " and ".join(budget_terms) + "." if budget_terms else ""
            depth_constraint = f"""

EXPERIMENTAL DEPTH CONSTRAINT (mandatory): Return a non-empty skill_tree with
EXACTLY {int(target_depth)} semantic Markdown heading levels. Do NOT use a Markdown heading for a
document title: every Markdown heading is a semantic tree node, and the first semantic heading is
depth 1. Every heading level from 1 through {int(target_depth)} must occur, and no heading may be
deeper. Do not use empty, dummy, or purely structural headings just to meet the depth. The depth is
an experimental condition, so action=\"keep\" is not allowed.{budget_constraint}
"""
            depth_execution_override = f"""
EXPERIMENTAL OVERRIDE: The general \"start shallow\" guidance below does not
override this condition. This run MUST contain evidence-grounded semantic
nodes at each level 1 through {int(target_depth)}. If a candidate is too
shallow, deepen it by adding meaningful child decisions or preconditions; do
not merely rename or repeat a parent node.
"""
        repair_section = ""
        if repair_candidate:
            feedback = repair_feedback or {}
            actual_depth = feedback.get("actual_depth")
            target = int(target_depth) if target_depth is not None else None
            if isinstance(actual_depth, int) and target is not None and actual_depth < target:
                direction = f"The candidate is TOO SHALLOW: its deepest semantic heading is level {actual_depth}, but the required deepest level is {target}. DEEPEN it by adding evidence-grounded child decisions or preconditions."
            elif isinstance(actual_depth, int) and target is not None and actual_depth > target:
                direction = f"The candidate is TOO DEEP: its deepest semantic heading is level {actual_depth}, but the required deepest level is {target}. SHALLOW it by merging or lifting overly deep nodes without deleting their useful rules."
            elif feedback.get("protocol_validation_errors"):
                direction = "The candidate has the requested heading depth but violates the progressive extension, evidence-grounding, or environment-semantics protocol. Repair only the listed violations while preserving all accepted parent lines."
            else:
                direction = "The candidate has an invalid heading structure. Rebuild it so every required semantic heading level is present and no heading deeper than the target occurs."
            errors = list(feedback.get("depth_validation_errors") or []) + list(feedback.get("protocol_validation_errors") or [])
            error_text = ", ".join(str(x) for x in errors) or "unspecified_depth_error"
            repair_section = f"""

DEPTH-REPAIR CANDIDATE: The following candidate was generated from the SAME evidence but did not
meet the requested heading depth.
DEPTH VALIDATION: {direction}
VALIDATION ERRORS: {error_text}
Preserve the same-evidence grounding and do not add dummy headings.
\"\"\"
{repair_candidate.strip()}
\"\"\"
"""

        return f"""You are the SKILL TREE author and editor for one small language-model agent.
The current environment is {self.environment_name}.
ENVIRONMENT CONTRACT: {self._domain_context()}

This agent keeps ONE skill tree PER goal family / task type. The tree you are editing here is ONLY for
the current goal family "{task_type}" and will be read at the TOP of the agent's prompt when a future
task is detected as the same family. Do not mix in rules for unrelated goal families.

Although the tree is task-family-specific, keep the content GENERAL within that family: infer the
goal, sub-goals, state phases, decision bottlenecks, and failure modes from the trajectories. Do not
hard-code benchmark labels, exact sampled entity IDs, layouts, product IDs, or dataset-specific
category names as if they were the method. The skill tree should teach the agent how to assess its own
situation and what to do next — the goal, the decision flow, and the concrete actions — using only
what the prompt already contains, with no externally injected state hints.

CURRENT SKILL TREE (exactly what the agent was shown this round):
\"\"\"
{cur}
\"\"\"

{consensus_section}

NEW SUCCESSFUL TRAJECTORIES (what worked):
{succ_txt}

NEW FAILED TRAJECTORIES (what went wrong):
{fail_txt}

FAILURE DIAGNOSES (root cause + which skill-tree part was missing/weak):
{diag_txt}
{depth_constraint}
{depth_execution_override}
{repair_section}
{full_evidence_section}
{progressive_section}

Do these steps IN ORDER:

1. INDUCE THE REGULARITY (reason, don't just patch). Look ACROSS the trajectories and diagnoses and
   ask: what general regularity explains success vs failure? Combine two sources of evidence:
   (a) DATA INDUCTION — what do the successful runs consistently do that the failed ones do not, and
   at which decision point do they diverge? Prefer a pattern seen MULTIPLE times over a one-off.
   (b) VERIFIED ENVIRONMENT CONTRACT — use only the explicit environment contract above for mechanics,
   and use general reasoning only to organize rules already supported by trajectories. Do not use
   unstated common sense to invent a sub-goal order, object location, state transition, or precondition.
   Use verified evidence to GENERALIZE beyond the few sampled trajectories so the rule transfers.
   State every regularity as a GENERAL principle grounded in a stated reason — never an
   instance-specific fix tied to one concrete entity or location.

   EVIDENCE SAFETY: keep a transition CONDITIONAL when the evidence shows that its validity depends
   on the current observation or carried-object state. Do not promote an action order seen in one
   trajectory into a universal prerequisite. When successful traces disagree about an order, write an
   observation-conditioned decision rule instead; when the evidence does not establish a transition,
   tell the agent to inspect its current feedback/admissible action rather than inventing a mechanic.

2. USAGE CRITIQUE (skip if there is no current skill tree). Judge how the agent USED the current
   skill tree: did it follow it, misread it, or ignore a section? IMPORTANT — also check whether the
   skill tree ITSELF caused the failure: is any wording ambiguous, over-specific, contradictory, or
   misleading enough to push the agent into the wrong action? If a failure traces back to your own
   text, FIX THE TEXT (that is a higher-priority edit than adding new rules).

3. DECIDE THE ACTION:
   - No current skill tree -> action="rewrite": author the FIRST version from the induced regularities.
     Start SHALLOW (see step 4) — do NOT pre-emptively add depth the evidence has not shown to need.
   - Current skill tree works and shows NO avoidable failures -> action="keep".
   - Otherwise -> action="refine": change ONLY the section(s) the diagnoses point at (use each
     diagnosis's patch_location), including fixing your own misleading wording.

4. HIERARCHICAL MARKDOWN — this is the core format. Write the skill tree as a TREE, using markdown
   heading depth for the branches: a heading nested one level deeper than its parent (a '##' under a
   '#', a '###' under a '##', and so on) is a REFINEMENT of that parent — it elaborates, clarifies, or
   details what the parent says. YOU decide what every node contains and HOW MANY levels there are;
   there is no fixed meaning for any level and no fixed number of levels. Let the content decide the
   shape — go exactly as deep as the material needs and no deeper.
   Organize this task-family tree with clear categories or bullet-like branches when helpful (for
   example by goal phase, state-assessment phase, decision bottleneck, or recurring mistake), then
   break each branch down step by step only as far as evidence justifies.
   DEEPEN BY JUDGEMENT, NOT BY RULE. Keep every branch as SHALLOW as it can be while still working.
   Add a child heading under a section ONLY when the evidence — a recurring failure, a diagnosis's
   patch_location, or a misread — shows the agent did NOT grasp that parent at its current depth.
   Well-understood sections stay shallow; different branches may sit at different depths at once.
   Depth follows demonstrated need, branch by branch — never deepen the whole document uniformly.
   When a diagnosis names a patch_location, put the fix under exactly that heading (or create the new
   heading it asks for).

5. LAYOUT & CONSTRAINTS: lead with the goal, then order sections in the natural flow the agent acts
   (assess state -> choose the right sub-goal -> act -> recognize completion and stop). One idea per
   line; most decision-critical rule first within each section; no duplication or contradiction across
   sections. Keep it SHORT — a small model pays a thinking tax per line; spend depth only where it
   earns its keep.

Return ONLY one JSON object, EXACTLY these fields:
  - "action":    "keep" | "refine" | "rewrite"
  - "level":     the deepest heading depth present, as a number (1 = only '#', 2 = a '##' exists,
                 3 = a '###' exists, ...)
  - "skill_tree": the FULL new skill tree MARKDOWN (empty string if action="keep")
  - "critique":  1-3 sentences: how the agent used the skill tree AND whether the skill tree's own
                 wording misled it (or "" if there was no current skill tree)
  - "changelog": 1 sentence naming which section(s) you changed or deepened, and why
{grounding_schema}
Return ONLY the JSON object, no other text."""

    def _format_playbook_history(self, history: List[Dict]) -> str:
        """渲染最近至多 2 版历史：每版的 changelog + 当时要修的失败类型 + 内容摘要，
        供大模型判断上一次修改是否奏效。内容截断以省 token。"""
        if not history:
            return "(none — this is an early version, no prior edits to compare against)"
        out = []
        for h in history[-2:]:
            rf = h.get("round_failures") or []
            fail_types = ", ".join(sorted({d.get("failure_type", "?") for d in rf})) or "(unknown)"
            content = (h.get("content") or "").strip()
            snippet = content if len(content) <= 500 else content[:500] + " …"
            out.append(f"--- version {h.get('version', '?')} (level={h.get('level', '?')}) ---\n  changelog: {h.get('changelog', '')}\n  was meant to fix these failure types: {fail_types}\n  its content:\n{snippet}")
        return "\n".join(out)

    def _format_diagnoses(self, diagnoses: List[Dict]) -> str:
        if not diagnoses:
            return "(none)"
        out = []
        for d in diagnoses[:8]:
            out.append(f"- [{d.get('failure_type', '?')}] {d.get('root_cause', '')} | evidence: {d.get('evidence', '')} | fix: {d.get('corrective_rule', '')} | skill_tree_gap: {d.get('skill_tree_gap', d.get('playbook_gap', ''))} | patch_location: {d.get('patch_location', '(unspecified)')}")
        return "\n".join(out)

    @staticmethod
    def _group_by_task_type(traces: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for tr in traces:
            tt = tr.get("task_type") or "unknown"
            grouped.setdefault(tt, []).append(tr)
        return grouped

    def _dump_json(self, dir_path: str, fname: str, obj: Any) -> None:
        try:
            path = os.path.join(dir_path, fname)
            with open(path, "w") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            print(f"[CloudAnalyzer] wrote → {path}")
        except Exception as e:
            print(f"[CloudAnalyzer] json dump failed ({fname}): {e}")

    @staticmethod
    def _usage_field(usage: Any, name: str) -> Any:
        """Read an SDK or mapping usage field without assuming one provider."""
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    def _record_call(self, purpose: str, prompt: str, response: str, usage: Any, task_type: Optional[str] = None) -> None:
        """Record hashes/token usage without putting API text in normal metrics."""
        import hashlib

        usage_reported = usage is not None and (self._usage_field(usage, "prompt_tokens") is not None or self._usage_field(usage, "completion_tokens") is not None)
        prompt_tokens = int(self._usage_field(usage, "prompt_tokens") or 0) if usage_reported else None
        completion_tokens = int(self._usage_field(usage, "completion_tokens") or 0) if usage_reported else None
        raw_cache_hit = self._usage_field(usage, "prompt_cache_hit_tokens")
        raw_cache_miss = self._usage_field(usage, "prompt_cache_miss_tokens")
        cache_usage_reported = raw_cache_hit is not None and raw_cache_miss is not None
        cache_hit_tokens = int(raw_cache_hit or 0) if cache_usage_reported else None
        cache_miss_tokens = int(raw_cache_miss or 0) if cache_usage_reported else None
        if cache_usage_reported:
            self.cache_usage_reported_calls = int(getattr(self, "cache_usage_reported_calls", 0) or 0) + 1
            self.total_prompt_cache_hit_tokens = int(getattr(self, "total_prompt_cache_hit_tokens", 0) or 0) + cache_hit_tokens
            self.total_prompt_cache_miss_tokens = int(getattr(self, "total_prompt_cache_miss_tokens", 0) or 0) + cache_miss_tokens
        else:
            # Do not infer a cache split from total prompt tokens.  Treating a
            # missing provider field as all-cache-miss would fabricate a bill.
            self.cache_usage_missing_calls = int(getattr(self, "cache_usage_missing_calls", 0) or 0) + 1
        if usage_reported:
            self.usage_reported_calls += 1
        else:
            self.usage_missing_calls += 1
            if task_type:
                self.usage_missing_calls_by_task_type[task_type] = self.usage_missing_calls_by_task_type.get(task_type, 0) + 1
            else:
                self.usage_missing_calls_mixed += 1
        self.call_audit.append(
            {
                "purpose": purpose,
                "task_type": task_type,
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens + completion_tokens) if usage_reported else None,
                "usage_reported": usage_reported,
                "usage_status": "reported" if usage_reported else "missing",
                "prompt_cache_hit_tokens": cache_hit_tokens,
                "prompt_cache_miss_tokens": cache_miss_tokens,
                "cache_usage_reported": cache_usage_reported,
                "cache_usage_status": "reported" if cache_usage_reported else "missing",
                # Keep payload-size accounting alongside provider token usage.  The
                # latter is authoritative when available; chars/4 is only a
                # deterministic inspection aid for prompt construction.
                "prompt_chars": len(prompt),
                "prompt_bytes_utf8": len(prompt.encode("utf-8")),
                "prompt_tokens_chars_div_4": max(1, len(prompt) // 4) if prompt else 0,
                "evidence_render_limits": dict(getattr(self, "evidence_render_limits", {"multiplier": 1})),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_sha256": hashlib.sha256((response or "").encode("utf-8")).hexdigest(),
            }
        )

    def _dump_text(self, dir_path: str, fname: str, text: str) -> None:
        """落盘发给云端大模型的原始 prompt 原文（人类可读，纯文本）。

        与结果产物（patches/diagnoses/skill tree）分开存，专门用于核对"大模型到底
        看到了什么输入"——尤其是共识前缀折叠、决策分叉点、失败诊断是否真的按预期
        拼进了 prompt。调用放在 API 调用之前，即使请求失败也留得下这份记录。
        """
        try:
            path = os.path.join(dir_path, fname)
            with open(path, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"[CloudAnalyzer] prompt dump failed ({fname}): {e}")

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
        consensus = batch.get("consensus_prefix") or []
        tree_evidence = batch.get("tree_evidence")
        succ_txt = self._format_difftraces(batch.get("success_samples", []), limit=self.evidence_render_limits["contrastive_success_examples"], consensus=consensus, tree_evidence=tree_evidence)
        fail_txt = self._format_difftraces(batch.get("failure_samples", []), limit=self.evidence_render_limits["contrastive_failure_examples"], consensus=consensus, tree_evidence=tree_evidence)
        fork_txt = self._format_forks(tree_evidence=tree_evidence)

        existing_titles = [s.get("title", "") for s in current_skills.get("general_skills", [])]
        for tt, skills in current_skills.get("task_specific_skills", {}).items():
            for s in skills:
                existing_titles.append(f"[{tt}] {s.get('title', '')}")

        example_ids = ", ".join(f'"dyn_{next_dyn_idx + j:03d}"' for j in range(self.max_new_skills_per_update))

        observed_types = sorted({str(trace.get("task_type") or "unknown") for trace in ((batch.get("success_samples", []) or []) + (batch.get("failure_samples", []) or []))})
        allowed_types = ", ".join(f'"{tt}"' for tt in observed_types)
        example_task_type = observed_types[0] if observed_types else "unknown"
        if self.environment_name.lower() == "webshop":
            seed_example = (
                f'{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Verify Before Purchase", '
                f'"scope": "task_specific", "task_type": "{example_task_type}", '
                '"principle": "Before buying, verify the product text, price, and every required option; select each matching option first.", '
                '"when_to_apply": "On a product page before click[buy now]."}'
            )
        else:
            seed_example = (
                f'{{"skill_id": "dyn_{next_dyn_idx:03d}", "title": "Open Before Search", '
                f'"scope": "task_specific", "task_type": "{example_task_type}", '
                '"principle": "If the target may be inside a closed container, open each closed container before searching its contents.", '
                '"when_to_apply": "When the goal object is not visible and closed containers remain."}'
            )

        return f"""You are an expert at distilling sequential-agent experience into reusable skills.
The current environment is {self.environment_name}.
ENVIRONMENT CONTRACT: {self._domain_context()}

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
Match the concise format of the existing hand-written seed skills.

Return ONLY a JSON array. Each skill MUST have EXACTLY these fields (no extra fields):
  - "skill_id":      one of {example_ids}
  - "title":         3-5 word title
  - "scope":         "general" or "task_specific"
  - "task_type":     one of [{allowed_types}] when task_specific, or "" if general. Never use "ALL".
  - "principle":     ONE or TWO plain sentences stating the rule. Keep it under 30 words. No JSON, no lists.
  - "when_to_apply": ONE short sentence naming the situation that triggers this skill. Under 20 words.

Example:
[{seed_example}]

Return ONLY the JSON array, no other text."""

    def _domain_context(self) -> str:
        """Short action/reward contract injected into every cloud prompt."""
        env = self.environment_name.lower()
        if env == "webshop":
            return (
                "Actions are exactly search[query] or click[visible text]. Search queries are free-form; "
                "click targets must be visible/clickable. The goal is to purchase a product satisfying "
                "all requested attributes, options, and price. click[buy now] terminates; only task_score "
                "1.0 is a success, while a lower terminal score is a failed partial match. Never invent "
                "ALFWorld rooms, receptacles, or manipulation actions."
            )
        if env == "alfworld":
            return (
                "Actions must be selected from the current state's provided admissible household actions; "
                "an action observed in another state is not automatically legal now. Inventory capacity is "
                "one object, so two-object tasks must take and deliver objects sequentially. State-transform "
                "commands act on the object currently held: clean OBJECT with sinkbasin, heat OBJECT with "
                "microwave, and cool OBJECT with fridge. Do not first relinquish the object into the appliance "
                "and then assume the transform remains admissible. A look-at-light goal requires holding the "
                "target and using the relevant desklamp; it does not require placing the target on the lamp. "
                "Open a closed receptacle before taking from it, and move the transformed/held object to the "
                "requested destination only after required transformations. These are environment action "
                "semantics, not hints about object identities or sampled layouts. Success means the requested "
                "object manipulation/state goal is completed. Do not invent shopping search, product-option, "
                "purchase actions, object locations, or unobserved state transitions."
            )
        return "Infer legal actions and the success criterion only from the supplied goals, observations, actions, rewards, and successful reference trajectories."

    @staticmethod
    def _fold_count(steps: List[Dict], consensus: Optional[List[str]]) -> int:
        """一条轨迹起始有多少步与共识动作逐位相同（这些是端侧已掌握的共识，可折叠）。"""
        if not consensus:
            return 0
        n = 0
        for i, s in enumerate(steps):
            if i < len(consensus) and (s.get("action") or "") == consensus[i]:
                n += 1
            else:
                break
        return n

    def _evidence_limit(self, name: str, default: int) -> int:
        """Read a render limit safely for prompt-only unit callers."""
        limits = getattr(self, "evidence_render_limits", {}) or {}
        return max(1, int(limits.get(name, default) or default))

    @staticmethod
    def _step_observation(step: Dict[str, Any]) -> tuple[str, str]:
        """Return the observation payload and an explicit state label.

        RawTrace stores the observation *before* the action in the same step.
        Compressed traces preserve that timing but may store a delta/reference.
        Keeping timing separate from compression prevents cloud prompts from
        interpreting a pre-action observation as the action's result.
        """
        if "observation" in step and "obs_delta" not in step:
            return str(step.get("observation", "") or ""), "full_observation"
        label = "full_observation" if step.get("obs_is_full") else "observation_delta"
        return str(step.get("obs_delta", "") or ""), label

    def _format_causal_transitions(
        self,
        steps: List[Dict[str, Any]],
        *,
        fold: int = 0,
    ) -> List[str]:
        """Render each action between its actual pre/post states without duplication."""
        if fold >= len(steps):
            return ["  (all transitions folded)"]
        limit = self._evidence_limit("steps_per_trace", 12)
        obs_chars = self._evidence_limit("observation_chars_per_step", 400)
        end = min(len(steps), fold + limit)
        first_text, first_kind = self._step_observation(steps[fold])
        first_number = steps[fold].get("step", fold + 1)
        lines = [f"  state_before_step_{first_number} [{first_kind}]: {first_text[:obs_chars]}"]
        for index in range(fold, end):
            step = steps[index]
            number = step.get("step", index + 1)
            lines.append(f"  step {number} action: {step.get('action', '')}")
            lines.append(f"    reward_after_action: {step.get('reward', 0) or 0}")
            if index + 1 < len(steps) and index + 1 < end:
                post_text, post_kind = self._step_observation(steps[index + 1])
                next_number = steps[index + 1].get("step", index + 2)
                lines.append(f"    state_after_action_and_before_step_{next_number} [{post_kind}]: {post_text[:obs_chars]}")
            elif index + 1 < len(steps):
                lines.append("    state_after_action: (not displayed because this renderer reached its step limit; do not infer it)")
            else:
                lines.append("    state_after_action: (terminal post-observation was not recorded; use only reward/outcome and do not infer hidden state)")
        return lines

    def _format_tree_coded_traces(
        self,
        traces: List[Dict],
        *,
        consensus: Optional[List[str]],
        tree_evidence: Dict,
        reflabeled: bool = False,
    ) -> str:
        """Render trace evidence through a trie node table, not flat actions.

        A node action is printed once.  Each rollout then references it via a
        compact ``N1>N7>...`` path and carries only outcome-bound observation
        deltas.  This preserves success/failure attribution while removing
        repeated action prefixes from every cloud prompt.
        """
        records = {str(record.get("u", "")): record for record in (tree_evidence.get("records") or [])}
        selected = []
        for index, trace in enumerate(traces, start=1):
            record = records.get(str(trace.get("traj_uid", "")))
            if record is None:
                # Compatibility with batches produced before the codec.
                return self._format_difftraces_flat(traces, consensus, reflabeled)
            selected.append((index, trace, record))

        node_table = {index: node for index, node in enumerate(tree_evidence.get("nodes") or [], start=1)}
        needed_ids = set()
        rendered = []
        for index, trace, record in selected:
            fold = self._fold_count(trace.get("steps", []) or [], consensus)
            path = list(record.get("q") or [])[fold:]
            observations = list(record.get("x") or [])[fold:]
            needed_ids.update(path)
            score = trace.get("task_score")
            score_text = f" task_score={score}" if score is not None else ""
            label = f"[ref={trace.get('_ref', '?')}]" if reflabeled else f"Trajectory {index} [{trace.get('outcome', '?')}]"
            lines = [f"\n{label}{score_text} task: {trace.get('task', '')}"]
            if fold:
                lines.append(f"  [consensus prefix ✓: {' → '.join((consensus or [])[:fold])}] (folded)")
            lines.append("  path: " + (" > ".join(f"N{node_id}" for node_id in path) or "(fully folded)"))
            lines.append("  transition semantics: each state attached to N is BEFORE N's action; the following node's state is the observed state AFTER the previous action")
            for node_id, payload in zip(path, observations):
                obs_text = str(payload[0] if isinstance(payload, list) and payload else "")
                is_full = bool(payload[1]) if isinstance(payload, list) and len(payload) > 1 else False
                text_label = "full_observation" if is_full else "observation_delta"
                lines.append(f"  N{node_id} | state_before_action [{text_label}]: {obs_text[: self._evidence_limit('observation_chars_per_step', 400)]}")
            if trace.get("dropped_loops"):
                lines.append(f"  (dropped {trace['dropped_loops']} looping actions)")
            rendered.append("\n".join(lines))

        node_lines = ["ACTION NODE TABLE (each action is defined once; paths reference N-id):"]
        for node_id in sorted(needed_ids):
            node = node_table.get(node_id)
            if not isinstance(node, list) or len(node) < 4:
                continue
            parent, action, success_count, failure_count = node[:4]
            node_lines.append(f"N{node_id}: parent=N{parent}; action={action}; succ={success_count}; fail={failure_count}")
        return "\n".join(node_lines + rendered)

    def _format_difftraces_flat(
        self,
        traces: List[Dict],
        consensus: Optional[List[str]],
        reflabeled: bool = False,
    ) -> str:
        """Legacy flat renderer used only for mixed/old batches."""
        out = []
        for i, tr in enumerate(traces, start=1):
            steps = tr.get("steps", [])
            fold = self._fold_count(steps, consensus)
            score = tr.get("task_score")
            score_text = f" task_score={score}" if score is not None else ""
            label = f"[ref={tr.get('_ref', '?')}]" if reflabeled else f"Trajectory {i} [{tr.get('outcome', '?')}]"
            lines = [f"\n{label}{score_text} task: {tr.get('task', '')}"]
            if fold:
                lines.append(f"  [consensus prefix ✓: {' → '.join(consensus[:fold])}] (folded)")
            lines.extend(self._format_causal_transitions(steps, fold=fold))
            out.append("\n".join(lines))
        return "\n".join(out)

    def _format_difftraces(self, traces: List[Dict], limit: int, consensus: Optional[List[str]] = None, tree_evidence: Optional[Dict] = None) -> str:
        if not traces:
            return "(none)"
        selected = traces[:limit]
        if tree_evidence:
            return self._format_tree_coded_traces(selected, consensus=consensus, tree_evidence=tree_evidence)
        out = []
        for i, tr in enumerate(selected):
            steps = tr.get("steps", [])
            fold = self._fold_count(steps, consensus)
            score = tr.get("task_score")
            score_text = f" task_score={score}" if score is not None else ""
            lines = [f"\nTrajectory {i + 1} [{tr.get('outcome', '?')}] [ref={tr.get('traj_uid', '?')}]{score_text} task: {tr.get('task', '')}"]
            if fold:
                lines.append(f"  [consensus prefix ✓: {' → '.join(consensus[:fold])}]  (folded, already mastered)")
            lines.extend(self._format_causal_transitions(steps, fold=fold))
            if tr.get("dropped_loops"):
                lines.append(f"  (dropped {tr['dropped_loops']} looping actions)")
            out.append("\n".join(lines))
        return "\n".join(out)

    @staticmethod
    def _tree_depth(markdown: str) -> int:
        """Return semantic Markdown heading depth; the document title is not special.

        CoSkill trees use headings as their node representation.  A non-heading
        preface is ignored, and malformed heading jumps do not invent levels.
        """
        depths = [len(line) - len(line.lstrip("#")) for line in (markdown or "").splitlines() if line.startswith("#") and line.lstrip("#").startswith(" ")]
        return max(depths, default=0)

    @staticmethod
    def _validate_tree_depth(
        markdown: str,
        target_depth: int,
        *,
        max_nodes: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate a fixed-depth tree without locally editing cloud output.

        A maximum heading depth alone is not enough: a lone ``###`` heading
        must not pass a three-level condition.  Every Markdown heading is
        treated as a semantic tree node; the cloud prompt asks it to keep any
        document title as plain text.
        """
        target = int(target_depth)
        heading_levels, empty_levels, jumps = [], [], []
        previous_level = None
        for line in (markdown or "").splitlines():
            if not (line.startswith("#") and line.lstrip("#").startswith(" ")):
                continue
            level = len(line) - len(line.lstrip("#"))
            heading_levels.append(level)
            if previous_level is not None and level > previous_level + 1:
                jumps.append((previous_level, level))
            previous_level = level
            if not line[level:].strip():
                empty_levels.append(level)
        present = sorted(set(heading_levels))
        expected = list(range(1, target + 1))
        missing = [level for level in expected if level not in present]
        too_deep = [level for level in present if level > target]
        errors = []
        if not heading_levels:
            errors.append("no_semantic_headings")
        if missing:
            errors.append("missing_heading_levels:" + ",".join(map(str, missing)))
        if too_deep:
            errors.append("heading_deeper_than_target:" + ",".join(map(str, too_deep)))
        if empty_levels:
            errors.append("empty_heading_labels:" + ",".join(map(str, empty_levels)))
        if jumps:
            errors.append("heading_level_jumps:" + ",".join(f"{parent}->{child}" for parent, child in jumps))
        if max_nodes is not None and len(heading_levels) > int(max_nodes):
            errors.append(f"node_budget_exceeded:{len(heading_levels)}>{int(max_nodes)}")
        rendered_chars = len(markdown or "")
        if max_chars is not None and rendered_chars > int(max_chars):
            errors.append(f"character_budget_exceeded:{rendered_chars}>{int(max_chars)}")
        return {
            "actual_depth": max(heading_levels, default=0),
            "node_count": len(heading_levels),
            "rendered_chars": rendered_chars,
            "heading_levels_present": present,
            "missing_heading_levels": missing,
            "depth_validation_errors": errors,
            "depth_valid": not errors,
        }

    def _format_forks(
        self,
        prefix_tree: Optional[Dict] = None,
        max_forks: Optional[int] = None,
        tree_evidence: Optional[Dict] = None,
    ) -> str:
        """从前缀树中找出分叉节点（children>1），展示各分支的成功/失败计数。

        分支标签 ``a`` 是归一化后的 action（实例编号已折成 "#"，见
        ``traces_pool._merge_prefix_tree``），代表的是"去 cabinet 还是 drawer"
        这类真正的决策分歧，而不是"去 cabinet 3 还是 cabinet 7"这种同一决策
        下的不同具体实例。``n_variants``>1 时附上几个具体实例样例，只作为
        grounding 提示，不应被当成分支本身。
        """
        max_forks = self.evidence_render_limits["decision_forks"] if max_forks is None else max_forks
        max_branches = self.evidence_render_limits["branches_per_fork"]
        forks: List[str] = []

        if tree_evidence:
            nodes = tree_evidence.get("nodes") or []
            children: Dict[int, List[tuple[int, List]]] = {}
            for node_id, node in enumerate(nodes, start=1):
                if isinstance(node, list) and len(node) >= 4:
                    children.setdefault(int(node[0]), []).append((node_id, node))
            for parent_id, child_nodes in children.items():
                if len(child_nodes) <= 1:
                    continue
                branch_desc = "; ".join(f"N{node_id}='{node[1]}' (succ={node[2]},fail={node[3]})" for node_id, node in child_nodes[:max_branches])
                forks.append(f"After N{parent_id}: {branch_desc}")
                if len(forks) >= max_forks:
                    break
            return "\n".join(forks) if forks else "(no clear divergence point)"

        prefix_tree = prefix_tree or {}

        def branch_label(a: str, c: Dict) -> str:
            n_variants = c.get("n_variants", 1)
            examples = c.get("example_actions") or []
            hint = f" [{n_variants} instance variants, e.g. {', '.join(examples)}]" if n_variants > 1 and examples else ""
            return f"'{a}'{hint} (succ={c['n_success']},fail={c['n_failure']})"

        def walk(node, path):
            if len(forks) >= max_forks:
                return
            children = node.get("children", {})
            if len(children) > 1:
                branch_desc = "; ".join(branch_label(a, c) for a, c in list(children.items())[:max_branches])
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

    def _parse_json_array(self, response: str) -> List[Dict]:
        """宽松解析首个 JSON 数组（不要求含 title 字段），用于诊断输出。"""
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError as e:
            print(f"[CloudAnalyzer] JSON array parse error: {e}")
        return []

    def _parse_json_object(self, response: str) -> Optional[Dict]:
        """解析首个 JSON 对象，用于 evolve_playbook 输出。"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                obj = json.loads(response[start:end])
                return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError as e:
            print(f"[CloudAnalyzer] JSON object parse error: {e}")
        return None

    def _normalize_patches(
        self,
        skills: List[Dict],
        start_idx: int,
        batch: Dict[str, Any],
    ) -> List[Dict]:
        """重分配 dyn_ ID、补全兼容字段（principle/when_to_apply）、附 evidence。"""
        n_succ = len(batch.get("success_samples", []))
        n_fail = len(batch.get("failure_samples", []))
        observed_types = {str(trace.get("task_type") or "unknown") for trace in ((batch.get("success_samples", []) or []) + (batch.get("failure_samples", []) or []))}
        out: List[Dict] = []
        for i, s in enumerate(skills):
            patch = dict(s)
            patch["skill_id"] = f"dyn_{start_idx + i:03d}"
            action_flow = patch.get("action_flow") or []
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
            if patch.get("scope") == "task_specific":
                patch_type = str(patch.get("task_type") or "")
                if patch_type not in observed_types:
                    # A mixed export has task_type=ALL; never create an
                    # unreachable `task_specific_skills["ALL"]` bucket.
                    if len(observed_types) == 1:
                        patch["task_type"] = next(iter(observed_types))
                    else:
                        patch["scope"] = "general"
                        patch["task_type"] = ""
            patch["evidence"] = {"from_success": n_succ, "from_failure": n_fail}
            out.append(patch)
        return out

    def _dump_patches(self, batch: Dict, patches: List[Dict]) -> None:
        try:
            bid = batch.get("batch_id", "unknown")[:8]
            path = os.path.join(self.output_dir, f"patches_{bid}.json")
            with open(path, "w") as f:
                json.dump({"batch_id": batch.get("batch_id"), "patches": patches}, f, ensure_ascii=False, indent=2)
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
        if not self.update_history and not self.playbook_history:
            return {"total_updates": 0, "total_patches": 0}
        n_kept = sum(1 for h in self.playbook_history if h.get("action") == "keep")
        # Per-task_type call/update counts, same attributability split as the
        # token counters above: evolve_playbook is already one call per
        # task_type, so this is a plain group-by rather than a heuristic.
        evolve_calls_by_task_type: Dict[str, int] = {}
        skill_tree_updates_by_task_type: Dict[str, int] = {}
        for h in self.playbook_history:
            tt = h.get("task_type") or "unknown"
            evolve_calls_by_task_type[tt] = evolve_calls_by_task_type.get(tt, 0) + 1
            if h.get("action") != "keep":
                skill_tree_updates_by_task_type[tt] = skill_tree_updates_by_task_type.get(tt, 0) + 1
        return {
            "total_updates": len(self.update_history),
            "total_patches": sum(h["num_patches"] for h in self.update_history),
            "all_skill_ids": [sid for h in self.update_history for sid in h["skill_ids"]],
            "large_model_prompt_tokens": self.total_prompt_tokens,
            "large_model_completion_tokens": self.total_completion_tokens,
            "large_model_total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "large_model_prompt_cache_hit_tokens": int(getattr(self, "total_prompt_cache_hit_tokens", 0) or 0),
            "large_model_prompt_cache_miss_tokens": int(getattr(self, "total_prompt_cache_miss_tokens", 0) or 0),
            "large_model_cache_usage_reported_calls": int(getattr(self, "cache_usage_reported_calls", 0) or 0),
            "large_model_cache_usage_missing_calls": int(getattr(self, "cache_usage_missing_calls", 0) or 0),
            # Per-task_type breakdown (evolve_playbook only - cleanly
            # attributable) plus an honest "mixed" bucket for calls that
            # cannot be split by task_type (contrastive_distill,
            # diagnose_failures). See CloudAnalyzer.__init__ for rationale.
            "large_model_prompt_tokens_by_task_type": dict(self.total_prompt_tokens_by_task_type),
            "large_model_completion_tokens_by_task_type": dict(self.total_completion_tokens_by_task_type),
            "large_model_prompt_tokens_mixed": self.total_prompt_tokens_mixed,
            "large_model_completion_tokens_mixed": self.total_completion_tokens_mixed,
            "large_model_total_tokens_mixed": self.total_prompt_tokens_mixed + self.total_completion_tokens_mixed,
            "large_model_usage_reported_calls": int(getattr(self, "usage_reported_calls", 0) or 0),
            "large_model_usage_missing_calls": int(getattr(self, "usage_missing_calls", 0) or 0),
            "large_model_usage_missing_calls_by_task_type": dict(getattr(self, "usage_missing_calls_by_task_type", {}) or {}),
            "large_model_usage_missing_calls_mixed": int(getattr(self, "usage_missing_calls_mixed", 0) or 0),
            # skill-tree 进化 / 失败诊断可观测
            "diagnose_calls": self.n_diagnose_calls,
            "evolve_calls": self.n_evolve_calls,
            "skill_tree_updates": self.n_evolve_calls - n_kept,
            "playbook_updates": self.n_evolve_calls - n_kept,
            "evolve_calls_by_task_type": evolve_calls_by_task_type,
            "skill_tree_updates_by_task_type": skill_tree_updates_by_task_type,
        }
