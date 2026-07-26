"""ALFWorld V4 progressive L0--L5 skill-tree depth experiment.

V4 removes V3's online growth phase.  It imports one frozen external corpus,
selects exactly 6 success + 6 failure trajectories per task family, and exposes
all selected transitions to the cloud without consensus folding or per-trace
truncation.  L1 is a minimal evidence-grounded tree.  L2--L5 are monotonic
extensions of the accepted previous level: every parent line must remain
verbatim and only new, cited semantic children may be inserted.

No node-count or character-count ceiling is applied.  Structural depth,
parent preservation, evidence references, and ALFWorld action semantics are
validated before an artifact becomes evaluation-eligible.  All arms use one
held-out fixed manifest and never update skills during evaluation.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.memory.hierarchical_skill_lib import HierarchicalSkillLib
from examples.playbook_evolve import fixed_trajectory_ablation as fixed
from mini_test_pen_shelf.env_utils import find_games_by_type

ARMS = tuple(f"skill_level_l{level}" for level in range(6))
TREE_DEPTHS = tuple(range(1, 6))
DEFAULT_INITIAL_TRACES_PER_TYPE = 12
DEFAULT_GENERATION_ATTEMPTS = 20


def _is_success(trace: dict[str, Any]) -> bool:
    return trace.get("outcome") == "success" or float(trace.get("episode_reward", 0) or 0) > 0


def select_initial_evidence(raw_path: Path, destination: Path, per_task: int) -> Path:
    """Select one deterministic, balanced, auditable evidence set."""
    if per_task < 2 or per_task % 2:
        raise ValueError("--initial_traces_per_type must be an even integer >= 2")
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"success": [], "failure": []})
    for trace in fixed._read_jsonl(raw_path):
        bucket = "success" if _is_success(trace) else "failure"
        by_type[str(trace.get("task_type", "unknown"))][bucket].append(trace)

    half = per_task // 2
    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "protocol": "v4_balanced_full_causal_evidence",
        "per_task": per_task,
        "success_per_task": half,
        "failure_per_task": half,
        "all_selected_traces_enter_every_tree_level": True,
        "consensus_prefix_folding": False,
        "step_truncation": False,
        "observation_truncation": False,
        "trajectory_compression_for_tree_authoring": False,
        "task_types": {},
    }
    for task_type in fixed.RUNTIME_TASK_TYPES:
        task_audit = {}
        for outcome in ("success", "failure"):
            available = by_type[task_type][outcome]
            if len(available) < half:
                raise ValueError(f"external corpus lacks {half} {outcome} traces for {task_type}; found {len(available)}")
            chosen, selection = fixed._stratified_trace_sample(
                available,
                half,
                salt=f"v4_initial:{task_type}:{outcome}",
            )
            selected.extend(chosen)
            task_audit[outcome] = selection
        audit["task_types"][task_type] = task_audit

    fixed._write_jsonl(destination, selected)
    fixed._write_json(destination.with_suffix(".selection.json"), audit)
    return destination


def create_eval_manifest(
    root: Path,
    data_root: Path,
    split: str,
    seed: int,
    games_per_type: int,
) -> Path:
    """Create one shared held-out manifest for every L0--L5 arm."""
    path = root / "manifests" / "eval_games.json"
    if path.exists():
        manifest = fixed._read_json(path)
        if manifest.get("games_per_task_type") != games_per_type or manifest.get("split") != split:
            raise RuntimeError("existing V4 evaluation manifest uses different settings; use a new --root")
        return path

    games: list[dict[str, str]] = []
    for offset, task_type in enumerate(fixed.TASK_TYPES):
        sampled = find_games_by_type(
            task_type,
            alfworld_data=str(data_root),
            split=split,
            sample_n=games_per_type,
            sample_seed=seed + offset,
            verbose=False,
        )
        if len(sampled) < games_per_type:
            raise RuntimeError(f"need {games_per_type} evaluation games for {task_type}, found {len(sampled)}")
        for index, (game_file, _) in enumerate(sampled, start=1):
            games.append(
                {
                    "label": f"eval_{task_type}_{index}",
                    "task_type": task_type,
                    "game_file": fixed._relative_game(game_file, data_root),
                }
            )
    resolved = [os.path.realpath(data_root / game["game_file"]) for game in games]
    if len(resolved) != len(set(resolved)):
        raise RuntimeError("V4 evaluation manifest contains duplicate game files")
    fixed._write_json(
        path,
        {
            "split": split,
            "role": "v4_shared_held_out_eval",
            "games_per_task_type": games_per_type,
            "games": games,
        },
    )
    return path


def _heading_paths(markdown: str, target_depth: int) -> list[str]:
    stack: list[tuple[int, str]] = []
    paths: list[str] = []
    for line in (markdown or "").splitlines():
        if not (line.startswith("#") and line.lstrip("#").startswith(" ")):
            continue
        depth = len(line) - len(line.lstrip("#"))
        label = line[depth:].strip()
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, label))
        if depth == target_depth:
            paths.append(" > ".join(value for _, value in stack))
    return paths


def _parent_is_verbatim_subsequence(parent: str, child: str) -> bool:
    """Allow inserted child branches but reject any parent edit or reordering."""
    parent_lines = [line.rstrip() for line in (parent or "").splitlines() if line.strip()]
    child_lines = [line.rstrip() for line in (child or "").splitlines() if line.strip()]
    position = 0
    for line in child_lines:
        if position < len(parent_lines) and line == parent_lines[position]:
            position += 1
    return position == len(parent_lines)


def _progressive_insertion_errors(parent: str, child: str, target_depth: int) -> list[str]:
    """Require every inserted line to belong to a newly added next-depth node."""
    parent_lines = [line.rstrip() for line in (parent or "").splitlines() if line.strip()]
    child_lines = [line.rstrip() for line in (child or "").splitlines() if line.strip()]
    parent_position = 0
    current_heading_depth = 0
    errors: list[str] = []
    for line in child_lines:
        heading_depth = len(line) - len(line.lstrip("#")) if line.startswith("#") else 0
        if heading_depth and line[heading_depth:].startswith(" "):
            current_heading_depth = heading_depth
        if parent_position < len(parent_lines) and line == parent_lines[parent_position]:
            parent_position += 1
            continue
        if heading_depth:
            if heading_depth != target_depth:
                errors.append(f"new_heading_not_exact_next_depth:{heading_depth}!={target_depth}:{line}")
        elif current_heading_depth != target_depth:
            errors.append(f"inserted_content_not_under_new_depth_{target_depth}_heading:{line}")
    return errors


def _grounding_errors(
    tree: str,
    depth: int,
    grounding: Any,
    traces: Iterable[dict[str, Any]],
) -> list[str]:
    """Verify that each newly introduced deepest heading cites shown evidence."""
    required = _heading_paths(tree, depth)
    if len(required) != len(set(required)):
        return ["duplicate_deepest_heading_path"]
    if not isinstance(grounding, list):
        return ["new_node_grounding_not_array"]
    trace_steps = {str(trace.get("traj_uid", "")): {int(step.get("step", index) or index) for index, step in enumerate(trace.get("steps") or [], start=1)} for trace in traces}
    supplied: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index, item in enumerate(grounding):
        if not isinstance(item, dict):
            errors.append(f"grounding_item_{index}_not_object")
            continue
        path = " > ".join(str(item.get("heading_path", "")).split(" > "))
        supplied[path] = item
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"grounding_missing_evidence:{path or index}")
            continue
        valid_reference = False
        for reference in evidence:
            if not isinstance(reference, dict):
                continue
            uid = str(reference.get("traj_ref", ""))
            try:
                step = int(reference.get("step"))
            except (TypeError, ValueError):
                continue
            if uid in trace_steps and step in trace_steps[uid]:
                valid_reference = True
            else:
                errors.append(f"grounding_unknown_reference:{uid}:step{step}")
        if not valid_reference:
            errors.append(f"grounding_has_no_valid_reference:{path or index}")
        if not str(item.get("supported_claim", "")).strip():
            errors.append(f"grounding_missing_supported_claim:{path or index}")
    for path in required:
        if path not in supplied:
            errors.append(f"deepest_heading_not_grounded:{path}")
    return errors


def _affirmative_contract_text(markdown: str) -> str:
    """Drop explicit warnings so a correct 'never do X' rule is not rejected."""
    negative = re.compile(r"\b(?:do not|don't|never|avoid|must not|should not)\b", re.I)
    return "\n".join(line for line in (markdown or "").splitlines() if not negative.search(line)).lower()


def validate_alfworld_tree_semantics(task_type: str, markdown: str) -> list[str]:
    """Reject known-impossible mechanics for every ALFWorld task family.

    This is a generic environment action contract, not a sampled-game oracle:
    it contains no object identities, locations, or held-out answers.
    """
    text = _affirmative_contract_text(markdown)
    errors: list[str] = []

    dual_inventory = re.compile(
        r"(?:both|two)\s+(?:target\s+)?objects?.{0,50}(?:inventory|carried|holding)"
        r"|(?:inventory|carry|hold).{0,50}(?:both|two)\s+(?:target\s+)?objects?",
        re.S,
    )
    if dual_inventory.search(text):
        errors.append("alfworld_inventory_capacity_one_violation")

    transform_contracts = {
        "heat": ("heat", "microwave"),
        "cool": ("cool", "fridge"),
        "clean": ("clean", "sinkbasin"),
    }
    if task_type in transform_contracts:
        action, appliance = transform_contracts[task_type]
        if action not in text or appliance not in text:
            errors.append(f"missing_required_transform_contract:{action}_with_{appliance}")
        relinquish_then_transform = re.compile(
            rf"(?:put|place|move).{{0,100}}(?:into|inside|to).{{0,40}}{appliance}"
            rf".{{0,180}}(?:then|next|after|before).{{0,60}}{action}",
            re.S,
        )
        if relinquish_then_transform.search(text):
            errors.append(f"transform_after_relinquishing_object_violation:{action}:{appliance}")

    if task_type == "look_at_obj_in_light":
        if "use" not in text or ("desklamp" not in text and "desk lamp" not in text):
            errors.append("missing_use_desklamp_contract")
        if re.search(
            r"(?:put|place|move).{0,100}(?:onto|on|into|to).{0,50}(?:desklamp|desk lamp)",
            text,
            re.S,
        ):
            errors.append("look_task_must_not_place_object_on_lamp")

    if task_type == "pick_two_obj_and_place":
        sequential_terms = (
            "one at a time",
            "sequential",
            "first object",
            "each object",
            "after placing",
            "return for the second",
            "repeat for the second",
            "before taking the second",
            "before picking up the second",
            "deliver it before",
            "place it before",
        )
        if not any(term in text for term in sequential_terms):
            errors.append("pick_two_missing_sequential_single_inventory_rule")

    if task_type == "pick_and_place" and not (("take" in text or "pick up" in text) and ("move" in text or "place" in text)):
        errors.append("pick_and_place_missing_take_then_deliver_contract")
    return errors


def _protocol_validation(
    task_type: str,
    parent: str | None,
    candidate: str,
    depth: int,
    grounding: Any,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    depth_result = CloudAnalyzer._validate_tree_depth(candidate, depth)
    errors = list(depth_result.get("depth_validation_errors", []))
    if parent:
        if not _parent_is_verbatim_subsequence(parent, candidate):
            errors.append("accepted_parent_tree_was_modified_or_reordered")
        if len(candidate.strip()) <= len(parent.strip()):
            errors.append("progressive_extension_added_no_assistance")
        errors.extend(_progressive_insertion_errors(parent, candidate, depth))
        accepted_parent_paths = set(_heading_paths(parent, depth - 1))
        for child_path in _heading_paths(candidate, depth):
            ancestor_path = " > ".join(child_path.split(" > ")[:-1])
            if ancestor_path not in accepted_parent_paths:
                errors.append(f"deepest_heading_not_attached_to_accepted_parent:{child_path}")
    errors.extend(_grounding_errors(candidate, depth, grounding, traces))
    errors.extend(validate_alfworld_tree_semantics(task_type, candidate))
    return {
        **depth_result,
        "protocol_validation_errors": sorted(set(errors)),
        "protocol_valid": not errors,
    }


def _configure_full_evidence(analyzer: CloudAnalyzer, traces: list[dict[str, Any]]) -> dict[str, Any]:
    max_steps = max((len(trace.get("steps") or []) for trace in traces), default=1)
    max_observation_chars = max(
        (len(str(step.get("observation", "") or "")) for trace in traces for step in (trace.get("steps") or [])),
        default=1,
    )
    analyzer.evidence_render_limits.update(
        {
            "tree_success_examples": len([trace for trace in traces if _is_success(trace)]),
            "tree_failure_examples": len([trace for trace in traces if not _is_success(trace)]),
            "steps_per_trace": max_steps,
            "observation_chars_per_step": max_observation_chars,
        }
    )
    return {
        "selected_traces": len(traces),
        "selected_traj_uids": [str(trace.get("traj_uid", "")) for trace in traces],
        "selected_success_traj_uids": [str(trace.get("traj_uid", "")) for trace in traces if _is_success(trace)],
        "selected_failure_traj_uids": [str(trace.get("traj_uid", "")) for trace in traces if not _is_success(trace)],
        "max_steps_per_selected_trace": max_steps,
        "max_observation_chars": max_observation_chars,
    }


def build_progressive_tree_artifacts(
    raw_path: Path,
    root: Path,
    *,
    max_attempts: int,
    max_completion_tokens: int,
) -> dict[str, Path]:
    """Build L1--L5 sequentially from one frozen evidence set."""
    raw = fixed._read_jsonl(raw_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in raw:
        grouped[str(trace.get("task_type", "unknown"))].append(trace)

    artifacts = root / "artifacts"
    result: dict[str, Path] = {}
    l0_manifest = artifacts / "skill_level_l0" / "artifact_manifest.json"
    result["skill_level_l0"] = l0_manifest if l0_manifest.exists() else fixed.build_l0_artifact(raw_path, l0_manifest.parent)

    for depth in TREE_DEPTHS:
        arm = f"skill_level_l{depth}"
        directory = artifacts / arm
        manifest_path = directory / "artifact_manifest.json"
        if manifest_path.exists():
            result[arm] = manifest_path
            continue

        parent_skills = fixed._read_json(artifacts / f"skill_level_l{depth - 1}" / "skills.json") if depth > 1 else fixed._empty_skill_bank()
        empty_path = directory / "empty_skills.json"
        fixed._write_json(empty_path, fixed._empty_skill_bank())
        lib = HierarchicalSkillLib(str(empty_path), retrieval_mode="template", enable_playbook=True)
        analyzer = CloudAnalyzer(
            output_dir=str(directory),
            environment_name="ALFWorld",
            max_completion_tokens=max_completion_tokens,
        )
        all_level_calls_start = len(analyzer.call_audit)
        status: dict[str, dict[str, Any]] = {}
        accounting: dict[str, Any] = {}

        for task_type in fixed.RUNTIME_TASK_TYPES:
            evidence = grouped[task_type]
            success = [trace for trace in evidence if _is_success(trace)]
            failure = [trace for trace in evidence if not _is_success(trace)]
            limits = _configure_full_evidence(analyzer, evidence)
            parent = ""
            if depth > 1:
                parent = str(((parent_skills.get("skill_trees", {}) or {}).get(task_type) or {}).get("content", ""))
                if not parent:
                    status[task_type] = {
                        "status": "blocked_missing_parent",
                        "attempts": 0,
                        "protocol_validation_errors": ["missing_accepted_parent_tree"],
                    }
                    continue

            candidate: dict[str, Any] | None = None
            validation: dict[str, Any] = {}
            repair_candidate: str | None = None
            attempts = 0
            while attempts < max_attempts:
                candidate = analyzer.evolve_playbook(
                    task_type=task_type,
                    current_playbook=parent or None,
                    success_traces=success,
                    failure_traces=failure,
                    diagnoses=[],
                    target_depth=depth,
                    repair_candidate=repair_candidate,
                    repair_feedback=(
                        {
                            "actual_depth": validation.get("actual_depth"),
                            "depth_validation_errors": validation.get("depth_validation_errors", []),
                            "protocol_validation_errors": validation.get("protocol_validation_errors", []),
                        }
                        if validation
                        else None
                    ),
                    max_success_examples=len(success),
                    max_failure_examples=len(failure),
                    max_tree_nodes=None,
                    max_tree_chars=None,
                    preserve_parent_tree=bool(parent),
                    render_full_trajectories=True,
                )
                attempts += 1
                tree = str((candidate or {}).get("skill_tree", "") or "")
                validation = _protocol_validation(
                    task_type,
                    parent or None,
                    tree,
                    depth,
                    (candidate or {}).get("new_node_grounding"),
                    evidence,
                )
                if candidate and validation["protocol_valid"]:
                    break
                repair_candidate = tree or repair_candidate

            if not candidate or not validation.get("protocol_valid"):
                status[task_type] = {
                    "status": "generation_failed",
                    "attempts": attempts,
                    "actual_depth": validation.get("actual_depth"),
                    "depth_validation_errors": validation.get("depth_validation_errors", []),
                    "protocol_validation_errors": validation.get("protocol_validation_errors", ["cloud_no_result"]),
                }
                continue

            record = lib.update_playbook(
                task_type,
                candidate["skill_tree"],
                candidate.get("level", depth),
                {
                    "generation_protocol": "v4_monotonic_progressive_depth",
                    "target_depth": depth,
                    "actual_depth": validation.get("actual_depth"),
                    "parent_depth": depth - 1 if parent else None,
                    "parent_content_sha256": fixed._sha256_text(parent) if parent else None,
                    "depth_generation_attempts": attempts,
                    "new_node_grounding": candidate.get("new_node_grounding", []),
                    "unsupported_claims": candidate.get("unsupported_claims", []),
                    "critique": candidate.get("critique", ""),
                    "changelog": candidate.get("changelog", ""),
                },
            )
            tree_stats = fixed._tree_stats(record["content"])
            parent_stats = fixed._tree_stats(parent) if parent else {"node_count": 0, "text_chars": 0}
            accounting[task_type] = {
                "full_tree": fixed._tree_node_token_accounting(record["content"]),
                "parent_chars": len(parent),
                "child_chars": len(record["content"]),
                "added_chars": len(record["content"]) - len(parent),
                "parent_nodes": parent_stats.get("node_count", 0),
                "child_nodes": tree_stats.get("node_count", 0),
                "added_nodes": tree_stats.get("node_count", 0) - parent_stats.get("node_count", 0),
                "evidence": limits,
            }
            status[task_type] = {
                "status": "ok",
                "version": record["version"],
                "attempts": attempts,
                "actual_depth": validation.get("actual_depth"),
                "parent_preserved_verbatim": True,
                "grounded_deepest_headings": len(_heading_paths(record["content"], depth)),
                "unsupported_claims": candidate.get("unsupported_claims", []),
            }

        failed = {task_type: info for task_type, info in status.items() if info.get("status") != "ok"}
        fixed._write_json(directory / "generation_status.json", status)
        fixed._write_json(directory / "tree_increment_accounting.json", accounting)
        result[arm] = fixed._save_artifact_manifest(
            directory,
            arm,
            raw_path,
            lib.skills,
            analyzer.call_audit[all_level_calls_start:],
            {
                "experiment_version": "v4",
                "generation_protocol": "same_tree_monotonic_progressive_extension",
                "skill_level": f"L{depth}",
                "target_depth": depth,
                "parent_arm": f"skill_level_l{depth - 1}" if depth > 1 else None,
                "tree_generation_status": status,
                "tree_generation_max_attempts": max_attempts,
                "tree_max_nodes": None,
                "tree_max_chars": None,
                "hard_tree_size_limits": False,
                "online_growth": False,
                "evidence_protocol": {
                    "selected_per_task": len(grouped[fixed.RUNTIME_TASK_TYPES[0]]),
                    "all_selected_traces_enter_prompt": True,
                    "full_steps": True,
                    "full_observations": True,
                    "causal_state_action_state_rendering": True,
                    "consensus_prefix_folding": False,
                },
                "validation_protocol": {
                    "exact_depth": True,
                    "parent_lines_verbatim_subsequence": True,
                    "deepest_heading_evidence_references": True,
                    "alfworld_semantic_contract_all_six_tasks": True,
                },
                "tree_increment_accounting": accounting,
                "status": "N.A." if failed else "ready",
                "evaluation_eligible": not bool(failed),
                "unavailable_reason": "v4_generation_or_protocol_validation_failed" if failed else None,
                "failed_task_types": sorted(failed),
            },
        )
    return result


def evaluate_arm(args: argparse.Namespace, root: Path, manifest: Path, arm: str) -> Path:
    output = root / "arms" / arm
    summary = output / "summary.json"
    if summary.exists():
        _audit_evaluation_context(root, arm, summary)
        return summary
    artifact_manifest = fixed._read_json(root / "artifacts" / arm / "artifact_manifest.json")
    if not artifact_manifest.get("evaluation_eligible", True):
        fixed._write_json(
            summary,
            {
                "status": "N.A.",
                "arm": arm,
                "evaluation_skipped": True,
                "reason": artifact_manifest.get("unavailable_reason"),
                "failed_task_types": artifact_manifest.get("failed_task_types", []),
                "total_episodes": 0,
                "wins": 0,
                "success_rate": None,
            },
        )
        return summary

    level = int(arm.rsplit("l", 1)[1])
    game_count = len(fixed._read_json(manifest)["games"])
    episodes = game_count * args.eval_rollouts_per_game
    command = fixed._driver_cmd(
        args,
        output,
        manifest,
        episodes,
        min(args.batch_rollout_size, episodes),
        int(level == 0),
        int(level > 0),
        0,
        0,
        [
            "--max_model_len",
            str(args.local_max_model_len),
            "--enable_hierarchy",
            str(int(level > 0)),
            "--skills_json",
            str(root / "artifacts" / arm / "skills.json"),
        ],
    )
    fixed._run(command, args.project_root)
    _audit_evaluation_context(root, arm, summary)
    return summary


def _audit_evaluation_context(root: Path, arm: str, summary_path: Path) -> None:
    """Invalidate an arm if the local model silently trimmed any prompt."""
    summary = fixed._read_json(summary_path)
    guard = summary.get("context_guard", {}) or {}
    prompt_trims = int(guard.get("prompt_trims", 0) or 0)
    trimmed_tokens = int(guard.get("trimmed_tokens", 0) or 0)
    valid = prompt_trims == 0
    summary["evaluation_protocol_valid"] = valid
    summary["evaluation_protocol_error"] = None if valid else f"local_context_guard_trimmed_prompts:{prompt_trims}:tokens:{trimmed_tokens}"
    fixed._write_json(summary_path, summary)
    if valid:
        return
    artifact_path = root / "artifacts" / arm / "artifact_manifest.json"
    artifact = fixed._read_json(artifact_path)
    artifact.update(
        {
            "status": "N.A.",
            "evaluation_eligible": False,
            "unavailable_reason": "local_prompt_context_trim_detected",
            "context_guard": guard,
        }
    )
    fixed._write_json(artifact_path, artifact)


def write_generation_metrics(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        path = root / "artifacts" / arm / "artifact_manifest.json"
        if not path.exists():
            continue
        artifact = fixed._read_json(path)
        calls = artifact.get("cloud_calls", []) or []
        row = {
            "arm": arm,
            "target_tree_depth": artifact.get("target_depth"),
            "status": artifact.get("status"),
            "online_growth": False,
            "cloud_calls": len(calls),
            "cloud_prompt_tokens": sum(int(call.get("prompt_tokens", 0) or 0) for call in calls),
            "cloud_completion_tokens": sum(int(call.get("completion_tokens", 0) or 0) for call in calls),
            "cloud_total_tokens": sum(int(call.get("total_tokens", 0) or 0) for call in calls),
            "cloud_usage_missing_calls": sum(int(call.get("usage_reported") is False) for call in calls),
            "tree_nodes": sum(int(info.get("node_count", 0) or 0) for info in (artifact.get("skill_trees", {}) or {}).values()),
            "tree_chars": sum(int(info.get("text_chars", 0) or 0) for info in (artifact.get("skill_trees", {}) or {}).values()),
        }
        rows.append(row)
        for task_type in fixed.RUNTIME_TASK_TYPES:
            scoped = [call for call in calls if call.get("task_type") == task_type]
            status = (artifact.get("tree_generation_status", {}) or {}).get(task_type, {})
            increment = (artifact.get("tree_increment_accounting", {}) or {}).get(task_type, {})
            task_rows.append(
                {
                    "arm": arm,
                    "target_tree_depth": artifact.get("target_depth"),
                    "task_type": task_type,
                    "status": status.get("status", artifact.get("status")),
                    "generation_attempts": status.get("attempts"),
                    "parent_preserved_verbatim": status.get("parent_preserved_verbatim"),
                    "grounded_deepest_headings": status.get("grounded_deepest_headings"),
                    "added_nodes": increment.get("added_nodes"),
                    "added_chars": increment.get("added_chars"),
                    "cloud_calls": len(scoped),
                    "cloud_prompt_tokens": sum(int(call.get("prompt_tokens", 0) or 0) for call in scoped),
                    "cloud_completion_tokens": sum(int(call.get("completion_tokens", 0) or 0) for call in scoped),
                    "cloud_total_tokens": sum(int(call.get("total_tokens", 0) or 0) for call in scoped),
                    "protocol_validation_errors": status.get("protocol_validation_errors", []),
                    "unsupported_claims": status.get("unsupported_claims", []),
                }
            )
    fixed._write_jsonl(root / "generation_metrics.jsonl", rows)
    fixed._write_jsonl(root / "generation_metrics_by_task.jsonl", task_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--external_raw_traces", required=True)
    parser.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument(
        "--phase",
        choices=("prepare", "evaluate", "summary", "all"),
        default="all",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial_traces_per_type",
        type=int,
        default=DEFAULT_INITIAL_TRACES_PER_TYPE,
    )
    parser.add_argument("--eval_games_per_type", type=int, default=3)
    parser.add_argument("--eval_rollouts_per_game", type=int, default=12)
    parser.add_argument("--batch_rollout_size", type=int, default=72)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument(
        "--local_max_model_len",
        type=int,
        default=int(os.environ.get("V4_LOCAL_MAX_MODEL_LEN", "16384")),
    )
    parser.add_argument(
        "--tree_generation_attempts",
        type=int,
        default=DEFAULT_GENERATION_ATTEMPTS,
    )
    parser.add_argument(
        "--tree_max_completion_tokens",
        type=int,
        default=int(os.environ.get("V4_TREE_MAX_COMPLETION_TOKENS", "8192")),
    )
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--retrieval_mode", choices=("template", "embedding"), default="template")
    parser.add_argument("--data_parallel_workers", type=int, default=1)
    parser.add_argument("--rollout_worker_gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("VLLM_ENFORCE_EAGER", "0")),
    )
    parser.add_argument("--log_trajectories", type=int, default=0)
    parser.add_argument("--driver_arg", action="append", default=[])
    args = parser.parse_args()

    if not args.alfworld_data:
        parser.error("--alfworld_data or ALFWORLD_DATA is required")
    if args.initial_traces_per_type != 12:
        parser.error("formal V4 requires exactly 12 initial traces per task")
    if (
        min(
            args.eval_games_per_type,
            args.eval_rollouts_per_game,
            args.batch_rollout_size,
            args.tree_generation_attempts,
            args.tree_max_completion_tokens,
            args.local_max_model_len,
        )
        < 1
    ):
        parser.error("V4 sizes and generation budgets must be positive")
    if args.local_max_model_len <= 4096:
        parser.error("--local_max_model_len must exceed the fixed 4096-token response budget")

    args.project_root = Path(__file__).resolve().parents[2]
    args.rollouts_per_type = args.eval_rollouts_per_game
    args.eval_groups_per_level = 1
    root = Path(args.root).resolve()
    data_root = Path(args.alfworld_data).resolve()
    root.mkdir(parents=True, exist_ok=True)

    imported = fixed.import_external_raw_traces(Path(args.external_raw_traces), root)
    evidence = select_initial_evidence(
        imported,
        root / "frozen" / "initial_evidence.jsonl",
        args.initial_traces_per_type,
    )
    eval_manifest = create_eval_manifest(
        root,
        data_root,
        args.split,
        args.sample_seed,
        args.eval_games_per_type,
    )
    config = {
        "experiment_kind": "alfworld_skill_tree_depth_v4",
        "arms": list(ARMS),
        "task_types": fixed.TASK_TYPES,
        "runtime_task_types": fixed.RUNTIME_TASK_TYPES,
        "protocol": {
            "online_growth": False,
            "same_tree_progressive_levels": True,
            "parent_lines_preserved_verbatim": True,
            "initial_external_traces_per_type": 12,
            "initial_success_per_type": 6,
            "initial_failure_per_type": 6,
            "all_selected_traces_enter_every_level": True,
            "complete_causal_transitions": True,
            "consensus_prefix_folding": False,
            "tree_size_limits": None,
            "tree_completion_token_ceiling": args.tree_max_completion_tokens,
            "local_max_model_len": args.local_max_model_len,
            "local_max_response_tokens": 4096,
            "local_prompt_token_budget": args.local_max_model_len - 4096,
            "context_guard_policy": "any_prompt_trim_invalidates_arm",
            "evaluation": {
                "held_out_games_per_task": args.eval_games_per_type,
                "rollouts_per_game": args.eval_rollouts_per_game,
            },
        },
        "external_source_game_ids": ("not available in generic raw_traces schema; evaluation is shared across arms but cannot prove non-overlap with the external source corpus"),
    }
    existing_config = root / "run_config.json"
    if existing_config.exists():
        existing = fixed._read_json(existing_config)
        if existing != config:
            raise RuntimeError("existing V4 root has different protocol settings; use a new --root")
    else:
        fixed._write_json(existing_config, config)

    if args.phase in ("prepare", "all"):
        build_progressive_tree_artifacts(
            evidence,
            root,
            max_attempts=args.tree_generation_attempts,
            max_completion_tokens=args.tree_max_completion_tokens,
        )
        write_generation_metrics(root)
        if args.phase == "prepare":
            return

    if args.phase in ("evaluate", "all"):
        missing = [arm for arm in ARMS if not (root / "artifacts" / arm / "artifact_manifest.json").exists()]
        if missing:
            raise RuntimeError(f"V4 artifacts missing for {missing}; run --phase prepare first")
        for arm in ARMS:
            evaluate_arm(args, root, eval_manifest, arm)

    if args.phase in ("summary", "evaluate", "all"):
        fixed.write_summary(
            root,
            {
                "eval_manifest": str(eval_manifest),
                "eval_sha256": fixed._sha256_path(eval_manifest),
                "evidence_sha256": fixed._sha256_path(evidence),
            },
        )
        write_generation_metrics(root)


if __name__ == "__main__":
    main()
