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
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
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


def _json_normalized(value: Any) -> Any:
    """Return the exact JSON round-trip representation used on disk."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _ensure_run_config_compatible(
    path: Path,
    config: dict[str, Any],
    rebuild_from_level: int | None,
) -> str:
    """Create, validate, or narrowly upgrade a resumable V4 run config."""
    expected = _json_normalized(config)
    if not path.exists():
        fixed._write_json(path, expected)
        return "created"

    existing = _json_normalized(fixed._read_json(path))
    if existing == expected:
        return "matched"

    if rebuild_from_level is not None:
        expected_legacy = _json_normalized(expected)
        expected_legacy["protocol"].pop("progressive_generation_output", None)
        if existing == expected_legacy:
            fixed._write_json(path, expected)
            print("[skill-tree-v4] upgraded compatible pre-delta run_config for suffix rebuild")
            return "upgraded"

    raise RuntimeError("existing V4 root has different protocol settings; use a new --root")


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


def _trace_step_index(traces: Iterable[dict[str, Any]]) -> dict[str, set[int]]:
    """Return exactly the numeric trajectory/step references rendered to the cloud."""
    result: dict[str, set[int]] = {}
    for trace in traces:
        uid = str(trace.get("traj_uid", ""))
        steps: set[int] = set()
        for index, step in enumerate(trace.get("steps") or [], start=1):
            raw = step.get("step")
            if raw is None or raw == "":
                raw = index
            try:
                steps.add(int(raw))
            except (TypeError, ValueError):
                # The causal renderer displays this transition, but the JSON
                # grounding schema deliberately permits only numeric steps.
                steps.add(index)
        result[uid] = steps
    return result


def _evidence_reference_catalog(traces: Iterable[dict[str, Any]]) -> str:
    """Give the model a compact allow-list instead of making it recopy long traces."""
    index = _trace_step_index(traces)
    return "\n".join(
        f"- traj_ref={uid}; valid_steps={','.join(str(step) for step in sorted(steps))}"
        for uid, steps in index.items()
    )


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
    trace_steps = _trace_step_index(traces)
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
            if uid not in trace_steps:
                errors.append(f"grounding_unknown_traj_uid:{uid}")
            elif step not in trace_steps[uid]:
                valid = ",".join(str(value) for value in sorted(trace_steps[uid]))
                errors.append(f"grounding_unknown_step:{uid}:step{step}:valid_steps={valid}")
            else:
                valid_reference = True
        if not valid_reference:
            errors.append(f"grounding_has_no_valid_reference:{path or index}")
        if not str(item.get("supported_claim", "")).strip():
            errors.append(f"grounding_missing_supported_claim:{path or index}")
    for path in required:
        if path not in supplied:
            errors.append(f"deepest_heading_not_grounded:{path}")
    return errors


def _progressive_patch_prompt(
    analyzer: CloudAnalyzer,
    task_type: str,
    parent: str,
    success: list[dict[str, Any]],
    failure: list[dict[str, Any]],
    target_depth: int,
    repair_patch: Any,
    repair_errors: Iterable[str],
) -> str:
    """Build a delta-only prompt; the cloud never rewrites an accepted parent."""
    base = analyzer._build_evolve_prompt(
        task_type,
        parent,
        success,
        failure,
        [],
        target_depth=target_depth,
        max_success_examples=len(success),
        max_failure_examples=len(failure),
        max_tree_nodes=None,
        max_tree_chars=None,
        preserve_parent_tree=True,
        render_full_trajectories=True,
    )
    # Remove the generic full-tree response schema before appending the V4
    # delta schema. A "final override" alone still left two contradictory JSON
    # contracts in one prompt and caused the model to regenerate the parent.
    schema_marker = "\nReturn ONLY one JSON object, EXACTLY these fields:"
    if schema_marker in base:
        base = base.split(schema_marker, 1)[0].rstrip()
    parent_paths = _heading_paths(parent, target_depth - 1)
    repair = ""
    if repair_patch is not None:
        repair = f"""

THE PREVIOUS DELTA PATCH WAS REJECTED. Replace it rather than copying its mistakes.
REJECTED PATCH:
{json.dumps(repair_patch, ensure_ascii=False, indent=2)}
PRECISE LOCAL VALIDATION ERRORS:
{json.dumps(list(repair_errors), ensure_ascii=False, indent=2)}
"""
    return f"""{base}

V4 DELTA-ONLY OUTPUT PROTOCOL:
The accepted parent tree is immutable. Do NOT return or reproduce `skill_tree`, and do not edit,
summarize, reorder, or quote parent lines. Return only new level-{target_depth} child nodes. The local
program will validate and insert them into the accepted parent deterministically.

Each `parent_heading_path` MUST exactly equal one path in this allow-list:
{json.dumps(parent_paths, ensure_ascii=False, indent=2)}

Each evidence reference MUST exactly use one traj_ref and numeric step from this allow-list:
{_evidence_reference_catalog(success + failure)}

Return ONLY one JSON object with exactly these fields:
{{
  "action": "refine",
  "level": {target_depth},
  "new_nodes": [
    {{
      "parent_heading_path": "exact allowed parent path",
      "heading": "plain heading text without # or newline",
      "body_lines": ["one evidence-supported instruction per string"],
      "evidence": [{{"traj_ref": "exact allowed id", "step": 1}}],
      "supported_claim": "the precise rule established by the cited transition(s)"
    }}
  ],
  "critique": "brief evidence-based reason for these refinements",
  "changelog": "brief list of parent paths deepened",
  "unsupported_claims": []
}}

Rules:
- `new_nodes` must be non-empty, and every item becomes exactly one new level-{target_depth} heading.
- `body_lines` must be non-empty and must not contain Markdown headings.
- Do not include generic shallow bullets outside a new node.
- Cite only allow-listed references. Never guess a nearby step or trajectory id.
- If a possible refinement lacks evidence, omit it and list it under `unsupported_claims`.
- The new child may clarify a parent condition but must never contradict the immutable parent.
{repair}
Return ONLY the delta JSON object."""


def _request_progressive_patch(
    analyzer: CloudAnalyzer,
    task_type: str,
    parent: str,
    success: list[dict[str, Any]],
    failure: list[dict[str, Any]],
    target_depth: int,
    repair_patch: Any = None,
    repair_errors: Iterable[str] = (),
) -> dict[str, Any] | None:
    """Call the cloud with the V4 delta schema while retaining provider usage audit."""
    prompt = _progressive_patch_prompt(
        analyzer,
        task_type,
        parent,
        success,
        failure,
        target_depth,
        repair_patch,
        repair_errors,
    )
    call_number = analyzer.n_evolve_calls
    if analyzer.playbook_io_dir is not None:
        analyzer._dump_text(
            analyzer.playbook_io_dir,
            f"evolve_skill_tree_patch_{task_type}_call{call_number:03d}.txt",
            prompt,
        )
    try:
        response = analyzer.client.chat.completions.create(
            model=analyzer.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=analyzer.max_completion_tokens,
        )
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        analyzer._record_call(
            "evolve_playbook_patch",
            prompt,
            content,
            usage,
            task_type=task_type,
        )
        if analyzer.playbook_io_dir is not None:
            analyzer._dump_text(
                analyzer.playbook_io_dir,
                f"evolve_skill_tree_patch_response_{task_type}_call{call_number:03d}.txt",
                content,
            )
        if usage:
            analyzer.total_prompt_tokens += int(usage.prompt_tokens or 0)
            analyzer.total_completion_tokens += int(usage.completion_tokens or 0)
            analyzer.total_prompt_tokens_by_task_type[task_type] = (
                analyzer.total_prompt_tokens_by_task_type.get(task_type, 0)
                + int(usage.prompt_tokens or 0)
            )
            analyzer.total_completion_tokens_by_task_type[task_type] = (
                analyzer.total_completion_tokens_by_task_type.get(task_type, 0)
                + int(usage.completion_tokens or 0)
            )
        analyzer.n_evolve_calls += 1
        obj = analyzer._parse_json_object(content)
        return obj if isinstance(obj, dict) else None
    except Exception as exc:
        print(f"[skill-tree-v4] progressive patch error ({analyzer.model}, {task_type}): {exc}")
        return None


def _parent_heading_locations(parent: str, depth: int) -> dict[str, tuple[int, int]]:
    """Map each exact deepest parent path to its subtree insertion boundary."""
    lines = parent.splitlines()
    stack: list[tuple[int, str]] = []
    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        if not (line.startswith("#") and line.lstrip("#").startswith(" ")):
            continue
        heading_depth = len(line) - len(line.lstrip("#"))
        label = line[heading_depth:].strip()
        while stack and stack[-1][0] >= heading_depth:
            stack.pop()
        stack.append((heading_depth, label))
        headings.append((index, heading_depth, " > ".join(value for _, value in stack)))

    locations: dict[str, tuple[int, int]] = {}
    for position, (line_index, heading_depth, path) in enumerate(headings):
        if heading_depth != depth:
            continue
        boundary = len(lines)
        for next_index, next_depth, _ in headings[position + 1 :]:
            if next_depth <= heading_depth:
                boundary = next_index
                break
        locations[path] = (line_index, boundary)
    return locations


def _merge_progressive_nodes(
    parent: str,
    target_depth: int,
    new_nodes: Any,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Validate a cloud delta and deterministically insert it into the parent."""
    if not isinstance(new_nodes, list):
        return parent, [], ["progressive_patch_new_nodes_not_array"]
    if not new_nodes:
        return parent, [], ["progressive_patch_added_no_nodes"]

    locations = _parent_heading_locations(parent, target_depth - 1)
    existing_paths = set(_heading_paths(parent, target_depth))
    added_paths: set[str] = set()
    insertions: dict[int, list[str]] = defaultdict(list)
    grounding: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, node in enumerate(new_nodes):
        label = f"node{index}"
        if not isinstance(node, dict):
            errors.append(f"progressive_patch_item_not_object:{label}")
            continue
        parent_path = " > ".join(
            part.strip()
            for part in str(node.get("parent_heading_path", "")).split(">")
        )
        if parent_path not in locations:
            errors.append(f"progressive_patch_unknown_parent_path:{parent_path or label}")
            continue
        heading = str(node.get("heading", "")).strip()
        if (
            not heading
            or "\n" in heading
            or heading.startswith("#")
            or " > " in heading
        ):
            errors.append(f"progressive_patch_invalid_heading:{parent_path}:{heading or label}")
            continue
        body_lines = node.get("body_lines")
        if not isinstance(body_lines, list) or not body_lines:
            errors.append(f"progressive_patch_body_lines_missing:{parent_path} > {heading}")
            continue
        cleaned_body: list[str] = []
        invalid_body = False
        for body in body_lines:
            text = str(body).strip()
            if not text or "\n" in text or re.match(r"^#+\s", text):
                invalid_body = True
                break
            cleaned_body.append(text)
        if invalid_body:
            errors.append(f"progressive_patch_invalid_body_line:{parent_path} > {heading}")
            continue

        child_path = f"{parent_path} > {heading}"
        if child_path in existing_paths or child_path in added_paths:
            errors.append(f"progressive_patch_duplicate_heading_path:{child_path}")
            continue
        added_paths.add(child_path)
        boundary = locations[parent_path][1]
        fragment = ["#" * target_depth + f" {heading}", *cleaned_body]
        if insertions[boundary]:
            insertions[boundary].append("")
        insertions[boundary].extend(fragment)
        grounding.append(
            {
                "heading_path": child_path,
                "evidence": node.get("evidence"),
                "supported_claim": node.get("supported_claim"),
            }
        )

    if not insertions:
        return parent, grounding, errors or ["progressive_patch_added_no_valid_nodes"]

    parent_lines = parent.splitlines()
    child_lines: list[str] = []
    for index in range(len(parent_lines) + 1):
        if index in insertions:
            if child_lines and child_lines[-1].strip():
                child_lines.append("")
            child_lines.extend(insertions[index])
            if index < len(parent_lines) and child_lines[-1].strip():
                child_lines.append("")
        if index < len(parent_lines):
            child_lines.append(parent_lines[index])
    return "\n".join(child_lines).strip(), grounding, errors


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
            repair_candidate: Any = None
            attempts = 0
            while attempts < max_attempts:
                if parent:
                    patch = _request_progressive_patch(
                        analyzer,
                        task_type,
                        parent,
                        success,
                        failure,
                        depth,
                        repair_patch=repair_candidate,
                        repair_errors=validation.get("protocol_validation_errors", []),
                    )
                    tree, grounding, patch_errors = _merge_progressive_nodes(
                        parent,
                        depth,
                        (patch or {}).get("new_nodes"),
                    )
                    candidate = (
                        {
                            "action": "refine",
                            "level": depth,
                            "skill_tree": tree,
                            "new_nodes": (patch or {}).get("new_nodes"),
                            "new_node_grounding": grounding,
                            "unsupported_claims": (patch or {}).get("unsupported_claims", []),
                            "critique": (patch or {}).get("critique", ""),
                            "changelog": (patch or {}).get("changelog", ""),
                        }
                        if patch
                        else None
                    )
                else:
                    patch_errors = []
                    candidate = analyzer.evolve_playbook(
                        task_type=task_type,
                        current_playbook=None,
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
                        preserve_parent_tree=False,
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
                if patch_errors:
                    validation["protocol_validation_errors"] = sorted(
                        set(validation.get("protocol_validation_errors", []))
                        | set(patch_errors)
                    )
                    validation["protocol_valid"] = False
                if candidate and validation["protocol_valid"]:
                    break
                repair_candidate = (
                    (candidate or {}).get("new_nodes")
                    if parent
                    else tree or repair_candidate
                )

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
                    "progressive_patch_nodes": candidate.get("new_nodes", []),
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
                "progressive_output_protocol": (
                    "full_tree_from_evidence"
                    if depth == 1
                    else "cloud_delta_nodes_plus_deterministic_local_merge"
                ),
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


def archive_invalid_suffix_for_rebuild(root: Path, start_level: int) -> Path | None:
    """Archive, never delete, the first incompatible/failed level and descendants."""
    if start_level < 1 or start_level > 5:
        raise ValueError("rebuild start level must be between 1 and 5")
    first_rebuild: int | None = None
    for depth in range(start_level, 6):
        manifest_path = (
            root
            / "artifacts"
            / f"skill_level_l{depth}"
            / "artifact_manifest.json"
        )
        expected = (
            "full_tree_from_evidence"
            if depth == 1
            else "cloud_delta_nodes_plus_deterministic_local_merge"
        )
        if not manifest_path.exists():
            first_rebuild = depth
            break
        manifest = fixed._read_json(manifest_path)
        if (
            manifest.get("status") != "ready"
            or not manifest.get("evaluation_eligible", False)
            or manifest.get("progressive_output_protocol") != expected
        ):
            first_rebuild = depth
            break
    if first_rebuild is None:
        print(
            f"[skill-tree-v4] L{start_level}-L5 already use the validated delta protocol; "
            "nothing to rebuild"
        )
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = root / "superseded" / f"{stamp}_rebuild_from_l{first_rebuild}"
    moved: list[dict[str, str]] = []
    for area in ("artifacts", "arms"):
        for depth in range(first_rebuild, 6):
            source = root / area / f"skill_level_l{depth}"
            if not source.exists():
                continue
            destination = archive / area / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append(
                {
                    "source": str(source.relative_to(root)),
                    "destination": str(destination.relative_to(root)),
                }
            )

    summary_names = (
        "generation_metrics.jsonl",
        "generation_metrics_by_task.jsonl",
        "metrics.jsonl",
        "metrics_by_task.jsonl",
        "ablation_summary.json",
        "ablation_summary.csv",
        "skill_level_by_task.csv",
    )
    for name in summary_names:
        source = root / name
        if not source.exists():
            continue
        destination = archive / "root_summaries" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(
            {
                "source": name,
                "destination": str(destination.relative_to(root)),
            }
        )
    fixed._write_json(
        archive / "rebuild_receipt.json",
        {
            "requested_start_level": start_level,
            "effective_start_level": first_rebuild,
            "reason": "replace_failed_or_pre_delta_v4_generation_without_touching_valid_prefix",
            "moved": moved,
        },
    )
    print(
        f"[skill-tree-v4] archived stale L{first_rebuild}-L5 outputs under {archive}; "
        f"preserving L0-L{first_rebuild - 1}"
    )
    return archive


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
    parser.add_argument(
        "--rebuild_from_level",
        type=int,
        choices=range(1, 6),
        default=None,
        help=(
            "Archive a failed/pre-fix suffix and regenerate it while preserving the valid "
            "L0..L(N-1) prefix. Use only with --phase prepare/all."
        ),
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
    if args.rebuild_from_level is not None and args.phase not in ("prepare", "all"):
        parser.error("--rebuild_from_level requires --phase prepare or --phase all")

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
        "task_types": list(fixed.TASK_TYPES),
        "runtime_task_types": list(fixed.RUNTIME_TASK_TYPES),
        "protocol": {
            "online_growth": False,
            "same_tree_progressive_levels": True,
            "parent_lines_preserved_verbatim": True,
            "progressive_generation_output": (
                "L1_full_tree_then_cloud_delta_nodes_plus_deterministic_local_merge"
            ),
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
    _ensure_run_config_compatible(existing_config, config, args.rebuild_from_level)

    if args.phase in ("prepare", "all"):
        if args.rebuild_from_level is not None:
            archive_invalid_suffix_for_rebuild(root, args.rebuild_from_level)
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
