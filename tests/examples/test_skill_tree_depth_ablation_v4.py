# ruff: noqa: E402

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from examples.playbook_evolve import fixed_trajectory_ablation as fixed
from examples.playbook_evolve.skill_tree_depth_ablation_v4 import (
    _audit_evaluation_context,
    _configure_full_evidence,
    _grounding_errors,
    _parent_is_verbatim_subsequence,
    _protocol_validation,
    select_initial_evidence,
    validate_alfworld_tree_semantics,
)


def _trace(uid, task_type, outcome, steps=3):
    rows = []
    for index in range(1, steps + 1):
        rows.append(
            {
                "step": index,
                "observation": f"state before {index}",
                "action": f"action {index}",
                "reward": int(outcome == "success" and index == steps),
            }
        )
    return {
        "traj_uid": uid,
        "task": f"{task_type} goal",
        "task_type": task_type,
        "outcome": outcome,
        "episode_reward": int(outcome == "success"),
        "steps": rows,
    }


def test_v4_selects_all_twelve_balanced_traces_per_task(tmp_path):
    source = tmp_path / "raw.jsonl"
    rows = []
    for task_type in fixed.RUNTIME_TASK_TYPES:
        for index in range(8):
            rows.append(_trace(f"{task_type}-s-{index}", task_type, "success", index + 2))
            rows.append(_trace(f"{task_type}-f-{index}", task_type, "failure", index + 2))
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = select_initial_evidence(source, tmp_path / "selected.jsonl", 12)
    output = fixed._read_jsonl(selected)
    for task_type in fixed.RUNTIME_TASK_TYPES:
        task_rows = [row for row in output if row["task_type"] == task_type]
        assert len(task_rows) == 12
        assert sum(row["outcome"] == "success" for row in task_rows) == 6
        assert sum(row["outcome"] == "failure" for row in task_rows) == 6
    audit = json.loads(selected.with_suffix(".selection.json").read_text())
    assert audit["all_selected_traces_enter_every_tree_level"] is True
    assert audit["step_truncation"] is False


def test_v4_context_audit_invalidates_an_arm_after_any_prompt_trim(tmp_path):
    summary = tmp_path / "arms" / "skill_level_l5" / "summary.json"
    artifact = tmp_path / "artifacts" / "skill_level_l5" / "artifact_manifest.json"
    fixed._write_json(
        summary,
        {
            "status": "done",
            "context_guard": {"prompt_trims": 2, "trimmed_tokens": 91},
        },
    )
    fixed._write_json(
        artifact,
        {"status": "ready", "evaluation_eligible": True},
    )
    _audit_evaluation_context(tmp_path, "skill_level_l5", summary)
    audited_summary = fixed._read_json(summary)
    audited_artifact = fixed._read_json(artifact)
    assert audited_summary["evaluation_protocol_valid"] is False
    assert "local_context_guard_trimmed_prompts:2:tokens:91" == (audited_summary["evaluation_protocol_error"])
    assert audited_artifact["status"] == "N.A."
    assert audited_artifact["evaluation_eligible"] is False
    assert audited_artifact["unavailable_reason"] == "local_prompt_context_trim_detected"


def test_cloud_full_evidence_prompt_is_causal_unfolded_and_unbounded():
    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.environment_name = "ALFWorld"
    analyzer.evidence_render_limits = {}
    successes = [_trace(f"s-{index}", "heat", "success", 14) for index in range(6)]
    failures = [_trace(f"f-{index}", "heat", "failure", 14) for index in range(6)]
    evidence_audit = _configure_full_evidence(analyzer, successes + failures)
    assert evidence_audit["selected_traj_uids"] == [f"s-{index}" for index in range(6)] + [f"f-{index}" for index in range(6)]
    parent = "# Transform while holding\nUse `heat OBJECT with microwave`."
    prompt = analyzer._build_evolve_prompt(
        "heat",
        parent,
        successes,
        failures,
        [],
        target_depth=2,
        max_success_examples=6,
        max_failure_examples=6,
        preserve_parent_tree=True,
        render_full_trajectories=True,
    )
    assert "Trajectory 6 [success] [ref=s-5]" in prompt
    assert "Trajectory 6 [failure] [ref=f-5]" in prompt
    assert "step 14 action: action 14" in prompt
    assert "state_before_step_1 [full_observation]" in prompt
    assert "state_after_action_and_before_step_2" in prompt
    assert "terminal post-observation was not recorded" in prompt
    assert "No consensus prefix is folded" in prompt
    assert "opening action is presumed mastered" in prompt
    assert "The agent has MASTERED these steps" not in prompt
    assert "Preserve every non-empty parent line VERBATIM" in prompt
    assert "at most" not in prompt


def test_parent_validation_allows_insertions_but_rejects_rewrites():
    parent = "# Goal\nKeep this rule.\n# Finish\nStop on success."
    child = "# Goal\nKeep this rule.\n## Conditional recovery\nInspect feedback.\n# Finish\nStop on success."
    rewritten = child.replace("Keep this rule.", "Change this rule.")
    assert _parent_is_verbatim_subsequence(parent, child)
    assert not _parent_is_verbatim_subsequence(parent, rewritten)


def test_grounding_requires_real_trace_and_step_for_each_deepest_heading():
    tree = "# Goal\nRule\n## Recovery\nRecover"
    traces = [_trace("trace-a", "heat", "success", 3)]
    good = [
        {
            "heading_path": "Goal > Recovery",
            "evidence": [{"traj_ref": "trace-a", "step": 2}],
            "supported_claim": "Step 2 shows recovery.",
        }
    ]
    assert _grounding_errors(tree, 2, good, traces) == []
    bad = [
        {
            "heading_path": "Goal > Recovery",
            "evidence": [{"traj_ref": "invented", "step": 99}],
            "supported_claim": "unsupported",
        }
    ]
    errors = _grounding_errors(tree, 2, bad, traces)
    assert any("grounding_unknown_reference" in error for error in errors)


def test_alfworld_semantic_validator_covers_deep_tree_failure_modes():
    broken_heat = "# Heat\nPut the target inside the microwave.\nThen close it and heat the target with microwave."
    assert "transform_after_relinquishing_object_violation:heat:microwave" in (validate_alfworld_tree_semantics("heat", broken_heat))
    correct_heat = "# Heat while holding\nOpen the microwave, then use `heat OBJECT with microwave` while still holding it.\n# Deliver\nMove it to the requested destination."
    assert validate_alfworld_tree_semantics("heat", correct_heat) == []

    broken_pick_two = "# Collect\nKeep both objects in inventory before going to the destination."
    assert "alfworld_inventory_capacity_one_violation" in (validate_alfworld_tree_semantics("pick_two_obj_and_place", broken_pick_two))
    correct_pick_two = "# Deliver sequentially\nHandle one at a time: take and place the first object, then repeat for the second object."
    assert validate_alfworld_tree_semantics("pick_two_obj_and_place", correct_pick_two) == []

    broken_look = "# Light\nMove the object onto the desk lamp, then use the desklamp."
    assert "look_task_must_not_place_object_on_lamp" in (validate_alfworld_tree_semantics("look_at_obj_in_light", broken_look))


def test_alfworld_semantic_validator_covers_remaining_task_families():
    broken_clean = "# Clean\nPut the object into the sinkbasin, then clean it."
    assert "transform_after_relinquishing_object_violation:clean:sinkbasin" in (validate_alfworld_tree_semantics("clean", broken_clean))
    correct_clean = "# Clean while holding\nUse `clean OBJECT with sinkbasin` while holding it."
    assert validate_alfworld_tree_semantics("clean", correct_clean) == []

    broken_cool = "# Cool\nPlace the object into the fridge, then cool it."
    assert "transform_after_relinquishing_object_violation:cool:fridge" in (validate_alfworld_tree_semantics("cool", broken_cool))
    correct_cool = "# Cool while holding\nUse `cool OBJECT with fridge` while holding it."
    assert validate_alfworld_tree_semantics("cool", correct_cool) == []

    broken_pick_place = "# Deliver\nExplore until done."
    assert "pick_and_place_missing_take_then_deliver_contract" in (validate_alfworld_tree_semantics("pick_and_place", broken_pick_place))
    correct_pick_place = "# Deliver\nTake the target, move to the requested destination, and place it."
    assert validate_alfworld_tree_semantics("pick_and_place", correct_pick_place) == []


def test_protocol_validation_combines_depth_parent_grounding_and_semantics():
    parent = "# Heat while holding\nUse `heat OBJECT with microwave` while holding it."
    child = "# Heat while holding\nUse `heat OBJECT with microwave` while holding it.\n## Recover from unavailable action\nInspect current feedback and admissible actions."
    traces = [_trace("heat-success", "heat", "success", 3)]
    grounding = [
        {
            "heading_path": "Heat while holding > Recover from unavailable action",
            "evidence": [{"traj_ref": "heat-success", "step": 2}],
            "supported_claim": "The transition supplies feedback after the action.",
        }
    ]
    validation = _protocol_validation("heat", parent, child, 2, grounding, traces)
    assert validation["protocol_valid"] is True
    rewritten = child.replace("while holding it", "after placing it inside")
    invalid = _protocol_validation("heat", parent, rewritten, 2, grounding, traces)
    assert invalid["protocol_valid"] is False
    assert "accepted_parent_tree_was_modified_or_reordered" in invalid["protocol_validation_errors"]

    detached = parent + "\n# Newly invented root\n## Recover from unavailable action\nInspect current feedback and admissible actions."
    detached_grounding = [
        {
            "heading_path": "Newly invented root > Recover from unavailable action",
            "evidence": [{"traj_ref": "heat-success", "step": 2}],
            "supported_claim": "The transition supplies feedback after the action.",
        }
    ]
    invalid = _protocol_validation("heat", parent, detached, 2, detached_grounding, traces)
    assert invalid["protocol_valid"] is False
    assert any(error.startswith("deepest_heading_not_attached_to_accepted_parent:") for error in invalid["protocol_validation_errors"])
    assert any(error.startswith("new_heading_not_exact_next_depth:") for error in invalid["protocol_validation_errors"])

    escaped_body = parent + "\nThis unsupported shallow rule is not under a new L2 heading.\n## Recovery\nInspect feedback."
    escaped_grounding = [
        {
            "heading_path": "Heat while holding > Recovery",
            "evidence": [{"traj_ref": "heat-success", "step": 2}],
            "supported_claim": "The transition supplies feedback after the action.",
        }
    ]
    invalid = _protocol_validation("heat", parent, escaped_body, 2, escaped_grounding, traces)
    assert invalid["protocol_valid"] is False
    assert any(error.startswith("inserted_content_not_under_new_depth_2_heading:") for error in invalid["protocol_validation_errors"])
