# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.playbook_evolve import fixed_trajectory_ablation as fixed
from examples.playbook_evolve import skill_tree_depth_ablation_v4 as v4
from examples.playbook_evolve import skill_tree_depth_v4_extend_validation as extension


def _game(task_type: str, index: int) -> dict:
    return {
        "label": f"eval_{task_type}_{index}",
        "task_type": task_type,
        "game_file": f"json_2.1.1/train/{task_type}/{index}/game.tw-pddl",
    }


def _manifest(games_per_task: int) -> dict:
    return {
        "split": "train",
        "role": "test",
        "games_per_task_type": games_per_task,
        "games": [
            _game(task_type, index)
            for task_type in fixed.TASK_TYPES
            for index in range(games_per_task)
        ],
    }


def _episode(task_type: str, won: bool, token_base: int) -> dict:
    return {
        "detected_type": fixed.TASK_TYPE_TO_RUNTIME[task_type],
        "won": won,
        "tokens_prompt": token_base,
        "tokens_response": 2,
        "tokens_total": token_base + 2,
        "step": 1,
    }


def _summary(games_per_task: int, *, extension_segment: bool) -> dict:
    rows = [
        _episode(task_type, (index + int(extension_segment)) % 2 == 0, 10 + index)
        for index, task_type in enumerate(fixed.TASK_TYPES)
        for _ in range(games_per_task)
    ]
    wins = sum(int(row["won"]) for row in rows)
    return {
        "status": "done",
        "group_size": 1,
        "completed_rollout_groups": games_per_task,
        "total_episodes": len(rows),
        "wins": wins,
        "success_rate": wins / len(rows),
        "per_game": rows,
        "fixed_game_files": [f"segment/{extension_segment}/{index}" for index in range(len(rows))],
        "token_usage": {
            "large_model": {
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "usage": {"reported_calls": 0, "missing_calls": 0},
            }
        },
        "context_guard": {"prompt_trims": 0, "trimmed_tokens": 0},
    }


def test_delta_manifest_is_balanced_strict_superset(tmp_path):
    source = tmp_path / "source.json"
    expanded = tmp_path / "expanded.json"
    delta = tmp_path / "delta.json"
    fixed._write_json(source, _manifest(1))
    fixed._write_json(expanded, _manifest(2))

    extension.create_delta_manifest(source, expanded, delta)

    payload = fixed._read_json(delta)
    assert payload["games_per_task_type"] == 1
    assert len(payload["games"]) == 6
    assert {game["game_file"] for game in payload["games"]} == {
        _game(task_type, 1)["game_file"] for task_type in fixed.TASK_TYPES
    }


def test_arm_selection_accepts_only_unique_v4_subset():
    assert extension.parse_arm_selection("skill_level_l0,skill_level_l3") == (
        "skill_level_l0",
        "skill_level_l3",
    )
    assert extension.parse_arm_selection("all") == v4.ARMS
    with pytest.raises(ValueError, match="unknown V4 arm"):
        extension.parse_arm_selection("skill_level_l6")
    with pytest.raises(ValueError, match="duplicates"):
        extension.parse_arm_selection("skill_level_l1,skill_level_l1")


def test_arm_resume_rejects_data_parallel_change(tmp_path):
    arm = "skill_level_l1"
    output = tmp_path / "validation_delta" / "arms" / arm
    fixed._write_json(
        tmp_path / "artifacts" / arm / "artifact_manifest.json",
        {"status": "ready", "evaluation_eligible": True},
    )
    fixed._write_json(
        output / "summary_partial.json",
        {"status": "running", "data_parallel_workers": 2},
    )
    args = argparse.Namespace(resume=1, data_parallel_workers=1)
    with pytest.raises(RuntimeError, match="checkpoint used DP=2"):
        extension.evaluate_delta_arm(
            args,
            tmp_path,
            tmp_path / "manifests" / "eval_games_delta.json",
            arm,
        )


def test_delta_manifest_rejects_non_superset(tmp_path):
    source = tmp_path / "source.json"
    expanded = tmp_path / "expanded.json"
    fixed._write_json(source, _manifest(2))
    fixed._write_json(expanded, _manifest(1))
    with pytest.raises(RuntimeError, match="strict superset"):
        extension.create_delta_manifest(source, expanded, tmp_path / "delta.json")


def test_source_evidence_must_not_overlap_expanded_eval(tmp_path):
    trace = {
        "traj_uid": "trace-1",
        "task_type": "heat",
        "steps": [{"observation": "same initial observation", "action": "look"}],
    }
    evidence = tmp_path / "evidence.jsonl"
    fixed._write_jsonl(evidence, [trace])
    fingerprint = v4._trace_initial_observation_fingerprint(trace)
    audit = tmp_path / "fingerprints.json"
    fixed._write_json(
        audit,
        {
            "unique_fingerprint_count": 1,
            "games": [{"initial_observation_sha256": fingerprint}],
        },
    )
    with pytest.raises(RuntimeError, match="overlaps"):
        extension.audit_source_evidence_exclusion(evidence, audit)


def test_merge_rebuilds_episode_and_token_ledger(tmp_path):
    arm = "skill_level_l0"
    source_manifest = tmp_path / "manifests" / "source_eval_games.json"
    expanded_manifest = tmp_path / "manifests" / "eval_games.json"
    delta_manifest = tmp_path / "manifests" / "eval_games_delta.json"
    fixed._write_json(source_manifest, _manifest(1))
    fixed._write_json(expanded_manifest, _manifest(2))
    fixed._write_json(
        delta_manifest,
        {**_manifest(1), "games": [_game(task_type, 1) for task_type in fixed.TASK_TYPES]},
    )
    baseline_path = tmp_path / "baseline_snapshot" / "arms" / arm / "summary.json"
    delta_path = tmp_path / "validation_delta" / "arms" / arm / "summary.json"
    fixed._write_json(baseline_path, _summary(1, extension_segment=False))
    fixed._write_json(delta_path, _summary(1, extension_segment=True))
    artifact = tmp_path / "artifacts" / arm / "artifact_manifest.json"
    fixed._write_json(
        artifact,
        {"status": "ready", "evaluation_eligible": True},
    )

    output = extension.merge_arm_summaries(tmp_path, expanded_manifest, arm)

    merged = fixed._read_json(output)
    assert merged["total_episodes"] == 12
    assert merged["wins"] == 6
    assert merged["success_rate"] == 0.5
    assert [row["step"] for row in merged["per_game"]] == list(range(1, 13))
    assert {
        row["validation_segment"] for row in merged["per_game"]
    } == {"source", "extension"}
    expected_total = sum(int(row["tokens_total"]) for row in merged["per_game"])
    assert merged["token_usage"]["small_model"]["total"] == expected_total
    assert merged["evaluation_protocol_valid"] is True


def test_partial_artifact_snapshot_resume_validates_only_existing_arm(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for arm in v4.ARMS:
        skills = source / "artifacts" / arm / "skills.json"
        fixed._write_json(skills, {"arm": arm})
        fixed._write_json(
            source / "artifacts" / arm / "artifact_manifest.json",
            {
                "status": "ready",
                "evaluation_eligible": True,
                "skills_sha256": fixed._sha256_path(skills),
            },
        )
        fixed._write_json(
            source / "arms" / arm / "summary.json",
            _summary(1, extension_segment=False),
        )
        fixed._write_jsonl(
            source / "arms" / arm / "group_metrics.jsonl",
            [{"step": 1, "metrics": {}}],
        )
    fixed._write_json(source / "run_config.json", {"protocol": {"evaluation": {"rollouts_per_game": 1}}})
    fixed._write_json(source / "manifests" / "eval_games.json", _manifest(1))
    fixed._write_jsonl(
        source / "frozen" / "initial_evidence.jsonl",
        [{"traj_uid": "x", "task_type": "heat", "steps": [{"observation": "x"}]}],
    )

    first_arm = v4.ARMS[0]
    copied = target / "artifacts" / first_arm
    copied.parent.mkdir(parents=True)
    fixed._write_json(copied / "skills.json", {"arm": first_arm})
    fixed._write_json(
        copied / "artifact_manifest.json",
        {
            "status": "ready",
            "evaluation_eligible": True,
            "skills_sha256": fixed._sha256_path(copied / "skills.json"),
        },
    )

    metadata = extension.stage_frozen_source(source, target)

    assert set(metadata["artifacts"]) == set(v4.ARMS)
    assert all((target / "artifacts" / arm / "skills.json").is_file() for arm in v4.ARMS)
