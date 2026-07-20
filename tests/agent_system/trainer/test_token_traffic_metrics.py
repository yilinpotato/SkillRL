"""Regression coverage for CoSkill's primary JSONL token traffic ledger."""

import json

from verl.trainer.ppo.metric_utils import KNOWN_TASK_TYPES
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _bare_trainer(metrics_path, large_snapshots):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._token_traffic_loaded = False
    trainer._token_traffic_totals = {}
    trainer._token_traffic_large_raw = None
    trainer._token_traffic_large_by_tt_raw = None
    trainer._token_traffic_large_mixed_raw = None
    snapshots = iter(large_snapshots)
    trainer._large_token_usage_snapshot = lambda: next(snapshots)
    return trainer


def test_token_traffic_backfills_old_primary_rows_and_tracks_new_deltas(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(json.dumps({
        "step": 3,
        "tokens/small_model/prompt": 100,
        "tokens/small_model/response": 10,
        "tokens/small_model/total": 110,
        "coskill/cloud/large_model_prompt_tokens": 7,
        "coskill/cloud/large_model_completion_tokens": 3,
        "coskill/cloud/large_model_total_tokens": 10,
    }) + "\n", encoding="utf-8")
    monkeypatch.setenv("JSONL_METRICS_PATH", str(metrics_path))
    trainer = _bare_trainer(metrics_path, [
        (4, 1, 5, {}, 0),
        (6, 2, 8, {}, 0),
    ])

    first = {
        "tokens/small_model/prompt": 5,
        "tokens/small_model/response": 2,
        "tokens/small_model/total": 7,
    }
    trainer._add_token_traffic_metrics(first)
    assert first["tokens/small_model/prompt_cumulative"] == 105
    assert first["tokens/small_model/response_cumulative"] == 12
    assert first["tokens/small_model/total_cumulative"] == 117
    assert first["tokens/large_model/prompt"] == 4
    assert first["tokens/large_model/completion"] == 1
    assert first["tokens/large_model/total_cumulative"] == 15

    second = {
        "tokens/small_model/prompt": 6,
        "tokens/small_model/response": 3,
        "tokens/small_model/total": 9,
    }
    trainer._add_token_traffic_metrics(second)
    assert second["tokens/small_model/prompt_cumulative"] == 111
    assert second["tokens/small_model/response_cumulative"] == 15
    assert second["tokens/small_model/total_cumulative"] == 126
    # Cloud provider counters are process-cumulative, so only 2/1/3 are new.
    assert second["tokens/large_model/prompt"] == 2
    assert second["tokens/large_model/completion"] == 1
    assert second["tokens/large_model/total"] == 3
    assert second["tokens/large_model/prompt_cumulative"] == 13
    assert second["tokens/large_model/completion_cumulative"] == 5
    assert second["tokens/large_model/total_cumulative"] == 18


def test_token_traffic_by_task_type_and_mixed_breakdown(tmp_path, monkeypatch):
    """Per-subtask (task_type) token breakdown, both small-model (step-local,
    already-disjoint) and cloud (cumulative provider counters reconciled into
    per-step deltas the same way the flat scalar total already is)."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("JSONL_METRICS_PATH", str(metrics_path))

    tt_a, tt_b = KNOWN_TASK_TYPES[0], KNOWN_TASK_TYPES[1]
    # Step 1: cloud usage so far is 10 for tt_a (evolve_playbook, attributable)
    # and 3 mixed (contrastive_distill/diagnose_failures, unattributable).
    trainer = _bare_trainer(metrics_path, [
        (10, 3, 13, {tt_a: 10}, 3),
        (18, 5, 23, {tt_a: 10, tt_b: 6}, 5),
    ])

    step1 = {
        "tokens/small_model/prompt": 20,
        "tokens/small_model/response": 5,
        "tokens/small_model/total": 25,
        f"tokens/small_model/by_task_type/{tt_a}/total": 25,
    }
    trainer._add_token_traffic_metrics(step1)

    # Small-model per-task_type sum must equal the raw total for this step -
    # the breakdown must never lose or double-count tokens.
    small_total_by_tt = sum(
        step1.get(f"tokens/small_model/by_task_type/{tt}/total_cumulative", 0)
        for tt in KNOWN_TASK_TYPES
    )
    assert small_total_by_tt == step1["tokens/small_model/total_cumulative"]
    assert step1[f"tokens/large_model/by_task_type/{tt_a}/total"] == 10
    assert step1[f"tokens/large_model/by_task_type/{tt_b}/total"] == 0
    assert step1["tokens/large_model/mixed/total"] == 3
    # Cloud total this step = attributable (tt_a) + mixed, no double counting.
    assert step1["tokens/large_model/total"] == 13

    step2 = {
        "tokens/small_model/prompt": 5,
        "tokens/small_model/response": 1,
        "tokens/small_model/total": 6,
        f"tokens/small_model/by_task_type/{tt_b}/total": 6,
    }
    trainer._add_token_traffic_metrics(step2)
    # tt_a: 10 -> 10 (no new cloud usage this step); tt_b: 0 -> 6 (new).
    assert step2[f"tokens/large_model/by_task_type/{tt_a}/total"] == 0
    assert step2[f"tokens/large_model/by_task_type/{tt_b}/total"] == 6
    assert step2["tokens/large_model/mixed/total"] == 2
    assert step2[f"tokens/large_model/by_task_type/{tt_a}/total_cumulative"] == 10
    assert step2[f"tokens/large_model/by_task_type/{tt_b}/total_cumulative"] == 6
    assert step2["tokens/large_model/mixed/total_cumulative"] == 5
