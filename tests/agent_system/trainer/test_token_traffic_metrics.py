"""Regression coverage for CoSkill's primary JSONL token traffic ledger."""

import json

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _bare_trainer(metrics_path, large_snapshots):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._token_traffic_loaded = False
    trainer._token_traffic_totals = {}
    trainer._token_traffic_large_raw = None
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
    trainer = _bare_trainer(metrics_path, [(4, 1, 5), (6, 2, 8)])

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
