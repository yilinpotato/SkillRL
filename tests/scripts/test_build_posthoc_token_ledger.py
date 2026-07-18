import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "build_posthoc_token_ledger.py"
SPEC = importlib.util.spec_from_file_location("posthoc_token_ledger", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_training_ledger_uses_latest_duplicate_step_and_keeps_perf_separate(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 1, "tokens/small_model/prompt": 10, "tokens/small_model/response": 2,
         "tokens/small_model/total": 12, "perf/total_num_tokens": 12, "episode/count": 2,
         "episode/action_count": 6, "episode/wins": 1},
        {"step": 1, "tokens/small_model/prompt": 11, "tokens/small_model/response": 3,
         "tokens/small_model/total": 14, "perf/total_num_tokens": 14, "episode/count": 2,
         "episode/action_count": 6, "episode/wins": 1},
        {"step": 2, "tokens/small_model/prompt": 20, "tokens/small_model/response": 4,
         "tokens/small_model/total": 24, "perf/total_num_tokens": 24, "episode/count": 2,
         "episode/action_count": 8, "episode/wins": 2,
         "coskill/cloud/large_model_total_tokens": 9},
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    entry = MODULE.training_entry("coskill", metrics, "alfworld", None)
    assert entry["small_model_total_tokens"] == 38
    assert entry["large_model_total_tokens"] == 9
    assert entry["online_model_traffic_tokens"] == 47
    assert entry["perf_batch_tokens"] == 38
    assert entry["episodes"] == 4
    assert entry["env_decisions"] == 14
    assert entry["successes"] == 3


def test_api_ledger_uses_provider_usage_and_task_summary_totals(tmp_path):
    summary = tmp_path / "overall_summary.json"
    summary.write_text(json.dumps({
        "one": {"episodes": 2, "success_rate": 0.5, "avg_episode_steps": 3,
                "total_prompt_tokens": 30, "total_completion_tokens": 9, "total_tokens": 39,
                "model": "deepseek"},
        "two": {"episodes": 2, "success_rate": 1.0, "avg_episode_steps": 2,
                "total_prompt_tokens": 40, "total_completion_tokens": 11, "total_tokens": 51,
                "model": "deepseek"},
    }), encoding="utf-8")

    entry = MODULE.api_entry("deepseek_full_skills", summary, "alfworld")
    assert entry["large_model_total_tokens"] == 90
    assert entry["episodes"] == 4
    assert entry["env_decisions"] == 10
    assert entry["successes"] == 3
    assert entry["normalised"]["online_tokens_per_episode"] == 22.5
