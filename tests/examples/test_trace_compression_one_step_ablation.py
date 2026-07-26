from types import SimpleNamespace

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from examples.playbook_evolve.trace_compression_one_step_ablation import (
    ROLL_OUTS_PER_TYPE,
    RUNTIME_TASK_TYPES,
    _uploaded_trace_payload,
    _validate_shared_raw,
    annotate_call_costs,
    build_token_waterfall,
    summarize_cloud_cost,
)


def test_uploaded_trace_payload_reports_actual_codec_version():
    payload = _uploaded_trace_payload({
        "tree_evidence": {
            "version": 2,
            "actions": ["look"],
            "nodes": [[0, 1, 1, 0]],
            "records": [],
        },
        "success_samples": [],
        "failure_samples": [],
    })

    assert payload["cloud_evidence_representation"] == "prefix_tree_codec_v2"


def test_cache_usage_audit_preserves_provider_split():
    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.usage_reported_calls = 0
    analyzer.usage_missing_calls = 0
    analyzer.usage_missing_calls_by_task_type = {}
    analyzer.usage_missing_calls_mixed = 0
    analyzer.cache_usage_reported_calls = 0
    analyzer.cache_usage_missing_calls = 0
    analyzer.total_prompt_cache_hit_tokens = 0
    analyzer.total_prompt_cache_miss_tokens = 0
    analyzer.call_audit = []
    analyzer.model = "deepseek-v4-flash"

    CloudAnalyzer._record_call(
        analyzer,
        "test",
        "prompt",
        "response",
        SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=64, prompt_cache_miss_tokens=36),
    )

    call = analyzer.call_audit[0]
    assert call["prompt_cache_hit_tokens"] == 64
    assert call["prompt_cache_miss_tokens"] == 36
    assert call["cache_usage_status"] == "reported"
    assert call["prompt_chars"] == len("prompt")
    assert call["prompt_tokens_chars_div_4"] == 1
    assert analyzer.total_prompt_cache_hit_tokens == 64
    assert analyzer.total_prompt_cache_miss_tokens == 36


def test_cost_summary_never_assumes_missing_cache_split_is_zero():
    complete = summarize_cloud_cost(
        [
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_cache_hit_tokens": 200,
                "prompt_cache_miss_tokens": 800,
            }
        ]
    )
    assert complete["observed_cache_billed_cost_usd"] is not None
    assert complete["all_input_cache_miss_cost_usd"] > complete["observed_cache_billed_cost_usd"]

    incomplete = summarize_cloud_cost(
        [
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_cache_hit_tokens": None,
                "prompt_cache_miss_tokens": None,
            }
        ]
    )
    assert incomplete["observed_cache_billed_cost_usd"] is None
    assert incomplete["all_input_cache_miss_cost_usd"] is not None


def test_token_waterfall_keeps_trace_estimates_and_provider_usage_distinct():
    batch = {
        "success_samples": [],
        "failure_samples": [],
        "compression": {
            "trace_stage_totals": {
                "raw": {"steps": 4, "chars": 400, "tokens": 100},
                "loop_filtered": {"steps": 3, "chars": 300, "tokens": 75},
                "encoded": {"steps": 3, "chars": 200, "tokens": 50},
            },
        },
    }
    waterfall = build_token_waterfall(
        batch,
        [{"prompt_chars": 500, "prompt_bytes_utf8": 500,
          "prompt_tokens_chars_div_4": 125, "prompt_tokens": 121}],
    )
    by_stage = {row["stage"]: row for row in waterfall["stages"]}
    assert waterfall["display_order"] == [
        "raw", "loop_filter", "obs_delta", "prefix_tree_context", "actual_cloud_prompt",
    ]
    assert by_stage["obs_delta"]["tokens_chars_div_4"] == 50
    assert by_stage["actual_cloud_prompt"]["provider_prompt_tokens"] == 121
    assert by_stage["actual_cloud_prompt"]["tokens_chars_div_4"] == 125


def test_full_trajectory_capture_rejects_old_one_action_corpus():
    one_action = [
        {"task_type": task_type, "steps": [{"step": 1}]}
        for task_type in RUNTIME_TASK_TYPES
        for _ in range(ROLL_OUTS_PER_TYPE)
    ]
    try:
        _validate_shared_raw(one_action)
    except RuntimeError as exc:
        assert "more than one environment step" in str(exc)
    else:
        raise AssertionError("one-action corpus must not satisfy full-trajectory protocol")

    full = [
        {"task_type": task_type, "steps": [{"step": 1}, {"step": 2}]}
        for task_type in RUNTIME_TASK_TYPES
        for _ in range(ROLL_OUTS_PER_TYPE)
    ]
    stats = _validate_shared_raw(full)
    assert stats["observed_steps_per_trace"]["total"] == len(full) * 2


def test_per_call_costs_preserve_missing_cache_split_as_not_available():
    rows = annotate_call_costs([
        {
            "purpose": "contrastive_distill",
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_cache_hit_tokens": 400,
            "prompt_cache_miss_tokens": 600,
        },
        {
            "purpose": "diagnose_failures",
            "prompt_tokens": 800,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": None,
            "prompt_cache_miss_tokens": None,
        },
    ])
    assert rows[0]["call_index"] == 1
    assert rows[0]["cost_status"] == "cache_split_reported"
    assert rows[0]["observed_cache_billed_cost_usd"] is not None
    assert rows[1]["cost_status"] == "provider_usage_without_cache_split"
    assert rows[1]["observed_cache_billed_cost_usd"] is None
    assert rows[1]["all_input_cache_miss_cost_usd"] is not None
