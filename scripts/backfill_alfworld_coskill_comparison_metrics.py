#!/usr/bin/env python3
"""Create canonical ALFWorld comparison metrics from old CoSkill group logs.

The source group_metrics.jsonl is never modified.  This is intended for a
completed CoSkill run that predates comparison_metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def as_int(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CoSkill group_metrics.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="default: sibling comparison_metrics.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output or args.input.with_name("comparison_metrics.jsonl")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {output}; pass --overwrite after checking it")

    totals = {
        "episodes": 0, "wins": 0, "actions": 0,
        "small_prompt": 0, "small_response": 0, "small_total": 0,
        "large_prompt": 0, "large_completion": 0, "large_total": 0,
        "cloud_round": 0,
    }
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            original = payload.get("metrics")
            if not isinstance(original, dict):
                raise ValueError(f"line {lineno}: no nested metrics object")
            metrics = dict(original)  # preserve every original metric field
            count = as_int(metrics.get("episode/count"))
            wins = as_int(metrics.get("episode/wins"))
            action_count = as_int(metrics.get("episode/action_count"))
            if action_count <= 0:
                action_count = as_int(metrics.get("episode/length/mean", 0.0) * count)
            small_prompt = as_int(metrics.get("tokens/small_model/prompt"))
            small_response = as_int(metrics.get("tokens/small_model/response"))
            small_total = as_int(metrics.get("tokens/small_model/total"))
            large_prompt = as_int(metrics.get("tokens/large_model/prompt"))
            large_completion = as_int(metrics.get("tokens/large_model/completion"))
            large_total = as_int(metrics.get("tokens/large_model/total"))

            totals["episodes"] += count
            totals["wins"] += wins
            totals["actions"] += action_count
            totals["small_prompt"] += small_prompt
            totals["small_response"] += small_response
            totals["small_total"] += small_total
            totals["large_prompt"] += large_prompt
            totals["large_completion"] += large_completion
            totals["large_total"] += large_total
            totals["cloud_round"] += int(bool(metrics.get("coskill/cloud_update_fired", False)))

            valid_ratio = float(metrics.get("episode/valid_action_ratio", 0.0) or 0.0)
            metrics.update({
                "comparison/schema_version": 1,
                "comparison/method": "coskill",
                "comparison/benchmark": "alfworld",
                "comparison/rollout_accounting": "active_env_decisions",
                "comparison/timing_cloud_update_measured": 1,
                "training/group": as_int(metrics.get("training/group", payload.get("step", 0))),
                "rollout/global_episode_end": totals["episodes"],
                "episode/count": count,
                "episode/generated_count": as_int(metrics.get("episode/generated_count", count)),
                "episode/wins": wins,
                "episode/success_rate": wins / max(count, 1),
                "episode/action_count": action_count,
                "episode/count_cumulative": totals["episodes"],
                "episode/wins_cumulative": totals["wins"],
                "episode/action_count_cumulative": totals["actions"],
                "episode/relaxed_valid_action_ratio": float(
                    metrics.get("episode/relaxed_valid_action_ratio", valid_ratio) or 0.0
                ),
                "tokens/small_model/accounting": "vllm_request_tokens_two_stage",
                "tokens/small_model/prompt_cumulative": totals["small_prompt"],
                "tokens/small_model/response_cumulative": totals["small_response"],
                "tokens/small_model/total_cumulative": totals["small_total"],
                "tokens/large_model/accounting": "provider_api_usage",
                "tokens/large_model/prompt_cumulative": totals["large_prompt"],
                "tokens/large_model/completion_cumulative": totals["large_completion"],
                "tokens/large_model/total_cumulative": totals["large_total"],
                "experiment/cloud_round": totals["cloud_round"],
                "skill_tree/n_nodes": as_int(metrics.get("skill_tree/n_nodes", 0)),
                "timing_s/rollout": float(metrics.get("timing_s/rollout", 0.0) or 0.0),
                "timing_s/cloud_update": float(metrics.get("timing_s/cloud_update", 0.0) or 0.0),
                "timing_s/group_total": float(metrics.get("timing_s/group_total", 0.0) or 0.0),
                "perf/total_num_tokens": small_total,
                "perf/throughput_episodes_per_second": count / max(float(metrics.get("timing_s/rollout", 0.0) or 0.0), 1e-9),
                "perf/throughput_small_tokens_per_second": small_total / max(float(metrics.get("timing_s/rollout", 0.0) or 0.0), 1e-9),
            })
            rows.append({
                "step": as_int(payload.get("step", metrics["training/group"])),
                "global_episode_end": totals["episodes"],
                "metrics": metrics,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, output)
    print(f"wrote {len(rows)} canonical groups to {output}")


if __name__ == "__main__":
    main()
