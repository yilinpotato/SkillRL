#!/usr/bin/env python3
"""Read-only validation for the shared ALFWorld comparison metric schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = {
    "comparison/schema_version", "comparison/method", "comparison/benchmark",
    "comparison/rollout_accounting", "training/group", "rollout/global_episode_end",
    "episode/count", "episode/generated_count", "episode/wins", "episode/success_rate",
    "episode/length/mean", "episode/length/max", "episode/length/min",
    "episode/action_count", "episode/valid_action_ratio",
    "episode/strict_valid_action_ratio", "episode/relaxed_valid_action_ratio",
    "episode/salvaged_action_ratio", "episode/fallback_action_ratio",
    "episode/count_cumulative", "episode/wins_cumulative",
    "episode/action_count_cumulative", "tokens/small_model/prompt",
    "tokens/small_model/response", "tokens/small_model/total",
    "tokens/small_model/accounting", "tokens/small_model/prompt_cumulative",
    "tokens/small_model/response_cumulative", "tokens/small_model/total_cumulative",
    "tokens/large_model/prompt", "tokens/large_model/completion",
    "tokens/large_model/total", "tokens/large_model/accounting",
    "tokens/large_model/prompt_cumulative",
    "tokens/large_model/completion_cumulative",
    "tokens/large_model/total_cumulative", "experiment/skill_tree_enabled",
    "experiment/skill_tree_evolve_enabled", "experiment/skill_bullets_enabled",
    "experiment/cloud_round", "coskill/cloud_update_fired", "skill_tree/n_nodes",
    "timing_s/rollout", "timing_s/cloud_update", "timing_s/group_total",
    "comparison/timing_cloud_update_measured", "perf/total_num_tokens",
    "perf/throughput_episodes_per_second", "perf/throughput_small_tokens_per_second",
}


def number(row, key):
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def validate(path: Path):
    previous = None
    method = None
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            row = payload.get("metrics")
            if not isinstance(row, dict):
                raise ValueError(f"line {lineno}: expected nested metrics object")
            missing = sorted(REQUIRED - row.keys())
            if missing:
                raise ValueError(f"line {lineno}: missing {', '.join(missing)}")
            if row["comparison/schema_version"] != 1:
                raise ValueError(f"line {lineno}: unsupported schema version")
            if row["comparison/benchmark"] != "alfworld":
                raise ValueError(f"line {lineno}: benchmark is not alfworld")
            if row["comparison/rollout_accounting"] != "active_env_decisions":
                raise ValueError(f"line {lineno}: wrong rollout accounting")
            if row["comparison/method"] not in {"coskill", "skillrl", "skill0"}:
                raise ValueError(f"line {lineno}: unknown method")
            if method is None:
                method = row["comparison/method"]
            elif method != row["comparison/method"]:
                raise ValueError(f"line {lineno}: multiple methods in one file")
            for key in REQUIRED:
                if key in {
                    "comparison/method", "comparison/benchmark", "comparison/rollout_accounting",
                    "tokens/small_model/accounting", "tokens/large_model/accounting",
                    "coskill/cloud_update_fired",
                }:
                    continue
                number(row, key)
            if abs(number(row, "tokens/small_model/total") - number(row, "tokens/small_model/prompt") - number(row, "tokens/small_model/response")) > 1e-6:
                raise ValueError(f"line {lineno}: small-model total != prompt + response")
            if abs(number(row, "tokens/large_model/total") - number(row, "tokens/large_model/prompt") - number(row, "tokens/large_model/completion")) > 1e-6:
                raise ValueError(f"line {lineno}: large-model total != prompt + completion")
            if number(row, "episode/wins") > number(row, "episode/count"):
                raise ValueError(f"line {lineno}: wins exceeds episode count")
            if abs(number(row, "episode/success_rate") - number(row, "episode/wins") / max(number(row, "episode/count"), 1.0)) > 1e-6:
                raise ValueError(f"line {lineno}: success rate != wins / count")
            if previous is not None:
                for key in (
                    "rollout/global_episode_end", "episode/wins_cumulative",
                    "episode/action_count_cumulative", "tokens/small_model/prompt_cumulative",
                    "tokens/small_model/response_cumulative", "tokens/small_model/total_cumulative",
                    "tokens/large_model/prompt_cumulative",
                    "tokens/large_model/completion_cumulative",
                    "tokens/large_model/total_cumulative", "experiment/cloud_round",
                ):
                    if number(row, key) < number(previous, key):
                        raise ValueError(f"line {lineno}: cumulative field decreased: {key}")
            previous = row
            rows += 1
    if rows == 0:
        raise ValueError("no metric rows")
    return method, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failures = 0
    for path in args.paths:
        try:
            method, rows = validate(path)
            print(f"OK {path}: method={method}, groups={rows}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"ERROR {path}: {exc}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
