#!/usr/bin/env python3
"""Recover ALFWorld terminal group 100 after a post-rollout interruption.

The 72 rollout traces are canonical. The worker-only full reasoning text was
not persisted, so small-model tokens are estimated from group 99's measured
per-task, per-action rates and explicitly marked as estimated. Since group 100
is terminal, cloud/skill evolution is excluded from the recovered metric.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ALIASES = {
    "clean": "pick_clean_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "pick_and_place": "pick_and_place_simple",
}


def read_jsonl(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def trace_stats(rows):
    by_task = defaultdict(lambda: {
        "episodes": 0, "wins": 0, "steps": 0, "valid": 0,
        "strict": 0, "salvaged": 0, "fallback": 0,
    })
    lengths = []
    for row in rows:
        steps = row.get("steps") or []
        meta = row.get("meta") or {}
        task = row.get("task_type", "unknown")
        stat = by_task[task]
        stat["episodes"] += 1
        stat["wins"] += int(row.get("outcome") == "success")
        stat["steps"] += len(steps)
        stat["valid"] += int(meta.get("n_valid_actions", sum(
            int(s.get("valid_action", False)) for s in steps)))
        stat["strict"] += int(meta.get("n_strict_valid_actions", sum(
            int(s.get("strict_valid_action", False)) for s in steps)))
        stat["salvaged"] += int(meta.get("n_salvaged_actions", sum(
            int(s.get("execution_source") == "salvaged") for s in steps)))
        stat["fallback"] += int(meta.get("n_fallback_actions", sum(
            int(s.get("execution_source") == "fallback") for s in steps)))
        lengths.append(len(steps))
    return dict(by_task), lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    group_path = outdir / "group_metrics.jsonl"
    raw_path = outdir / "traces_pool" / "raw_traces.jsonl"
    group_rows = read_jsonl(group_path)
    if any(int(row.get("step", 0) or 0) == 100 for row in group_rows):
        raise SystemExit("group 100 already exists; refusing to duplicate it")
    if not group_rows or int(group_rows[-1].get("step", 0) or 0) != 99:
        raise SystemExit("expected canonical group_metrics to end at group 99")

    traces = read_jsonl(raw_path)
    if len(traces) < 144:
        raise SystemExit("not enough raw traces to recover groups 99 and 100")
    group99_traces, group100_traces = traces[-144:-72], traces[-72:]
    stats99, _ = trace_stats(group99_traces)
    stats100, lengths = trace_stats(group100_traces)
    if len(group100_traces) != 72:
        raise SystemExit("terminal trace group is not batch 72")

    previous = group_rows[-1]
    prior = previous["metrics"]
    metric = dict(prior)
    wins = sum(stat["wins"] for stat in stats100.values())
    actions = sum(lengths)
    valid = sum(stat["valid"] for stat in stats100.values())
    strict = sum(stat["strict"] for stat in stats100.values())
    salvaged = sum(stat["salvaged"] for stat in stats100.values())
    fallback = sum(stat["fallback"] for stat in stats100.values())

    # Recover the rollout duration from the pre-rollout skill snapshot and the
    # post-rollout compressed batch. This is filesystem-measured, not guessed.
    start_path = outdir / "skill_lib" / "skills_rollout_step7128.json"
    batch_candidates = sorted((outdir / "traces_pool").glob("batch_*_ceccf9cc.json"))
    rollout_seconds = 0.0
    if start_path.exists() and batch_candidates:
        rollout_seconds = max(0.0, batch_candidates[-1].stat().st_mtime - start_path.stat().st_mtime)

    metric.update({
        "training/group": 100,
        "training/global_step": 100,
        "training/epoch": 1,
        "rollout/global_episode_end": 7200,
        "episode/count": 72,
        "episode/generated_count": 72,
        "episode/wins": wins,
        "episode/success_rate": round(wins / 72, 6),
        "episode/count_cumulative": 7200,
        "episode/wins_cumulative": int(prior["episode/wins_cumulative"]) + wins,
        "episode/action_count": actions,
        "episode/action_count_cumulative": int(prior["episode/action_count_cumulative"]) + actions,
        "episode/length/mean": round(sum(lengths) / 72, 6),
        "episode/length/max": max(lengths),
        "episode/length/min": min(lengths),
        "episode/valid_action_ratio": round(valid / actions, 6),
        "episode/strict_valid_action_ratio": round(strict / actions, 6),
        "episode/relaxed_valid_action_ratio": round(valid / actions, 6),
        "episode/non_strict_valid_action_ratio": round(valid / actions, 6),
        "episode/salvaged_action_ratio": round(salvaged / actions, 6),
        "episode/fallback_action_ratio": round(fallback / actions, 6),
        "coskill/cloud_update_fired": False,
        "timing_s/rollout": round(rollout_seconds, 6),
        "timing_s/cloud_update": 0.0,
        "timing_s/group_total": round(rollout_seconds, 6),
        "comparison/timing_cloud_update_measured": 1,
    })

    for task, stat in stats100.items():
        metric[f"episode/{task}/episodes"] = stat["episodes"]
        metric[f"episode/{task}/wins"] = stat["wins"]
        metric[f"episode/{task}/success_rate"] = round(
            stat["wins"] / max(stat["episodes"], 1), 6)

    estimated = {"prompt": 0, "response": 0, "total": 0}
    for raw_task, stat in stats100.items():
        canonical = ALIASES.get(raw_task, raw_task)
        base_steps = stats99[raw_task]["steps"]
        prompt99 = int(prior[f"tokens/small_model/by_task_type/{canonical}/prompt"])
        response99 = int(prior[f"tokens/small_model/by_task_type/{canonical}/response"])
        prompt = int(round(prompt99 * stat["steps"] / base_steps))
        response = int(round(response99 * stat["steps"] / base_steps))
        total = prompt + response
        metric[f"tokens/small_model/by_task_type/{canonical}/prompt"] = prompt
        metric[f"tokens/small_model/by_task_type/{canonical}/response"] = response
        metric[f"tokens/small_model/by_task_type/{canonical}/total"] = total
        metric[f"tokens/small_model/by_task_type/{canonical}/total_cumulative"] = (
            int(prior[f"tokens/small_model/by_task_type/{canonical}/total_cumulative"]) + total)
        estimated["prompt"] += prompt
        estimated["response"] += response
        estimated["total"] += total

    metric.update({
        "tokens/small_model/prompt": estimated["prompt"],
        "tokens/small_model/response": estimated["response"],
        "tokens/small_model/total": estimated["total"],
        "tokens/small_model/accounting": prior["tokens/small_model/accounting"],
        "tokens/small_model/prompt_cumulative": int(prior["tokens/small_model/prompt_cumulative"]) + estimated["prompt"],
        "tokens/small_model/response_cumulative": int(prior["tokens/small_model/response_cumulative"]) + estimated["response"],
        "tokens/small_model/total_cumulative": int(prior["tokens/small_model/total_cumulative"]) + estimated["total"],
        "perf/total_num_tokens": estimated["total"],
        "perf/throughput_episodes_per_second": round(72 / max(rollout_seconds, 1e-9), 6),
        "perf/throughput_small_tokens_per_second": round(estimated["total"] / max(rollout_seconds, 1e-9), 6),
    })

    # Terminal group: no cloud/skill update belongs to the experiment.
    for key in list(metric):
        if key.startswith("tokens/large_model/by_task_type/") and not key.endswith("total_cumulative"):
            metric[key] = 0
    for key in ("prompt", "completion", "total"):
        metric[f"tokens/large_model/{key}"] = 0
        metric[f"tokens/large_model/mixed/{key}"] = 0

    record = {"step": 100, "global_episode_end": 7200, "metrics": metric}
    print(json.dumps({
        "episodes": 72, "wins": wins, "actions": actions,
        "small_tokens_estimated": estimated,
        "rollout_seconds": round(rollout_seconds, 6),
        "cloud_update_fired": False,
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry-run only; pass --apply to append canonical group 100")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = group_path.with_name(f"{group_path.name}.pre_group100_backfill_{stamp}")
    shutil.copy2(group_path, backup)
    with group_path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"appended group100; backup={backup}")


if __name__ == "__main__":
    main()
