#!/usr/bin/env python3
"""Splice a targeted pick2 no-RL suffix into the published six-task metrics.

This intentionally replaces the 12 ``pick_two_obj_and_place`` episodes in
each source group 51--100 (600 records total). All 6,600 non-replaced source
episodes remain unchanged. It is therefore valid for six-task per-task success
and small-model-token comparisons, but is *not* a newly executed joint six-task
CoSkill trajectory: cloud updates after group 50 saw pick2 traces only. The
manifest deliberately marks cloud-token accounting as non-comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PICK2 = "pick_two_obj_and_place"
SOURCE_GROUPS = 100
SOURCE_EPISODES_PER_GROUP = 72
PREFIX_GROUPS = 50
PICK2_PER_GROUP = 12
SUFFIX_EPISODES = (SOURCE_GROUPS - PREFIX_GROUPS) * PICK2_PER_GROUP


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"Expected numeric {label}, got {value!r}")
    return value


def metrics(row: dict[str, Any], label: str) -> dict[str, Any]:
    value = row.get("metrics")
    if not isinstance(value, dict):
        raise SystemExit(f"{label} has no metrics object")
    return value


def task_of(row: dict[str, Any], label: str) -> str:
    value = metrics(row, label).get("episode/detected_type")
    if not isinstance(value, str):
        raise SystemExit(f"{label} has no episode/detected_type")
    return value


def token(row: dict[str, Any], name: str) -> int | float:
    return number(metrics(row, "episode row").get(name, 0), name)


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate_source(source_metrics: list[dict[str, Any]], source_groups: list[dict[str, Any]]) -> None:
    if len(source_metrics) < SOURCE_GROUPS * SOURCE_EPISODES_PER_GROUP:
        raise SystemExit("Source metrics.jsonl has fewer than 7,200 episodes")
    if len(source_groups) < SOURCE_GROUPS:
        raise SystemExit("Source group_metrics.jsonl has fewer than 100 groups")
    expected_steps = list(range(1, SOURCE_GROUPS * SOURCE_EPISODES_PER_GROUP + 1))
    got_steps = [int(number(row.get("step"), "source episode step")) for row in source_metrics[: len(expected_steps)]]
    if got_steps != expected_steps:
        raise SystemExit("Source episode steps must begin at contiguous 1..7200")
    expected_groups = list(range(1, SOURCE_GROUPS + 1))
    got_groups = [int(number(row.get("step"), "source group step")) for row in source_groups[:SOURCE_GROUPS]]
    if got_groups != expected_groups:
        raise SystemExit("Source group steps must begin at contiguous 1..100")
    for group in range(PREFIX_GROUPS + 1, SOURCE_GROUPS + 1):
        start = (group - 1) * SOURCE_EPISODES_PER_GROUP
        block = source_metrics[start : start + SOURCE_EPISODES_PER_GROUP]
        count = sum(task_of(row, f"source group {group}") == PICK2 for row in block)
        if count != PICK2_PER_GROUP:
            raise SystemExit(f"Source group {group} has {count} pick2 episodes, expected {PICK2_PER_GROUP}")


def validate_suffix(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    if len(rows) != SUFFIX_EPISODES:
        raise SystemExit(f"Pick2 suffix must contain {SUFFIX_EPISODES} episodes, found {len(rows)}")
    steps = [int(number(row.get("step"), "suffix episode step")) for row in rows]
    if steps != list(range(1, SUFFIX_EPISODES + 1)):
        raise SystemExit("Pick2 suffix episode steps must be contiguous 1..600")
    if any(task_of(row, "pick2 suffix") != PICK2 for row in rows):
        raise SystemExit("Pick2 suffix contains an episode of another task type")
    if len(groups) != 50:
        raise SystemExit(f"Pick2 suffix must contain 50 group rows, found {len(groups)}")
    for index, row in enumerate(groups, 1):
        if int(number(row.get("step"), "suffix group step")) != index:
            raise SystemExit("Pick2 suffix group steps must be contiguous 1..50")
        if int(number(row.get("global_episode_end"), "suffix group endpoint")) != index * PICK2_PER_GROUP:
            raise SystemExit("Each pick2 suffix group must end after exactly 12 episodes")


def replace_episode_rows(
    source: list[dict[str, Any]], suffix: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged = [copy.deepcopy(row) for row in source[: SOURCE_GROUPS * SOURCE_EPISODES_PER_GROUP]]
    replacement_map: list[dict[str, Any]] = []
    cursor = 0
    for group in range(PREFIX_GROUPS + 1, SOURCE_GROUPS + 1):
        start = (group - 1) * SOURCE_EPISODES_PER_GROUP
        for source_index in range(start, start + SOURCE_EPISODES_PER_GROUP):
            if task_of(merged[source_index], f"source group {group}") != PICK2:
                continue
            replacement = copy.deepcopy(suffix[cursor])
            replacement_map.append({
                "source_step": int(number(merged[source_index]["step"], "source replacement step")),
                "source_group": group,
                "source_task_type": PICK2,
                "suffix_step": int(number(suffix[cursor]["step"], "suffix replacement step")),
                "suffix_virtual_group": cursor // PICK2_PER_GROUP + 1,
            })
            replacement["step"] = merged[source_index]["step"]
            merged[source_index] = replacement
            cursor += 1
    if cursor != SUFFIX_EPISODES:
        raise AssertionError(f"Internal replacement mismatch: used {cursor} suffix rows")
    return merged, replacement_map


def recompute_episode_cumulatives(rows: list[dict[str, Any]]) -> None:
    episodes_by_task: Counter[str] = Counter()
    wins_by_task: Counter[str] = Counter()
    total_wins = 0
    for index, row in enumerate(rows, 1):
        row["step"] = index
        item = metrics(row, f"merged episode {index}")
        task = task_of(row, f"merged episode {index}")
        won = int(bool(item.get("episode/won")))
        episodes_by_task[task] += 1
        wins_by_task[task] += won
        total_wins += won
        item["episode/running_total_episodes"] = index
        item["episode/running_total_wins"] = total_wins
        item["episode/success_rate"] = ratio(total_wins, index)
        # A driver normally emits all task counters seen so far. Reconstruct the
        # same schema from the replacement sequence, removing stale suffix-only
        # counters before inserting the current six-task ledger.
        for key in list(item):
            if key.startswith("episode/") and key.endswith(("/episodes", "/wins", "/success_rate")):
                del item[key]
        for seen_task in sorted(episodes_by_task):
            prefix = f"episode/{seen_task}"
            item[f"{prefix}/episodes"] = episodes_by_task[seen_task]
            item[f"{prefix}/wins"] = wins_by_task[seen_task]
            item[f"{prefix}/success_rate"] = ratio(wins_by_task[seen_task], episodes_by_task[seen_task])


def aggregate_group(block: list[dict[str, Any]], original: dict[str, Any], group: int) -> dict[str, Any]:
    """Retain source-only cloud fields, recompute episode and small-token fields."""
    row = copy.deepcopy(original)
    row["step"] = group
    row["global_episode_end"] = group * SOURCE_EPISODES_PER_GROUP
    item = metrics(row, f"source group {group}")
    won = sum(int(bool(metrics(ep, "episode").get("episode/won"))) for ep in block)
    lengths = [number(metrics(ep, "episode").get("episode/length", 0), "episode/length") for ep in block]
    actions = sum(lengths)
    prompt = sum(token(ep, "tokens/small_model/prompt") for ep in block)
    response = sum(token(ep, "tokens/small_model/response") for ep in block)
    total = sum(token(ep, "tokens/small_model/total") for ep in block)
    valid = sum(number(metrics(ep, "episode").get("episode/valid_actions", 0), "episode/valid_actions") for ep in block)
    strict = sum(number(metrics(ep, "episode").get("episode/strict_valid_actions", 0), "episode/strict_valid_actions") for ep in block)
    salvaged = sum(number(metrics(ep, "episode").get("episode/salvaged_actions", 0), "episode/salvaged_actions") for ep in block)
    fallback = sum(number(metrics(ep, "episode").get("episode/fallback_actions", 0), "episode/fallback_actions") for ep in block)
    item.update({
        "episode/count": len(block),
        "episode/generated_count": len(block),
        "episode/wins": won,
        "episode/success_rate": ratio(won, len(block)),
        "episode/action_count": actions,
        "episode/length/mean": ratio(sum(lengths), len(lengths)),
        "episode/length/max": max(lengths),
        "episode/length/min": min(lengths),
        "episode/valid_action_ratio": ratio(valid, actions),
        "episode/strict_valid_action_ratio": ratio(strict, actions),
        "episode/relaxed_valid_action_ratio": ratio(valid, actions),
        "episode/non_strict_valid_action_ratio": ratio(valid, actions),
        "episode/salvaged_action_ratio": ratio(salvaged, actions),
        "episode/fallback_action_ratio": ratio(fallback, actions),
        "tokens/small_model/prompt": prompt,
        "tokens/small_model/response": response,
        "tokens/small_model/total": total,
        "comparison/hybrid_pick2_replacement": 1,
        "comparison/hybrid_cloud_token_semantics": "not_a_single_joint_six_task_run",
    })
    # Cloud-token fields come from the source mixed run and cannot be replaced
    # by a pick2-only suffix without fabricating a joint trajectory.
    return row


def recompute_group_cumulatives(rows: list[dict[str, Any]]) -> None:
    cumulative = defaultdict(float)
    for row in rows:
        item = metrics(row, "merged group")
        for key, group_key in (
            ("episode/count_cumulative", "episode/count"),
            ("episode/wins_cumulative", "episode/wins"),
            ("episode/action_count_cumulative", "episode/action_count"),
            ("tokens/small_model/prompt_cumulative", "tokens/small_model/prompt"),
            ("tokens/small_model/response_cumulative", "tokens/small_model/response"),
            ("tokens/small_model/total_cumulative", "tokens/small_model/total"),
        ):
            cumulative[key] += number(item.get(group_key, 0), group_key)
            item[key] = int(cumulative[key]) if float(cumulative[key]).is_integer() else cumulative[key]


def repaired_summary(rows: list[dict[str, Any]], final_group: dict[str, Any]) -> dict[str, Any]:
    """A compact, explicit summary for analysis without claiming joint execution."""
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "wins": 0})
    small_by_task: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"prompt": 0, "response": 0, "total": 0}
    )
    for row in rows:
        item = metrics(row, "merged episode")
        task = task_of(row, "merged episode")
        by_task[task]["episodes"] += 1
        by_task[task]["wins"] += int(bool(item.get("episode/won")))
        for key, field in (("tokens/small_model/prompt", "prompt"),
                           ("tokens/small_model/response", "response"),
                           ("tokens/small_model/total", "total")):
            small_by_task[task][field] += number(item.get(key, 0), key)
    item = metrics(final_group, "final group")
    return {
        "status": "repaired_hybrid_metrics",
        "total_episodes": len(rows),
        "wins": item["episode/wins_cumulative"],
        "success_rate": ratio(item["episode/wins_cumulative"], item["episode/count_cumulative"]),
        "completed_rollout_groups": SOURCE_GROUPS,
        "per_task": {
            task: {
                **counts,
                "success_rate": ratio(counts["wins"], counts["episodes"]),
                "small_model_tokens": dict(small_by_task[task]),
            }
            for task, counts in sorted(by_task.items())
        },
        "small_model_tokens": {
            "prompt": item["tokens/small_model/prompt_cumulative"],
            "response": item["tokens/small_model/response_cumulative"],
            "total": item["tokens/small_model/total_cumulative"],
            "accounting": "vllm_request_tokens_single_pass",
        },
        "large_model_tokens": {
            "semantics": "not a joint six-task rerun; inspect source and pick2 suffix separately",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True, help="Original full six-task noRL output")
    parser.add_argument("--pick2-suffix-run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_metrics = read_jsonl(args.source_run_dir / "metrics.jsonl")
    source_groups = read_jsonl(args.source_run_dir / "group_metrics.jsonl")
    suffix_metrics = read_jsonl(args.pick2_suffix_run_dir / "metrics.jsonl")
    suffix_groups = read_jsonl(args.pick2_suffix_run_dir / "group_metrics.jsonl")
    validate_source(source_metrics, source_groups)
    validate_suffix(suffix_metrics, suffix_groups)

    merged_metrics, replacement_map = replace_episode_rows(source_metrics, suffix_metrics)
    recompute_episode_cumulatives(merged_metrics)
    merged_groups = [
        aggregate_group(
            merged_metrics[(group - 1) * SOURCE_EPISODES_PER_GROUP : group * SOURCE_EPISODES_PER_GROUP],
            source_groups[group - 1],
            group,
        )
        for group in range(1, SOURCE_GROUPS + 1)
    ]
    recompute_group_cumulatives(merged_groups)

    final = metrics(merged_groups[-1], "final group")
    manifest = {
        "kind": "hybrid_six_task_metrics_with_pick2_groups_51_100_replaced",
        "source_run_dir": str(args.source_run_dir.resolve()),
        "pick2_suffix_run_dir": str(args.pick2_suffix_run_dir.resolve()),
        "source_groups_retained": [1, 50],
        "source_non_pick2_episodes_retained": 6600,
        "pick2_replaced_virtual_groups": [51, 100],
        "pick2_replaced_episodes": SUFFIX_EPISODES,
        "replacement_map": "pick2_replacement_map.jsonl",
        "merged_episodes": len(merged_metrics),
        "merged_groups": len(merged_groups),
        "merged_success_rate": final["episode/wins_cumulative"] / final["episode/count_cumulative"],
        "small_model_token_semantics": "exact splice: original non-pick2 episode token counts plus re-sampled pick2 episode token counts",
        "large_model_token_semantics": "not comparable as a joint six-task execution; source cloud fields are retained and pick2 suffix cloud costs remain available in the suffix run",
        "warning": "Do not label this as a freshly executed full six-task noRL trajectory. It is a per-task replacement/splice for repairing pick2 results.",
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.outdir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.outdir / "metrics.jsonl", merged_metrics)
    write_jsonl(args.outdir / "group_metrics.jsonl", merged_groups)
    write_jsonl(args.outdir / "pick2_replacement_map.jsonl", replacement_map)
    with (args.outdir / "merge_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.outdir / "summary_repaired.json").open("w", encoding="utf-8") as handle:
        json.dump(repaired_summary(merged_metrics, merged_groups[-1]), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote hybrid six-task metrics to {args.outdir}")


if __name__ == "__main__":
    main()
