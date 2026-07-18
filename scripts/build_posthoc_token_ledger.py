#!/usr/bin/env python3
"""Build a non-destructive, cross-method token ledger from completed runs.

This tool intentionally keeps two non-interchangeable quantities separate:

* ``online_model_traffic``: actual prompt + generated/provider tokens used to
  obtain agent decisions.  This is the quantity to normalize by episodes,
  environment decisions, or successful episodes.
* ``perf_batch_tokens``: local GRPO batch tokens summed over optimizer steps.
  This is a training-scale/throughput quantity, not an API evaluation cost.

It never alters its inputs.  Legacy metrics lacking episode/action counts stay
``null`` rather than being guessed from a nominal rollout size.

Examples:
  python scripts/build_posthoc_token_ledger.py \
    --benchmark alfworld \
    --train coskill=/path/to/CoSkill/metrics.jsonl \
    --train skillrl=/path/to/SkillRL/metrics.jsonl \
    --train skill0=/path/to/Skill0/metrics.jsonl \
    --api-eval deepseek_full_skills=/path/to/alfworld_api_eval.overall_summary.json \
    --output /path/to/token_ledger.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "posthoc-token-ledger/v1"


def number(value: Any) -> float | None:
    """Return a finite numeric scalar, without turning missing data into zero."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    result = number(value)
    return int(round(result)) if result is not None else None


def add(values: Iterable[float | int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return int(round(sum(present))) if present else None


def ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def parse_named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    name, raw_path = text.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("use non-empty NAME=PATH")
    return name, Path(raw_path).expanduser()


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    """Read flat or tracker-wrapped metric rows and keep the last duplicate step.

    A resumed run commonly writes a checkpoint boundary twice.  Retaining the
    later record prevents counting that logical optimizer step twice.  A run
    whose counter truly resets must be supplied as a separate ``--train`` input.
    """
    latest: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    anonymous: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            metrics = payload.get("metrics", payload)
            if not isinstance(metrics, dict):
                raise ValueError(f"{path}:{line_number}: no metrics object")

            step_key = None
            for key in ("training/global_step", "training/group", "step"):
                value = integer(metrics.get(key))
                if value is not None:
                    step_key = (key, value)
                    break
            if step_key is None:
                value = integer(payload.get("step"))
                step_key = ("payload/step", value) if value is not None else None
            if step_key is None:
                anonymous.append((line_number, metrics))
            else:
                latest[step_key] = (line_number, metrics)

    # Log order is the safe order for a resumed run; the retained duplicate
    # carries its final position in the file.
    retained = anonymous + list(latest.values())
    return [metrics for _, metrics in sorted(retained, key=lambda item: item[0])]


def cumulative_or_sum(rows: list[dict[str, Any]], raw_key: str, cumulative_key: str) -> tuple[int | None, str | None]:
    cumulative_values = [integer(row.get(cumulative_key)) for row in rows]
    cumulative_values = [value for value in cumulative_values if value is not None]
    if cumulative_values:
        return max(cumulative_values), f"latest_max:{cumulative_key}"
    raw_values = [integer(row.get(raw_key)) for row in rows]
    raw_values = [value for value in raw_values if value is not None]
    if raw_values:
        return sum(raw_values), f"sum:{raw_key}"
    return None, None


def legacy_snapshot(rows: list[dict[str, Any]], key: str) -> int | None:
    """Cloud counters in old logs are cumulative snapshots, never deltas."""
    values = [integer(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def training_entry(name: str, path: Path, benchmark: str, fallback_episodes_per_row: int | None) -> dict[str, Any]:
    rows = load_metric_rows(path)
    if not rows:
        raise ValueError(f"{path}: no metric rows")

    small_prompt, prompt_source = cumulative_or_sum(
        rows, "tokens/small_model/prompt", "tokens/small_model/prompt_cumulative"
    )
    small_response, response_source = cumulative_or_sum(
        rows, "tokens/small_model/response", "tokens/small_model/response_cumulative"
    )
    small_total, small_source = cumulative_or_sum(
        rows, "tokens/small_model/total", "tokens/small_model/total_cumulative"
    )
    if small_total is None and small_prompt is not None and small_response is not None:
        small_total = small_prompt + small_response
        small_source = "derived:small_prompt_plus_small_response"

    large_prompt, large_prompt_source = cumulative_or_sum(
        rows, "tokens/large_model/prompt", "tokens/large_model/prompt_cumulative"
    )
    large_completion, large_completion_source = cumulative_or_sum(
        rows, "tokens/large_model/completion", "tokens/large_model/completion_cumulative"
    )
    large_total, large_source = cumulative_or_sum(
        rows, "tokens/large_model/total", "tokens/large_model/total_cumulative"
    )
    if large_total is None:
        large_total = legacy_snapshot(rows, "coskill/cloud/large_model_total_tokens")
        if large_total is not None:
            large_source = "legacy_cumulative_snapshot:coskill/cloud/large_model_total_tokens"
    if large_total is None and large_prompt is not None and large_completion is not None:
        large_total = large_prompt + large_completion
        large_source = "derived:large_prompt_plus_large_completion"

    episodes, episode_source = cumulative_or_sum(rows, "episode/count", "episode/count_cumulative")
    if episodes is None and fallback_episodes_per_row is not None:
        episodes = fallback_episodes_per_row * len(rows)
        episode_source = f"assumption:{fallback_episodes_per_row}_episodes_per_unique_metric_row"
    actions, action_source = cumulative_or_sum(rows, "episode/action_count", "episode/action_count_cumulative")
    wins, wins_source = cumulative_or_sum(rows, "episode/wins", "episode/wins_cumulative")
    perf_total = add(integer(row.get("perf/total_num_tokens")) for row in rows)

    online_total = add((small_total, large_total))
    return {
        "name": name,
        "method": name.split("_", 1)[0],
        "benchmark": benchmark,
        "scope": "training_rollouts",
        "token_source": "native_local_tokenizer_and_logged_provider_usage",
        "source_files": [str(path)],
        "unique_metric_rows": len(rows),
        "episodes": episodes,
        "env_decisions": actions,
        "successes": wins,
        "success_rate": ratio(wins, episodes),
        "small_model_prompt_tokens": small_prompt,
        "small_model_response_tokens": small_response,
        "small_model_total_tokens": small_total,
        "large_model_prompt_tokens": large_prompt,
        "large_model_completion_tokens": large_completion,
        "large_model_total_tokens": large_total,
        "online_model_traffic_tokens": online_total,
        "perf_batch_tokens": perf_total,
        "perf_matches_small_traffic": (perf_total == small_total) if perf_total is not None and small_total is not None else None,
        "token_sources": {
            "small_prompt": prompt_source,
            "small_response": response_source,
            "small_total": small_source,
            "large_prompt": large_prompt_source,
            "large_completion": large_completion_source,
            "large_total": large_source,
            "episodes": episode_source,
            "env_decisions": action_source,
            "successes": wins_source,
        },
        "normalised": {
            "online_tokens_per_episode": ratio(online_total, episodes),
            "online_tokens_per_env_decision": ratio(online_total, actions),
            "online_tokens_per_success": ratio(online_total, wins),
        },
        "limitations": [
            "perf_batch_tokens is retained as a separate training-scale metric; do not add it to online_model_traffic_tokens.",
            "Native token counts remain model-tokenizer-specific.",
        ],
    }


def api_entry(name: str, path: Path, benchmark: str) -> dict[str, Any]:
    """Load an API evaluator's overall task-summary object without re-tokenizing."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path}: expected a non-empty task summary object")
    task_summaries = [value for value in payload.values() if isinstance(value, dict)]
    if not task_summaries:
        raise ValueError(f"{path}: no task summary objects")

    episodes = add(integer(item.get("episodes")) for item in task_summaries)
    prompt = add(integer(item.get("total_prompt_tokens")) for item in task_summaries)
    completion = add(integer(item.get("total_completion_tokens")) for item in task_summaries)
    total = add(integer(item.get("total_tokens")) for item in task_summaries)
    decisions_float = sum(
        (number(item.get("episodes")) or 0.0) * (number(item.get("avg_episode_steps")) or 0.0)
        for item in task_summaries
    )
    decisions = int(round(decisions_float)) if decisions_float else None
    wins_float = sum(
        (number(item.get("episodes")) or 0.0) * (number(item.get("success_rate")) or 0.0)
        for item in task_summaries
    )
    wins = int(round(wins_float)) if wins_float else None
    models = sorted({str(item.get("model")) for item in task_summaries if item.get("model")})

    return {
        "name": name,
        "method": name.split("_", 1)[0],
        "benchmark": benchmark,
        "scope": "heldout_api_evaluation",
        "token_source": "provider_api_usage",
        "model": models[0] if len(models) == 1 else models,
        "source_files": [str(path)],
        "unique_metric_rows": None,
        "episodes": episodes,
        "env_decisions": decisions,
        "successes": wins,
        "success_rate": ratio(wins, episodes),
        "small_model_prompt_tokens": None,
        "small_model_response_tokens": None,
        "small_model_total_tokens": None,
        "large_model_prompt_tokens": prompt,
        "large_model_completion_tokens": completion,
        "large_model_total_tokens": total,
        "online_model_traffic_tokens": total,
        "perf_batch_tokens": None,
        "perf_matches_small_traffic": None,
        "token_sources": {
            "large_prompt": "sum:provider_usage.total_prompt_tokens",
            "large_completion": "sum:provider_usage.total_completion_tokens",
            "large_total": "sum:provider_usage.total_tokens",
            "episodes": "sum:task_summary.episodes",
            "env_decisions": "derived:sum(episodes*avg_episode_steps)",
            "successes": "derived:round(sum(episodes*success_rate))",
        },
        "normalised": {
            "online_tokens_per_episode": ratio(total, episodes),
            "online_tokens_per_env_decision": ratio(total, decisions),
            "online_tokens_per_success": ratio(total, wins),
        },
        "limitations": [
            "Provider usage may include reasoning tokens not visible in raw text.",
            "This is an evaluation total, not a local GRPO training batch metric.",
        ],
    }


def csv_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "name", "method", "benchmark", "scope", "token_source", "episodes", "env_decisions", "successes",
        "success_rate", "small_model_total_tokens", "large_model_total_tokens", "online_model_traffic_tokens",
        "perf_batch_tokens", "perf_matches_small_traffic",
    )
    rows = []
    for entry in entries:
        row = {key: entry.get(key) for key in columns}
        row.update(entry["normalised"])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=("alfworld", "webshop"))
    parser.add_argument("--train", action="append", default=[], type=parse_named_path, metavar="NAME=METRICS_JSONL")
    parser.add_argument("--api-eval", action="append", default=[], type=parse_named_path, metavar="NAME=OVERALL_SUMMARY_JSON")
    parser.add_argument("--output", required=True, type=Path, help="destination token_ledger.json")
    parser.add_argument("--csv", type=Path, default=None, help="default: sibling token_ledger.csv")
    parser.add_argument(
        "--fallback-episodes-per-row", type=int, default=None,
        help="only for legacy logs with no episode count; explicitly records this assumption",
    )
    args = parser.parse_args()
    if not args.train and not args.api_eval:
        parser.error("provide at least one --train or --api-eval input")
    if args.fallback_episodes_per_row is not None and args.fallback_episodes_per_row <= 0:
        parser.error("--fallback-episodes-per-row must be positive")

    entries = []
    for name, path in args.train:
        entries.append(training_entry(name, path, args.benchmark, args.fallback_episodes_per_row))
    for name, path in args.api_eval:
        entries.append(api_entry(name, path, args.benchmark))
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": args.benchmark,
        "comparison_rule": (
            "Compare online_model_traffic_tokens only after normalising by the same episode or environment-decision unit. "
            "Keep perf_batch_tokens in a separate training-scale panel."
        ),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = args.csv or args.output.with_suffix(".csv")
    rows = csv_rows(entries)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
