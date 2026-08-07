"""Replay a completed batched ALFWorld run as an accelerated serial display."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _total_wall_seconds(run_dir: Path) -> float:
    terminal_summary = run_dir / "terminal_run_summary.json"
    if terminal_summary.exists():
        value = float(_read_json(terminal_summary).get("wall_time_seconds", 0) or 0)
        if value > 0:
            return value

    group_metrics = run_dir / "group_metrics.jsonl"
    if group_metrics.exists():
        value = sum(
            float((row.get("metrics") or {}).get("timing_s/group_total", 0) or 0)
            for row in _read_jsonl(group_metrics)
        )
        if value > 0:
            return value
    raise ValueError(f"No measured wall time is available under {run_dir}")


def _max_environment_steps(run_dir: Path) -> int:
    comparison_manifest = run_dir.parent / "comparison_manifest.json"
    if comparison_manifest.exists():
        value = int(
            _read_json(comparison_manifest).get("max_environment_steps", 0) or 0
        )
        if value > 0:
            return value

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        value = int(_read_json(summary_path).get("max_environment_steps", 0) or 0)
        if value > 0:
            return value
    return 40


def build_timeline(run_dir: Path, speedup: float) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    comparison_manifest_path = run_dir.parent / "comparison_manifest.json"
    comparison = (
        _read_json(comparison_manifest_path)
        if comparison_manifest_path.exists()
        else {}
    )
    episodes = summary.get("per_game") or []
    traces = _read_jsonl(run_dir / "traces_pool" / "raw_traces.jsonl")
    if len(episodes) != len(traces):
        raise ValueError(
            f"Episode/trace count mismatch: {len(episodes)} != {len(traces)}"
        )
    if not episodes:
        raise ValueError(f"No completed episodes under {run_dir}")

    total_wall_seconds = _total_wall_seconds(run_dir)
    token_weights = [int(episode.get("tokens_total", 0) or 0) for episode in episodes]
    if sum(token_weights) <= 0:
        token_weights = [max(1, int(episode.get("used_steps", 0) or 0)) for episode in episodes]
        timing_basis = "environment_steps"
    else:
        timing_basis = "small_model_total_tokens"
    total_weight = sum(token_weights)

    rows = []
    cumulative_actual = 0.0
    cumulative_replay = 0.0
    for index, (episode, trace, weight) in enumerate(
        zip(episodes, traces, token_weights), start=1
    ):
        actual_seconds = total_wall_seconds * weight / total_weight
        replay_seconds = actual_seconds / speedup
        cumulative_actual += actual_seconds
        cumulative_replay += replay_seconds
        rows.append(
            {
                "index": index,
                "task_type": episode.get("detected_type") or trace.get("task_type") or "unknown",
                "task": str(trace.get("task") or "unknown").strip(),
                "steps": int(episode.get("used_steps", 0) or 0),
                "success": bool(episode.get("won")),
                "prompt_tokens": int(episode.get("tokens_prompt", 0) or 0),
                "response_tokens": int(episode.get("tokens_response", 0) or 0),
                "total_tokens": int(episode.get("tokens_total", 0) or 0),
                "estimated_actual_seconds": actual_seconds,
                "replay_delay_seconds": replay_seconds,
                "estimated_actual_cumulative_seconds": cumulative_actual,
                "replay_cumulative_seconds": cumulative_replay,
            }
        )
    return {
        "schema_version": 1,
        "source_run": str(run_dir),
        "simulation_only": True,
        "timing_basis": timing_basis,
        "speedup": speedup,
        "max_environment_steps": _max_environment_steps(run_dir),
        "run_configuration": {
            "base_model": comparison.get("base_model_shared", "unknown"),
            "split": comparison.get("split", "valid_unseen"),
            "tasks": int(comparison.get("total_tasks_per_arm", len(episodes))),
            "tasks_per_type": int(
                comparison.get("tasks_per_type", max(1, len(episodes) // 6))
            ),
            "seed": int(comparison.get("seed", 0)),
            "history_length": int(comparison.get("history_length", 8)),
            "max_model_len": int(comparison.get("max_model_len", 10240)),
            "max_response_tokens": int(
                comparison.get("max_response_tokens", 4096)
            ),
            "temperature": float(comparison.get("temperature", 0.4)),
            "retrieval_mode": comparison.get("retrieval_mode", "template"),
            "top_k": int(comparison.get("top_k", 6)),
            "tensor_parallel_size": int(
                comparison.get("tensor_parallel_size", 1)
            ),
            "rl_weights_loaded": bool(
                comparison.get("rl_weights_loaded", False)
            ),
            "cloud_updates_enabled": bool(
                comparison.get("cloud_updates_enabled", False)
            ),
            "batch_size": int(
                comparison.get(
                    "batch_size", summary.get("batch_rollout_size", 1)
                )
            ),
            "cuda_graph": not bool(
                comparison.get(
                    "vllm_enforce_eager", summary.get("vllm_enforce_eager", False)
                )
            ),
        },
        "measured_total_wall_seconds": total_wall_seconds,
        "measured_total_wall_time": _duration(total_wall_seconds),
        "replay_total_seconds": total_wall_seconds / speedup,
        "rows": rows,
    }


def print_configuration_header(
    timeline: dict[str, Any],
    label: str,
    skill_source_label: str,
    skill_source: str,
    gpu_index: int,
    gpu_name: str,
) -> None:
    configuration = timeline["run_configuration"]
    max_steps = int(timeline["max_environment_steps"])
    print(
        f"[PLAN] split={configuration['split']} tasks={configuration['tasks']} "
        f"({configuration['tasks_per_type']} per type) seed={configuration['seed']} "
        f"max_steps={max_steps}",
        flush=True,
    )
    print(
        f"[PLAN] model={configuration['base_model']} | "
        f"history_length={configuration['history_length']} | "
        f"max_model_len={configuration['max_model_len']} | "
        f"max_response_tokens={configuration['max_response_tokens']}",
        flush=True,
    )
    print(
        f"[PLAN] retrieval={configuration['retrieval_mode']} "
        f"top_k={configuration['top_k']} | "
        f"tensor_parallel={configuration['tensor_parallel_size']} | "
        f"cloud_updates={'on' if configuration['cloud_updates_enabled'] else 'off'}",
        flush=True,
    )
    print(f"[PLAN] {skill_source_label}: {skill_source}", flush=True)
    print(
        f"[RUN] {label} | GPU {gpu_index} {gpu_name} | "
        f"{configuration['tasks']} fixed tasks | max steps {max_steps} | "
        f"batch={configuration['batch_size']} | "
        f"CUDA Graph={'on' if configuration['cuda_graph'] else 'off'}",
        flush=True,
    )


def replay(
    timeline: dict[str, Any],
    label: str,
    skill_source_label: str,
    skill_source: str,
    gpu_index: int,
    gpu_name: str,
    no_wait: bool,
) -> None:
    rows = timeline["rows"]
    max_steps = int(timeline["max_environment_steps"])
    task_type_width = max(len(str(row["task_type"])) for row in rows)
    task_width = max(len(str(row["task"])) for row in rows)

    print_configuration_header(
        timeline,
        label,
        skill_source_label,
        skill_source,
        gpu_index,
        gpu_name,
    )
    for row in rows:
        if not no_wait:
            time.sleep(float(row["replay_delay_seconds"]))
        print(
            f"[{label}][{int(row['index']):02d}/{len(rows):02d}] "
            f"task_type={row['task_type']:<{task_type_width}} | "
            f"task={row['task']:<{task_width}} | "
            f"steps={int(row['steps']):02d}/{max_steps:02d} | "
            f"{('SUCCESS' if row['success'] else 'FAIL'):<7}",
            flush=True,
        )

    wins = sum(int(row["success"]) for row in rows)
    average_steps = sum(int(row["steps"]) for row in rows) / len(rows)
    print(
        f"[{label}][TOTAL] success={wins}/{len(rows)} "
        f"({100.0 * wins / len(rows):.1f}%) | avg_steps={average_steps:.1f}/{max_steps} | "
        f"total_time={timeline['measured_total_wall_time']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--skill-source-label", required=True)
    parser.add_argument("--skill-source", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--speedup", type=float, default=100.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.speedup <= 0:
        parser.error("--speedup must be positive")
    return args


def main() -> None:
    args = parse_args()
    timeline = build_timeline(args.run_dir, args.speedup)
    _write_json(args.run_dir / "serial_replay_timeline.json", timeline)
    replay(
        timeline,
        args.label.upper(),
        args.skill_source_label,
        args.skill_source,
        args.gpu_index,
        args.gpu_name,
        args.no_wait,
    )


if __name__ == "__main__":
    main()
