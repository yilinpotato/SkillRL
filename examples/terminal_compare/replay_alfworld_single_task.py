"""Replay every environment action from one task in a completed ALFWorld run."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_alfworld_serial import (  # noqa: E402
    _duration,
    _read_jsonl,
    _write_json,
    build_timeline,
    print_configuration_header,
)


def build_single_task_timeline(
    run_dir: Path, task_index: int, speedup: float
) -> dict[str, Any]:
    timeline = build_timeline(run_dir, speedup)
    rows = timeline["rows"]
    if task_index < 1 or task_index > len(rows):
        raise ValueError(f"--task-index must be between 1 and {len(rows)}")

    traces = _read_jsonl(run_dir / "traces_pool" / "raw_traces.jsonl")
    row = rows[task_index - 1]
    trace = traces[task_index - 1]
    steps = trace.get("steps") or []
    if len(steps) != int(row["steps"]):
        raise ValueError(
            f"Task {task_index} step count mismatch: {len(steps)} != {row['steps']}"
        )
    step_delay = float(row["replay_delay_seconds"]) / max(len(steps), 1)
    return {
        "schema_version": 1,
        "source_run": timeline["source_run"],
        "simulation_only": True,
        "speedup": speedup,
        "max_environment_steps": timeline["max_environment_steps"],
        "run_configuration": timeline["run_configuration"],
        "task_index": task_index,
        "task_count": len(rows),
        "task_type": row["task_type"],
        "task": row["task"],
        "success": row["success"],
        "used_steps": row["steps"],
        "prompt_tokens": row["prompt_tokens"],
        "response_tokens": row["response_tokens"],
        "total_tokens": row["total_tokens"],
        "estimated_actual_task_seconds": row["estimated_actual_seconds"],
        "estimated_actual_task_time": _duration(row["estimated_actual_seconds"]),
        "replay_task_seconds": row["replay_delay_seconds"],
        "replay_step_delay_seconds": step_delay,
        "steps": [
            {
                **step,
                "replay_delay_seconds": step_delay,
            }
            for step in steps
        ],
    }


def replay_single_task(
    timeline: dict[str, Any],
    label: str,
    skill_source_label: str,
    skill_source: str,
    gpu_index: int,
    gpu_name: str,
    no_wait: bool,
) -> None:
    print_configuration_header(
        timeline,
        label,
        skill_source_label,
        skill_source,
        gpu_index,
        gpu_name,
    )
    print(
        f"[SELECT] task={timeline['task_index']:02d}/{timeline['task_count']:02d} | "
        f"task_type={timeline['task_type']} | task={timeline['task']}",
        flush=True,
    )
    print(
        f"[REPLAY] simulated single-task action display | "
        f"speedup={timeline['speedup']:g}x | "
        f"estimated_task_time={timeline['estimated_actual_task_time']}",
        flush=True,
    )

    steps = timeline["steps"]
    action_width = max(len(str(step.get("action") or "")) for step in steps)
    max_steps = int(timeline["max_environment_steps"])
    for step in steps:
        if not no_wait:
            time.sleep(float(step["replay_delay_seconds"]))
        print(
            f"[{label}][STEP {int(step['step']):02d}/{max_steps:02d}] "
            f"action={str(step.get('action') or ''):<{action_width}}",
            flush=True,
        )

    print(
        f"[{label}][TASK TOTAL] steps={len(steps)}/{max_steps} | "
        f"{'SUCCESS' if timeline['success'] else 'FAIL'} | "
        f"estimated_task_time={timeline['estimated_actual_task_time']}",
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
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--speedup", type=float, default=100.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.speedup <= 0:
        parser.error("--speedup must be positive")
    return args


def main() -> None:
    args = parse_args()
    timeline = build_single_task_timeline(
        args.run_dir, args.task_index, args.speedup
    )
    output_path = args.run_dir / f"single_task_replay_{args.task_index:02d}.json"
    _write_json(output_path, timeline)
    replay_single_task(
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
