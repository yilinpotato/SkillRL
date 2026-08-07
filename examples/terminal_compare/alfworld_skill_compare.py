"""Run a frozen ALFWorld comparison of CoSkill trees and SkillRL flat skills.

Both arms use the same base model, fixed game manifest, decoder settings, and
environment/action projection.  The only arm-specific input is the skill
context injected into the prompt.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))


TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

RUNTIME_TASK_TYPES = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_skill_artifacts(coskill_path: Path, skillrl_path: Path) -> dict[str, Any]:
    coskill = _read_json(coskill_path)
    skillrl = _read_json(skillrl_path)
    trees = coskill.get("skill_trees") or {}
    missing_trees = [RUNTIME_TASK_TYPES[task_type] for task_type in TASK_TYPES if RUNTIME_TASK_TYPES[task_type] not in trees]
    if missing_trees:
        raise ValueError(f"CoSkill artifact is missing task trees: {missing_trees}")
    if skillrl.get("skill_trees"):
        raise ValueError("SkillRL comparison artifact must contain flat skills only")
    if not skillrl.get("general_skills") or not skillrl.get("task_specific_skills"):
        raise ValueError("SkillRL artifact does not contain the expected flat skill library")
    return {
        "coskill": {
            "path": str(coskill_path),
            "sha256": _sha256(coskill_path),
            "tree_count": len(trees),
            "training_group": 50,
            "training_rollouts": 3600,
            "injection": "task_skill_tree_only",
        },
        "skillrl": {
            "path": str(skillrl_path),
            "sha256": _sha256(skillrl_path),
            "general_skill_count": len(skillrl.get("general_skills") or []),
            "task_skill_count": sum(len(values) for values in (skillrl.get("task_specific_skills") or {}).values()),
            "training_step": 50,
            "training_rollouts": 3600,
            "injection": "flat_skills_only",
        },
    }


def create_fixed_manifest(
    output_path: Path,
    data_root: Path,
    split: str,
    tasks_per_type: int,
    seed: int,
) -> dict[str, Any]:
    from mini_test_pen_shelf.env_utils import find_games_by_type

    games: list[dict[str, Any]] = []
    for offset, task_type in enumerate(TASK_TYPES):
        selected = find_games_by_type(
            task_type,
            alfworld_data=str(data_root),
            split=split,
            sample_n=tasks_per_type,
            sample_seed=seed + offset,
            verbose=False,
        )
        if len(selected) != tasks_per_type:
            raise RuntimeError(
                f"Expected {tasks_per_type} games for {task_type}, found {len(selected)}"
            )
        for index, (game_file, _trajectory) in enumerate(selected, start=1):
            games.append(
                {
                    "label": f"{task_type}_{index}",
                    "task_type": task_type,
                    "game_file": os.path.relpath(Path(game_file).resolve(), data_root.resolve()),
                }
            )

    manifest = {
        "schema_version": 1,
        "benchmark": "alfworld",
        "split": split,
        "seed": seed,
        "tasks_per_type": tasks_per_type,
        "total_tasks": len(games),
        "games": games,
    }
    _write_json(output_path, manifest)
    return manifest


def _gpu_state(index: int) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"Unexpected nvidia-smi output: {query}")
    gpu_uuid = fields[5]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    active = [line.strip() for line in processes if line.strip().startswith(gpu_uuid + ",")]
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "memory_used_mib": int(fields[2]),
        "memory_total_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "uuid": gpu_uuid,
        "compute_processes": active,
    }


def require_idle_gpu(index: int) -> dict[str, Any]:
    state = _gpu_state(index)
    if state["compute_processes"]:
        raise RuntimeError(
            f"GPU {index} has active compute processes; refusing to start: "
            + "; ".join(state["compute_processes"])
        )
    if state["utilization_percent"] > 5 or state["memory_used_mib"] > 1024:
        raise RuntimeError(f"GPU {index} is not idle: {state}")
    return state


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _driver_command(
    args: argparse.Namespace,
    arm: str,
    output_dir: Path,
    manifest_path: Path,
    skills_path: Path,
) -> list[str]:
    is_coskill = arm == "coskill"
    total_tasks = len(_read_json(manifest_path)["games"])
    return [
        sys.executable,
        "-u",
        "-m",
        "examples.playbook_evolve.run_playbook_evolve",
        "--outdir",
        str(output_dir),
        "--model_path",
        str(args.model_path),
        "--fixed_games_manifest",
        str(manifest_path),
        "--group_size",
        "1",
        "--batch_rollout_size",
        str(args.batch_size),
        "--max_episodes",
        str(total_tasks),
        "--epochs",
        "1",
        "--max_steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--history_length",
        "8",
        "--gpu_mem_util",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        "1",
        "--vllm_max_num_seqs",
        str(args.batch_size),
        "--vllm_enforce_eager",
        str(args.enforce_eager),
        "--max_model_len",
        "10240",
        "--max_tokens",
        "4096",
        "--temperature",
        str(args.temperature),
        "--skills_json",
        str(skills_path),
        "--retrieval_mode",
        "template",
        "--top_k",
        "6",
        "--enable_hierarchy",
        "0",
        "--enable_coskill",
        "0" if is_coskill else "1",
        "--enable_skill_tree",
        "1" if is_coskill else "0",
        "--enable_skill_tree_evolve",
        "0",
        "--enable_failure_analysis",
        "0",
        "--enable_cloud_updates",
        "0",
        "--log_trajectories",
        "0",
    ]


def run_arm(
    args: argparse.Namespace,
    arm: str,
    output_dir: Path,
    manifest_path: Path,
    skills_path: Path,
) -> None:
    state = require_idle_gpu(args.gpu)
    output_dir.mkdir(parents=True, exist_ok=False)
    command = _driver_command(args, arm, output_dir, manifest_path, skills_path)
    task_count = len(_read_json(manifest_path)["games"])
    environment = os.environ.copy()
    environment.update(
        {
            "ALFWORLD_DATA": str(args.alfworld_data),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONUNBUFFERED": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    print(
        f"[RUN] {arm.upper()} | GPU {state['index']} {state['name']} | "
        f"{task_count} fixed tasks | max steps {args.max_steps} | "
        f"batch={args.batch_size} | CUDA Graph={'off' if args.enforce_eager else 'on'}",
        flush=True,
    )
    log_path = output_dir / "driver.log"
    trace_path = output_dir / "traces_pool" / "raw_traces.jsonl"
    trace_offset = 0
    pending_traces: list[dict[str, Any]] = []
    run_started = time.monotonic()
    episode_pattern = re.compile(
        r"\[driver\] step=(\d+)\s+(\S+)\s+won=(True|False)\s+steps=(\d+)"
    )

    def read_new_traces() -> None:
        nonlocal trace_offset
        if not trace_path.exists():
            return
        with trace_path.open(encoding="utf-8") as trace_handle:
            trace_handle.seek(trace_offset)
            for trace_line in trace_handle:
                if trace_line.strip():
                    pending_traces.append(json.loads(trace_line))
            trace_offset = trace_handle.tell()

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=args.project_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            episode = episode_pattern.search(line)
            if not episode:
                continue
            sequence, task_type, success_text, steps_text = episode.groups()
            read_new_traces()
            trace = pending_traces.pop(0) if pending_traces else {}
            task = str(trace.get("task") or "unknown").strip()
            success = success_text == "True"
            print(
                f"[{arm.upper()}][{int(sequence):02d}/{task_count:02d}] "
                f"task_type={task_type} | task={task} | "
                f"steps={int(steps_text):02d}/{args.max_steps:02d} | "
                f"{'SUCCESS' if success else 'FAIL'}",
                flush=True,
            )
        process.wait()
    if process.returncode:
        raise RuntimeError(f"{arm} failed with exit code {process.returncode}; see {log_path}")

    total_seconds = time.monotonic() - run_started
    summary = _read_json(output_dir / "summary.json")
    episodes = summary.get("per_game") or []
    wins = sum(int(bool(episode.get("won"))) for episode in episodes)
    average_steps = sum(int(episode.get("used_steps", 0)) for episode in episodes) / max(len(episodes), 1)
    final = {
        "arm": arm,
        "tasks": len(episodes),
        "wins": wins,
        "success_rate": wins / max(len(episodes), 1),
        "average_steps": average_steps,
        "wall_time_seconds": total_seconds,
        "wall_time": _duration(total_seconds),
    }
    _write_json(output_dir / "terminal_run_summary.json", final)
    print(
        f"[{arm.upper()}][TOTAL] success={wins}/{len(episodes)} "
        f"({100.0 * wins / max(len(episodes), 1):.1f}%) | "
        f"avg_steps={average_steps:.1f}/{args.max_steps} | "
        f"total_time={_duration(total_seconds)}",
        flush=True,
    )


def _load_episode_rows(arm_dir: Path) -> list[dict[str, Any]]:
    summary = _read_json(arm_dir / "summary.json")
    trace_path = arm_dir / "traces_pool" / "raw_traces.jsonl"
    traces = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(json.loads(line))
    episodes = summary.get("per_game") or []
    if len(episodes) != len(traces):
        raise RuntimeError(
            f"Episode/trace count mismatch in {arm_dir}: {len(episodes)} != {len(traces)}"
        )
    rows = []
    for episode, trace in zip(episodes, traces):
        rows.append(
            {
                "task_type": episode.get("detected_type") or trace.get("task_type") or "unknown",
                "task": str(trace.get("task") or "unknown").strip(),
                "steps": int(episode.get("used_steps", 0)),
                "success": bool(episode.get("won")),
            }
        )
    return rows


def _indexed_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    indexed = {}
    for row in rows:
        base = (row["task_type"], row["task"])
        counts[base] = counts.get(base, 0) + 1
        indexed[(base[0], base[1], counts[base])] = row
    return indexed


def build_combined_rows(output_root: Path) -> list[dict[str, Any]]:
    coskill = _indexed_rows(_load_episode_rows(output_root / "coskill"))
    skillrl = _indexed_rows(_load_episode_rows(output_root / "skillrl"))
    if set(coskill) != set(skillrl):
        missing_coskill = sorted(set(skillrl) - set(coskill))
        missing_skillrl = sorted(set(coskill) - set(skillrl))
        raise RuntimeError(
            f"Arms did not evaluate identical tasks; missing CoSkill={missing_coskill}, "
            f"missing SkillRL={missing_skillrl}"
        )
    task_order = {task_type: index for index, task_type in enumerate(RUNTIME_TASK_TYPES.values())}
    combined = []
    for key in sorted(coskill, key=lambda item: (task_order.get(item[0], 99), item[1], item[2])):
        left, right = coskill[key], skillrl[key]
        combined.append(
            {
                "task_type": key[0],
                "task": key[1],
                "coskill_steps": left["steps"],
                "coskill_success": left["success"],
                "skillrl_steps": right["steps"],
                "skillrl_success": right["success"],
            }
        )
    return combined


def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def render_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    comparison_path = output_root / "comparison_manifest.json"
    step_limit = 40
    if comparison_path.exists():
        step_limit = int(_read_json(comparison_path).get("max_environment_steps", 40))
    print(f"\nALFWorld Frozen-Skill Comparison (same base model, {len(rows)} fixed tasks)")
    print(
        "CoSkill: tree at group 50 | SkillRL: flat skills at step 50 | "
        f"step limit: 0-{step_limit}"
    )
    header = (
        f"{'#':>2}  {'Task type':<24}  {'Task':<52}  "
        f"{'CoSkill steps':>13}  {'Success':>7}  {'SkillRL steps':>13}  {'Success':>7}"
    )
    print(header)
    print("-" * len(header))
    for index, row in enumerate(rows, start=1):
        print(
            f"{index:>2}  {_fit(row['task_type'], 24):<24}  {_fit(row['task'], 52):<52}  "
            f"{row['coskill_steps']:>13}/{step_limit}  "
            f"{('YES' if row['coskill_success'] else 'NO'):>7}  "
            f"{row['skillrl_steps']:>13}/{step_limit}  "
            f"{('YES' if row['skillrl_success'] else 'NO'):>7}"
        )
    coskill_wins = sum(row["coskill_success"] for row in rows)
    skillrl_wins = sum(row["skillrl_success"] for row in rows)
    coskill_steps = sum(row["coskill_steps"] for row in rows) / max(len(rows), 1)
    skillrl_steps = sum(row["skillrl_steps"] for row in rows) / max(len(rows), 1)
    print("-" * len(header))
    print(
        f"TOTAL  tasks={len(rows)} | CoSkill success={coskill_wins}/{len(rows)} "
        f"avg_steps={coskill_steps:.1f} | SkillRL success={skillrl_wins}/{len(rows)} "
        f"avg_steps={skillrl_steps:.1f}"
    )
    timing = {}
    for arm in ("coskill", "skillrl"):
        timing_path = output_root / arm / "terminal_run_summary.json"
        if timing_path.exists():
            timing[arm] = _read_json(timing_path).get("wall_time", "unknown")
    if timing:
        print(
            f"WALL TIME | CoSkill={timing.get('coskill', 'unknown')} | "
            f"SkillRL={timing.get('skillrl', 'unknown')}"
        )

    _write_json(output_root / "terminal_summary.json", {"rows": rows, "wall_time": timing})
    with (output_root / "terminal_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    project_root = DEFAULT_PROJECT_ROOT
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--coskill-skills",
        type=Path,
        default=workspace_root / "coskill-skilltree noRL alfworld fix" / "skill_lib" / "skills_checkpoint_step3600.json",
    )
    parser.add_argument(
        "--skillrl-skills",
        type=Path,
        default=workspace_root / "skillRL alfworld" / "updated_skills_step50.json",
    )
    parser.add_argument("--split", choices=("valid_seen", "valid_unseen"), default="valid_unseen")
    parser.add_argument("--arm", choices=("all", "coskill", "skillrl"), default="all")
    parser.add_argument("--tasks-per-type", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", type=int, choices=(0, 1), default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    args.alfworld_data = args.alfworld_data.resolve()
    args.model_path = args.model_path.resolve()
    args.coskill_skills = args.coskill_skills.resolve()
    args.skillrl_skills = args.skillrl_skills.resolve()
    if args.tasks_per_type < 1:
        parser.error("--tasks-per-type must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_steps < 1 or args.max_steps > 40:
        parser.error("--max-steps must be between 1 and 40")
    return args


def prepare_shared_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    artifact_info = validate_skill_artifacts(args.coskill_skills, args.skillrl_skills)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "fixed_tasks.json"
    lock_path = args.output_root / ".prepare.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            expected = {
                "split": args.split,
                "seed": args.seed,
                "tasks_per_type": args.tasks_per_type,
                "total_tasks": len(TASK_TYPES) * args.tasks_per_type,
            }
            observed = {key: manifest.get(key) for key in expected}
            if observed != expected:
                raise ValueError(
                    f"Existing shared manifest does not match this run: {observed} != {expected}"
                )
        else:
            manifest = create_fixed_manifest(
                manifest_path,
                args.alfworld_data,
                args.split,
                args.tasks_per_type,
                args.seed,
            )
        comparison = {
            "schema_version": 1,
            "purpose": "frozen_skill_context_terminal_comparison",
            "base_model_shared": str(args.model_path),
            "rl_weights_loaded": False,
            "max_environment_steps": args.max_steps,
            "history_length": 8,
            "max_model_len": 10240,
            "max_response_tokens": 4096,
            "retrieval_mode": "template",
            "top_k": 6,
            "tensor_parallel_size": 1,
            "cloud_updates_enabled": False,
            "skill_tree_evolve_enabled": False,
            "batch_size": args.batch_size,
            "vllm_enforce_eager": bool(args.enforce_eager),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature,
            "seed": args.seed,
            "split": args.split,
            "tasks_per_type": args.tasks_per_type,
            "total_tasks_per_arm": manifest["total_tasks"],
            "fixed_manifest": str(manifest_path),
            "fixed_manifest_sha256": _sha256(manifest_path),
            "artifacts": artifact_info,
        }
        _write_json(args.output_root / "comparison_manifest.json", comparison)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return manifest_path, manifest


def main() -> None:
    args = parse_args()
    if args.render_only:
        rows = build_combined_rows(args.output_root)
        render_summary(args.output_root, rows)
        return

    for path, label in (
        (args.project_root, "project root"),
        (args.alfworld_data / "json_2.1.1", "ALFWorld data"),
        (args.model_path, "model"),
        (args.coskill_skills, "CoSkill group-50 skills"),
        (args.skillrl_skills, "SkillRL step-50 skills"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if args.arm == "all" and args.output_root.exists() and not args.preflight:
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_root}; use --overwrite to replace it"
            )
        shutil.rmtree(args.output_root)
    manifest_path, manifest = prepare_shared_inputs(args)
    print(
        f"[PLAN] split={args.split} tasks={manifest['total_tasks']} "
        f"({args.tasks_per_type} per type) seed={args.seed} max_steps={args.max_steps}",
        flush=True,
    )
    if args.arm in {"all", "coskill"}:
        print(f"[PLAN] CoSkill tree: {args.coskill_skills}", flush=True)
    if args.arm in {"all", "skillrl"}:
        print(f"[PLAN] SkillRL flat skills: {args.skillrl_skills}", flush=True)
    if args.preflight:
        print(f"[PREFLIGHT] ready; manifest written to {manifest_path}")
        return

    if args.arm in {"coskill", "skillrl"}:
        arm_dir = args.output_root / args.arm
        if arm_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Arm output already exists: {arm_dir}; use --overwrite to replace it"
                )
            shutil.rmtree(arm_dir)
        skills_path = args.coskill_skills if args.arm == "coskill" else args.skillrl_skills
        run_arm(args, args.arm, arm_dir, manifest_path, skills_path)
        return

    run_arm(args, "coskill", args.output_root / "coskill", manifest_path, args.coskill_skills)
    run_arm(args, "skillrl", args.output_root / "skillrl", manifest_path, args.skillrl_skills)
    rows = build_combined_rows(args.output_root)
    render_summary(args.output_root, rows)


if __name__ == "__main__":
    main()
