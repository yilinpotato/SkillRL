"""Independent ALFWorld L0--L5 representation experiment with real online growth.

V3 intentionally does *not* derive levels by truncating one L5 document.  It
starts every arm from the same stratified external corpus, keeps arm state
isolated, then executes five real fixed-game rounds.  Each round supplies two
fresh games per task family and forces exactly one cloud update after the full
12-episode group.  L0 grows only flat skills; L1--L5 grow only a tree with an
exact depth and hard size budgets.  Final evaluation is held out by game file.
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from examples.playbook_evolve import fixed_trajectory_ablation as fixed
from mini_test_pen_shelf.env_utils import find_games_by_type


ARMS = tuple(f"skill_level_l{level}" for level in range(6))


def _copy_json(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _is_success(trace: Dict[str, Any]) -> bool:
    return trace.get("outcome") == "success" or float(trace.get("episode_reward", 0) or 0) > 0


def select_initial_evidence(raw_path: Path, destination: Path, per_task: int) -> Path:
    """Select equal success/failure evidence per task with stable diversity."""
    if per_task < 2 or per_task % 2:
        raise ValueError("--initial_traces_per_type must be an even integer >= 2")
    by_type: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"success": [], "failure": []})
    for trace in fixed._read_jsonl(raw_path):
        by_type[trace["task_type"]]["success" if _is_success(trace) else "failure"].append(trace)
    selected: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {"per_task": per_task, "outcome_balance": "equal_success_failure"}
    half = per_task // 2
    for task_type in fixed.RUNTIME_TASK_TYPES:
        result: Dict[str, Any] = {}
        for outcome in ("success", "failure"):
            bucket = by_type[task_type][outcome]
            if len(bucket) < half:
                raise ValueError(
                    f"external corpus lacks {half} {outcome} traces for {task_type}; found {len(bucket)}")
            chosen, sample_audit = fixed._stratified_trace_sample(
                bucket, half, salt=f"v3_initial:{task_type}:{outcome}")
            selected.extend(chosen)
            result[outcome] = sample_audit
        audit[task_type] = result
    fixed._write_jsonl(destination, selected)
    fixed._write_json(destination.with_suffix(".selection.json"), audit)
    return destination


def create_growth_manifests(root: Path, data_root: Path, split: str, seed: int,
                            rounds: int, eval_games_per_type: int) -> tuple[List[Path], Path]:
    """Create disjoint online-round and held-out evaluation games once."""
    manifest_dir = root / "manifests"
    round_paths = [manifest_dir / f"online_round_{round_index}.json" for round_index in range(1, rounds + 1)]
    eval_path = manifest_dir / "eval_games.json"
    if all(path.exists() for path in [*round_paths, eval_path]):
        return round_paths, eval_path
    online: List[List[Dict[str, str]]] = [[] for _ in range(rounds)]
    evaluation: List[Dict[str, str]] = []
    requested = rounds * 2 + eval_games_per_type
    for offset, task_type in enumerate(fixed.TASK_TYPES):
        games = find_games_by_type(task_type, alfworld_data=str(data_root), split=split,
                                   sample_n=requested, sample_seed=seed + offset, verbose=False)
        if len(games) < requested:
            raise RuntimeError(f"need {requested} distinct games for {task_type}, found {len(games)}")
        paths = [os.path.realpath(game_file) for game_file, _ in games]
        if len(set(paths)) != requested:
            raise RuntimeError(f"sampler returned duplicate game files for {task_type}")
        for round_index in range(rounds):
            for within_round, (game_file, _) in enumerate(games[round_index * 2:round_index * 2 + 2], start=1):
                online[round_index].append({"label": f"online_r{round_index + 1}_{task_type}_{within_round}",
                                            "task_type": task_type,
                                            "game_file": fixed._relative_game(game_file, data_root)})
        for index, (game_file, _) in enumerate(games[rounds * 2:], start=1):
            evaluation.append({"label": f"eval_{task_type}_{index}", "task_type": task_type,
                               "game_file": fixed._relative_game(game_file, data_root)})
    for index, path in enumerate(round_paths):
        fixed._write_json(path, {"split": split, "role": "online_growth_round", "round": index + 1,
                                 "games_per_task_type": 2, "games": online[index]})
    fixed._write_json(eval_path, {"split": split, "role": "held_out_eval",
                                  "games_per_task_type": eval_games_per_type, "games": evaluation})
    all_paths = []
    for path in [*round_paths, eval_path]:
        games = fixed._read_json(path)["games"]
        all_paths.extend(os.path.realpath(data_root / game["game_file"]) for game in games)
    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("online/evaluation manifests overlap; use a fresh root")
    return round_paths, eval_path


def _set_final_artifact_state(root: Path, arm: str, source_skills: Path, round_summary: Path | None = None) -> None:
    artifact_dir = root / "artifacts" / arm
    _copy_json(source_skills, artifact_dir / "skills.json")
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = fixed._read_json(manifest_path)
    manifest["skills_sha256"] = fixed._sha256_path(artifact_dir / "skills.json")
    if round_summary and round_summary.exists():
        summary = fixed._read_json(round_summary)
        manifest.setdefault("online_round_summaries", []).append({
            "path": str(round_summary),
            "large_model": (summary.get("token_usage", {}) or {}).get("large_model", {}),
            "final_coskill_metrics": summary.get("final_coskill_metrics", {}),
        })
    fixed._write_json(manifest_path, manifest)


def _validate_tree_state(skills_path: Path, depth: int, nodes: int, chars: int) -> List[str]:
    skills = fixed._read_json(skills_path)
    errors = []
    for task_type in fixed.RUNTIME_TASK_TYPES:
        content = ((skills.get("skill_trees", {}) or {}).get(task_type) or {}).get("content", "")
        validation = CloudAnalyzer._validate_tree_depth(content, depth, max_nodes=nodes, max_chars=chars)
        if not validation["depth_valid"]:
            errors.append(f"{task_type}: {';'.join(validation['depth_validation_errors'])}")
    return errors


def _run_round(args, root: Path, arm: str, manifest: Path, round_index: int) -> Path:
    level = int(arm.rsplit("l", 1)[1])
    outdir = root / "growth" / arm / f"round_{round_index}"
    summary = outdir / "summary.json"
    if summary.exists():
        return summary
    skills = root / "artifacts" / arm / "skills.json"
    extra = ["--skills_json", str(skills), "--enable_hierarchy", str(int(level > 0)),
             "--cloud_update_every", "1", "--checkpoint_every_groups", "1",
             "--skill_tree_evolve_min_samples", "2", "--required_tree_depth", str(level),
             "--tree_depth_repair_attempts", str(args.tree_generation_attempts),
             "--tree_max_nodes", str(args.tree_max_nodes), "--tree_max_chars", str(args.tree_max_chars)]
    cmd = fixed._driver_cmd(args, outdir, manifest, 12, 12,
                            int(level == 0), int(level > 0), int(level > 0), 1, extra)
    fixed._run(cmd, args.project_root)
    final = outdir / "skill_lib" / "skills_latest_final.json"
    if not final.exists():
        raise RuntimeError(f"online round completed without a final skill checkpoint: {final}")
    if level:
        errors = _validate_tree_state(final, level, args.tree_max_nodes, args.tree_max_chars)
        if errors:
            raise RuntimeError(f"round {round_index} generated an invalid L{level} tree: {' | '.join(errors)}")
    _set_final_artifact_state(root, arm, final, summary)
    return summary


def _evaluate(args, root: Path, eval_manifest: Path, arm: str) -> None:
    output = root / "arms" / arm
    if (output / "summary.json").exists():
        return
    missing_rounds = [index for index in range(1, args.rounds + 1)
                      if not (root / "growth" / arm / f"round_{index}" / "summary.json").exists()]
    if missing_rounds:
        raise RuntimeError(
            f"cannot evaluate {arm}: online growth rounds are missing {missing_rounds}; run --phase grow first")
    level = int(arm.rsplit("l", 1)[1])
    games = fixed._read_json(eval_manifest)["games"]
    previous = args.rollouts_per_type
    args.rollouts_per_type = args.eval_rollouts_per_game
    try:
        cmd = fixed._driver_cmd(args, output, eval_manifest, len(games) * args.eval_rollouts_per_game,
                                min(72, len(games) * args.eval_rollouts_per_game), int(level == 0),
                                int(level > 0), 0, 0,
                                ["--enable_hierarchy", str(int(level > 0)), "--skills_json",
                                 str(root / "artifacts" / arm / "skills.json")])
        fixed._run(cmd, args.project_root)
    finally:
        args.rollouts_per_type = previous


def _growth_token_rows(root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Write an explicit online-growth token ledger beside final evaluation metrics.

    Driver summaries already report provider usage for each isolated round.
    We preserve their per-task/mixed attribution rather than pretending a mixed
    diagnosis/distillation call belongs to every task type.
    """
    arm_rows, task_rows = [], []
    for arm in ARMS:
        artifact = fixed._read_json(root / "artifacts" / arm / "artifact_manifest.json")
        evaluation = fixed._read_json(root / "arms" / arm / "summary.json")
        rounds = artifact.get("online_round_summaries", []) or []
        cloud = {"prompt": 0, "completion": 0, "total": 0, "mixed_prompt": 0,
                 "mixed_completion": 0, "mixed_total": 0}
        by_task = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0,
                                       "evolve_calls": 0, "tree_updates": 0})
        for record in rounds:
            usage = record.get("large_model", {}) or {}
            metrics = record.get("final_coskill_metrics", {}) or {}
            for key in ("prompt", "completion", "total"):
                cloud[key] += int(usage.get(key, 0) or 0)
            prompts = metrics.get("large_model_prompt_tokens_by_task_type", {}) or {}
            completions = metrics.get("large_model_completion_tokens_by_task_type", {}) or {}
            calls = metrics.get("evolve_calls_by_task_type", {}) or {}
            updates = metrics.get("skill_tree_updates_by_task_type", {}) or {}
            for task_type in set(prompts) | set(completions) | set(calls) | set(updates):
                bucket = by_task[task_type]
                bucket["prompt"] += int(prompts.get(task_type, 0) or 0)
                bucket["completion"] += int(completions.get(task_type, 0) or 0)
                bucket["total"] += int(prompts.get(task_type, 0) or 0) + int(completions.get(task_type, 0) or 0)
                bucket["evolve_calls"] += int(calls.get(task_type, 0) or 0)
                bucket["tree_updates"] += int(updates.get(task_type, 0) or 0)
            cloud["mixed_prompt"] += int(metrics.get("large_model_prompt_tokens_mixed", 0) or 0)
            cloud["mixed_completion"] += int(metrics.get("large_model_completion_tokens_mixed", 0) or 0)
            cloud["mixed_total"] += int(metrics.get("large_model_total_tokens_mixed", 0) or 0)
        arm_rows.append({"arm": arm, "target_tree_depth": artifact.get("target_depth"),
                         "online_rounds_completed": len(rounds),
                         "online_large_model_prompt_tokens": cloud["prompt"],
                         "online_large_model_completion_tokens": cloud["completion"],
                         "online_large_model_total_tokens": cloud["total"],
                         "online_mixed_large_model_total_tokens": cloud["mixed_total"],
                         "eval_episodes": evaluation.get("total_episodes", 0),
                         "eval_wins": evaluation.get("wins", 0),
                         "eval_success_rate": evaluation.get("success_rate")})
        for task_type in fixed.RUNTIME_TASK_TYPES:
            task_rows.append({"arm": arm, "target_tree_depth": artifact.get("target_depth"),
                              "task_type": task_type, **by_task[task_type],
                              "mixed_calls_not_split_across_tasks": True})
    fixed._write_jsonl(root / "growth_metrics.jsonl", arm_rows)
    fixed._write_jsonl(root / "growth_metrics_by_task.jsonl", task_rows)
    return arm_rows, task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--external_raw_traces", required=True)
    parser.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--phase", choices=("prepare", "grow", "evaluate", "summary", "all"), default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial_traces_per_type", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--eval_games_per_type", type=int, default=3)
    parser.add_argument("--eval_rollouts_per_game", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--tree_generation_attempts", type=int, default=20)
    parser.add_argument("--tree_max_nodes", type=int, default=16)
    parser.add_argument("--tree_max_chars", type=int, default=1800)
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--retrieval_mode", choices=("template", "embedding"), default="template")
    parser.add_argument("--data_parallel_workers", type=int, default=1)
    parser.add_argument("--rollout_worker_gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--gpu_mem_util", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")))
    parser.add_argument("--vllm_enforce_eager", type=int, choices=(0, 1), default=int(os.environ.get("VLLM_ENFORCE_EAGER", "0")))
    parser.add_argument("--log_trajectories", type=int, default=0)
    parser.add_argument("--driver_arg", action="append", default=[])
    args = parser.parse_args()
    if not args.alfworld_data:
        parser.error("--alfworld_data or ALFWORLD_DATA is required")
    if args.rounds < 1 or args.eval_games_per_type < 1 or args.eval_rollouts_per_game < 1:
        parser.error("rounds and evaluation sizes must be positive")
    if args.tree_generation_attempts < 1 or args.tree_max_nodes < 1 or args.tree_max_chars < 1:
        parser.error("tree budgets and repair attempts must be positive")
    args.project_root = Path(__file__).resolve().parents[2]
    args.rollouts_per_type = 1  # exactly one rollout for each of the two online games/task/round
    root, data_root = Path(args.root).resolve(), Path(args.alfworld_data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw = fixed.import_external_raw_traces(Path(args.external_raw_traces), root)
    initial = select_initial_evidence(raw, root / "frozen" / "initial_evidence.jsonl", args.initial_traces_per_type)
    rounds, evaluation = create_growth_manifests(root, data_root, args.split, args.sample_seed,
                                                 args.rounds, args.eval_games_per_type)
    config = {"experiment_kind": "alfworld_skill_tree_growth_v3", "task_types": fixed.TASK_TYPES,
              "runtime_task_types": fixed.RUNTIME_TASK_TYPES, "arms": list(ARMS),
              "protocol": {"initial_external_traces_per_type": args.initial_traces_per_type,
                           "initial_balance": "equal_success_failure", "online_rounds": args.rounds,
                           "online_games_per_task_per_round": 2, "online_rollouts_per_task_per_round": 2,
                           "cloud_updates": "forced_once_after_each_full_round batch", "tree_bounds": {
                               "max_depth": "exact arm level", "max_nodes": args.tree_max_nodes,
                               "max_chars": args.tree_max_chars, "repair_attempts": args.tree_generation_attempts},
                           "evaluation": {"held_out_games_per_task": args.eval_games_per_type,
                                          "rollouts_per_game": args.eval_rollouts_per_game}},
              "external_source_game_ids": "not available in generic raw_traces schema; held-out is strict for online manifests only"}
    fixed._write_json(root / "run_config.json", config)
    if args.phase == "prepare":
        for arm in ARMS:
            path = root / "artifacts" / arm / "artifact_manifest.json"
            if path.exists():
                continue
            level = int(arm.rsplit("l", 1)[1])
            if level == 0:
                fixed.build_l0_artifact(initial, path.parent)
            else:
                fixed.build_tree_artifact(initial, path.parent, level, args.tree_generation_attempts,
                                          args.tree_max_nodes, args.tree_max_chars)
        return
    # Always ensure preparation is complete before a resumable later phase.
    for arm in ARMS:
        path = root / "artifacts" / arm / "artifact_manifest.json"
        if not path.exists():
            level = int(arm.rsplit("l", 1)[1])
            if level == 0:
                fixed.build_l0_artifact(initial, path.parent)
            else:
                fixed.build_tree_artifact(initial, path.parent, level, args.tree_generation_attempts,
                                          args.tree_max_nodes, args.tree_max_chars)
    if args.phase in ("grow", "all"):
        for round_index, manifest in enumerate(rounds, start=1):
            for arm in ARMS:
                _run_round(args, root, arm, manifest, round_index)
    if args.phase == "grow":
        return
    if args.phase in ("evaluate", "all"):
        for arm in ARMS:
            _evaluate(args, root, evaluation, arm)
    fixed.write_summary(root, {"online_round_manifests": [str(path) for path in rounds],
                                "eval_manifest": str(evaluation)})
    _growth_token_rows(root)


if __name__ == "__main__":
    main()
