"""Extend held-out validation for frozen ALFWorld V4 L0--L5 artifacts.

This controller never calls the cloud and never regenerates a skill artifact.
It snapshots an existing V4 run, proves that the old evaluation manifest is a
strict subset of a larger deterministic manifest, evaluates only the added
games, then merges the immutable baseline and delta episode ledgers into a new
self-contained comparison root.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.playbook_evolve import fixed_trajectory_ablation as fixed
from examples.playbook_evolve import skill_tree_depth_ablation_v4 as v4


def _artifact_fingerprint(directory: Path, arm: str) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    skills_path = directory / "skills.json"
    if not manifest_path.is_file() or not skills_path.is_file():
        raise FileNotFoundError(f"V4 artifact is incomplete for {arm}: {directory}")
    manifest = fixed._read_json(manifest_path)
    skills_sha256 = fixed._sha256_path(skills_path)
    if manifest.get("skills_sha256") != skills_sha256:
        raise RuntimeError(f"V4 skills checksum mismatch for {arm}: {directory}")
    return {
        "artifact_manifest_sha256": fixed._sha256_path(manifest_path),
        "skills_sha256": skills_sha256,
        "status": manifest.get("status"),
        "evaluation_eligible": manifest.get("evaluation_eligible", True),
        "generation_protocol": manifest.get("generation_protocol"),
    }


def _artifact_fingerprints(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in v4.ARMS:
        directory = root / "artifacts" / arm
        result[arm] = _artifact_fingerprint(directory, arm)
    return result


def _copy_file_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if fixed._sha256_path(source) != fixed._sha256_path(destination):
            raise RuntimeError(f"existing validation-extension snapshot differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_frozen_source(source_root: Path, root: Path) -> dict[str, Any]:
    """Copy immutable artifacts and baseline summaries without touching source."""
    if source_root == root:
        raise ValueError("validation extension root must differ from source_root")
    source_config = source_root / "run_config.json"
    source_eval = source_root / "manifests" / "eval_games.json"
    source_evidence = source_root / "frozen" / "initial_evidence.jsonl"
    for required in (source_config, source_eval, source_evidence):
        if not required.is_file():
            raise FileNotFoundError(f"source V4 run is missing {required}")

    source_artifacts = _artifact_fingerprints(source_root)
    for arm in v4.ARMS:
        source = source_root / "artifacts" / arm
        destination = root / "artifacts" / arm
        if destination.exists():
            copied = _artifact_fingerprint(destination, arm)
            if copied != source_artifacts[arm]:
                raise RuntimeError(f"copied artifact differs from source for {arm}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)

        source_summary = source_root / "arms" / arm / "summary.json"
        source_groups = source_root / "arms" / arm / "group_metrics.jsonl"
        if not source_summary.is_file() or not source_groups.is_file():
            raise FileNotFoundError(f"source V4 evaluation is incomplete for {arm}")
        _copy_file_verified(
            source_summary,
            root / "baseline_snapshot" / "arms" / arm / "summary.json",
        )
        _copy_file_verified(
            source_groups,
            root / "baseline_snapshot" / "arms" / arm / "group_metrics.jsonl",
        )

    _copy_file_verified(
        source_config,
        root / "baseline_snapshot" / "run_config.json",
    )
    _copy_file_verified(
        source_eval,
        root / "manifests" / "source_eval_games.json",
    )
    _copy_file_verified(
        source_evidence,
        root / "frozen" / "source_initial_evidence.jsonl",
    )
    source_selection = source_evidence.with_suffix(".selection.json")
    if source_selection.is_file():
        _copy_file_verified(
            source_selection,
            root / "frozen" / "source_initial_evidence.selection.json",
        )

    return {
        "source_root": str(source_root),
        "source_run_config_sha256": fixed._sha256_path(source_config),
        "source_eval_manifest_sha256": fixed._sha256_path(source_eval),
        "source_evidence_sha256": fixed._sha256_path(source_evidence),
        "artifacts": source_artifacts,
    }


def create_delta_manifest(
    source_manifest_path: Path,
    expanded_manifest_path: Path,
    destination: Path,
) -> Path:
    """Require prefix-preserving expansion and write only newly added games."""
    source = fixed._read_json(source_manifest_path)
    expanded = fixed._read_json(expanded_manifest_path)
    source_games = source.get("games") or []
    expanded_games = expanded.get("games") or []
    source_files = {str(game["game_file"]) for game in source_games}
    expanded_files = {str(game["game_file"]) for game in expanded_games}
    if not source_files < expanded_files:
        raise RuntimeError(
            "expanded evaluation manifest must be a strict superset of the "
            "source manifest; use the same split/sample_seed and a larger count"
        )
    delta_games = [
        game for game in expanded_games if str(game["game_file"]) not in source_files
    ]
    per_task: dict[str, int] = defaultdict(int)
    for game in delta_games:
        per_task[str(game["task_type"])] += 1
    if set(per_task) != set(fixed.TASK_TYPES) or len(set(per_task.values())) != 1:
        raise RuntimeError(f"validation delta is not task-balanced: {dict(per_task)}")
    fixed._write_json(
        destination,
        {
            "split": expanded.get("split"),
            "role": "v4_incremental_held_out_eval_delta",
            "source_eval_manifest_sha256": fixed._sha256_path(source_manifest_path),
            "expanded_eval_manifest_sha256": fixed._sha256_path(expanded_manifest_path),
            "games_per_task_type": next(iter(per_task.values())),
            "games": delta_games,
        },
    )
    return destination


def audit_source_evidence_exclusion(
    evidence_path: Path,
    fingerprint_path: Path,
) -> dict[str, Any]:
    held_out = v4._load_eval_observation_fingerprints(fingerprint_path)
    overlaps = []
    evidence = fixed._read_jsonl(evidence_path)
    for trace in evidence:
        fingerprint = v4._trace_initial_observation_fingerprint(trace)
        if fingerprint and fingerprint in held_out:
            overlaps.append(
                {
                    "traj_uid": str(trace.get("traj_uid", "")),
                    "task_type": str(trace.get("task_type", "unknown")),
                    "initial_observation_sha256": fingerprint,
                }
            )
    audit = {
        "policy": "exact_initial_observation_sha256",
        "source_evidence_count": len(evidence),
        "expanded_eval_unique_fingerprints": len(held_out),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }
    if overlaps:
        raise RuntimeError(
            "source tree evidence overlaps the expanded held-out evaluation; "
            "the frozen artifacts cannot be used for a valid extension"
        )
    return audit


def evaluate_delta_arm(
    args: argparse.Namespace,
    root: Path,
    delta_manifest: Path,
    arm: str,
) -> Path:
    output = root / "validation_delta" / "arms" / arm
    summary = output / "summary.json"
    if summary.exists():
        return summary
    artifact = fixed._read_json(
        root / "artifacts" / arm / "artifact_manifest.json"
    )
    if not artifact.get("evaluation_eligible", True):
        fixed._write_json(
            summary,
            {
                "status": "N.A.",
                "arm": arm,
                "evaluation_skipped": True,
                "reason": artifact.get("unavailable_reason"),
                "total_episodes": 0,
                "wins": 0,
                "success_rate": None,
            },
        )
        return summary

    level = int(arm.rsplit("l", 1)[1])
    episodes = len(fixed._read_json(delta_manifest)["games"]) * args.eval_rollouts_per_game
    command = fixed._driver_cmd(
        args,
        output,
        delta_manifest,
        episodes,
        min(args.batch_rollout_size, episodes),
        int(level == 0),
        int(level > 0),
        0,
        0,
        [
            "--max_model_len",
            str(args.local_max_model_len),
            "--enable_hierarchy",
            str(int(level > 0)),
            "--skills_json",
            str(root / "artifacts" / arm / "skills.json"),
        ],
    )
    fixed._run(command, args.project_root)
    v4._audit_evaluation_context(root, arm, summary)
    return summary


def _sum_numeric_tree(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, dict) and isinstance(right, dict):
        return {
            key: _sum_numeric_tree(left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
        }
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return left + right
    if left == right:
        return left
    raise RuntimeError(f"cannot merge incompatible token accounting values: {left!r} != {right!r}")


def merge_arm_summaries(
    root: Path,
    expanded_manifest_path: Path,
    arm: str,
) -> Path:
    baseline_path = root / "baseline_snapshot" / "arms" / arm / "summary.json"
    delta_path = root / "validation_delta" / "arms" / arm / "summary.json"
    baseline = fixed._read_json(baseline_path)
    delta = fixed._read_json(delta_path)
    if baseline.get("status") != "done" or delta.get("status") != "done":
        raise RuntimeError(f"cannot merge incomplete validation summaries for {arm}")
    source_manifest = fixed._read_json(
        root / "manifests" / "source_eval_games.json"
    )
    source_expected = len(source_manifest.get("games") or []) * int(
        baseline.get("group_size", 0) or 0
    )
    delta_manifest = fixed._read_json(root / "manifests" / "eval_games_delta.json")
    delta_expected = len(delta_manifest.get("games") or []) * int(
        delta.get("group_size", 0) or 0
    )
    if int(baseline.get("total_episodes", 0) or 0) != source_expected:
        raise RuntimeError(
            f"source validation episode count mismatch for {arm}: "
            f"expected {source_expected}, found {baseline.get('total_episodes')}"
        )
    if int(delta.get("total_episodes", 0) or 0) != delta_expected:
        raise RuntimeError(
            f"delta validation episode count mismatch for {arm}: "
            f"expected {delta_expected}, found {delta.get('total_episodes')}"
        )

    combined = dict(baseline)
    per_game: list[dict[str, Any]] = []
    task_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "wins": 0}
    )
    running_wins = 0
    for segment, rows in (
        ("source", baseline.get("per_game") or []),
        ("extension", delta.get("per_game") or []),
    ):
        for row in rows:
            merged = dict(row)
            merged["validation_segment"] = segment
            merged["segment_step"] = row.get("step")
            merged["step"] = len(per_game) + 1
            task_type = str(merged.get("detected_type", "unknown"))
            task_counts[task_type]["episodes"] += 1
            task_counts[task_type]["wins"] += int(bool(merged.get("won")))
            running_wins += int(bool(merged.get("won")))
            merged["running_total_episodes"] = len(per_game) + 1
            merged["running_total_wins"] = running_wins
            merged["task_type_episodes"] = task_counts[task_type]["episodes"]
            merged["task_type_wins"] = task_counts[task_type]["wins"]
            per_game.append(merged)

    expected_episodes = (
        len(fixed._read_json(expanded_manifest_path)["games"])
        * int(baseline.get("group_size", 0) or 0)
    )
    if expected_episodes <= 0 or len(per_game) != expected_episodes:
        raise RuntimeError(
            f"combined validation episode count mismatch for {arm}: "
            f"expected {expected_episodes}, found {len(per_game)}"
        )

    small_by_type = fixed._small_tokens_from_episodes(per_game)
    small_prompt = sum(int(value["prompt"]) for value in small_by_type.values())
    small_response = sum(int(value["response"]) for value in small_by_type.values())
    small_total = sum(int(value["total"]) for value in small_by_type.values())
    baseline_large = ((baseline.get("token_usage") or {}).get("large_model") or {})
    delta_large = ((delta.get("token_usage") or {}).get("large_model") or {})
    context_baseline = baseline.get("context_guard") or {}
    context_delta = delta.get("context_guard") or {}
    prompt_trims = int(context_baseline.get("prompt_trims", 0) or 0) + int(
        context_delta.get("prompt_trims", 0) or 0
    )
    trimmed_tokens = int(context_baseline.get("trimmed_tokens", 0) or 0) + int(
        context_delta.get("trimmed_tokens", 0) or 0
    )
    wins = sum(int(bool(row.get("won"))) for row in per_game)
    expanded = fixed._read_json(expanded_manifest_path)
    combined.update(
        {
            "status": "done",
            "per_game": per_game,
            "total_episodes": len(per_game),
            "wins": wins,
            "success_rate": round(wins / len(per_game), 6),
            "completed_rollout_groups": int(
                baseline.get("completed_rollout_groups", 0) or 0
            )
            + int(delta.get("completed_rollout_groups", 0) or 0),
            "fixed_games_manifest": str(expanded_manifest_path),
            "fixed_game_files": list(
                dict.fromkeys(
                    list(baseline.get("fixed_game_files") or [])
                    + list(delta.get("fixed_game_files") or [])
                )
            ),
            "num_games_per_type": expanded.get("games_per_task_type"),
            "total_games_combined_pool": len(expanded.get("games") or []),
            "token_usage": {
                "small_model": {
                    "prompt": small_prompt,
                    "response": small_response,
                    "total": small_total,
                    "accounting": "vllm_request_tokens_single_pass",
                    "by_task_type": small_by_type,
                    "by_task_type_total": small_total,
                    "by_task_type_reconciliation_error": 0,
                },
                "large_model": _sum_numeric_tree(
                    baseline_large,
                    delta_large,
                ),
            },
            "context_guard": {
                "prompt_trims": prompt_trims,
                "trimmed_tokens": trimmed_tokens,
                "protocol_valid": prompt_trims == 0,
            },
            "evaluation_protocol_valid": prompt_trims == 0,
            "evaluation_protocol_error": (
                None
                if prompt_trims == 0
                else f"local_context_guard_trimmed_prompts:{prompt_trims}:tokens:{trimmed_tokens}"
            ),
            "validation_extension": {
                "baseline_summary_sha256": fixed._sha256_path(baseline_path),
                "delta_summary_sha256": fixed._sha256_path(delta_path),
                "baseline_episodes": int(baseline.get("total_episodes", 0) or 0),
                "delta_episodes": int(delta.get("total_episodes", 0) or 0),
                "artifacts_regenerated": False,
                "cloud_calls": 0,
            },
        }
    )

    output = root / "arms" / arm / "summary.json"
    fixed._write_json(output, combined)
    fixed._write_jsonl(
        root / "arms" / arm / "group_metrics.jsonl",
        [
            {
                "step": combined["completed_rollout_groups"],
                "global_episode_end": len(per_game),
                "metrics": {
                    "episode/count_cumulative": len(per_game),
                    "episode/wins_cumulative": wins,
                    "episode/success_rate": combined["success_rate"],
                    "tokens/small_model/prompt_cumulative": small_prompt,
                    "tokens/small_model/response_cumulative": small_response,
                    "tokens/small_model/total_cumulative": small_total,
                    "tokens/small_model/accounting": "vllm_request_tokens_single_pass",
                    "tokens/large_model/prompt_cumulative": int(
                        combined["token_usage"]["large_model"].get("prompt", 0) or 0
                    ),
                    "tokens/large_model/completion_cumulative": int(
                        combined["token_usage"]["large_model"].get("completion", 0) or 0
                    ),
                    "tokens/large_model/total_cumulative": int(
                        combined["token_usage"]["large_model"].get("total", 0) or 0
                    ),
                    "perf/total_num_tokens": small_total,
                },
            }
        ],
    )
    v4._audit_evaluation_context(root, arm, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument(
        "--phase",
        choices=("prepare", "evaluate", "summary", "all"),
        default="all",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_games_per_type", type=int, default=5)
    parser.add_argument("--eval_rollouts_per_game", type=int, default=12)
    parser.add_argument("--batch_rollout_size", type=int, default=72)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument(
        "--local_max_model_len",
        type=int,
        default=int(os.environ.get("V4_LOCAL_MAX_MODEL_LEN", "16384")),
    )
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument(
        "--retrieval_mode",
        choices=("template", "embedding"),
        default="template",
    )
    parser.add_argument("--data_parallel_workers", type=int, default=1)
    parser.add_argument(
        "--rollout_worker_gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("VLLM_ENFORCE_EAGER", "0")),
    )
    parser.add_argument("--log_trajectories", type=int, default=0)
    parser.add_argument("--driver_arg", action="append", default=[])
    args = parser.parse_args()

    if not args.alfworld_data:
        parser.error("--alfworld_data or ALFWORLD_DATA is required")
    if args.eval_games_per_type < 2 or args.eval_rollouts_per_game < 1:
        parser.error("expanded validation sizes must be positive")
    if args.local_max_model_len <= 4096:
        parser.error("--local_max_model_len must exceed the 4096 response budget")
    args.project_root = Path(__file__).resolve().parents[2]
    args.rollouts_per_type = args.eval_rollouts_per_game

    root = Path(args.root).resolve()
    source_root = Path(args.source_root).resolve()
    data_root = Path(args.alfworld_data).resolve()
    root.mkdir(parents=True, exist_ok=True)

    source_metadata = stage_frozen_source(source_root, root)
    expanded_manifest = v4.create_eval_manifest(
        root,
        data_root,
        args.split,
        args.sample_seed,
        args.eval_games_per_type,
    )
    fingerprints = v4.create_eval_observation_fingerprints(
        root,
        data_root,
        expanded_manifest,
        args.sample_seed,
    )
    delta_manifest = create_delta_manifest(
        root / "manifests" / "source_eval_games.json",
        expanded_manifest,
        root / "manifests" / "eval_games_delta.json",
    )
    overlap_audit = audit_source_evidence_exclusion(
        root / "frozen" / "source_initial_evidence.jsonl",
        fingerprints,
    )
    source_eval = fixed._read_json(
        root / "manifests" / "source_eval_games.json"
    )
    source_rollouts = int(
        (
            fixed._read_json(root / "baseline_snapshot" / "run_config.json")
            .get("protocol", {})
            .get("evaluation", {})
            .get("rollouts_per_game", 0)
        )
        or 0
    )
    if source_rollouts != args.eval_rollouts_per_game:
        raise RuntimeError(
            f"source used {source_rollouts} rollouts/game but extension requested "
            f"{args.eval_rollouts_per_game}; unequal replication cannot be merged"
        )
    config = {
        "experiment_kind": "alfworld_skill_tree_depth_v4_validation_extension",
        "arms": list(v4.ARMS),
        "artifact_reuse": {
            **source_metadata,
            "cloud_regeneration": False,
            "tree_regeneration": False,
            "l0_regeneration": False,
            "source_l0_protocol_is_preserved": True,
        },
        "evaluation": {
            "source_games_per_task": source_eval.get("games_per_task_type"),
            "expanded_games_per_task": args.eval_games_per_type,
            "delta_games_per_task": fixed._read_json(delta_manifest).get(
                "games_per_task_type"
            ),
            "rollouts_per_game": args.eval_rollouts_per_game,
            "sample_seed": args.sample_seed,
            "split": args.split,
            "source_evidence_exclusion": overlap_audit,
        },
        "comparison_scope": (
            "incremental held-out validation of frozen source artifacts; the "
            "source L0 generation protocol is not rewritten"
        ),
    }
    v4._ensure_run_config_compatible(root / "run_config.json", config, None)
    fixed._write_json(
        root / "validation_extension_receipt.json",
        {
            **config,
            "expanded_eval_manifest_sha256": fixed._sha256_path(expanded_manifest),
            "delta_eval_manifest_sha256": fixed._sha256_path(delta_manifest),
        },
    )

    if args.phase == "prepare":
        return
    if args.phase in ("evaluate", "all"):
        for arm in v4.ARMS:
            evaluate_delta_arm(args, root, delta_manifest, arm)
    if args.phase in ("summary", "evaluate", "all"):
        for arm in v4.ARMS:
            delta_summary = (
                root / "validation_delta" / "arms" / arm / "summary.json"
            )
            if not delta_summary.exists():
                raise RuntimeError(
                    f"delta evaluation missing for {arm}; run --phase evaluate first"
                )
            merge_arm_summaries(root, expanded_manifest, arm)
        fixed.write_summary(
            root,
            {
                "source_eval_manifest": str(
                    root / "manifests" / "source_eval_games.json"
                ),
                "expanded_eval_manifest": str(expanded_manifest),
                "delta_eval_manifest": str(delta_manifest),
                "artifact_source_root": str(source_root),
            },
        )
        v4.write_generation_metrics(root)


if __name__ == "__main__":
    main()
