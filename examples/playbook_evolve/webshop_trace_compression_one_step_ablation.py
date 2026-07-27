"""WebShop one-training-step CoSkill trace-compression cost ablation.

The rollout unit follows the frozen no-RL WebShop driver rather than the
ALFWorld task-type manifest: one training group samples 12 distinct training
goals and repeats each goal six times, yielding 72 complete trajectories with
at most 15 search/click actions.  That immutable raw corpus is then fanned out
to two otherwise identical cloud updates:

* ``compression_on`` uses loop filtering, observation deltas, and the compact
  trajectory-prefix-tree codec;
* ``compression_off`` uploads flat, full-observation trajectories.

Both arms therefore have identical WebShop tasks, model actions, rewards,
terminal task scores, and raw trace SHA-256.  Arm-specific success rates are
not claimed because no post-update rollout is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_system.memory import HierarchicalSkillLib
from agent_system.memory.coskill_loop import CoSkillCloudLoop
from examples.playbook_evolve import trace_compression_one_step_ablation as common

DISTINCT_GOALS = 12
ROLLOUTS_PER_GOAL = 6
TOTAL_ROLLOUTS = DISTINCT_GOALS * ROLLOUTS_PER_GOAL
MAX_ENVIRONMENT_STEPS = 15
WEBSHOP_TASK_TYPES = (
    "accessories",
    "apparel",
    "beauty_health",
    "electronics",
    "footwear",
    "home_decor",
    "other",
)
ARMS = common.ARMS
# Match the executable syntax accepted by WebShop's runtime projection.  The
# projection deliberately salvages model outputs such as ``click [search]``;
# those actions must remain in the immutable raw corpus even when the
# environment later treats the target as invalid/no-op evidence.
_ACTION_RE = re.compile(r"^(search|click)\s*\[[^\r\n]*\]$", re.IGNORECASE)


def _validate_shared_raw(raw: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the WebShop sampling/reward contract before reusing a corpus."""
    if len(raw) != TOTAL_ROLLOUTS:
        raise RuntimeError(
            f"expected exactly {TOTAL_ROLLOUTS} captured traces, found {len(raw)}"
        )

    goal_counts: Counter[int] = Counter()
    category_counts: Counter[str] = Counter()
    category_wins: Counter[str] = Counter()
    category_scores: defaultdict[str, float] = defaultdict(float)
    goal_contract: dict[int, tuple[str, str]] = {}
    observed_steps: list[int] = []
    seen_uids: set[str] = set()
    total_score = 0.0
    wins = 0
    action_counts: Counter[str] = Counter()

    for index, trace in enumerate(raw, start=1):
        uid = str(trace.get("traj_uid", "") or "")
        if not uid or uid in seen_uids:
            raise RuntimeError(f"capture trace {index} has missing/duplicate traj_uid={uid!r}")
        seen_uids.add(uid)

        meta = trace.get("meta") or {}
        if str(meta.get("environment", "")).lower() != "webshop":
            raise RuntimeError(
                f"capture trace {index} is not tagged meta.environment=WebShop"
            )
        try:
            goal_index = int(meta["goal_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"capture trace {index} has invalid WebShop goal_index"
            ) from exc

        task = str(trace.get("task", "") or "").strip()
        task_type = str(trace.get("task_type", "") or "")
        if not task:
            raise RuntimeError(f"capture trace {index} has an empty instruction")
        if task_type not in WEBSHOP_TASK_TYPES:
            raise RuntimeError(
                f"capture trace {index} has unknown WebShop task_type={task_type!r}"
            )
        prior = goal_contract.setdefault(goal_index, (task, task_type))
        if prior != (task, task_type):
            raise RuntimeError(
                f"goal {goal_index} changed instruction/category across replicas"
            )

        steps = trace.get("steps") or []
        step_count = len(steps)
        if step_count < 1 or step_count > MAX_ENVIRONMENT_STEPS:
            raise RuntimeError(
                f"capture trace {index} has {step_count} steps; expected "
                f"[1, {MAX_ENVIRONMENT_STEPS}]"
            )
        observed_steps.append(step_count)
        for step_number, step in enumerate(steps, start=1):
            action = str(step.get("action", "") or "").strip()
            match = _ACTION_RE.fullmatch(action)
            if not match:
                raise RuntimeError(
                    f"capture trace {index} step {step_number} violates the "
                    f"WebShop search/click action contract: {action!r}"
                )
            action_counts[match.group(1).lower()] += 1

        try:
            task_score = float(meta.get("task_score", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"capture trace {index} has non-numeric task_score"
            ) from exc
        if not 0.0 <= task_score <= 1.0:
            raise RuntimeError(
                f"capture trace {index} task_score={task_score} is outside [0, 1]"
            )
        success = trace.get("outcome") == "success"
        if success != (abs(task_score - 1.0) < 1e-8):
            raise RuntimeError(
                f"capture trace {index} outcome/task_score disagree: "
                f"{trace.get('outcome')!r} vs {task_score}"
            )

        goal_counts[goal_index] += 1
        category_counts[task_type] += 1
        category_wins[task_type] += int(success)
        category_scores[task_type] += task_score
        wins += int(success)
        total_score += task_score

    if len(goal_counts) != DISTINCT_GOALS:
        raise RuntimeError(
            f"expected {DISTINCT_GOALS} distinct WebShop goals, got {len(goal_counts)}"
        )
    wrong_replicas = {
        goal: count
        for goal, count in goal_counts.items()
        if count != ROLLOUTS_PER_GOAL
    }
    if wrong_replicas:
        raise RuntimeError(
            f"expected {ROLLOUTS_PER_GOAL} replicas per goal, got {wrong_replicas}"
        )
    if max(observed_steps, default=0) <= 1:
        raise RuntimeError(
            "full-trajectory protocol requires at least one trace with more "
            "than one environment step"
        )

    per_category = {}
    for category in sorted(category_counts):
        episodes = category_counts[category]
        category_successes = category_wins[category]
        per_category[category] = {
            "episodes": episodes,
            "wins": category_successes,
            "success_rate": category_successes / episodes,
            "mean_task_score": category_scores[category] / episodes,
        }
    return {
        "rollouts": len(raw),
        "distinct_goals": len(goal_counts),
        "replicas_per_goal": ROLLOUTS_PER_GOAL,
        "goal_indices": sorted(goal_counts),
        "max_environment_steps": MAX_ENVIRONMENT_STEPS,
        "observed_steps_per_trace": {
            "min": min(observed_steps),
            "max": max(observed_steps),
            "total": sum(observed_steps),
            "mean": sum(observed_steps) / len(observed_steps),
        },
        "actions": dict(sorted(action_counts.items())),
        "capture_diagnostics": {
            "wins": wins,
            "success_rate": wins / len(raw),
            "mean_task_score": total_score / len(raw),
            "per_task_type": per_category,
        },
    }


def _driver_cmd(
    args: argparse.Namespace,
    capture_dir: Path,
    initial_skills: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "examples.playbook_evolve.run_webshop_evolve",
        "--outdir",
        str(capture_dir),
        "--model_path",
        str(args.model_path),
        "--webshop_file_path",
        str(args.webshop_file_path),
        "--webshop_attr_path",
        str(args.webshop_attr_path),
        "--train_data_size",
        str(DISTINCT_GOALS),
        "--val_data_size",
        str(max(32, args.data_parallel_workers)),
        "--validation_every_groups",
        "0",
        "--validation_before_train",
        "0",
        "--group_size",
        str(ROLLOUTS_PER_GOAL),
        "--total_groups",
        "1",
        "--max_episodes",
        str(TOTAL_ROLLOUTS),
        "--max_steps",
        str(MAX_ENVIRONMENT_STEPS),
        "--seed",
        str(args.seed),
        "--data_parallel_workers",
        str(args.data_parallel_workers),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--pipeline_parallel_size",
        str(args.pipeline_parallel_size),
        "--vllm_max_num_seqs",
        str(args.vllm_max_num_seqs),
        "--vllm_enforce_eager",
        str(args.vllm_enforce_eager),
        "--checkpoint_every_groups",
        "0",
        "--enable_cloud_updates",
        "0",
        "--history_length",
        "8",
        "--prompt_char_limit",
        str(args.prompt_char_limit),
        "--max_model_len",
        str(args.max_model_len),
        "--max_tokens",
        str(args.max_tokens),
        "--think_budget",
        str(args.think_budget),
        "--action_budget",
        str(args.action_budget),
        "--temperature",
        str(args.temperature),
        "--gpu_mem_util",
        str(args.gpu_mem_util),
        "--skills_json",
        str(initial_skills),
        "--retrieval_mode",
        args.retrieval_mode,
        "--top_k",
        "6",
        "--enable_hierarchy",
        "1",
        "--enable_coskill",
        "1",
        "--enable_skill_tree",
        "1",
        "--enable_skill_tree_evolve",
        "0",
        "--enable_failure_analysis",
        "1",
        "--trace_enable_loop_filter",
        "1",
        "--trace_enable_obs_delta",
        "1",
        "--trace_enable_prefix_tree",
        "1",
        "--trace_enable_consensus_prefix",
        "1",
        "--trace_cloud_evidence_mode",
        "tree_only",
        "--log_trajectories",
        "0",
    ]
    if args.rollout_worker_gpus:
        command += ["--rollout_worker_gpus", args.rollout_worker_gpus]
    command.extend(args.driver_arg)
    return command


def capture_once(
    args: argparse.Namespace,
    root: Path,
    initial_skills: Path,
) -> Path:
    raw_path = root / "shared" / "raw_traces.jsonl"
    if raw_path.exists():
        _validate_shared_raw(common._read_jsonl(raw_path))
        return raw_path

    capture_dir = root / "capture"
    source = capture_dir / "traces_pool" / "raw_traces.jsonl"
    if source.exists():
        print(
            "[webshop-train-step-trace-ablation] recovering completed capture "
            f"from {source}"
        )
        raw = common._read_jsonl(source)
        integrity = _validate_shared_raw(raw)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, raw_path)
        common._write_json(
            capture_dir / "capture_integrity.json",
            {
                "protocol": (
                    "shared_12_distinct_webshop_goals_x_6_replicas_"
                    "one_training_group_full_trajectories"
                ),
                "raw_traces": str(raw_path),
                "raw_traces_sha256": common._sha256_path(raw_path),
                "recovered_completed_capture": True,
                **integrity,
            },
        )
        return raw_path

    command = _driver_cmd(args, capture_dir, initial_skills)
    print("[webshop-train-step-trace-ablation] capture:", " ".join(command))
    subprocess.run(command, cwd=args.project_root, check=True)
    if not source.exists():
        raise RuntimeError(f"capture completed without raw trace log: {source}")
    raw = common._read_jsonl(source)
    integrity = _validate_shared_raw(raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, raw_path)
    common._write_json(
        capture_dir / "capture_integrity.json",
        {
            "protocol": (
                "shared_12_distinct_webshop_goals_x_6_replicas_"
                "one_training_group_full_trajectories"
            ),
            "raw_traces": str(raw_path),
            "raw_traces_sha256": common._sha256_path(raw_path),
            **integrity,
        },
    )
    return raw_path


def _expected_cloud_call_purposes(raw: list[dict[str, Any]]) -> Counter[str]:
    """Return the normal CoSkill cloud-call set for this frozen batch."""
    by_type: Counter[str] = Counter(
        str(trace.get("task_type") or "unknown") for trace in raw
    )
    expected: Counter[str] = Counter()
    if raw:
        expected["contrastive_distill"] = 1
    if any(trace.get("outcome") != "success" for trace in raw):
        expected["diagnose_failures"] = 1
    expected["evolve_playbook"] = sum(
        int(count >= 6) for count in by_type.values()
    )
    return expected


def _recover_complete_cloud_calls(
    arm_dir: Path,
    raw: list[dict[str, Any]],
    skills_path: Path,
) -> list[dict[str, Any]] | None:
    """Recover a billed cloud update interrupted during local reporting."""
    if not skills_path.exists():
        return None
    for audit_path in (
        arm_dir / "cloud_io" / "call_audit.json",
        arm_dir / "cloud_io" / "call_audit_live.json",
    ):
        if not audit_path.exists():
            continue
        if (
            audit_path.name == "call_audit_live.json"
            and skills_path.stat().st_mtime_ns <= audit_path.stat().st_mtime_ns
        ):
            # All responses may be checkpointed just before the final parsed
            # tree is installed and saved.  Recover only when the skill bank
            # was persisted after the last live provider-usage snapshot.
            continue
        calls = common._read_json(audit_path)
        if not isinstance(calls, list):
            continue
        actual = Counter(str(call.get("purpose", "")) for call in calls)
        if actual == _expected_cloud_call_purposes(raw):
            print(
                "[webshop-train-step-trace-ablation] recovering completed "
                f"cloud update from {audit_path}"
            )
            return calls
    return None


def build_arm(
    root: Path,
    arm: str,
    raw_path: Path,
    initial_skills: Path,
    retrieval_mode: str,
    capture_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    arm_dir = root / "arms" / arm
    result_path = arm_dir / "arm_result.json"
    if result_path.exists():
        return common._read_json(result_path)

    flags = ARMS[arm]
    raw = common._read_jsonl(raw_path)
    skills_path = arm_dir / "skill_lib" / f"skills_step{TOTAL_ROLLOUTS}.json"
    reporting_pool = common._pool(raw, arm_dir / "trace_payload", flags)
    batch = reporting_pool.export_batch(
        trigger_reason="one_webshop_training_group_shared_capture"
    )
    common._write_json(arm_dir / "compressed_batch.json", batch)

    recovered_calls = _recover_complete_cloud_calls(arm_dir, raw, skills_path)
    if recovered_calls is None:
        cloud_pool = common._pool(raw, arm_dir, flags)
        library = HierarchicalSkillLib(
            str(initial_skills),
            retrieval_mode=retrieval_mode,
            enable_playbook=True,
        )
        loop = CoSkillCloudLoop(
            output_dir=str(arm_dir),
            enable_coskill=True,
            enable_playbook_evolve=True,
            enable_failure_analysis=True,
            environment_name="WebShop",
        )
        fired = loop.maybe_update(
            cloud_pool,
            library,
            TOTAL_ROLLOUTS,
            force_reason="one_webshop_training_group_shared_capture",
        )
        if not fired:
            raise RuntimeError(f"{arm} did not execute its required one cloud update")
        if not skills_path.exists():
            raise RuntimeError(f"{arm} cloud update did not persist its skill library")
        analyzer = loop.cloud_analyzer
        raw_calls = getattr(analyzer, "call_audit", []) or []
    else:
        raw_calls = recovered_calls
    calls = common.annotate_call_costs(raw_calls)
    common._write_json(arm_dir / "cloud_io" / "call_audit.json", calls)
    call_cost_path = arm_dir / "cloud_io" / "call_costs.csv"
    if calls:
        with call_cost_path.open("w", newline="") as handle:
            fields = [
                "call_index",
                "purpose",
                "task_type",
                "model",
                "pricing_model",
                "prompt_tokens",
                "completion_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
                "observed_cache_billed_cost_usd",
                "all_input_cache_miss_cost_usd",
                "cost_status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(calls)

    waterfall = common.build_token_waterfall(batch, calls)
    common._write_json(arm_dir / "token_waterfall.json", waterfall)
    result = {
        "arm": arm,
        "protocol": (
            "same_shared_raw_12_webshop_goals_x_6_replicas_72_"
            "full_trajectories_then_one_coskill_cloud_update"
        ),
        "raw_traces_sha256": common._sha256_path(raw_path),
        "compression_flags": flags,
        "capture_diagnostics_shared_across_arms": capture_diagnostics,
        "capture_upload_payload": common._capture_payload_stats(raw),
        "uploaded_trace_payload": common._uploaded_trace_payload(batch),
        "token_waterfall": waterfall,
        "token_waterfall_path": str(arm_dir / "token_waterfall.json"),
        "cloud_cost": common.summarize_cloud_cost(calls),
        "cloud_call_audit_path": str(arm_dir / "cloud_io" / "call_audit.json"),
        "cloud_call_costs_path": str(call_cost_path),
        "skills_path": str(skills_path),
        "skill_sha256": common._sha256_path(skills_path),
        "cloud_update_fired": True,
    }
    common._write_json(result_path, result)
    return result


def write_reports(root: Path) -> dict[str, Any]:
    on = common._read_json(root / "arms" / "compression_on" / "arm_result.json")
    off = common._read_json(root / "arms" / "compression_off" / "arm_result.json")
    if on["raw_traces_sha256"] != off["raw_traces_sha256"]:
        raise RuntimeError("compression arms do not share the same raw trace SHA-256")

    on_payload = on["uploaded_trace_payload"]
    off_payload = off["uploaded_trace_payload"]
    on_cost = on["cloud_cost"]
    off_cost = off["cloud_cost"]
    on_waterfall = common._waterfall_by_stage(on)
    off_waterfall = common._waterfall_by_stage(off)

    waterfall_rows = [
        {"arm": result["arm"], **stage}
        for result in (on, off)
        for stage in result["token_waterfall"]["stages"]
    ]
    common._write_json(
        root / "token_waterfall.json",
        {
            "protocol": on["protocol"],
            "raw_trace_sha256": on["raw_traces_sha256"],
            "normal_compression": on["token_waterfall"],
            "all_compression_off": off["token_waterfall"],
        },
    )
    with (root / "token_waterfall.csv").open("w", newline="") as handle:
        fields = sorted({key for row in waterfall_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(waterfall_rows)

    comparison = {
        "protocol": on["protocol"],
        "direction": (
            "positive all_off_minus_on means normal CoSkill compression is "
            "smaller or cheaper"
        ),
        "raw_trace_sha256": on["raw_traces_sha256"],
        "capture_diagnostics_shared_across_arms": on[
            "capture_diagnostics_shared_across_arms"
        ],
        "uploaded_trace_payload_delta": {
            "trace_evidence_chars": common._difference(
                on_payload["trace_evidence"]["chars"],
                off_payload["trace_evidence"]["chars"],
            ),
            "trace_evidence_tokens_chars_div_4": common._difference(
                on_payload["trace_evidence"]["tokens_chars_div_4"],
                off_payload["trace_evidence"]["tokens_chars_div_4"],
            ),
            "cloud_batch_chars": common._difference(
                on_payload["cloud_batch"]["chars"],
                off_payload["cloud_batch"]["chars"],
            ),
            "cloud_batch_tokens_chars_div_4": common._difference(
                on_payload["cloud_batch"]["tokens_chars_div_4"],
                off_payload["cloud_batch"]["tokens_chars_div_4"],
            ),
        },
        "provider_token_delta": {
            "prompt_tokens": common._difference(
                on_cost["prompt_tokens"], off_cost["prompt_tokens"]
            ),
            "completion_tokens": common._difference(
                on_cost["completion_tokens"], off_cost["completion_tokens"]
            ),
            "cache_hit_tokens": common._difference(
                on_cost["prompt_cache_hit_tokens"],
                off_cost["prompt_cache_hit_tokens"],
            ),
            "cache_miss_tokens": common._difference(
                on_cost["prompt_cache_miss_tokens"],
                off_cost["prompt_cache_miss_tokens"],
            ),
        },
        "cost_delta_usd": {
            "observed_cache_billed": common._difference(
                on_cost["observed_cache_billed_cost_usd"],
                off_cost["observed_cache_billed_cost_usd"],
            ),
            "all_input_cache_miss": common._difference(
                on_cost["all_input_cache_miss_cost_usd"],
                off_cost["all_input_cache_miss_cost_usd"],
            ),
        },
        "token_waterfall_delta": {
            stage: {
                "chars": common._difference(
                    on_waterfall.get(stage, {}).get("chars"),
                    off_waterfall.get(stage, {}).get("chars"),
                ),
                "tokens_chars_div_4": common._difference(
                    on_waterfall.get(stage, {}).get("tokens_chars_div_4"),
                    off_waterfall.get(stage, {}).get("tokens_chars_div_4"),
                ),
                "provider_prompt_tokens": common._difference(
                    on_waterfall.get(stage, {}).get("provider_prompt_tokens"),
                    off_waterfall.get(stage, {}).get("provider_prompt_tokens"),
                ),
            }
            for stage in (
                "raw",
                "loop_filter",
                "obs_delta",
                "prefix_tree_context",
                "actual_cloud_prompt",
            )
        },
    }
    common._write_json(root / "compression_comparison.json", comparison)

    diagnostics = on["capture_diagnostics_shared_across_arms"]
    metrics = []
    for result in (on, off):
        payload = result["uploaded_trace_payload"]
        cost = result["cloud_cost"]
        metrics.append(
            {
                "step": TOTAL_ROLLOUTS,
                "metrics": {
                    "experiment/name": (
                        "webshop_train_step_trace_compression_ablation"
                    ),
                    "experiment/arm": result["arm"],
                    "experiment/rollouts": TOTAL_ROLLOUTS,
                    "experiment/distinct_goals": DISTINCT_GOALS,
                    "experiment/rollouts_per_goal": ROLLOUTS_PER_GOAL,
                    "experiment/max_environment_steps": MAX_ENVIRONMENT_STEPS,
                    "experiment/task_success_metric": (
                        "shared_capture_diagnostic_only"
                    ),
                    "episode/wins": diagnostics["wins"],
                    "episode/success_rate": diagnostics["success_rate"],
                    "episode/mean_task_score": diagnostics["mean_task_score"],
                    "trace_upload/evidence_chars": payload["trace_evidence"]["chars"],
                    "trace_upload/evidence_tokens_chars_div_4": payload[
                        "trace_evidence"
                    ]["tokens_chars_div_4"],
                    "trace_upload/batch_chars": payload["cloud_batch"]["chars"],
                    "trace_upload/batch_tokens_chars_div_4": payload[
                        "cloud_batch"
                    ]["tokens_chars_div_4"],
                    **{
                        f"trace_waterfall/{row['stage']}/tokens_chars_div_4": row[
                            "tokens_chars_div_4"
                        ]
                        for row in result["token_waterfall"]["stages"]
                    },
                    "trace_waterfall/actual_cloud_prompt/provider_tokens": (
                        common._waterfall_by_stage(result)["actual_cloud_prompt"][
                            "provider_prompt_tokens"
                        ]
                    ),
                    "tokens/large_model/prompt": cost["prompt_tokens"],
                    "tokens/large_model/completion": cost["completion_tokens"],
                    "tokens/large_model/cache_hit": cost[
                        "prompt_cache_hit_tokens"
                    ],
                    "tokens/large_model/cache_miss": cost[
                        "prompt_cache_miss_tokens"
                    ],
                    "cost/deepseek_v4_flash/observed_usd": cost[
                        "observed_cache_billed_cost_usd"
                    ],
                    "cost/deepseek_v4_flash/all_miss_usd": cost[
                        "all_input_cache_miss_cost_usd"
                    ],
                },
            }
        )
    with (root / "metrics.jsonl").open("w") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cost_rows = []
    for result in (on, off):
        payload = result["uploaded_trace_payload"]
        cost = result["cloud_cost"]
        cost_rows.append(
            {
                "arm": result["arm"],
                "trace_evidence_chars": payload["trace_evidence"]["chars"],
                "trace_evidence_tokens_chars_div_4": payload["trace_evidence"][
                    "tokens_chars_div_4"
                ],
                "cloud_prompt_tokens": cost["prompt_tokens"],
                "cloud_completion_tokens": cost["completion_tokens"],
                "cache_hit_tokens": cost["prompt_cache_hit_tokens"],
                "cache_miss_tokens": cost["prompt_cache_miss_tokens"],
                "observed_cache_billed_cost_usd": cost[
                    "observed_cache_billed_cost_usd"
                ],
                "all_input_cache_miss_cost_usd": cost[
                    "all_input_cache_miss_cost_usd"
                ],
            }
        )
    with (root / "cost_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)

    summary = {
        "status": "done",
        "environment": "WebShop",
        "arms": [on, off],
        "comparison": comparison,
        "metrics": str(root / "metrics.jsonl"),
        "cost_table": str(root / "cost_comparison.csv"),
        "token_waterfall": str(root / "token_waterfall.json"),
        "token_waterfall_table": str(root / "token_waterfall.csv"),
    }
    common._write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--phase",
        choices=("capture", "arms", "report", "all"),
        default="all",
    )
    parser.add_argument("--webshop_file_path", required=True)
    parser.add_argument("--webshop_attr_path", required=True)
    parser.add_argument(
        "--skills_json",
        default="memory_data/webshop/claude_style_skills.json",
    )
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--retrieval_mode",
        choices=("template", "embedding"),
        default="template",
    )
    parser.add_argument("--data_parallel_workers", type=int, default=1)
    parser.add_argument("--rollout_worker_gpus", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0)
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
    parser.add_argument("--prompt_char_limit", type=int, default=24000)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--think_budget", type=int, default=3840)
    parser.add_argument("--action_budget", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--driver_arg", action="append", default=[])
    args = parser.parse_args()

    if not args.model_path:
        parser.error("--model_path or MODEL_PATH is required")
    cloud_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if cloud_model != common.DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION["model"]:
        parser.error(
            "this cost table is priced only for deepseek-v4-flash; "
            f"got DEEPSEEK_MODEL={cloud_model!r}"
        )
    if args.data_parallel_workers < 1:
        parser.error("--data_parallel_workers must be positive")
    if DISTINCT_GOALS < args.data_parallel_workers:
        parser.error("12 distinct goals must be >= data_parallel_workers")

    args.project_root = Path(__file__).resolve().parents[2]
    root = Path(args.root).expanduser().resolve()
    args.webshop_file_path = Path(args.webshop_file_path).expanduser().resolve()
    args.webshop_attr_path = Path(args.webshop_attr_path).expanduser().resolve()
    source_skills = Path(args.skills_json).expanduser().resolve()
    for path in (
        args.webshop_file_path,
        args.webshop_attr_path,
        source_skills,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    data_dir = args.webshop_file_path.parent
    for required in (
        data_dir / "items_human_ins.json",
        data_dir.parent / "search_engine" / "indexes",
    ):
        if not required.exists():
            parser.error(f"required WebShop asset does not exist: {required}")

    root.mkdir(parents=True, exist_ok=True)
    initial_skills = common._ensure_initial_skills(root, source_skills)
    config = {
        "experiment_kind": "webshop_train_step_trace_compression_ablation",
        "protocol": (
            "shared_12_distinct_goals_x_6_replicas_72_full_trajectories_"
            "then_two_independent_cloud_updates"
        ),
        "rollouts": TOTAL_ROLLOUTS,
        "distinct_goals": DISTINCT_GOALS,
        "rollouts_per_goal": ROLLOUTS_PER_GOAL,
        "max_environment_steps": MAX_ENVIRONMENT_STEPS,
        "task_types": list(WEBSHOP_TASK_TYPES),
        "arms": ARMS,
        "seed": args.seed,
        "webshop_file_path": str(args.webshop_file_path),
        "webshop_attr_path": str(args.webshop_attr_path),
        "initial_skills": str(initial_skills),
        "initial_skills_sha256": common._sha256_path(initial_skills),
        "data_parallel_workers": args.data_parallel_workers,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "model_path": str(args.model_path),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "cloud_evidence_multiplier": float(
            os.environ.get("COSKILL_CLOUD_EVIDENCE_MULTIPLIER", "1")
        ),
        "cloud_model": cloud_model,
        "deepseek_pricing": common.DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION,
    }
    config_path = root / "run_config.json"
    if config_path.exists() and common._read_json(config_path) != config:
        raise RuntimeError(
            "existing root has a different WebShop trace-ablation "
            "configuration; use a new --root"
        )
    common._write_json(config_path, config)

    raw_path = root / "shared" / "raw_traces.jsonl"
    if args.phase in ("capture", "all"):
        raw_path = capture_once(args, root, initial_skills)
    if args.phase == "capture":
        return
    if not raw_path.exists():
        raise RuntimeError(
            "--phase arms/report requires a completed capture under --root"
        )
    integrity = _validate_shared_raw(common._read_jsonl(raw_path))
    diagnostics = integrity["capture_diagnostics"]
    if args.phase in ("arms", "all"):
        for arm in ARMS:
            build_arm(
                root,
                arm,
                raw_path,
                initial_skills,
                args.retrieval_mode,
                diagnostics,
            )
    if args.phase == "arms":
        return
    write_reports(root)


if __name__ == "__main__":
    main()
