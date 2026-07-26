"""ALFWorld one-training-step CoSkill trace-compression cost ablation.

It captures one complete 72-rollout training group (up to 40 environment
actions per rollout) exactly once, then fans that immutable raw trace corpus
out to two otherwise identical CoSkill cloud updates:

* ``compression_on``: normal loop/delta/prefix/consensus processing;
* ``compression_off``: every trace-payload transformation disabled.

The result isolates cloud upload size and cache-aware API cost from rollout
sampling.  It never lets an arm's updated skill library feed another rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from agent_system.memory import HierarchicalSkillLib, TracesPool
from agent_system.memory.coskill_loop import CoSkillCloudLoop
from examples.playbook_evolve import fixed_trajectory_ablation as fixed

RUNTIME_TASK_TYPES = fixed.RUNTIME_TASK_TYPES
ROLL_OUTS_PER_TYPE = 12
TOTAL_ROLLOUTS = len(RUNTIME_TASK_TYPES) * ROLL_OUTS_PER_TYPE
MAX_ENVIRONMENT_STEPS = 40
ARMS = {
    "compression_on": {
        "enable_loop_filter": True,
        "enable_obs_delta": True,
        "enable_prefix_tree": True,
        "enable_consensus_prefix": True,
    },
    "compression_off": {
        "enable_loop_filter": False,
        "enable_obs_delta": False,
        "enable_prefix_tree": False,
        "enable_consensus_prefix": False,
    },
}

# Official DeepSeek V4 Flash prices, captured 2026-07-25.  The URL and all
# three rates are emitted in every report so future readers do not mistake a
# price simulation for an invoice.
DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION = {
    "model": "deepseek-v4-flash",
    "currency": "USD",
    "unit": "USD_per_1M_tokens",
    "prompt_cache_hit": 0.0028,
    "prompt_cache_miss": 0.14,
    "completion": 0.28,
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "snapshot_date": "2026-07-25",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_stats(value: Any) -> dict[str, int]:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "chars": len(text),
        "bytes_utf8": len(text.encode("utf-8")),
        "tokens_chars_div_4": max(1, len(text) // 4) if text else 0,
    }


def _capture_payload_stats(raw: list[dict[str, Any]]) -> dict[str, int]:
    return _json_stats({"raw_traces": raw})


def _pool(raw: Iterable[dict[str, Any]], outdir: Path, flags: dict[str, bool]) -> TracesPool:
    pool = TracesPool(
        output_dir=str(outdir),
        min_samples=1,
        enable_loop_filter=flags["enable_loop_filter"],
        enable_obs_delta=flags["enable_obs_delta"],
        enable_prefix_tree=flags["enable_prefix_tree"],
        enable_consensus_prefix=flags["enable_consensus_prefix"],
        cloud_evidence_mode=(
            "tree_only" if flags["enable_prefix_tree"] else "flat"
        ),
    )
    for trace in raw:
        pool.add_trace(trace)
    return pool


def _uploaded_trace_payload(batch: dict[str, Any]) -> dict[str, Any]:
    """Separate cloud-facing evidence from local compatibility state.

    With prefix compression enabled, the remote evidence is ``tree_evidence``:
    one action-node table plus per-rollout node-id paths and deltas.  The flat
    samples stay local so the loop can group task types and persist artifacts,
    but they are deliberately not counted as cloud upload payload.
    """
    flat_evidence = {
        "success_samples": batch.get("success_samples", []),
        "failure_samples": batch.get("failure_samples", []),
        "consensus_prefix": batch.get("consensus_prefix"),
    }
    evidence = batch.get("tree_evidence") or flat_evidence
    cloud_batch = TracesPool.project_cloud_batch(batch)
    codec_version = int((batch.get("tree_evidence") or {}).get("version", 0) or 0)
    return {
        "trace_evidence": _json_stats(evidence),
        "cloud_evidence_representation": (
            f"prefix_tree_codec_v{codec_version}"
            if codec_version else "flat_trajectories"),
        "local_flat_samples": _json_stats(flat_evidence),
        "cloud_batch": _json_stats(cloud_batch),
        "compressed_batch": _json_stats(batch),
        "trace_stage_totals": (batch.get("compression", {}) or {}).get("trace_stage_totals", {}),
        "prefix_tree": (batch.get("compression", {}) or {}).get("prefix_tree"),
        "consensus_prefix": (batch.get("compression", {}) or {}).get("consensus_prefix", {}),
    }


def _stage_row(name: str, value: dict[str, Any], *, token_source: str) -> dict[str, Any]:
    """Normalize one waterfall stage without conflating token definitions."""
    return {
        "stage": name,
        "steps": value.get("steps"),
        "chars": value.get("chars"),
        "bytes_utf8": value.get("bytes_utf8"),
        "tokens_chars_div_4": value.get("tokens"),
        "token_source": token_source,
    }


def build_token_waterfall(batch: dict[str, Any], calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Account for each compression stage and the prompts actually sent.

    ``raw`` through ``obs_delta`` are trace-only estimates from TracesPool.
    ``prefix_tree_context`` is the complete trace evidence supplied to the
    cloud analyzer (samples plus optional prefix/consensus structures).
    ``actual_cloud_prompt`` includes prompt instructions and may repeat
    evidence across calls, so it is intentionally not forced to be monotonic.
    """
    compression = batch.get("compression", {}) or {}
    stages = compression.get("trace_stage_totals", {}) or {}
    evidence = _uploaded_trace_payload(batch)["trace_evidence"]
    rows = list(calls)
    prompt_chars = sum(int(row.get("prompt_chars") or 0) for row in rows)
    prompt_bytes = sum(int(row.get("prompt_bytes_utf8") or 0) for row in rows)
    prompt_estimated = sum(int(row.get("prompt_tokens_chars_div_4") or 0) for row in rows)
    provider_prompt_values = [row.get("prompt_tokens") for row in rows]
    provider_prompt_complete = bool(rows) and all(value is not None for value in provider_prompt_values)

    waterfall = [
        _stage_row("raw", stages.get("raw", {}), token_source="chars_div_4_trace_only"),
        _stage_row("loop_filter", stages.get("loop_filtered", {}), token_source="chars_div_4_trace_only"),
        _stage_row("obs_delta", stages.get("encoded", {}), token_source="chars_div_4_trace_only"),
        {
            "stage": "prefix_tree_context",
            "steps": stages.get("encoded", {}).get("steps"),
            "chars": evidence["chars"],
            "bytes_utf8": evidence["bytes_utf8"],
            "tokens_chars_div_4": evidence["tokens_chars_div_4"],
            "token_source": "chars_div_4_trace_evidence",
        },
        {
            "stage": "actual_cloud_prompt",
            "steps": None,
            "chars": prompt_chars,
            "bytes_utf8": prompt_bytes,
            "tokens_chars_div_4": prompt_estimated,
            "provider_prompt_tokens": sum(int(value) for value in provider_prompt_values) if provider_prompt_complete else None,
            "provider_prompt_tokens_status": "reported" if provider_prompt_complete else "missing",
            "cloud_call_count": len(rows),
            "token_source": "provider_usage_when_reported_else_chars_div_4",
        },
    ]
    return {
        "display_order": [row["stage"] for row in waterfall],
        "note": (
            "The final cloud-prompt stage includes instructions and may repeat "
            "evidence across multiple calls; do not interpret it as a strict "
            "subset of prefix_tree_context."
        ),
        "stages": waterfall,
    }


def annotate_call_costs(
    calls: Iterable[dict[str, Any]],
    pricing: dict[str, Any] = DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION,
) -> list[dict[str, Any]]:
    """Add per-request provider usage and reproducible price simulations.

    ``observed_cache_billed_cost_usd`` is emitted only when the response
    contains DeepSeek's hit/miss split.  It is a reconstruction from returned
    usage and the recorded public tariff, not an account-balance query.
    ``all_input_cache_miss_cost_usd`` remains available when ordinary input and
    output usage is reported but the cache split is absent.
    """
    annotated = []
    million = 1_000_000.0
    for index, raw in enumerate(calls, start=1):
        row = dict(raw)
        prompt = row.get("prompt_tokens")
        completion = row.get("completion_tokens")
        hit = row.get("prompt_cache_hit_tokens")
        miss = row.get("prompt_cache_miss_tokens")
        usage_complete = prompt is not None and completion is not None
        cache_complete = usage_complete and hit is not None and miss is not None
        row["call_index"] = index
        row["pricing_model"] = pricing["model"]
        row["observed_cache_billed_cost_usd"] = (
            (int(hit) * float(pricing["prompt_cache_hit"])
             + int(miss) * float(pricing["prompt_cache_miss"])
             + int(completion) * float(pricing["completion"])) / million
            if cache_complete else None
        )
        row["all_input_cache_miss_cost_usd"] = (
            (int(prompt) * float(pricing["prompt_cache_miss"])
             + int(completion) * float(pricing["completion"])) / million
            if usage_complete else None
        )
        row["cost_status"] = (
            "cache_split_reported" if cache_complete
            else "provider_usage_without_cache_split" if usage_complete
            else "provider_usage_missing"
        )
        annotated.append(row)
    return annotated


def summarize_cloud_cost(calls: Iterable[dict[str, Any]], pricing: dict[str, Any] = DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION) -> dict[str, Any]:
    """Produce observed and cache-neutral cost estimates from provider usage.

    Observed cost is only emitted when *every* cloud call returned cache split
    fields.  All-miss cost uses prompt/completion totals and remains useful for
    comparing payload compression even when DeepSeek omitted cache metadata.
    """
    rows = list(calls)
    hit = miss = completion = prompt = 0
    cache_complete = bool(rows)
    usage_complete = bool(rows)
    for row in rows:
        row_prompt = row.get("prompt_tokens")
        row_completion = row.get("completion_tokens")
        if row_prompt is None or row_completion is None:
            usage_complete = False
        else:
            prompt += int(row_prompt)
            completion += int(row_completion)
        row_hit = row.get("prompt_cache_hit_tokens")
        row_miss = row.get("prompt_cache_miss_tokens")
        if row_hit is None or row_miss is None:
            cache_complete = False
        else:
            hit += int(row_hit)
            miss += int(row_miss)
    denominator = 1_000_000.0
    observed = None
    if cache_complete and usage_complete:
        observed = (hit * float(pricing["prompt_cache_hit"]) + miss * float(pricing["prompt_cache_miss"]) + completion * float(pricing["completion"])) / denominator
    all_miss = None
    if usage_complete:
        all_miss = (prompt * float(pricing["prompt_cache_miss"]) + completion * float(pricing["completion"])) / denominator
    return {
        "pricing": dict(pricing),
        "cloud_call_count": len(rows),
        "provider_usage_complete": usage_complete,
        "cache_split_complete": cache_complete,
        "prompt_cache_hit_tokens": hit if cache_complete else None,
        "prompt_cache_miss_tokens": miss if cache_complete else None,
        "prompt_tokens": prompt if usage_complete else None,
        "completion_tokens": completion if usage_complete else None,
        "observed_cache_billed_cost_usd": observed,
        "all_input_cache_miss_cost_usd": all_miss,
        "observed_cost_accounting": ("provider_prompt_cache_hit_tokens_plus_prompt_cache_miss_tokens_plus_completion_tokens" if observed is not None else "N.A._provider_cache_split_missing"),
        "all_miss_cost_accounting": ("provider_prompt_tokens_repriced_as_cache_miss_plus_completion_tokens" if all_miss is not None else "N.A._provider_usage_missing"),
    }


def _validate_shared_raw(raw: list[dict[str, Any]]) -> dict[str, Any]:
    if len(raw) != TOTAL_ROLLOUTS:
        raise RuntimeError(f"expected exactly {TOTAL_ROLLOUTS} captured traces, found {len(raw)}")
    counts = {task_type: 0 for task_type in RUNTIME_TASK_TYPES}
    invalid_steps = []
    observed_step_counts = []
    for index, trace in enumerate(raw, start=1):
        task_type = str(trace.get("task_type", ""))
        if task_type not in counts:
            raise RuntimeError(f"capture trace {index} has unknown task_type={task_type!r}")
        counts[task_type] += 1
        step_count = len(trace.get("steps") or [])
        observed_step_counts.append(step_count)
        if step_count < 1 or step_count > MAX_ENVIRONMENT_STEPS:
            invalid_steps.append(index)
    if invalid_steps:
        raise RuntimeError(
            "full-trajectory protocol violated; trace step counts must be in "
            f"[1, {MAX_ENVIRONMENT_STEPS}]: " + ", ".join(map(str, invalid_steps[:10])))
    # A pre-existing one-environment-action capture is not valid evidence for
    # trajectory compression: it contains no temporal redundancy for loop or
    # observation-delta compression.  Reject it rather than silently reusing
    # the old one-step corpus under the corrected protocol.
    if max(observed_step_counts, default=0) <= 1:
        raise RuntimeError(
            "full-trajectory protocol requires at least one trace with more "
            "than one environment step; do not reuse a one-action capture")
    if any(value != ROLL_OUTS_PER_TYPE for value in counts.values()):
        raise RuntimeError(f"expected {ROLL_OUTS_PER_TYPE} captures per task, got {counts}")
    return {
        "rollouts": len(raw),
        "per_task_type": counts,
        "max_environment_steps": MAX_ENVIRONMENT_STEPS,
        "observed_steps_per_trace": {
            "min": min(observed_step_counts),
            "max": max(observed_step_counts),
            "total": sum(observed_step_counts),
        },
    }


def _driver_cmd(args: argparse.Namespace, capture_dir: Path, manifest: Path, initial_skills: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "examples.playbook_evolve.run_playbook_evolve",
        "--outdir",
        str(capture_dir),
        "--fixed_games_manifest",
        str(manifest),
        "--epochs",
        "1",
        "--group_size",
        str(ROLL_OUTS_PER_TYPE),
        "--max_episodes",
        str(TOTAL_ROLLOUTS),
        "--batch_rollout_size",
        str(TOTAL_ROLLOUTS),
        "--max_steps",
        str(MAX_ENVIRONMENT_STEPS),
        "--seed",
        str(args.seed),
        "--retrieval_mode",
        args.retrieval_mode,
        "--enable_coskill",
        "1",
        "--enable_skill_tree",
        "1",
        "--enable_skill_tree_evolve",
        "0",
        "--enable_cloud_updates",
        "0",
        "--enable_failure_analysis",
        "1",
        "--log_trajectories",
        "0",
        "--gpu_mem_util",
        str(args.gpu_mem_util),
        "--vllm_enforce_eager",
        str(args.vllm_enforce_eager),
        "--skills_json",
        str(initial_skills),
    ]
    if args.model_path:
        command += ["--model_path", args.model_path]
    if args.data_parallel_workers:
        command += ["--data_parallel_workers", str(args.data_parallel_workers)]
    if args.rollout_worker_gpus:
        command += ["--rollout_worker_gpus", args.rollout_worker_gpus]
    command.extend(args.driver_arg)
    return command


def _ensure_initial_skills(root: Path, source: Path) -> Path:
    target = root / "shared" / "initial_skills.json"
    if target.exists():
        if _sha256_path(target) != _sha256_path(source):
            raise RuntimeError("initial skills source changed under an existing root; use a new --root")
        return target
    try:
        _read_json(source)
    except Exception as exc:
        raise RuntimeError(f"--skills_json is not readable JSON: {source}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def capture_once(args: argparse.Namespace, root: Path, manifest: Path, initial_skills: Path) -> Path:
    raw_path = root / "shared" / "raw_traces.jsonl"
    if raw_path.exists():
        _validate_shared_raw(_read_jsonl(raw_path))
        return raw_path
    capture_dir = root / "capture"
    command = _driver_cmd(args, capture_dir, manifest, initial_skills)
    print("[train-step-trace-ablation] capture:", " ".join(command))
    subprocess.run(command, cwd=args.project_root, check=True)
    source = capture_dir / "traces_pool" / "raw_traces.jsonl"
    if not source.exists():
        raise RuntimeError(f"capture completed without raw trace log: {source}")
    raw = _read_jsonl(source)
    integrity = _validate_shared_raw(raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, raw_path)
    _write_json(
        root / "capture" / "capture_integrity.json",
        {
            "protocol": "shared_72_rollouts_x_one_training_group_full_trajectories",
            "raw_traces": str(raw_path),
            "raw_traces_sha256": _sha256_path(raw_path),
            **integrity,
        },
    )
    return raw_path


def build_arm(root: Path, arm: str, raw_path: Path, initial_skills: Path, retrieval_mode: str) -> dict[str, Any]:
    arm_dir = root / "arms" / arm
    result_path = arm_dir / "arm_result.json"
    if result_path.exists():
        return _read_json(result_path)
    flags = ARMS[arm]
    raw = _read_jsonl(raw_path)
    reporting_pool = _pool(raw, arm_dir / "trace_payload", flags)
    batch = reporting_pool.export_batch(trigger_reason="one_training_group_shared_capture")
    _write_json(arm_dir / "compressed_batch.json", batch)

    # A distinct pool is necessary because export_batch intentionally drains
    # the first one.  Both pools receive the same immutable rows and flags.
    cloud_pool = _pool(raw, arm_dir, flags)
    library = HierarchicalSkillLib(str(initial_skills), retrieval_mode=retrieval_mode, enable_playbook=True)
    loop = CoSkillCloudLoop(
        output_dir=str(arm_dir),
        enable_coskill=True,
        enable_playbook_evolve=True,
        enable_failure_analysis=True,
        environment_name="ALFWorld",
    )
    fired = loop.maybe_update(cloud_pool, library, TOTAL_ROLLOUTS, force_reason="one_training_group_shared_capture")
    if not fired:
        raise RuntimeError(f"{arm} did not execute its required one cloud update")
    skills_path = arm_dir / "skill_lib" / f"skills_step{TOTAL_ROLLOUTS}.json"
    if not skills_path.exists():
        raise RuntimeError(f"{arm} cloud update did not persist its skill library")
    analyzer = loop.cloud_analyzer
    calls = annotate_call_costs(getattr(analyzer, "call_audit", []) or [])
    _write_json(arm_dir / "cloud_io" / "call_audit.json", calls)
    call_cost_path = arm_dir / "cloud_io" / "call_costs.csv"
    if calls:
        with call_cost_path.open("w", newline="") as handle:
            fields = [
                "call_index", "purpose", "task_type", "model", "pricing_model",
                "prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens", "observed_cache_billed_cost_usd",
                "all_input_cache_miss_cost_usd", "cost_status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(calls)
    waterfall = build_token_waterfall(batch, calls)
    _write_json(arm_dir / "token_waterfall.json", waterfall)
    result = {
        "arm": arm,
        "protocol": "same_shared_raw_72_full_trajectories_then_one_coskill_cloud_update",
        "raw_traces_sha256": _sha256_path(raw_path),
        "compression_flags": flags,
        "capture_upload_payload": _capture_payload_stats(raw),
        "uploaded_trace_payload": _uploaded_trace_payload(batch),
        "token_waterfall": waterfall,
        "token_waterfall_path": str(arm_dir / "token_waterfall.json"),
        "cloud_cost": summarize_cloud_cost(calls),
        "cloud_call_audit_path": str(arm_dir / "cloud_io" / "call_audit.json"),
        "cloud_call_costs_path": str(call_cost_path),
        "skills_path": str(skills_path),
        "skill_sha256": _sha256_path(skills_path),
        "cloud_update_fired": True,
    }
    _write_json(result_path, result)
    return result


def _difference(on: float | None, off: float | None) -> dict[str, float | None]:
    if on is None or off is None:
        return {"all_off_minus_on": None, "on_saves_percent_of_all_off": None}
    delta = off - on
    return {
        "all_off_minus_on": delta,
        "on_saves_percent_of_all_off": (delta / off * 100.0) if off else None,
    }


def _waterfall_by_stage(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["stage"]: row
        for row in (result.get("token_waterfall", {}) or {}).get("stages", [])
    }


def write_reports(root: Path) -> dict[str, Any]:
    on = _read_json(root / "arms" / "compression_on" / "arm_result.json")
    off = _read_json(root / "arms" / "compression_off" / "arm_result.json")
    on_payload = on["uploaded_trace_payload"]
    off_payload = off["uploaded_trace_payload"]
    on_cost = on["cloud_cost"]
    off_cost = off["cloud_cost"]
    on_waterfall = _waterfall_by_stage(on)
    off_waterfall = _waterfall_by_stage(off)
    waterfall_rows = []
    for arm_result in (on, off):
        for stage in (arm_result.get("token_waterfall", {}) or {}).get("stages", []):
            waterfall_rows.append({"arm": arm_result["arm"], **stage})
    _write_json(
        root / "token_waterfall.json",
        {
            "protocol": on["protocol"],
            "raw_trace_sha256": on["raw_traces_sha256"],
            "normal_compression": on["token_waterfall"],
            "all_compression_off": off["token_waterfall"],
        },
    )
    with (root / "token_waterfall.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in waterfall_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(waterfall_rows)
    comparison = {
        "protocol": on["protocol"],
        "direction": "positive all_off_minus_on means normal CoSkill compression is smaller/cheaper",
        "raw_trace_sha256": on["raw_traces_sha256"],
        "uploaded_trace_payload_delta": {
            "trace_evidence_chars": _difference(on_payload["trace_evidence"]["chars"], off_payload["trace_evidence"]["chars"]),
            "trace_evidence_tokens_chars_div_4": _difference(on_payload["trace_evidence"]["tokens_chars_div_4"], off_payload["trace_evidence"]["tokens_chars_div_4"]),
            "compressed_batch_chars": _difference(on_payload["compressed_batch"]["chars"], off_payload["compressed_batch"]["chars"]),
            "compressed_batch_tokens_chars_div_4": _difference(on_payload["compressed_batch"]["tokens_chars_div_4"], off_payload["compressed_batch"]["tokens_chars_div_4"]),
        },
        "provider_token_delta": {
            "prompt_tokens": _difference(on_cost["prompt_tokens"], off_cost["prompt_tokens"]),
            "completion_tokens": _difference(on_cost["completion_tokens"], off_cost["completion_tokens"]),
            "cache_hit_tokens": _difference(on_cost["prompt_cache_hit_tokens"], off_cost["prompt_cache_hit_tokens"]),
            "cache_miss_tokens": _difference(on_cost["prompt_cache_miss_tokens"], off_cost["prompt_cache_miss_tokens"]),
        },
        "cost_delta_usd": {
            "observed_cache_billed": _difference(on_cost["observed_cache_billed_cost_usd"], off_cost["observed_cache_billed_cost_usd"]),
            "all_input_cache_miss": _difference(on_cost["all_input_cache_miss_cost_usd"], off_cost["all_input_cache_miss_cost_usd"]),
        },
        "token_waterfall_delta": {
            stage: {
                "chars": _difference(on_waterfall.get(stage, {}).get("chars"), off_waterfall.get(stage, {}).get("chars")),
                "tokens_chars_div_4": _difference(on_waterfall.get(stage, {}).get("tokens_chars_div_4"), off_waterfall.get(stage, {}).get("tokens_chars_div_4")),
                "provider_prompt_tokens": _difference(on_waterfall.get(stage, {}).get("provider_prompt_tokens"), off_waterfall.get(stage, {}).get("provider_prompt_tokens")),
            }
            for stage in ("raw", "loop_filter", "obs_delta", "prefix_tree_context", "actual_cloud_prompt")
        },
    }
    _write_json(root / "compression_comparison.json", comparison)
    metrics = []
    for result in (on, off):
        payload, cost = result["uploaded_trace_payload"], result["cloud_cost"]
        metrics.append(
            {
                "step": TOTAL_ROLLOUTS,
                "metrics": {
                    "experiment/name": "alfworld_train_step_trace_compression_ablation",
                    "experiment/arm": result["arm"],
                    "experiment/rollouts": TOTAL_ROLLOUTS,
                    "experiment/max_environment_steps": MAX_ENVIRONMENT_STEPS,
                    "experiment/task_success_metric": "capture_group_diagnostic_only",
                    "trace_upload/evidence_chars": payload["trace_evidence"]["chars"],
                    "trace_upload/evidence_tokens_chars_div_4": payload["trace_evidence"]["tokens_chars_div_4"],
                    "trace_upload/batch_chars": payload["cloud_batch"]["chars"],
                    "trace_upload/batch_tokens_chars_div_4": payload["cloud_batch"]["tokens_chars_div_4"],
                    **{
                        f"trace_waterfall/{row['stage']}/tokens_chars_div_4": row["tokens_chars_div_4"]
                        for row in result["token_waterfall"]["stages"]
                    },
                    "trace_waterfall/actual_cloud_prompt/provider_tokens": _waterfall_by_stage(result)["actual_cloud_prompt"]["provider_prompt_tokens"],
                    "tokens/large_model/prompt": cost["prompt_tokens"],
                    "tokens/large_model/completion": cost["completion_tokens"],
                    "tokens/large_model/cache_hit": cost["prompt_cache_hit_tokens"],
                    "tokens/large_model/cache_miss": cost["prompt_cache_miss_tokens"],
                    "cost/deepseek_v4_flash/observed_usd": cost["observed_cache_billed_cost_usd"],
                    "cost/deepseek_v4_flash/all_miss_usd": cost["all_input_cache_miss_cost_usd"],
                },
            }
        )
    with (root / "metrics.jsonl").open("w") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = []
    for result in (on, off):
        payload, cost = result["uploaded_trace_payload"], result["cloud_cost"]
        rows.append(
            {
                "arm": result["arm"],
                "trace_evidence_chars": payload["trace_evidence"]["chars"],
                "trace_evidence_tokens_chars_div_4": payload["trace_evidence"]["tokens_chars_div_4"],
                "cloud_prompt_tokens": cost["prompt_tokens"],
                "cloud_completion_tokens": cost["completion_tokens"],
                "cache_hit_tokens": cost["prompt_cache_hit_tokens"],
                "cache_miss_tokens": cost["prompt_cache_miss_tokens"],
                "observed_cache_billed_cost_usd": cost["observed_cache_billed_cost_usd"],
                "all_input_cache_miss_cost_usd": cost["all_input_cache_miss_cost_usd"],
            }
        )
    with (root / "cost_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "done",
        "arms": [on, off],
        "comparison": comparison,
        "metrics": str(root / "metrics.jsonl"),
        "cost_table": str(root / "cost_comparison.csv"),
        "token_waterfall": str(root / "token_waterfall.json"),
        "token_waterfall_table": str(root / "token_waterfall.csv"),
    }
    _write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=("capture", "arms", "report", "all"), default="all")
    parser.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--skills_json", default="memory_data/alfworld/claude_style_skills.json")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--retrieval_mode", choices=("template", "embedding"), default="template")
    parser.add_argument("--data_parallel_workers", type=int, default=1)
    parser.add_argument("--rollout_worker_gpus", default=None)
    parser.add_argument("--gpu_mem_util", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")))
    parser.add_argument("--vllm_enforce_eager", type=int, choices=(0, 1), default=int(os.environ.get("VLLM_ENFORCE_EAGER", "0")))
    parser.add_argument("--driver_arg", action="append", default=[])
    args = parser.parse_args()
    if not args.alfworld_data:
        parser.error("--alfworld_data or ALFWORLD_DATA is required")
    cloud_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if cloud_model != DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION["model"]:
        parser.error(f"this cost table is priced only for deepseek-v4-flash; got DEEPSEEK_MODEL={cloud_model!r}")
    args.project_root = Path(__file__).resolve().parents[2]
    root = Path(args.root).expanduser().resolve()
    data_root = Path(args.alfworld_data).expanduser().resolve()
    source_skills = Path(args.skills_json).expanduser().resolve()
    if not source_skills.is_file():
        parser.error(f"--skills_json does not exist: {source_skills}")
    if not (data_root / "json_2.1.1").is_dir():
        parser.error(f"ALFWorld data is invalid: {data_root}")
    root.mkdir(parents=True, exist_ok=True)
    initial_skills = _ensure_initial_skills(root, source_skills)
    capture_manifest, _unused_eval_manifest = fixed.create_manifests(root, data_root, args.split, args.sample_seed, eval_games_per_type=1)
    config = {
        "experiment_kind": "alfworld_train_step_trace_compression_ablation",
        "protocol": "shared_72_rollouts_x_one_training_group_full_trajectories_then_two_independent_cloud_updates",
        "rollouts": TOTAL_ROLLOUTS,
        "rollouts_per_task_type": ROLL_OUTS_PER_TYPE,
        "max_environment_steps": MAX_ENVIRONMENT_STEPS,
        "arms": ARMS,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "manifest": str(capture_manifest),
        "manifest_sha256": _sha256_path(capture_manifest),
        "initial_skills": str(initial_skills),
        "initial_skills_sha256": _sha256_path(initial_skills),
        "cloud_model": cloud_model,
        "deepseek_pricing": DEEPSEEK_V4_FLASH_PRICING_USD_PER_MILLION,
    }
    config_path = root / "run_config.json"
    if config_path.exists() and _read_json(config_path) != config:
        raise RuntimeError("existing root has a different train-step trace-ablation configuration; use a new --root")
    _write_json(config_path, config)
    raw_path = root / "shared" / "raw_traces.jsonl"
    if args.phase in ("capture", "all"):
        raw_path = capture_once(args, root, capture_manifest, initial_skills)
    if args.phase == "capture":
        return
    if not raw_path.exists():
        raise RuntimeError("--phase arms/report requires completed --phase capture under the same --root")
    _validate_shared_raw(_read_jsonl(raw_path))
    if args.phase in ("arms", "all"):
        for arm in ARMS:
            build_arm(root, arm, raw_path, initial_skills, args.retrieval_mode)
    if args.phase == "arms":
        return
    write_reports(root)


if __name__ == "__main__":
    main()
