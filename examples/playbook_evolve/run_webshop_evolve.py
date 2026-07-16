"""Frozen-model CoSkill evolution on WebShop, aligned with WebShop GRPO.

One rollout group follows the CoSkill ALFWorld comparison standard:
``train_data_size=12`` distinct shopping instructions, each sampled
``group_size=6`` times (72 episodes).  The main process is the only owner of
TracesPool and cloud state; persistent GPU workers finish a full group before a
watermark may trigger DeepSeek.  Outputs intentionally mirror the ALFWorld
no-RL driver: traces_pool/, cloud_io/, skill_lib/, trajectories/,
metrics.jsonl, group_metrics.jsonl, checkpoints/, summary*.json and
coskill_status.json.
"""

import argparse
import atexit
import fcntl
import json
import multiprocessing as mp
import os
import re
import subprocess
import time
import traceback
import uuid

import numpy as np

from agent_system.environments.env_package.webshop.projection import webshop_projection
from agent_system.memory import CoSkillCloudLoop, HierarchicalSkillLib, TracesPool
from mini_test_pen_shelf.webshop_utils import (
    LocalBatchWebShopEnv,
    WebShopObsBuilder,
    extract_webshop_task,
    format_webshop_observation,
    webshop_trace_observation,
)


def _append_jsonl(path, obj):
    with open(path, "a") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _atomic_write_lines(path, lines):
    """Atomically replace a small JSONL prefix while preserving durability."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    os.replace(tmp, path)


def _acquire_run_lock(outdir):
    """Ensure exactly one CoSkill WebShop driver owns an output directory."""
    path = os.path.join(outdir, ".run_webshop_evolve.lock")
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another CoSkill WebShop driver already owns {outdir}. "
            f"Do not launch a second RESUME=1 process into the same OUTPUT_DIR. "
            f"Inspect {path} and the running process first."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"[webshop-driver] acquired output lock {path}", flush=True)
    return handle


def _trim_uncheckpointed_jsonl_tail(path, keep_lines, label):
    """Keep a checkpoint-consistent JSONL prefix and archive any newer tail.

    A stopped rollout can have already appended per-episode diagnostics after
    the most recent checkpoint.  Resuming from that checkpoint must not append
    a second copy of those records to the main logs.  The discarded tail is
    retained beside the log for post-mortem inspection.
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) < keep_lines:
        raise RuntimeError(
            f"Cannot resume: {label} has only {len(lines)} rows, but the "
            f"checkpoint requires {keep_lines}. Do not resume into this outdir."
        )
    if len(lines) <= keep_lines:
        return
    suffix = time.strftime("%Y%m%d_%H%M%S")
    archived = f"{path}.uncheckpointed_tail_{suffix}"
    with open(archived, "w", encoding="utf-8") as handle:
        handle.writelines(lines[keep_lines:])
    _atomic_write_lines(path, lines[:keep_lines])
    print(f"[webshop-driver][resume] archived {len(lines) - keep_lines} "
          f"uncheckpointed {label} rows to {archived}")


def _load_resume_state(args):
    """Load a checkpoint-consistent WebShop CoSkill continuation state."""
    state = {
        "resume": False,
        "skills_json_path": args.skills_json,
        "completed_groups": 0,
        "per_game": [],
        "wins": 0,
        "score_sum": 0.0,
        "category_stats": {},
        "cloud_update_steps": [],
        "small_cumulative": {"prompt": 0, "response": 0, "total": 0},
        "validation_history": [],
        "validation_small_cumulative": {"prompt": 0, "response": 0, "total": 0},
        "large_cumulative": {"prompt": 0, "completion": 0, "total": 0},
    }
    if not args.resume:
        return state

    summary_path = os.path.join(args.outdir, "summary_partial.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"--resume 1 requires {summary_path}. Resume is explicit to avoid "
            "silently mixing a new run with an old output directory."
        )
    with open(summary_path, encoding="utf-8") as handle:
        previous = json.load(handle)

    per_game = list(previous.get("per_game") or [])
    completed_groups = int(previous.get("completed_rollout_groups", 0) or 0)
    saved_episodes = int(previous.get("total_episodes", len(per_game)) or 0)
    if saved_episodes != len(per_game):
        raise RuntimeError(
            f"Resume checkpoint is inconsistent: total_episodes={saved_episodes}, "
            f"but per_game has {len(per_game)} rows."
        )

    skill_path = previous.get("skill_lib_checkpoint")
    candidates = [skill_path] if skill_path else []
    candidates.extend([
        os.path.join(args.outdir, "skill_lib", "skills_latest_checkpoint.json"),
        os.path.join(args.outdir, "skill_lib", "skills_latest_rollout.json"),
    ])
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            state["skills_json_path"] = candidate
            break
    else:
        raise FileNotFoundError(
            "--resume 1 found progress metadata but no skill library checkpoint "
            f"under {os.path.join(args.outdir, 'skill_lib')}"
        )

    category_stats = {}
    for row in per_game:
        category = row.get("detected_type", "unknown")
        stat = category_stats.setdefault(category, {
            "episodes": 0, "wins": 0, "task_score_sum": 0.0,
        })
        stat["episodes"] += 1
        stat["wins"] += int(bool(row.get("won", False)))
        stat["task_score_sum"] += float(row.get("task_score", 0.0) or 0.0)

    token_usage = previous.get("token_usage") or {}
    small_usage = token_usage.get("small_model") or {}
    validation_small_usage = small_usage.get("validation") or {}
    large_usage = token_usage.get("large_model") or {}
    state.update({
        "resume": True,
        "completed_groups": completed_groups,
        "per_game": per_game,
        "wins": int(previous.get("wins", sum(s["wins"] for s in category_stats.values())) or 0),
        "score_sum": sum(s["task_score_sum"] for s in category_stats.values()),
        "category_stats": category_stats,
        "cloud_update_steps": [int(step) for step in previous.get("cloud_update_steps", [])],
        "small_cumulative": {
            key: int(small_usage.get(key, 0) or 0)
            for key in ("prompt", "response", "total")
        },
        "validation_history": list(previous.get("validation_history") or []),
        "validation_small_cumulative": {
            key: int(validation_small_usage.get(key, 0) or 0)
            for key in ("prompt", "response", "total")
        },
        "large_cumulative": {
            key: int(large_usage.get(key, 0) or 0)
            for key in ("prompt", "completion", "total")
        },
    })

    # Make the primary append-only logs match the checkpoint before adding new
    # rows. The archived files retain diagnostics from a partially completed
    # rollout group that will be re-run after resume.
    _trim_uncheckpointed_jsonl_tail(
        os.path.join(args.outdir, "metrics.jsonl"), len(per_game), "episode metric"
    )
    _trim_uncheckpointed_jsonl_tail(
        os.path.join(args.outdir, "group_metrics.jsonl"), completed_groups, "group metric"
    )
    _trim_uncheckpointed_jsonl_tail(
        os.path.join(args.outdir, "validation_metrics.jsonl"),
        len(state["validation_history"]), "validation metric"
    )
    _trim_uncheckpointed_jsonl_tail(
        os.path.join(args.outdir, "traces_pool", "raw_traces.jsonl"),
        len(per_game), "raw trace"
    )
    print(f"[webshop-driver][resume] checkpoint={summary_path} "
          f"groups={completed_groups} episodes={len(per_game)} "
          f"skill_lib<-{state['skills_json_path']}")
    return state


def _rehydrate_traces_pool(traces_pool, outdir, state):
    """Restore pending trace-pool state without duplicating its raw JSONL log."""
    if not state["resume"] or not state["per_game"]:
        return
    raw_path = os.path.join(outdir, "traces_pool", "raw_traces.jsonl")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(
            f"Cannot restore pending CoSkill traces: {raw_path} is missing."
        )
    records = []
    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cannot restore pending CoSkill traces: malformed JSON at "
                    f"{raw_path}:{line_number}") from exc
    expected = len(state["per_game"])
    if len(records) != expected:
        raise RuntimeError(
            f"Cannot restore pending CoSkill traces: expected {expected} records "
            f"from the checkpoint, found {len(records)}."
        )

    raw_log_path = traces_pool._raw_log_path
    output_dir = traces_pool.output_dir
    traces_pool._raw_log_path = None
    traces_pool.output_dir = None
    update_steps = set(state["cloud_update_steps"])
    try:
        for episode_index, raw_trace in enumerate(records, start=1):
            traces_pool.add_trace(raw_trace)
            if episode_index in update_steps:
                # A previous successful cloud update exported/cleared the pool.
                # Repeating only that local clear reconstructs the pending pool
                # without invoking the cloud or rewriting any artifacts.
                traces_pool.export_batch(trigger_reason="resume_rehydrate")
    finally:
        traces_pool._raw_log_path = raw_log_path
        traces_pool.output_dir = output_dir
    print(f"[webshop-driver][resume] rehydrated trace pool from {expected} traces; "
          f"pending_since_last_cloud_update={traces_pool.stats()['pending_added']}")


def _split_int(total, parts):
    base, remainder = divmod(total, parts)
    return [base + int(i < remainder) for i in range(parts)]


def _parse_gpu_ids(raw):
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _resolve_worker_gpu_groups(args):
    """Map visible GPUs to one disjoint vLLM group per DP worker.

    A group has ``tensor_parallel_size * pipeline_parallel_size`` GPUs.  This
    keeps the existing DP task split stable when a four-GPU run uses DP=2,
    TP=2, while still allowing an explicit DP=4, TP=1 throughput topology.
    """
    if args.rollout_worker_gpus:
        gpu_ids = _parse_gpu_ids(args.rollout_worker_gpus)
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_ids = _parse_gpu_ids(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        gpu_ids = [line.strip() for line in raw.splitlines() if line.strip()]

    if args.tensor_parallel_size < 1 or args.pipeline_parallel_size < 1:
        raise ValueError("tensor_parallel_size and pipeline_parallel_size must both be >= 1")
    devices_per_worker = args.tensor_parallel_size * args.pipeline_parallel_size
    required_gpus = args.data_parallel_workers * devices_per_worker
    if len(gpu_ids) < required_gpus:
        raise ValueError(
            f"Need {required_gpus} GPUs for DP={args.data_parallel_workers}, "
            f"TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}; "
            f"found {gpu_ids}")
    gpu_ids = gpu_ids[:required_gpus]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Rollout GPU list contains duplicates: {gpu_ids}")
    return [
        ",".join(gpu_ids[start:start + devices_per_worker])
        for start in range(0, required_gpus, devices_per_worker)
    ]


def _token_delta(after, before):
    prompt = max(0, int(after.get("prompt", 0)) - int(before.get("prompt", 0)))
    response = max(0, int(after.get("response", 0)) - int(before.get("response", 0)))
    return {"prompt": prompt, "response": response, "total": prompt + response}


def _phase_stats(rows):
    phases = {}
    for row in rows:
        phase = phases.setdefault(str(row.get("cloud_round_used", 0)), {
            "episodes": 0, "wins": 0, "task_score_sum": 0.0, "by_task_type": {},
        })
        phase["episodes"] += 1
        phase["wins"] += int(row.get("won", False))
        phase["task_score_sum"] += float(row.get("task_score", 0.0))
        category = row.get("detected_type", "unknown")
        cat = phase["by_task_type"].setdefault(category, {
            "episodes": 0, "wins": 0, "task_score_sum": 0.0,
        })
        cat["episodes"] += 1
        cat["wins"] += int(row.get("won", False))
        cat["task_score_sum"] += float(row.get("task_score", 0.0))
    for phase in phases.values():
        phase["success_rate"] = round(phase["wins"] / max(phase["episodes"], 1), 6)
        phase["mean_task_score"] = round(
            phase.pop("task_score_sum") / max(phase["episodes"], 1), 6)
        for cat in phase["by_task_type"].values():
            cat["success_rate"] = round(cat["wins"] / max(cat["episodes"], 1), 6)
            cat["mean_task_score"] = round(
                cat.pop("task_score_sum") / max(cat["episodes"], 1), 6)
    return phases


def _validation_metrics(results, *, group_id, global_episode, token_delta,
                        token_cumulative):
    """Summarize a held-out WebShop pass in the common metric schema."""
    count = len(results)
    lengths = [int(item.get("used", 0) or 0) for item in results]
    wins = sum(int(bool(item.get("won", False))) for item in results)
    scores = [float(item.get("task_score", 0.0) or 0.0) for item in results]
    valid = sum(int(item.get("n_valid", 0) or 0) for item in results)
    relaxed = sum(int(item.get("n_relaxed_valid", 0) or 0) for item in results)
    total_actions = sum(lengths)
    metric = {
        "validation/group": int(group_id),
        "validation/global_episode": int(global_episode),
        "validation/episode/count": count,
        "validation/episode/wins": wins,
        "validation/episode/success_rate": round(wins / max(count, 1), 6),
        "validation/episode/task_score/mean": round(sum(scores) / max(count, 1), 6),
        "validation/episode/task_score/max": max(scores or [0.0]),
        "validation/episode/task_score/min": min(scores or [0.0]),
        "validation/episode/length/mean": round(sum(lengths) / max(count, 1), 6),
        "validation/episode/length/max": max(lengths or [0]),
        "validation/episode/length/min": min(lengths or [0]),
        "validation/episode/strict_valid_action_ratio": round(
            valid / max(total_actions, 1), 6),
        "validation/episode/relaxed_valid_action_ratio": round(
            relaxed / max(total_actions, 1), 6),
        "validation/tokens/small_model/prompt": int(token_delta["prompt"]),
        "validation/tokens/small_model/response": int(token_delta["response"]),
        "validation/tokens/small_model/total": int(token_delta["total"]),
        "validation/tokens/small_model/prompt_cumulative": int(token_cumulative["prompt"]),
        "validation/tokens/small_model/response_cumulative": int(token_cumulative["response"]),
        "validation/tokens/small_model/total_cumulative": int(token_cumulative["total"]),
        "validation/rollout_accounting": "heldout_active_env_decisions",
    }
    by_category = {}
    for item in results:
        category = str(item.get("task_type", "unknown"))
        stat = by_category.setdefault(category, {"count": 0, "wins": 0, "score": 0.0})
        stat["count"] += 1
        stat["wins"] += int(bool(item.get("won", False)))
        stat["score"] += float(item.get("task_score", 0.0) or 0.0)
    for category, stat in sorted(by_category.items()):
        prefix = f"validation/episode/{category}"
        metric[f"{prefix}/count"] = stat["count"]
        metric[f"{prefix}/wins"] = stat["wins"]
        metric[f"{prefix}/success_rate"] = round(stat["wins"] / max(stat["count"], 1), 6)
        metric[f"{prefix}/task_score/mean"] = round(stat["score"] / max(stat["count"], 1), 6)
    return metric


def _dump_episode(outdir, episode_idx, result):
    directory = os.path.join(outdir, "trajectories")
    os.makedirs(directory, exist_ok=True)
    status = "WIN" if result["won"] else "FAIL"
    category = result["task_type"]
    base = f"ep{episode_idx:04d}_{category}_{status}_{result['used']}steps"
    playbook = result.get("playbook_record")
    tree_meta = None
    if playbook:
        tree_meta = {
            "version": playbook.get("version"),
            "level": playbook.get("level"),
            "n_nodes": len(playbook.get("nodes") or {}),
        }
    payload = {
        "episode": episode_idx,
        "task": result["raw_trace"]["task"],
        "task_type": category,
        "outcome": status,
        "task_score": result["task_score"],
        "goal_index": result.get("goal_index"),
        "used_steps": result["used"],
        "skill_tree_used": tree_meta,
        "skill_ids_used": list(result.get("injected") or []),
        "steps": result.get("logrows") or [],
    }
    with open(os.path.join(directory, base + "_episode.json"), "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "=" * 78,
        f"Episode #{episode_idx} category={category} [{status} / {result['used']} steps]",
        f"task_score={result['task_score']:.4f} goal_index={result.get('goal_index')}",
        f"task: {result['raw_trace']['task']}",
        f"skill tree: {tree_meta or '(none)'}",
        "=" * 78,
    ]
    for step in result.get("logrows") or []:
        lines.extend([
            "",
            f"-- step {step['step']:>2} [{'OK' if step['valid'] else 'INVALID'}] "
            f"action={step['action']!r} valid_action={step.get('valid_action', False)} "
            f"strict_valid_action={step.get('strict_valid_action', step['valid'])} "
            f"source={step.get('execution_source', 'unknown')} reward={step.get('reward', 0)} "
            f"task_score={step.get('task_score', 0)} won={step.get('won', False)}",
            "   obs: " + " ".join(str(step.get("obs", "")).split())[:1200],
            "   raw_model_output: " + repr(step.get("raw_model_output", ""))[:4000],
        ])
    with open(os.path.join(directory, base + "_trajectory.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    prompts = [
        "#" * 78,
        f"# Episode #{episode_idx} WebShop prompts",
        f"# task: {result['raw_trace']['task']} | {status} score={result['task_score']:.4f}",
        "#" * 78,
    ]
    for step in result.get("logrows") or []:
        prompts.extend([
            "", "/" * 78,
            f"// step {step['step']} action={step['action']!r} "
            f"valid_action={step.get('valid_action', False)} "
            f"strict_valid_action={step.get('strict_valid_action', step['valid'])} "
            f"source={step.get('execution_source', 'unknown')}",
            "/" * 78, step.get("prompt", ""),
            "\n// raw_model_output\n" + str(step.get("raw_model_output", "")),
        ])
    with open(os.path.join(directory, base + "_prompts.txt"), "w") as handle:
        handle.write("\n".join(prompts) + "\n")


def _tag_block(text, tag):
    match = re.search(
        rf"<{tag}>.*?</{tag}>", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    return match.group(0) if match else ""


def _dump_thinking_samples(outdir, group_id, ingested, max_samples):
    """Save a few complete, human-readable thought trajectories.

    This is observation-only: the selected model outputs were already sampled
    and executed.  It neither changes the prompt nor enters TracesPool, reward,
    cloud updates, or the skill library.
    """
    if max_samples <= 0:
        return
    jsonl_path = os.path.join(outdir, "thinking_samples.jsonl")
    text_path = os.path.join(outdir, "thinking_samples.txt")
    saved = 0
    for row, result in ingested:
        if saved >= max_samples:
            break
        steps = result.get("raw_trace", {}).get("steps", [])
        step_records = []
        for step in steps:
            raw_output = str(step.get("raw_model_output", ""))
            step_records.append({
                "step": step.get("step"),
                "observation": step.get("observation", ""),
                "strict_valid_action": bool(step.get("strict_valid_action", False)),
                "valid_action": bool(step.get("valid_action", False)),
                "execution_source": step.get("execution_source"),
                "executed_action": step.get("action"),
                "reward": float(step.get("reward", 0.0) or 0.0),
                "task_score": float(step.get("task_score", 0.0) or 0.0),
                "think": _tag_block(raw_output, "think"),
                "action_block": _tag_block(raw_output, "action"),
                "raw_model_output": raw_output,
            })
        if not any(step["raw_model_output"] for step in step_records):
            continue
        payload = {
            "group": group_id,
            "episode": row["step"],
            "goal_index": row.get("goal_index"),
            "task_type": row["detected_type"],
            "task": result["raw_trace"].get("task"),
            "won": bool(result["won"]),
            "task_score": float(result["task_score"]),
            "steps": step_records,
        }
        _append_jsonl(jsonl_path, payload)
        with open(text_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{'=' * 78}\n"
                f"group={group_id} episode={row['step']} goal={row.get('goal_index')} "
                f"type={row['detected_type']} won={bool(result['won'])} "
                f"score={float(result['task_score']):.4f}\n"
                f"task: {payload['task']}\n"
            )
            for step in step_records:
                handle.write(
                    f"\n-- step {step['step']} action={step['executed_action']!r} "
                    f"strict_valid={step['strict_valid_action']} "
                    f"valid={step['valid_action']} source={step['execution_source']} "
                    f"reward={step['reward']:.4f} task_score={step['task_score']:.4f}\n"
                    f"[OBSERVATION]\n{step['observation']}\n"
                    f"[THINK]\n{step['think'] or '(no complete <think> block)'}\n"
                    f"[ACTION]\n{step['action_block'] or '(no complete <action> block)'}\n"
                )
        saved += 1
    if saved:
        print(f"[webshop-driver] saved {saved} thought samples to {text_path}", flush=True)


def rollout_webshop_group(env, agent, skill_lib, args, group_id, worker_tag="",
                          goal_indices=None, sampling_temperature=None,
                          sampling_seed=None):
    """Roll out one WebShop batch without mutating the supplied skill library.

    Training calls use the historical sampler.  Validation passes explicit
    held-out goal indices plus request-local sampling parameters, so it cannot
    enter TracesPool/cloud updates or perturb the training rollout RNG stream.
    """
    observations, infos = env.reset(group_id=group_id, goal_indices=goal_indices)
    batch_size = len(observations)
    tasks = [extract_webshop_task(observation) for observation in observations]
    formatted = [
        format_webshop_observation(observation, task)
        for observation, task in zip(observations, tasks)
    ]
    builders = [
        WebShopObsBuilder(
            mem_lib=skill_lib,
            history_length=args.history_length,
            with_skills=bool(args.enable_coskill),
            top_k=args.top_k,
            enable_skill_tree=bool(args.enable_skill_tree),
            prompt_char_limit=args.prompt_char_limit,
        ) for _ in range(batch_size)
    ]
    categories, injected_ids, tree_records = [], [], []
    for builder, task in zip(builders, tasks):
        builder.reset(task)
        category = (builder.retrieved or {}).get("task_type", "unknown")
        categories.append(category)
        injected_ids.append((builder.retrieved or {}).get("injected_skill_ids", []) or [])
        record = skill_lib.get_playbook_record(category) if bool(args.enable_skill_tree) else None
        tree_records.append(record)

    category_counts = {key: categories.count(key) for key in sorted(set(categories))}
    print(f"[webshop-batch]{worker_tag} group={group_id} size={batch_size} "
          f"categories={category_counts}")
    steps = [[] for _ in range(batch_size)]
    logrows = [[] for _ in range(batch_size)]
    won = [False] * batch_size
    done = [False] * batch_size
    used = [0] * batch_size
    valid_count = [0] * batch_size
    relaxed_valid_count = [0] * batch_size
    task_scores = [0.0] * batch_size
    last_action = [None] * batch_size
    repeat = [0] * batch_size
    goal_indices = [info.get("goal_index") for info in infos]

    for step_index in range(1, args.max_steps + 1):
        active = [i for i in range(batch_size) if not done[i]]
        if not active:
            break
        started = time.time()
        prompts = [
            builders[i].build(
                formatted[i], infos[i]["available_actions"], init=(step_index == 1))
            for i in active
        ]
        generated = agent.act_batch_with_meta(
            prompts,
            temperature=sampling_temperature,
            sampling_seed=sampling_seed,
        )
        raw_outputs = [item[0] for item in generated]
        forced = [bool(item[1]) for item in generated]
        actions, valid_flags, action_details = webshop_projection(
            list(raw_outputs), return_details=True)

        action_by_index = dict(zip(active, actions))
        valid_by_index = {i: bool(flag) for i, flag in zip(active, valid_flags)}
        action_detail_by_index = dict(zip(active, action_details))
        forced_by_index = dict(zip(active, forced))
        raw_output_by_index = dict(zip(active, raw_outputs))
        full_actions = [action_by_index.get(i, "click[__inactive__]")
                        for i in range(batch_size)]
        next_observations, rewards, dones, next_infos = env.step(full_actions)

        for i in active:
            action = action_by_index[i]
            valid = valid_by_index[i]
            action_detail = action_detail_by_index[i]
            if valid:
                valid_count[i] += 1
            if action_detail["valid_action"]:
                relaxed_valid_count[i] += 1
            raw_score = float(next_infos[i].get("task_score", 0.0) or 0.0)
            slot_won = bool(next_infos[i].get("won", False))
            slot_done = bool(dones[i])
            task_scores[i] = raw_score if slot_done else task_scores[i]
            trace_observation = webshop_trace_observation(formatted[i])
            steps[i].append({
                "step": step_index,
                "observation": trace_observation,
                "raw_model_output": raw_output_by_index[i],
                "action": action,
                "reward": float(rewards[i] or 0.0),
                "task_score": raw_score,
                "valid_action": action_detail["valid_action"],
                "strict_valid_action": valid,
                "execution_source": action_detail["execution_source"],
            })
            if bool(args.log_trajectories):
                logrows[i].append({
                    "step": step_index,
                    "prompt": prompts[active.index(i)],
                    "raw_model_output": raw_output_by_index[i],
                    "action": action,
                    "valid": valid,
                    "valid_action": action_detail["valid_action"],
                    "strict_valid_action": valid,
                    "execution_source": action_detail["execution_source"],
                    "forced": forced_by_index[i],
                    "obs": next_observations[i],
                    "reward": float(rewards[i] or 0.0),
                    "task_score": raw_score,
                    "won": slot_won,
                })
            builders[i].record(formatted[i], action)
            used[i] = step_index
            won[i] = slot_won
            repeat[i] = repeat[i] + 1 if action == last_action[i] else 0
            last_action[i] = action
            if slot_done or repeat[i] >= args.repeat_stop_threshold:
                done[i] = True

        observations = next_observations
        formatted = [
            format_webshop_observation(observation, task)
            for observation, task in zip(observations, tasks)
        ]
        infos = next_infos
        print(f"  [webshop-batch]{worker_tag} group={group_id} step={step_index}/{args.max_steps} "
              f"active={len(active)} done={sum(done)}/{batch_size} "
              f"won={sum(won)}/{batch_size} ({time.time() - started:.1f}s)")

    results = []
    for i in range(batch_size):
        raw_trace = {
            "traj_uid": str(uuid.uuid4()),
            "task": tasks[i],
            "task_type": categories[i],
            "outcome": "success" if won[i] else "failure",
            "episode_reward": 10.0 if won[i] else 0.0,
            "steps": steps[i],
            "meta": {
                "environment": "WebShop",
                "skill_ids_used": injected_ids[i],
                "model_version": "frozen",
                "task_score": task_scores[i],
                "goal_index": goal_indices[i],
                "n_valid_actions": relaxed_valid_count[i],
                "valid_action_ratio": relaxed_valid_count[i] / max(used[i], 1),
                "n_strict_valid_actions": valid_count[i],
                "strict_valid_action_ratio": valid_count[i] / max(used[i], 1),
            },
        }
        results.append({
            "won": won[i],
            "used": used[i] or args.max_steps,
            "task_score": task_scores[i],
            "goal_index": goal_indices[i],
            "raw_trace": raw_trace,
            "task_type": categories[i],
            "injected": injected_ids[i],
            "n_valid": valid_count[i],
            "n_relaxed_valid": relaxed_valid_count[i],
            "logrows": logrows[i],
            "playbook_record": tree_records[i],
        })
    return results


def _rollout_worker(worker_id, gpu_group, args_dict, base_task_count,
                    base_task_offset, val_base_task_count, val_base_task_offset,
                    input_queue, output_queue):
    group_id = None
    try:
        # ``spawn`` copied this worker's disjoint GPU mask from the parent at
        # ``start()``.  It may name one GPU (DP) or a TP/PP GPU group.
        # Do not rewrite it here: a second assignment can be interpreted in a
        # different CUDA visibility namespace by vLLM's EngineCore descendants
        # and remap worker 1 back onto GPU 0.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices != str(gpu_group):
            raise RuntimeError(
                f"worker {worker_id} expected CUDA_VISIBLE_DEVICES={gpu_group!r}, "
                f"got {visible_devices!r}; refusing unsafe vLLM launch")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        args = argparse.Namespace(**args_dict)
        env = LocalBatchWebShopEnv(
            seed=args.seed,
            base_task_count=base_task_count,
            group_size=args.group_size,
            base_task_offset=base_task_offset,
            total_base_tasks=args.train_data_size,
            file_path=args.webshop_file_path,
            attr_path=args.webshop_attr_path,
        )
        val_env = LocalBatchWebShopEnv(
            seed=args.seed + 1000,
            base_task_count=val_base_task_count,
            group_size=1,
            base_task_offset=val_base_task_offset,
            total_base_tasks=args.val_data_size,
            file_path=args.webshop_file_path,
            attr_path=args.webshop_attr_path,
        )
        from mini_test_pen_shelf.agent_vllm import VLLMAgent
        validation_enabled = bool(
            args.validation_before_train or args.validation_every_groups > 0)
        required_max_num_seqs = max(
            env.batch_size, val_env.batch_size if validation_enabled else 0)
        vllm_max_num_seqs = int(args.vllm_max_num_seqs or required_max_num_seqs)
        if vllm_max_num_seqs < required_max_num_seqs:
            raise ValueError(
                f"vllm_max_num_seqs={vllm_max_num_seqs} is smaller than "
                f"this worker's largest active batch={required_max_num_seqs}")
        agent = VLLMAgent(
            model_path=args.model_path,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=not args.no_thinking,
            seed=args.seed + worker_id,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            max_num_seqs=vllm_max_num_seqs,
            enforce_eager=bool(args.vllm_enforce_eager),
            no_wait=args.nowait,
            think_budget=args.think_budget,
            action_budget=args.action_budget,
        )
        print(f"[webshop-worker{worker_id}] ready gpu_group={gpu_group} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"TP={args.tensor_parallel_size} PP={args.pipeline_parallel_size} "
              f"max_num_seqs={vllm_max_num_seqs} "
              f"enforce_eager={bool(args.vllm_enforce_eager)} "
              f"train_base_tasks={base_task_count} train_batch={env.batch_size} "
              f"val_base_tasks={val_base_task_count} val_batch={val_env.batch_size} "
              f"goals={env.num_goals}")
        while True:
            command = input_queue.get()
            if command is None:
                env.close()
                val_env.close()
                return
            group_id = int(command["group_id"])
            phase = str(command.get("phase", "train"))
            if phase not in {"train", "validation"}:
                raise ValueError(f"Unknown WebShop rollout phase: {phase}")
            tokens_before = agent.get_token_usage()
            skill_lib = HierarchicalSkillLib(
                skills_json_path=command["skill_path"],
                retrieval_mode=args.retrieval_mode,
                embedding_model_path=args.embedding_model_path,
                enable_hierarchy=bool(args.enable_hierarchy),
                stable_cycles_l1=args.stable_cycles_l1,
                stable_cycles_l2=args.stable_cycles_l2,
                success_l1=args.success_l1,
                demote_threshold=args.demote_threshold,
                min_calls=args.min_calls,
                enable_playbook=bool(args.enable_skill_tree),
            )
            results = rollout_webshop_group(
                val_env if phase == "validation" else env,
                agent,
                skill_lib,
                args,
                group_id,
                worker_tag=(f" worker{worker_id} gpu_group={gpu_group} "
                            f"phase={phase}"),
                goal_indices=command.get("goal_indices"),
                sampling_temperature=(args.validation_temperature
                                      if phase == "validation" else None),
                sampling_seed=(int(args.validation_seed) + group_id
                               if phase == "validation" else None),
            )
            output_queue.put({
                "worker_id": worker_id,
                "group_id": group_id,
                "phase": phase,
                "results": results,
                "small_model_tokens": _token_delta(agent.get_token_usage(), tokens_before),
                "error": None,
            })
    except Exception:
        output_queue.put({
            "worker_id": worker_id, "group_id": group_id, "results": [],
            "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
            "error": traceback.format_exc(),
        })


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_size", type=int, default=12)
    parser.add_argument("--val_data_size", type=int, default=32,
                        help="Number of fixed held-out WebShop goals from split [0, 500)")
    parser.add_argument("--validation_every_groups", type=int, default=5,
                        help="Run held-out validation every N train groups; 0 disables it")
    parser.add_argument("--validation_before_train", type=int, choices=[0, 1], default=1,
                        help="Run a held-out validation pass at group 0")
    parser.add_argument("--validation_temperature", type=float, default=0.4,
                        help="Sampling temperature for held-out validation only")
    parser.add_argument("--validation_seed", type=int, default=1000,
                        help="Request-local seed for held-out validation sampling")
    parser.add_argument("--group_size", type=int, default=6)
    parser.add_argument("--total_groups", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat_stop_threshold", type=int, default=6)
    parser.add_argument("--webshop_file_path", required=True)
    parser.add_argument("--webshop_attr_path", required=True)
    parser.add_argument("--data_parallel_workers", type=int, default=2)
    parser.add_argument("--rollout_worker_gpus", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0,
                        help="0 uses this worker's rollout batch size; a positive override must be no smaller")
    parser.add_argument("--vllm_enforce_eager", type=int, choices=[0, 1], default=1,
                        help="0 enables vLLM CUDA Graphs after warm-up; 1 is eager-only")
    parser.add_argument("--checkpoint_every_groups", type=int, default=2)
    parser.add_argument("--cloud_update_every", type=int, default=0)
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument("--prompt_char_limit", type=int, default=13000)

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu_mem_util", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=6768)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--think_budget", type=int, default=640)
    parser.add_argument("--action_budget", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--nowait", action="store_true")

    parser.add_argument("--skills_json", default="memory_data/webshop/claude_style_skills.json")
    parser.add_argument("--retrieval_mode", default="template", choices=["template", "embedding"])
    parser.add_argument("--embedding_model_path", default=None)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--enable_hierarchy", type=int, default=1)
    parser.add_argument("--stable_cycles_l1", type=int, default=3)
    parser.add_argument("--stable_cycles_l2", type=int, default=5)
    parser.add_argument("--success_l1", type=float, default=0.7)
    parser.add_argument("--demote_threshold", type=float, default=0.3)
    parser.add_argument("--min_calls", type=int, default=10)
    parser.add_argument("--enable_coskill", type=int, default=1)
    parser.add_argument("--enable_skill_tree", type=int, default=1)
    parser.add_argument("--enable_skill_tree_evolve", type=int, default=1)
    parser.add_argument("--enable_failure_analysis", type=int, default=1)
    parser.add_argument("--max_new_skills", type=int, default=3)
    parser.add_argument("--skill_tree_evolve_min_samples", type=int, default=6)
    parser.add_argument("--capacity_watermark", type=int, default=50000)
    parser.add_argument("--perf_watermark", type=float, default=0.6)
    parser.add_argument("--min_samples", type=int, default=16)
    parser.add_argument("--loop_threshold", type=int, default=3)
    parser.add_argument("--coskill_debug", type=int, default=0)
    parser.add_argument("--log_trajectories", type=int, default=0)
    parser.add_argument("--think_trace_samples_per_group", type=int, default=0,
                        help="Save this many complete think/action episodes at audit groups; 0 disables")
    parser.add_argument("--think_trace_every_groups", type=int, default=10,
                        help="Audit group interval for think samples; group 1 is always included")
    parser.add_argument("--resume", type=int, choices=[0, 1], default=0,
                        help="1 resumes from summary_partial.json and its skill checkpoint in --outdir")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    run_lock = _acquire_run_lock(args.outdir)
    atexit.register(run_lock.close)
    resume_state = _load_resume_state(args)
    for path in (args.webshop_file_path, args.webshop_attr_path, args.skills_json):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    webshop_data_dir = os.path.dirname(os.path.realpath(args.webshop_file_path))
    required_assets = [
        os.path.join(webshop_data_dir, "items_human_ins.json"),
        os.path.join(os.path.dirname(webshop_data_dir), "search_engine", "indexes"),
    ]
    for path in required_assets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required WebShop asset not found: {path}")
    if args.train_data_size < args.data_parallel_workers:
        raise ValueError("train_data_size must be >= data_parallel_workers")
    if args.val_data_size <= 0 or args.val_data_size > 500:
        raise ValueError("val_data_size must be in [1, 500] for WebShop's held-out split")
    if args.val_data_size < args.data_parallel_workers:
        raise ValueError("val_data_size must be >= data_parallel_workers")
    if args.validation_every_groups < 0:
        raise ValueError("validation_every_groups must be >= 0")
    if args.think_budget + args.action_budget > args.max_tokens:
        raise ValueError("think_budget + action_budget must be <= max_tokens for WebShop alignment")
    if args.think_trace_samples_per_group < 0:
        raise ValueError("think_trace_samples_per_group must be >= 0")
    if args.think_trace_every_groups < 1:
        raise ValueError("think_trace_every_groups must be >= 1")

    enable_tree = bool(args.enable_skill_tree)
    enable_tree_evolve = enable_tree and bool(args.enable_skill_tree_evolve)
    batch_rollout_size = args.train_data_size * args.group_size
    episode_cap = args.max_episodes if args.max_episodes > 0 else (
        args.total_groups * batch_rollout_size)
    if resume_state["completed_groups"] > args.total_groups:
        raise ValueError(
            f"Resume checkpoint already completed {resume_state['completed_groups']} groups, "
            f"but total_groups={args.total_groups}. Increase TOTAL_GROUPS or start a new run."
        )
    if len(resume_state["per_game"]) > episode_cap:
        raise ValueError(
            f"Resume checkpoint already contains {len(resume_state['per_game'])} episodes, "
            f"but max_episodes={episode_cap}. Increase MAX_EPISODES or start a new run."
        )
    print(f"[webshop-driver] groups={args.total_groups} train_data_size={args.train_data_size} "
          f"group_size={args.group_size} batch={batch_rollout_size} "
          f"max_episodes={episode_cap} max_steps={args.max_steps} "
          f"validation={args.val_data_size} heldout goals every "
          f"{args.validation_every_groups or 'disabled'} groups")

    skill_lib = HierarchicalSkillLib(
        skills_json_path=resume_state["skills_json_path"],
        retrieval_mode=args.retrieval_mode,
        embedding_model_path=args.embedding_model_path,
        enable_hierarchy=bool(args.enable_hierarchy),
        stable_cycles_l1=args.stable_cycles_l1,
        stable_cycles_l2=args.stable_cycles_l2,
        success_l1=args.success_l1,
        demote_threshold=args.demote_threshold,
        min_calls=args.min_calls,
        enable_playbook=enable_tree,
    )
    traces_pool = TracesPool(
        capacity_watermark=args.capacity_watermark,
        perf_watermark=args.perf_watermark,
        min_samples=args.min_samples,
        loop_threshold=args.loop_threshold,
        output_dir=args.outdir,
    )
    _rehydrate_traces_pool(traces_pool, args.outdir, resume_state)
    cloud_loop = CoSkillCloudLoop(
        output_dir=args.outdir,
        enable_coskill=bool(args.enable_coskill),
        enable_playbook_evolve=enable_tree_evolve,
        enable_failure_analysis=bool(args.enable_failure_analysis),
        max_new_skills=args.max_new_skills,
        playbook_evolve_min_samples=args.skill_tree_evolve_min_samples,
        coskill_debug=bool(args.coskill_debug),
        environment_name="WebShop",
    )
    analyzer = getattr(cloud_loop, "cloud_analyzer", None)
    if analyzer is not None and resume_state["resume"]:
        analyzer.total_prompt_tokens = resume_state["large_cumulative"]["prompt"]
        analyzer.total_completion_tokens = resume_state["large_cumulative"]["completion"]

    worker_gpu_groups = _resolve_worker_gpu_groups(args)
    base_splits = _split_int(args.train_data_size, args.data_parallel_workers)
    val_base_splits = _split_int(args.val_data_size, args.data_parallel_workers)
    validation_goal_indices = np.random.RandomState(args.validation_seed).choice(
        np.arange(500), size=args.val_data_size, replace=False).tolist()
    validation_goal_shards = []
    val_offset = 0
    for count in val_base_splits:
        validation_goal_shards.append(validation_goal_indices[val_offset:val_offset + count])
        val_offset += count
    validation_enabled = bool(
        args.validation_before_train or args.validation_every_groups > 0)
    vllm_max_num_seqs_by_worker = [
        int(args.vllm_max_num_seqs or max(
            train_count * args.group_size,
            val_count if validation_enabled else 0,
        ))
        for train_count, val_count in zip(base_splits, val_base_splits)
    ]

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    input_queues, processes = [], []
    offset = 0
    for worker_id, (gpu_group, base_count, val_base_count) in enumerate(
            zip(worker_gpu_groups, base_splits, val_base_splits)):
        input_queue = context.Queue(maxsize=2)
        process = context.Process(
            target=_rollout_worker,
            args=(worker_id, gpu_group, vars(args).copy(), base_count, offset,
                  val_base_count, sum(val_base_splits[:worker_id]),
                  input_queue, output_queue),
            daemon=False,
        )
        # ``spawn`` copies the parent's environment at ``start()`` time.  The
        # old implementation changed CUDA_VISIBLE_DEVICES only inside the
        # target function.  That was too late for vLLM EngineCore descendants
        # on some cluster launches, so both replicas could see and fill GPU 0.
        # Snapshot this worker's disjoint GPU mask before spawning, then
        # restore the driver's original multi-GPU mask immediately afterwards.
        parent_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_group)
        try:
            process.start()
        finally:
            if parent_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_visible
        input_queues.append(input_queue)
        processes.append(process)
        offset += base_count
    print(f"[webshop-driver] data_parallel_workers={args.data_parallel_workers} "
          f"gpu_groups={worker_gpu_groups} TP={args.tensor_parallel_size} "
          f"PP={args.pipeline_parallel_size} base_task_splits={base_splits} "
          f"worker_batches={[count * args.group_size for count in base_splits]} "
          f"val_task_splits={val_base_splits} fixed_val_goals={args.val_data_size}")

    def shutdown():
        for queue in input_queues:
            try:
                queue.put(None)
            except Exception:
                pass
        for process in processes:
            try:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass

    atexit.register(shutdown)
    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    group_metrics_path = os.path.join(args.outdir, "group_metrics.jsonl")
    validation_metrics_path = os.path.join(args.outdir, "validation_metrics.jsonl")
    per_game = list(resume_state["per_game"])
    category_stats = dict(resume_state["category_stats"])
    wins = resume_state["wins"]
    score_sum = resume_state["score_sum"]
    global_episode = len(per_game)
    cloud_update_steps = list(resume_state["cloud_update_steps"])
    cloud_updates = len(cloud_update_steps)
    small_cumulative = dict(resume_state["small_cumulative"])
    validation_history = list(resume_state["validation_history"])
    validation_small_cumulative = dict(resume_state["validation_small_cumulative"])

    def large_tokens():
        analyzer = getattr(cloud_loop, "cloud_analyzer", None)
        if analyzer is None:
            return {"prompt": 0, "completion": 0, "total": 0}
        prompt = int(getattr(analyzer, "total_prompt_tokens", 0) or 0)
        completion = int(getattr(analyzer, "total_completion_tokens", 0) or 0)
        return {"prompt": prompt, "completion": completion,
                "total": prompt + completion}

    def tree_snapshot():
        snapshot = {}
        for category, record in (getattr(skill_lib, "task_playbooks", {}) or {}).items():
            if isinstance(record, dict):
                snapshot[category] = {
                    "version": record.get("version", 0),
                    "level": record.get("level"),
                    "n_nodes": len(record.get("nodes") or {}),
                }
        return snapshot

    def run_validation(validation_group, *, global_episode):
        """Evaluate the current frozen policy/skill snapshot on held-out goals.

        This function never calls ``record_usage``, ``TracesPool.add_trace``,
        or the cloud loop.  It is therefore reporting-only and cannot leak
        test trajectories into future skill updates.
        """
        validation_started = time.time()
        skill_dir = os.path.join(args.outdir, "skill_lib")
        os.makedirs(skill_dir, exist_ok=True)
        skill_path = os.path.join(
            skill_dir, f"skills_validation_group{validation_group:04d}.json")
        skill_lib.save_skills(skill_path)
        for queue, goal_shard in zip(input_queues, validation_goal_shards):
            queue.put({
                "phase": "validation",
                "group_id": validation_group,
                "skill_path": skill_path,
                "goal_indices": goal_shard,
            })
        replies = []
        for _ in input_queues:
            reply = output_queue.get()
            if reply.get("error"):
                raise RuntimeError(
                    f"WebShop validation worker {reply.get('worker_id')} failed:\n"
                    f"{reply['error']}")
            if reply.get("phase") != "validation":
                raise RuntimeError(f"Expected validation reply, got {reply.get('phase')!r}")
            replies.append(reply)
        replies.sort(key=lambda item: item["worker_id"])
        results = []
        token_delta = {"prompt": 0, "response": 0, "total": 0}
        for reply in replies:
            results.extend(reply["results"])
            for key in token_delta:
                token_delta[key] += int(reply["small_model_tokens"].get(key, 0))
        if len(results) != args.val_data_size:
            raise RuntimeError(
                f"Validation expected {args.val_data_size} episodes, got {len(results)}")
        for key in validation_small_cumulative:
            validation_small_cumulative[key] += token_delta[key]
        metric = _validation_metrics(
            results,
            group_id=validation_group,
            global_episode=global_episode,
            token_delta=token_delta,
            token_cumulative=validation_small_cumulative,
        )
        metric["validation/timing_s/rollout"] = round(time.time() - validation_started, 6)
        metric["validation/temperature"] = float(args.validation_temperature)
        metric["validation/fixed_goal_split"] = "WebShop[0:500)"
        row = {
            "step": int(validation_group),
            "global_episode_end": int(global_episode),
            "metrics": metric,
        }
        validation_history.append(row)
        _append_jsonl(validation_metrics_path, row)
        print(f"[webshop-driver] validation group={validation_group} "
              f"episodes={len(results)} wins={metric['validation/episode/wins']} "
              f"success={100 * metric['validation/episode/success_rate']:.1f}% "
              f"score={metric['validation/episode/task_score/mean']:.3f} "
              f"tokens={token_delta['total']}", flush=True)
        return metric

    def summary(status, completed_groups, reason=None):
        trees = tree_snapshot()
        return {
            "status": status,
            "checkpoint_reason": reason,
            "environment": "WebShop",
            "reference_config": "examples/grpo_trainer/run_webshop_skills.sh",
            "total_groups_target": args.total_groups,
            "completed_rollout_groups": completed_groups,
            "train_data_size": args.train_data_size,
            "val_data_size": args.val_data_size,
            "validation_every_groups": args.validation_every_groups,
            "validation_before_train": bool(args.validation_before_train),
            "validation_temperature": args.validation_temperature,
            "validation_seed": args.validation_seed,
            "validation_goal_split": "WebShop[0:500)",
            "group_size": args.group_size,
            "batch_rollout_size": batch_rollout_size,
            "max_steps": args.max_steps,
            "max_episodes": episode_cap,
            "data_parallel_workers": args.data_parallel_workers,
            "data_parallel_base_task_splits": base_splits,
            "rollout_worker_gpu_groups": worker_gpu_groups,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "vllm_max_num_seqs": args.vllm_max_num_seqs or max(
                vllm_max_num_seqs_by_worker),
            "vllm_max_num_seqs_by_worker": vllm_max_num_seqs_by_worker,
            "vllm_enforce_eager": bool(args.vllm_enforce_eager),
            "checkpoint_every_groups": args.checkpoint_every_groups,
            "skill_tree_enabled": enable_tree,
            "skill_tree_evolve_enabled": enable_tree_evolve,
            "skill_bullets_enabled": bool(args.enable_coskill),
            "cloud_update_every": args.cloud_update_every,
            "cloud_update_steps": cloud_update_steps,
            "total_episodes": len(per_game),
            "wins": wins,
            "success_rate": round(wins / max(len(per_game), 1), 6),
            "mean_task_score": round(score_sum / max(len(per_game), 1), 6),
            "skill_tree_versions": {key: value["version"] for key, value in trees.items()},
            "skill_tree_nodes": {key: value["n_nodes"] for key, value in trees.items()},
            "token_usage": {
                "small_model": {
                    **dict(small_cumulative),
                    "validation": dict(validation_small_cumulative),
                    "including_validation": {
                        key: int(small_cumulative[key]) + int(validation_small_cumulative[key])
                        for key in small_cumulative
                    },
                },
                "large_model": large_tokens(),
            },
            "final_coskill_metrics": cloud_loop.metrics(traces_pool, skill_lib),
            "phase_stats": _phase_stats(per_game),
            "validation_history": validation_history,
            "per_game": per_game,
        }

    def save_checkpoint(completed_groups):
        save_dir = os.path.join(args.outdir, "skill_lib")
        os.makedirs(save_dir, exist_ok=True)
        skill_path = os.path.join(save_dir, f"skills_checkpoint_step{global_episode}.json")
        skill_lib.save_skills(skill_path)
        skill_lib.save_skills(os.path.join(save_dir, "skills_latest_checkpoint.json"))
        payload = summary("running", completed_groups,
                          f"group_interval_{args.checkpoint_every_groups}")
        payload["skill_lib_checkpoint"] = skill_path
        _atomic_json_dump(payload, os.path.join(args.outdir, "summary_partial.json"))
        checkpoint_path = os.path.join(
            args.outdir, "checkpoints", f"step{global_episode:06d}.json")
        _atomic_json_dump(payload, checkpoint_path)
        print(f"[webshop-driver] checkpoint saved groups={completed_groups} "
              f"episodes={global_episode} -> {checkpoint_path}", flush=True)

    completed_groups = resume_state["completed_groups"]
    if (args.validation_before_train and completed_groups == 0
            and not validation_history):
        run_validation(0, global_episode=global_episode)
        # Persist the initial held-out result.  If the job is interrupted
        # before group 1, RESUME=1 retains the same validation timeline.
        save_checkpoint(0)
    try:
        for group_id in range(completed_groups + 1, args.total_groups + 1):
            if global_episode >= episode_cap:
                break
            group_started = time.time()
            skill_dir = os.path.join(args.outdir, "skill_lib")
            os.makedirs(skill_dir, exist_ok=True)
            skill_path = os.path.join(skill_dir, f"skills_rollout_step{global_episode}.json")
            skill_lib.save_skills(skill_path)
            skill_lib.save_skills(os.path.join(skill_dir, "skills_latest_rollout.json"))
            for queue in input_queues:
                queue.put({"group_id": group_id, "skill_path": skill_path})

            replies = []
            for _ in input_queues:
                reply = output_queue.get()
                if reply.get("error"):
                    raise RuntimeError(
                        f"WebShop rollout worker {reply.get('worker_id')} failed:\n{reply['error']}")
                replies.append(reply)
            replies.sort(key=lambda item: item["worker_id"])
            group_results = []
            small_tokens = {"prompt": 0, "response": 0, "total": 0}
            for reply in replies:
                group_results.extend(reply["results"])
                for key in small_tokens:
                    small_tokens[key] += int(reply["small_model_tokens"].get(key, 0))
            rollout_seconds = time.time() - group_started
            remaining = episode_cap - global_episode
            counted_results = group_results[:remaining]

            ingested = []
            for result in counted_results:
                global_episode += 1
                traces_pool.add_trace(result["raw_trace"])
                skill_lib.record_usage(
                    result.get("injected") or [], result["won"], result["task_type"])
                skill_lib.record_playbook_usage(result["task_type"], result["won"])
                wins += int(result["won"])
                score_sum += float(result["task_score"])
                cat = category_stats.setdefault(result["task_type"], {
                    "episodes": 0, "wins": 0, "task_score_sum": 0.0,
                })
                cat["episodes"] += 1
                cat["wins"] += int(result["won"])
                cat["task_score_sum"] += float(result["task_score"])
                row = {
                    "step": global_episode,
                    "group": group_id,
                    "detected_type": result["task_type"],
                    "won": bool(result["won"]),
                    "task_score": float(result["task_score"]),
                    "used_steps": int(result["used"]),
                    "valid_actions": int(result["n_valid"]),
                    "strict_valid_actions": int(result["n_valid"]),
                    "relaxed_valid_actions": int(result["n_relaxed_valid"]),
                    "goal_index": result.get("goal_index"),
                    "cloud_round_used": cloud_updates,
                    "skill_ids_used": list(result.get("injected") or []),
                    "running_total_episodes": len(per_game) + 1,
                    "running_total_wins": wins,
                    "running_task_score_sum": score_sum,
                    "task_type_episodes": cat["episodes"],
                    "task_type_wins": cat["wins"],
                    "task_type_score_sum": cat["task_score_sum"],
                }
                per_game.append(row)
                ingested.append((row, result))
                if bool(args.log_trajectories):
                    _dump_episode(args.outdir, global_episode, result)

            if args.think_trace_samples_per_group and (
                group_id == 1 or group_id % args.think_trace_every_groups == 0
            ):
                _dump_thinking_samples(
                    args.outdir,
                    group_id,
                    ingested,
                    args.think_trace_samples_per_group,
                )

            force_reason = None
            if args.cloud_update_every > 0 and group_id % args.cloud_update_every == 0:
                force_reason = f"group_interval_{args.cloud_update_every}"
            large_before = large_tokens()
            cloud_started = time.time()
            fired = cloud_loop.maybe_update(
                traces_pool, skill_lib, global_episode, force_reason=force_reason)
            cloud_seconds = time.time() - cloud_started
            large_after = large_tokens()
            if fired:
                cloud_updates += 1
                cloud_update_steps.append(global_episode)

            for index, (row, result) in enumerate(ingested):
                category = row["detected_type"]
                cat = category_stats[category]
                tree = result.get("playbook_record")
                _append_jsonl(metrics_path, {
                    "step": row["step"],
                    "metrics": {
                        "training/group": group_id,
                        "episode/detected_type": category,
                        "episode/won": row["won"],
                        "episode/task_score": row["task_score"],
                        "episode/length": row["used_steps"],
                        "episode/valid_action_ratio": round(
                            row["valid_actions"] / max(row["used_steps"], 1), 6),
                        "episode/strict_valid_action_ratio": round(
                            row["strict_valid_actions"] / max(row["used_steps"], 1), 6),
                        "episode/relaxed_valid_action_ratio": round(
                            row["relaxed_valid_actions"] / max(row["used_steps"], 1), 6),
                        "episode/success_rate": round(
                            row["running_total_wins"] /
                            max(row["running_total_episodes"], 1), 6),
                        "episode/mean_task_score": round(
                            row["running_task_score_sum"] /
                            max(row["running_total_episodes"], 1), 6),
                        f"episode/{category}/episodes": row["task_type_episodes"],
                        f"episode/{category}/wins": row["task_type_wins"],
                        f"episode/{category}/success_rate": round(
                            row["task_type_wins"] / max(row["task_type_episodes"], 1), 6),
                        f"episode/{category}/mean_task_score": round(
                            row["task_type_score_sum"] /
                            max(row["task_type_episodes"], 1), 6),
                        "experiment/skill_tree_enabled": int(enable_tree),
                        "experiment/skill_tree_evolve_enabled": int(enable_tree_evolve),
                        "experiment/skill_bullets_enabled": int(bool(args.enable_coskill)),
                        "parallel/data_parallel_workers": args.data_parallel_workers,
                        "parallel/tensor_parallel_size": args.tensor_parallel_size,
                        "parallel/pipeline_parallel_size": args.pipeline_parallel_size,
                        "parallel/vllm_enforce_eager": bool(args.vllm_enforce_eager),
                        "experiment/cloud_round_used": row["cloud_round_used"],
                        "coskill/cloud_update_fired": bool(
                            fired and index == len(ingested) - 1),
                        "skill_tree/version": tree.get("version") if tree else 0,
                        "skill_tree/level": tree.get("level") if tree else None,
                        "skill_tree/n_nodes": len(tree.get("nodes") or {}) if tree else 0,
                        **cloud_loop.metrics(traces_pool, skill_lib),
                    },
                })

            for key in small_cumulative:
                small_cumulative[key] += small_tokens[key]
            large_delta = {
                "prompt": large_after["prompt"] - large_before["prompt"],
                "completion": large_after["completion"] - large_before["completion"],
            }
            large_delta["total"] = large_delta["prompt"] + large_delta["completion"]
            group_rows = [row for row, _ in ingested]
            lengths = [row["used_steps"] for row in group_rows]
            group_wins = sum(int(row["won"]) for row in group_rows)
            group_score = sum(row["task_score"] for row in group_rows)
            action_count = sum(lengths)
            action_count_cumulative = sum(
                int(row.get("used_steps", 0) or 0) for row in per_game)
            group_metric = {
                "training/group": group_id,
                "training/global_step": group_id,
                "rollout/global_episode_end": global_episode,
                "episode/count": len(group_rows),
                "episode/generated_count": len(group_results),
                "episode/wins": group_wins,
                "episode/success_rate": round(group_wins / max(len(group_rows), 1), 6),
                "episode/count_cumulative": global_episode,
                "episode/wins_cumulative": wins,
                "episode/action_count": action_count,
                "episode/action_count_cumulative": action_count_cumulative,
                "episode/task_score/mean": round(group_score / max(len(group_rows), 1), 6),
                "episode/task_score/max": max([row["task_score"] for row in group_rows] or [0]),
                "episode/task_score/min": min([row["task_score"] for row in group_rows] or [0]),
                "episode/length/mean": round(sum(lengths) / max(len(lengths), 1), 6),
                "episode/length/max": max(lengths or [0]),
                "episode/length/min": min(lengths or [0]),
                "episode/valid_action_ratio": round(
                    sum(row["valid_actions"] for row in group_rows) /
                    max(sum(lengths), 1), 6),
                "episode/strict_valid_action_ratio": round(
                    sum(row["strict_valid_actions"] for row in group_rows) /
                    max(sum(lengths), 1), 6),
                "episode/relaxed_valid_action_ratio": round(
                    sum(row["relaxed_valid_actions"] for row in group_rows) /
                    max(sum(lengths), 1), 6),
                "experiment/skill_tree_enabled": int(enable_tree),
                "experiment/skill_tree_evolve_enabled": int(enable_tree_evolve),
                "experiment/skill_bullets_enabled": int(bool(args.enable_coskill)),
                "experiment/rl_enabled": 0,
                "experiment/tree_rl_internalize_enabled": 0,
                "parallel/data_parallel_workers": args.data_parallel_workers,
                "parallel/tensor_parallel_size": args.tensor_parallel_size,
                "parallel/pipeline_parallel_size": args.pipeline_parallel_size,
                "parallel/vllm_enforce_eager": bool(args.vllm_enforce_eager),
                "experiment/cloud_round": cloud_updates,
                "coskill/cloud_update_fired": bool(fired),
                "tokens/small_model/prompt": small_tokens["prompt"],
                "tokens/small_model/response": small_tokens["response"],
                "tokens/small_model/total": small_tokens["total"],
                "tokens/small_model/accounting": "vllm_request_tokens_two_stage",
                "tokens/small_model/prompt_cumulative": small_cumulative["prompt"],
                "tokens/small_model/response_cumulative": small_cumulative["response"],
                "tokens/small_model/total_cumulative": small_cumulative["total"],
                "tokens/large_model/prompt": large_delta["prompt"],
                "tokens/large_model/completion": large_delta["completion"],
                "tokens/large_model/total": large_delta["total"],
                "tokens/large_model/accounting": "provider_api_usage",
                "tokens/large_model/prompt_cumulative": large_after["prompt"],
                "tokens/large_model/completion_cumulative": large_after["completion"],
                "tokens/large_model/total_cumulative": large_after["total"],
                "timing_s/rollout": round(rollout_seconds, 6),
                "timing_s/cloud_update": round(cloud_seconds, 6),
                "timing_s/group_total": round(time.time() - group_started, 6),
                "perf/throughput_episodes_per_second": round(
                    len(group_rows) / max(rollout_seconds, 1e-9), 6),
                "perf/throughput_small_tokens_per_second": round(
                    small_tokens["total"] / max(rollout_seconds, 1e-9), 6),
                "perf/total_num_tokens": small_tokens["total"],
                # The comparison schema is embedded in the primary group log
                # (rather than mirrored into a second comparison_metrics file).
                "comparison/schema_version": 1,
                "comparison/method": "coskill",
                "comparison/benchmark": "webshop",
                "comparison/rollout_accounting": "active_env_decisions",
                "comparison/timing_cloud_update_measured": 1,
                **cloud_loop.metrics(traces_pool, skill_lib),
            }
            group_categories = {}
            for row in group_rows:
                stat = group_categories.setdefault(row["detected_type"], {
                    "episodes": 0, "wins": 0, "score": 0.0,
                })
                stat["episodes"] += 1
                stat["wins"] += int(row["won"])
                stat["score"] += row["task_score"]
            for category, stat in sorted(group_categories.items()):
                group_metric[f"episode/{category}/episodes"] = stat["episodes"]
                group_metric[f"episode/{category}/wins"] = stat["wins"]
                group_metric[f"episode/{category}/success_rate"] = round(
                    stat["wins"] / max(stat["episodes"], 1), 6)
                group_metric[f"episode/{category}/mean_task_score"] = round(
                    stat["score"] / max(stat["episodes"], 1), 6)
            validation_metric = None
            if (args.validation_every_groups > 0
                    and group_id % args.validation_every_groups == 0):
                validation_metric = run_validation(group_id, global_episode=global_episode)
                group_metric.update(validation_metric)
                group_metric["validation/ran"] = 1
            else:
                group_metric["validation/ran"] = 0
            group_metric["tokens/small_model/total_including_validation"] = (
                int(group_metric["tokens/small_model/total"])
                + int((validation_metric or {}).get(
                    "validation/tokens/small_model/total", 0)))
            _append_jsonl(group_metrics_path, {
                "step": group_id,
                "global_episode_end": global_episode,
                "metrics": group_metric,
            })
            completed_groups = group_id
            print(f"[webshop-driver] group{group_id} episodes={len(group_rows)} "
                  f"wins={group_wins} success={100 * group_wins / max(len(group_rows), 1):.1f}% "
                  f"mean_score={group_score / max(len(group_rows), 1):.3f} "
                  f"small_tokens={small_tokens['total']} large_tokens={large_delta['total']} "
                  f"rollout={rollout_seconds:.1f}s")
            if (args.checkpoint_every_groups > 0
                    and completed_groups % args.checkpoint_every_groups == 0):
                save_checkpoint(completed_groups)
    finally:
        shutdown()

    final = summary("done", completed_groups)
    save_dir = os.path.join(args.outdir, "skill_lib")
    final_skill_path = os.path.join(save_dir, f"skills_final_step{global_episode}.json")
    skill_lib.save_skills(final_skill_path)
    skill_lib.save_skills(os.path.join(save_dir, "skills_latest_final.json"))
    final["skill_lib_checkpoint"] = final_skill_path
    _atomic_json_dump(final, os.path.join(args.outdir, "summary.json"))
    print(f"[webshop-driver] done groups={completed_groups} episodes={len(per_game)} "
          f"success={100 * wins / max(len(per_game), 1):.1f}% "
          f"mean_task_score={score_sum / max(len(per_game), 1):.3f}")
    print(f"[webshop-driver] outputs under {args.outdir}")


if __name__ == "__main__":
    main()
