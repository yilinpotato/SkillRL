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
import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
import uuid

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
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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


def rollout_webshop_group(env, agent, skill_lib, args, group_id, worker_tag=""):
    observations, infos = env.reset(group_id=group_id)
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
        generated = agent.act_batch_with_meta(prompts)
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
                    base_task_offset, input_queue, output_queue):
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
        from mini_test_pen_shelf.agent_vllm import VLLMAgent
        vllm_max_num_seqs = int(args.vllm_max_num_seqs or env.batch_size)
        if vllm_max_num_seqs < env.batch_size:
            raise ValueError(
                f"vllm_max_num_seqs={vllm_max_num_seqs} is smaller than "
                f"this worker's rollout batch={env.batch_size}")
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
            no_wait=args.nowait,
            think_budget=args.think_budget,
            action_budget=args.action_budget,
        )
        print(f"[webshop-worker{worker_id}] ready gpu_group={gpu_group} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"TP={args.tensor_parallel_size} PP={args.pipeline_parallel_size} "
              f"max_num_seqs={vllm_max_num_seqs} "
              f"base_tasks={base_task_count} batch={env.batch_size} goals={env.num_goals}")
        while True:
            command = input_queue.get()
            if command is None:
                env.close()
                return
            group_id = int(command["group_id"])
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
                env, agent, skill_lib, args, group_id,
                worker_tag=f" worker{worker_id} gpu_group={gpu_group}",
            )
            output_queue.put({
                "worker_id": worker_id,
                "group_id": group_id,
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
                        help="Recorded comparison setting; no-RL driver, like ALFWorld, runs train rollout only")
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
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)
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
    if args.think_budget + args.action_budget > args.max_tokens:
        raise ValueError("think_budget + action_budget must be <= max_tokens for WebShop alignment")

    enable_tree = bool(args.enable_skill_tree)
    enable_tree_evolve = enable_tree and bool(args.enable_skill_tree_evolve)
    batch_rollout_size = args.train_data_size * args.group_size
    episode_cap = args.max_episodes if args.max_episodes > 0 else (
        args.total_groups * batch_rollout_size)
    print(f"[webshop-driver] groups={args.total_groups} train_data_size={args.train_data_size} "
          f"group_size={args.group_size} batch={batch_rollout_size} "
          f"max_episodes={episode_cap} max_steps={args.max_steps}")

    skill_lib = HierarchicalSkillLib(
        skills_json_path=args.skills_json,
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

    worker_gpu_groups = _resolve_worker_gpu_groups(args)
    base_splits = _split_int(args.train_data_size, args.data_parallel_workers)

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    input_queues, processes = [], []
    offset = 0
    for worker_id, (gpu_group, base_count) in enumerate(zip(worker_gpu_groups, base_splits)):
        input_queue = context.Queue(maxsize=2)
        process = context.Process(
            target=_rollout_worker,
            args=(worker_id, gpu_group, vars(args).copy(), base_count, offset,
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
          f"worker_batches={[count * args.group_size for count in base_splits]}")

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
    per_game, category_stats = [], {}
    wins = 0
    score_sum = 0.0
    global_episode = 0
    cloud_updates = 0
    cloud_update_steps = []
    small_cumulative = {"prompt": 0, "response": 0, "total": 0}

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
                count * args.group_size for count in base_splits),
            "vllm_max_num_seqs_by_worker": [
                args.vllm_max_num_seqs or count * args.group_size
                for count in base_splits
            ],
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
                "small_model": dict(small_cumulative),
                "large_model": large_tokens(),
            },
            "final_coskill_metrics": cloud_loop.metrics(traces_pool, skill_lib),
            "phase_stats": _phase_stats(per_game),
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
        _atomic_json_dump(payload, os.path.join(
            args.outdir, "checkpoints", f"step{global_episode:06d}.json"))

    completed_groups = 0
    try:
        for group_id in range(1, args.total_groups + 1):
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
            group_metric = {
                "training/group": group_id,
                "training/global_step": group_id,
                "rollout/global_episode_end": global_episode,
                "episode/count": len(group_rows),
                "episode/generated_count": len(group_results),
                "episode/wins": group_wins,
                "episode/success_rate": round(group_wins / max(len(group_rows), 1), 6),
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
                "parallel/data_parallel_workers": args.data_parallel_workers,
                "parallel/tensor_parallel_size": args.tensor_parallel_size,
                "parallel/pipeline_parallel_size": args.pipeline_parallel_size,
                "experiment/cloud_round": cloud_updates,
                "coskill/cloud_update_fired": bool(fired),
                "tokens/small_model/prompt": small_tokens["prompt"],
                "tokens/small_model/response": small_tokens["response"],
                "tokens/small_model/total": small_tokens["total"],
                "tokens/small_model/prompt_cumulative": small_cumulative["prompt"],
                "tokens/small_model/response_cumulative": small_cumulative["response"],
                "tokens/small_model/total_cumulative": small_cumulative["total"],
                "tokens/large_model/prompt": large_delta["prompt"],
                "tokens/large_model/completion": large_delta["completion"],
                "tokens/large_model/total": large_delta["total"],
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
