""

import argparse
import atexit
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
import uuid

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.frozen_executor.alfworld_prompt import AlfWorldObsBuilder
from agent_system.frozen_executor.alfworld_runtime import (
    _TASK_TYPE_TO_ID,
    extract_task,
    find_games_by_type,
    load_tw_config_types,
    make_batch_env,
    make_single_env,
)
from agent_system.memory import CoSkillCloudLoop, HierarchicalSkillLib, TracesPool
from agent_system.task_taxonomy import (
    ALFWORLD_DATASET_TO_TASK_TYPE,
    canonicalize_alfworld_task_type,
    task_types_for_benchmark,
)

SMALL_MODEL_TOKEN_ACCOUNTING = "vllm_request_tokens_single_pass"








TASK_TYPE_TO_RUNTIME = ALFWORLD_DATASET_TO_TASK_TYPE
NORL_TASK_TYPES = task_types_for_benchmark("alfworld")


def _trim_uncheckpointed_group_metrics(path, keep_lines):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) < keep_lines:
        raise RuntimeError(
            f"Cannot resume: group_metrics.jsonl has {len(lines)} rows, "
            f"but the checkpoint requires {keep_lines}."
        )
    if len(lines) == keep_lines:
        return
    suffix = time.strftime("%Y%m%d_%H%M%S")
    archived = f"{path}.uncheckpointed_tail_{suffix}"
    with open(archived, "w", encoding="utf-8") as handle:
        handle.writelines(lines[keep_lines:])
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.writelines(lines[:keep_lines])
    os.replace(temporary, path)
    print(
        f"[driver][resume] archived {len(lines) - keep_lines} uncheckpointed "
        f"group metric rows to {archived}"
    )


def _canonical_task_type(task_type):
    ""
    return canonicalize_alfworld_task_type(task_type)


def _canonicalize_token_breakdown(by_task_type, token_keys):
    ""
    normalized = {tt: {key: 0 for key in token_keys} for tt in NORL_TASK_TYPES}
    for raw_tt, usage in (by_task_type or {}).items():
        bucket = normalized[_canonical_task_type(raw_tt)]
        for key in token_keys:
            bucket[key] += int((usage or {}).get(key, 0) or 0)
    return normalized


def _canonicalize_count_breakdown(by_task_type):
    normalized = {tt: 0 for tt in NORL_TASK_TYPES}
    for raw_tt, value in (by_task_type or {}).items():
        normalized[_canonical_task_type(raw_tt)] += int(value or 0)
    return normalized


def _trace_compression_metric_fields(args):
    ""
    flags = {
        "enable_loop_filter": bool(args.trace_enable_loop_filter),
        "enable_obs_delta": bool(args.trace_enable_obs_delta),
        "enable_prefix_tree": bool(args.trace_enable_prefix_tree),
        "enable_consensus_prefix": bool(args.trace_enable_consensus_prefix),
    }
    condition = "all_on" if all(flags.values()) else ("all_off" if not any(flags.values()) else "partial")
    return {
        "experiment/trace_compression/condition": condition,
        "experiment/trace_compression/cloud_evidence_mode": getattr(
            args, "trace_cloud_evidence_mode", "tree_only"
        ),
        **{f"experiment/trace_compression/{name}": int(enabled) for name, enabled in flags.items()},
    }


def _load_fixed_games_manifest(manifest_path, alfworld_data=None):
    ""
    with open(manifest_path) as f:
        payload = json.load(f)
    entries = payload.get("games") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"fixed games manifest is empty: {manifest_path}")

    data_root = alfworld_data or os.environ.get("ALFWORLD_DATA")
    games = []
    task_types = []
    tids = []
    seen_paths = set()
    for i, entry in enumerate(entries, 1):
        if isinstance(entry, str):
            entry = {"game_file": entry}
        if not isinstance(entry, dict) or not entry.get("game_file"):
            raise ValueError(f"manifest game #{i} needs a game_file")
        raw_path = os.path.expandvars(os.path.expanduser(entry["game_file"]))
        if not os.path.isabs(raw_path):
            if not data_root:
                raise ValueError("relative game_file requires ALFWORLD_DATA")
            raw_path = os.path.join(data_root, raw_path)
        game_file = os.path.realpath(raw_path)
        if not os.path.isfile(game_file):
            raise FileNotFoundError(f"fixed game not found: {game_file}")
        if game_file in seen_paths:
            raise ValueError(f"duplicate fixed game: {game_file}")
        seen_paths.add(game_file)

        traj_path = os.path.join(os.path.dirname(game_file), "traj_data.json")
        with open(traj_path) as f:
            traj = json.load(f)
        with open(game_file) as f:
            game_data = json.load(f)
        task_type = traj.get("task_type")
        expected = entry.get("task_type")
        if expected and task_type != expected:
            raise ValueError(f"manifest game #{i} expected {expected}, got {task_type}: {game_file}")
        tid = _TASK_TYPE_TO_ID.get(task_type)
        if tid is None:
            raise ValueError(f"unsupported task_type {task_type}: {game_file}")
        if not game_data.get("solvable", False):
            raise ValueError(f"fixed game is not solvable: {game_file}")

        label = entry.get("label") or f"game_{i}"
        print(f"[driver] fixed game {label}: {task_type} -> {game_file}")
        games.append((game_file, traj))
        if task_type not in task_types:
            task_types.append(task_type)
        if tid not in tids:
            tids.append(tid)
    return games, task_types, tids


def _load_resume_state(args):
    ""
    state = {
        "resume": False,
        "skills_json_path": args.skills_json,
        "epoch0": 0,
        "ep_i0": 0,
        "completed_groups": 0,
        "wins": 0,
        "per_game": [],
        "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
        "context_guard": {"prompt_trims": 0, "trimmed_tokens": 0},
        "small_model_token_accounting": SMALL_MODEL_TOKEN_ACCOUNTING,
        "large_model_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "small_model_tokens_by_tt": {tt: {"prompt": 0, "response": 0, "total": 0} for tt in NORL_TASK_TYPES},
        "large_model_tokens_by_tt": {tt: {"prompt": 0, "completion": 0, "total": 0} for tt in NORL_TASK_TYPES},
        "large_model_tokens_mixed": {"prompt": 0, "completion": 0, "total": 0},
        "large_model_usage": {
            "reported_calls": 0,
            "missing_calls": 0,
            "missing_calls_by_task_type": {tt: 0 for tt in NORL_TASK_TYPES},
            "missing_calls_mixed": 0,
        },
        "cloud_updates": 0,
        "cloud_update_steps": [],
    }
    if not args.resume:
        return state

    summary_path = os.path.join(args.outdir, "summary_partial.json")
    if not os.path.isfile(summary_path):
        print(
            f"[driver][resume] --resume 1 was requested, but {summary_path} "
            "does not exist. Starting a fresh run from --skills_json at episode 0."
        )
        return state

    with open(summary_path) as f:
        prev_summary = json.load(f)

    skill_dir = os.path.join(args.outdir, "skill_lib")
    for candidate in ("skills_latest_checkpoint.json", "skills_latest_rollout.json"):
        candidate_path = os.path.join(skill_dir, candidate)
        if os.path.isfile(candidate_path):
            state["skills_json_path"] = candidate_path
            break
    else:
        print(
            f"[driver][resume] no skill-library checkpoint was found in {skill_dir}; "
            f"loading the skill library from the seed file {args.skills_json} "
            "while restoring the remaining progress."
        )

    state["resume"] = True
    state["epoch0"] = max(0, int(prev_summary.get("current_epoch", 1)) - 1)
    state["ep_i0"] = max(0, int(prev_summary.get("next_episode_index_in_epoch", 1)) - 1)
    state["completed_groups"] = int(prev_summary.get("completed_rollout_groups", 0) or 0)
    state["wins"] = int(prev_summary.get("wins", 0) or 0)
    state["per_game"] = list(prev_summary.get("per_game", []) or [])
    saved_small_tokens = (prev_summary.get("token_usage", {}) or {}).get("small_model", {}) or {}
    state["small_model_tokens"] = {key: int(saved_small_tokens.get(key, 0) or 0) for key in ("prompt", "response", "total")}
    saved_context_guard = prev_summary.get("context_guard", {}) or {}
    state["context_guard"] = {key: int(saved_context_guard.get(key, 0) or 0) for key in ("prompt_trims", "trimmed_tokens")}
    previous_accounting = saved_small_tokens.get("accounting")
    if not previous_accounting and state["small_model_tokens"]["total"]:
        previous_accounting = "vllm_request_tokens_two_stage"
    if previous_accounting and previous_accounting != SMALL_MODEL_TOKEN_ACCOUNTING:
        state["small_model_token_accounting"] = f"mixed:{previous_accounting}+{SMALL_MODEL_TOKEN_ACCOUNTING}"
    saved_large_tokens = (prev_summary.get("token_usage", {}) or {}).get("large_model", {}) or {}
    state["large_model_tokens"] = {key: int(saved_large_tokens.get(key, 0) or 0) for key in ("prompt", "completion", "total")}



    state["small_model_tokens_by_tt"] = _canonicalize_token_breakdown(
        saved_small_tokens.get("by_task_type", {}) or {},
        ("prompt", "response", "total"),
    )
    state["large_model_tokens_by_tt"] = _canonicalize_token_breakdown(
        saved_large_tokens.get("by_task_type", {}) or {},
        ("prompt", "completion", "total"),
    )
    saved_large_mixed = saved_large_tokens.get("mixed", {}) or {}
    state["large_model_tokens_mixed"] = {key: int(saved_large_mixed.get(key, 0) or 0) for key in ("prompt", "completion", "total")}
    saved_usage = saved_large_tokens.get("usage", {}) or {}
    state["large_model_usage"] = {
        "reported_calls": int(saved_usage.get("reported_calls", 0) or 0),
        "missing_calls": int(saved_usage.get("missing_calls", 0) or 0),
        "missing_calls_by_task_type": _canonicalize_count_breakdown(saved_usage.get("missing_calls_by_task_type", {}) or {}),
        "missing_calls_mixed": int(saved_usage.get("missing_calls_mixed", 0) or 0),
    }



    if state["per_game"]:
        rebuilt_small = {tt: {"prompt": 0, "response": 0, "total": 0} for tt in NORL_TASK_TYPES}
        for episode in state["per_game"]:
            bucket = rebuilt_small[_canonical_task_type(episode.get("detected_type"))]
            for key, field in (("prompt", "tokens_prompt"), ("response", "tokens_response"), ("total", "tokens_total")):
                bucket[key] += int(episode.get(field, 0) or 0)
        state["small_model_tokens_by_tt"] = rebuilt_small
    state["cloud_update_steps"] = list(prev_summary.get("cloud_update_steps", []) or [])
    state["cloud_updates"] = len(state["cloud_update_steps"])

    print(
        f"[driver][resume] restored from {summary_path}: "
        f"epoch={state['epoch0'] + 1} ep_i={state['ep_i0']} "
        f"completed_groups={state['completed_groups']} "
        f"wins={state['wins']}/{len(state['per_game'])} "
        f"skill_lib<-{state['skills_json_path']}"
    )
    print(
        "[driver][resume] note: TextWorld determines game order through its "
        "internal shuffled_cycle. Resume alignment rebuilds the environment and "
        "calls reset() once per completed episode. It is reproducible with the "
        "same seed and game_files, but it is not a mathematically exact replay guarantee."
    )
    return state


def rollout_episode(env, agent, builder, max_steps, tag="", keep_logrows=True):
    ""
    import time as _time

    obs_list, infos = env.reset()
    obs_text = obs_list[0]
    adm = infos["admissible_commands"][0]
    task = extract_task(obs_text)


    builder.reset(task)
    task_type = (builder.retrieved or {}).get("task_type", "unknown")
    injected_ids = (builder.retrieved or {}).get("injected_skill_ids", []) or []
    print(f"[rollout]{tag} task={task}")

    steps = []
    logrows = []
    won = False
    step = 0
    n_valid = 0
    n_strict_valid = 0
    last_action = None
    repeat = 0

    for step in range(1, max_steps + 1):
        _t0 = _time.time()
        prompt = builder.build(obs_text, adm, init=(step == 1))
        raw, forced = agent.act_with_meta(prompt)
        _, _think = None, None


        actions, valids, action_details = alfworld_projection([raw], [adm], return_details=True)
        action = actions[0]
        valid = bool(valids[0])
        action_detail = action_details[0]
        if valid:
            n_valid += 1
        if action_detail["strict_valid_action"]:
            n_strict_valid += 1

        nobs_list, scores, dones, ninfos = env.step([action])
        nobs = nobs_list[0]
        nadm = ninfos["admissible_commands"][0]
        reward = float(scores[0]) if scores is not None else 0.0
        done = bool(dones[0])
        won = bool(ninfos.get("won", [False])[0])
        print(f"  [rollout]{tag} step={step}/{max_steps} action={action!r} valid={valid} forced={forced} won={won} ({_time.time() - _t0:.1f}s)")


        steps.append(
            {
                "step": step,
                "observation": obs_text,
                "action": action,
                "reward": reward,
                "valid_action": valid,
                "non_strict_valid_action": valid,
                "strict_valid_action": action_detail["strict_valid_action"],
                "execution_source": action_detail["execution_source"],
                "direct_admissible_action": action_detail["direct_admissible_action"],
            }
        )
        if keep_logrows:
            logrows.append(
                {
                    "step": step,
                    "prompt": prompt,
                    "action": action,
                    "valid": valid,
                    "valid_action": valid,
                    "non_strict_valid_action": valid,
                    "strict_valid_action": action_detail["strict_valid_action"],
                    "execution_source": action_detail["execution_source"],
                    "direct_admissible_action": action_detail["direct_admissible_action"],
                    "forced": bool(forced),
                    "obs": nobs,
                    "reward": reward,
                    "won": won,
                }
            )

        builder.record(obs_text, action)
        obs_text, adm = nobs, nadm

        repeat = repeat + 1 if action == last_action else 0
        last_action = action
        if done or won:
            break

        if repeat >= 6:
            break

    raw_trace = {
        "traj_uid": str(uuid.uuid4()),
        "task": task,
        "task_type": task_type,
        "outcome": "success" if won else "failure",
        "episode_reward": 1.0 if won else 0.0,
        "steps": steps,
        "meta": {
            "skill_ids_used": injected_ids,
            "model_version": "frozen",
            "n_valid_actions": n_valid,
            "valid_action_ratio": n_valid / max(step, 1),
            "n_non_strict_valid_actions": n_valid,
            "non_strict_valid_action_ratio": n_valid / max(step, 1),
            "n_strict_valid_actions": n_strict_valid,
            "strict_valid_action_ratio": n_strict_valid / max(step, 1),
            "n_salvaged_actions": sum(s["execution_source"] == "salvaged" for s in steps),
            "n_fallback_actions": sum(s["execution_source"] == "fallback" for s in steps),
        },
    }
    return won, step, raw_trace, task_type, injected_ids, n_valid, logrows


def _stable_game_id(game_file):
    normalized = str(game_file).replace("\\", "/")
    marker = "/json_2.1.1/"
    return normalized.split(marker, 1)[1] if marker in normalized else normalized


def _fixed_request_seed(base_seed, game_id, replica_index, step):
    payload = f"{int(base_seed)}|{game_id}|{int(replica_index)}|{int(step)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def rollout_batch_group(env, agent, skill_lib, args, batch_size, tag="", fixed_replica_offsets=None):
    ""
    import time as _time

    obs_list, infos = env.reset()

    adms = infos["admissible_commands"]
    tasks = [extract_task(o) for o in obs_list]
    fixed_seed_rows = None
    if args.fixed_games_manifest:
        game_files = infos.get("extra.gamefile")
        if not game_files or len(game_files) != batch_size:
            raise RuntimeError("fixed manifest rollout requires extra.gamefile for every slot")
        offsets = fixed_replica_offsets or {}
        seen = {}
        fixed_seed_rows = []
        for game_file in game_files:
            game_id = _stable_game_id(game_file)
            replica_index = int(offsets.get(game_id, 0)) + seen.get(game_id, 0)
            seen[game_id] = seen.get(game_id, 0) + 1
            fixed_seed_rows.append((game_id, replica_index))
    builders = [AlfWorldObsBuilder(mem_lib=skill_lib, with_skills=bool(args.enable_coskill), top_k=args.top_k, history_length=args.history_length) for _ in range(batch_size)]
    task_types = []
    injected_ids = []
    playbook_records = []
    for i, b in enumerate(builders):
        b.reset(tasks[i])
        tt = (b.retrieved or {}).get("task_type", "unknown")
        task_types.append(tt)
        injected_ids.append((b.retrieved or {}).get("injected_skill_ids", []) or [])
        pb_rec = skill_lib.get_playbook_record(tt) if args.enable_skill_tree and hasattr(skill_lib, "get_playbook_record") else None
        playbook_records.append(pb_rec)

    print(f"[rollout-batch]{tag} size={batch_size} tasks={ {t: task_types.count(t) for t in sorted(set(task_types))} }")

    steps = [[] for _ in range(batch_size)]
    logrows = [[] for _ in range(batch_size)]
    won = [False for _ in range(batch_size)]
    done = [False for _ in range(batch_size)]
    used = [0 for _ in range(batch_size)]
    n_valid = [0 for _ in range(batch_size)]
    n_strict_valid = [0 for _ in range(batch_size)]
    last_action = [None for _ in range(batch_size)]
    repeat = [0 for _ in range(batch_size)]

    keep_logrows = bool(args.log_trajectories)






    episode_tokens = [{"prompt": 0, "response": 0, "total": 0} for _ in range(batch_size)]

    for step in range(1, args.max_steps + 1):
        active = [i for i in range(batch_size) if not done[i]]
        if not active:
            break

        _t0 = _time.time()
        prompts = [builders[i].build(obs_list[i], adms[i], init=(step == 1)) for i in active]
        request_seeds = [_fixed_request_seed(args.seed, fixed_seed_rows[i][0], fixed_seed_rows[i][1], step) for i in active] if fixed_seed_rows is not None else None
        seed_by_idx = {i: request_seeds[j] for j, i in enumerate(active)} if request_seeds is not None else {}
        raw_forced = agent.act_batch_with_meta(prompts, sampling_seeds=request_seeds)
        raws = [x[0] for x in raw_forced]
        forceds = [x[1] for x in raw_forced]
        per_request_tokens = getattr(agent, "last_batch_request_tokens", None) or []
        for local_i, i in enumerate(active):
            if local_i >= len(per_request_tokens):
                continue
            rt = per_request_tokens[local_i]
            episode_tokens[i]["prompt"] += rt.get("prompt", 0)
            episode_tokens[i]["response"] += rt.get("response", 0)
            episode_tokens[i]["total"] += rt.get("total", 0)
        active_adms = [adms[i] for i in active]


        actions, valids, action_details = alfworld_projection(raws, active_adms, return_details=True)

        full_actions = []
        action_by_idx = {}
        valid_by_idx = {}
        forced_by_idx = {}
        action_detail_by_idx = {}
        raw_by_idx = {}
        for local_i, i in enumerate(active):
            action = actions[local_i]
            valid = bool(valids[local_i])
            if valid:
                n_valid[i] += 1
            if action_details[local_i]["strict_valid_action"]:
                n_strict_valid[i] += 1
            action_by_idx[i] = action
            valid_by_idx[i] = valid
            forced_by_idx[i] = bool(forceds[local_i])
            action_detail_by_idx[i] = action_details[local_i]
            raw_by_idx[i] = raws[local_i]

        for i in range(batch_size):
            if i in action_by_idx:
                full_actions.append(action_by_idx[i])
            else:

                adm = adms[i] if i < len(adms) else []
                full_actions.append("look" if "look" in adm else (adm[0] if adm else "look"))

        nobs_list, scores, dones, ninfos = env.step(full_actions)
        nadms = ninfos["admissible_commands"]
        wins = ninfos.get("won", [False] * batch_size)

        for i in active:
            reward = float(scores[i]) if scores is not None else 0.0
            slot_won = bool(wins[i])
            slot_done = bool(dones[i]) or slot_won
            action = action_by_idx[i]
            action_detail = action_detail_by_idx[i]

            steps[i].append(
                {
                    "step": step,
                    "observation": obs_list[i],
                    "action": action,
                    "reward": reward,
                    "valid_action": valid_by_idx[i],
                    "non_strict_valid_action": valid_by_idx[i],
                    "strict_valid_action": action_detail["strict_valid_action"],
                    "execution_source": action_detail["execution_source"],
                    "direct_admissible_action": action_detail["direct_admissible_action"],
                    "sampling_seed": seed_by_idx.get(i),
                }
            )
            if keep_logrows:
                logrows[i].append(
                    {
                        "step": step,
                        "prompt": prompts[active.index(i)],
                        "action": action,
                        "valid": valid_by_idx[i],
                        "valid_action": valid_by_idx[i],
                        "non_strict_valid_action": valid_by_idx[i],
                        "strict_valid_action": action_detail["strict_valid_action"],
                        "execution_source": action_detail["execution_source"],
                        "direct_admissible_action": action_detail["direct_admissible_action"],
                        "sampling_seed": seed_by_idx.get(i),
                        "forced": forced_by_idx[i],
                        "obs": nobs_list[i],
                        "reward": reward,
                        "won": slot_won,
                    }
                )

            builders[i].record(obs_list[i], action)
            used[i] = step
            won[i] = slot_won
            repeat[i] = repeat[i] + 1 if action == last_action[i] else 0
            last_action[i] = action
            if slot_done or repeat[i] >= 6:
                done[i] = True

        obs_list = nobs_list
        adms = nadms
        n_done = sum(done)
        n_won = sum(won)
        print(f"  [rollout-batch]{tag} step={step}/{args.max_steps} active={len(active)} done={n_done}/{batch_size} won={n_won}/{batch_size} ({_time.time() - _t0:.1f}s)")

    episodes = []
    for i in range(batch_size):
        raw_trace = {
            "traj_uid": str(uuid.uuid4()),
            "task": tasks[i],
            "task_type": task_types[i],
            "outcome": "success" if won[i] else "failure",
            "episode_reward": 1.0 if won[i] else 0.0,
            "steps": steps[i],
            "meta": {
                "skill_ids_used": injected_ids[i],
                "model_version": "frozen",
                "fixed_game_id": (fixed_seed_rows[i][0] if fixed_seed_rows else None),
                "fixed_replica_index": (fixed_seed_rows[i][1] if fixed_seed_rows else None),
                "n_valid_actions": n_valid[i],
                "valid_action_ratio": n_valid[i] / max(used[i], 1),
                "n_non_strict_valid_actions": n_valid[i],
                "non_strict_valid_action_ratio": n_valid[i] / max(used[i], 1),
                "n_strict_valid_actions": n_strict_valid[i],
                "strict_valid_action_ratio": n_strict_valid[i] / max(used[i], 1),
                "n_salvaged_actions": sum(s.get("execution_source") == "salvaged" for s in steps[i]),
                "n_fallback_actions": sum(s.get("execution_source") == "fallback" for s in steps[i]),
            },
        }
        episodes.append(
            {
                "won": won[i],
                "used": used[i] or args.max_steps,
                "raw_trace": raw_trace,
                "task_type": task_types[i],
                "injected": injected_ids[i],
                "n_valid": n_valid[i],
                "logrows": logrows[i],
                "playbook_record": playbook_records[i],
                "small_model_tokens": episode_tokens[i],
            }
        )
    return episodes


def _split_int(total: int, parts: int):
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _fixed_manifest_dp_plan(game_files, replicas_per_game: int, workers: int):
    ""
    games = list(game_files)
    if not games or replicas_per_game < 1 or workers < 1:
        raise ValueError("fixed manifest DP planning requires games, replicas and workers")
    workers = min(int(workers), len(games) * int(replicas_per_game))
    plan = []
    if workers <= len(games):
        assignments = [games[i::workers] for i in range(workers)]
        plan = [(assigned, len(assigned) * replicas_per_game) for assigned in assignments]
    else:
        worker_counts = _split_int(workers, len(games))
        for game, game_workers in zip(games, worker_counts):
            for chunk in _split_int(replicas_per_game, game_workers):
                plan.append(([game], chunk))
    assert len(plan) == workers
    assert sum(batch for _, batch in plan) == len(games) * replicas_per_game
    return plan


def _token_delta(after, before):
    ""
    prompt = max(0, int(after.get("prompt", 0)) - int(before.get("prompt", 0)))
    response = max(0, int(after.get("response", 0)) - int(before.get("response", 0)))
    return {"prompt": prompt, "response": response, "total": prompt + response}


def _context_guard_delta(after, before):
    ""
    return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in ("prompt_trims", "trimmed_tokens")}


def _dp_rollout_worker(worker_id, gpu_id, args_dict, game_files, task_type_ids, fixed_batch_size, fixed_replica_offsets, in_q, out_q, resume_skip_groups=0):
    ""
    try:




        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices != str(gpu_id):
            raise RuntimeError(f"worker {worker_id} expected CUDA_VISIBLE_DEVICES={gpu_id!r}, got {visible_devices!r}; refusing unsafe vLLM launch")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        args = argparse.Namespace(**args_dict)
        args.tensor_parallel_size = 1

        config = load_tw_config_types(task_type_ids, num_games=len(game_files))
        env = make_batch_env(game_files, config, batch_size=fixed_batch_size, seed=int(args.seed) + 1009 * (worker_id + 1))
        if resume_skip_groups > 0:
            _t0 = time.time()
            for _ in range(resume_skip_groups):
                env.reset()
            print(f"[dp-worker{worker_id}][resume] skipped {resume_skip_groups} reset() calls to align with prior run ({time.time() - _t0:.1f}s)")

        from agent_system.frozen_executor.vllm_agent import VLLMAgent

        required_max_num_seqs = fixed_batch_size
        vllm_max_num_seqs = int(args.vllm_max_num_seqs or required_max_num_seqs)
        if vllm_max_num_seqs < required_max_num_seqs:
            raise ValueError(f"vllm_max_num_seqs={vllm_max_num_seqs} is smaller than worker {worker_id}'s rollout batch={required_max_num_seqs}")
        agent = VLLMAgent(
            model_path=args.model_path,
            gpu_memory_utilization=args.gpu_mem_util,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=not args.no_thinking,
            seed=int(args.seed) + worker_id,
            no_wait=args.nowait,
            think_budget=args.think_budget,
            max_num_seqs=vllm_max_num_seqs,
            enforce_eager=bool(args.vllm_enforce_eager),
        )
        print(f"[dp-worker{worker_id}] ready gpu={gpu_id} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} batch={fixed_batch_size} max_num_seqs={vllm_max_num_seqs} enforce_eager={bool(args.vllm_enforce_eager)} games={len(game_files)}")

        while True:
            cmd = in_q.get()
            if cmd is None:
                print(f"[dp-worker{worker_id}] shutdown")
                agent.close()
                return
            group_id = cmd.get("group_id")
            skill_path = cmd["skill_path"]
            try:
                tokens_before = agent.get_token_usage()
                context_guard_before = agent.get_context_guard_usage()
                skill_lib = HierarchicalSkillLib(
                    skills_json_path=skill_path,
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
                tag = f" group{group_id} worker{worker_id} gpu={gpu_id}"
                results = rollout_batch_group(
                    env,
                    agent,
                    skill_lib,
                    args,
                    fixed_batch_size,
                    tag=tag,
                    fixed_replica_offsets=fixed_replica_offsets,
                )
                token_usage = _token_delta(agent.get_token_usage(), tokens_before)
                context_guard_usage = _context_guard_delta(agent.get_context_guard_usage(), context_guard_before)
                out_q.put(
                    {
                        "worker_id": worker_id,
                        "group_id": group_id,
                        "results": results,
                        "small_model_tokens": token_usage,
                        "context_guard": context_guard_usage,
                        "error": None,
                    }
                )
            except Exception:
                out_q.put(
                    {
                        "worker_id": worker_id,
                        "group_id": group_id,
                        "results": [],
                        "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
                        "context_guard": {"prompt_trims": 0, "trimmed_tokens": 0},
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        out_q.put(
            {
                "worker_id": worker_id,
                "group_id": None,
                "results": [],
                "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
                "context_guard": {"prompt_trims": 0, "trimmed_tokens": 0},
                "error": traceback.format_exc(),
            }
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--task_types",
        default=(
            "pick_and_place_simple,look_at_obj_in_light,"
            "pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,"
            "pick_cool_then_place_in_recep,pick_two_obj_and_place"
        ),
        help=(
            "Comma-separated dataset task types. The default includes all six "
            "types; success is determined by env won for every type."
        ),
    )
    ap.add_argument("--fixed_games_manifest", default=None, help="JSON manifest of fixed tasks. When provided, task_types, num_games, and sample are ignored so OFF/ON comparisons can reuse identical game.tw-pddl files.")
    ap.add_argument("--num_games", type=int, default=-1, help="Games per task type. Values <=0 use all games, as required for full evaluation without sampling.")
    ap.add_argument("--group_size", type=int, default=6, help="Rollouts per game, corresponding to env.rollout.n.")
    ap.add_argument("--batch_rollout_size", type=int, default=1, help="Values >1 enable synchronous batched rollout across multiple ALFWorld environments. Cloud updates occur only after a complete group. Set 72 to approximate GRPO with train_data_size=12 and group_size=6.")
    ap.add_argument("--data_parallel_workers", type=int, default=1, help="Values >1 enable multi-process rollout data parallelism. Each worker owns one GPU and one vLLM replica; the driver aggregates trajectories and triggers cloud updates.")
    ap.add_argument("--rollout_worker_gpus", default=None, help="Comma-separated GPU list for rollout workers, for example '0,1'. By default, use the first data_parallel_workers devices from CUDA_VISIBLE_DEVICES or nvidia-smi.")
    ap.add_argument("--sample", action="store_true", default=False, help="When num_games>0, sample that many games evenly across object types. Full evaluation does not require sampling.")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1, help="Number of passes over the complete game set, allowing repeated skill-tree evolution.")
    ap.add_argument("--max_episodes", type=int, default=0, help="Hard cap on total episodes, independent of epochs and remaining games. Values <=0 run all epochs over the full game pool. Use a small value such as 20 for a smoke test.")
    ap.add_argument("--cloud_update_every", type=int, default=0, help="Force one cloud update at every fixed N-episode boundary. Values <=0 use only the two watermark triggers. Keep 0 for production; use a positive value for controlled A/B timing.")
    ap.add_argument("--checkpoint_every_groups", type=int, default=0, help="Save a lightweight summary and skill-library snapshot every N rollout groups without forcing a cloud update. Values <=0 save only at the end.")
    ap.add_argument("--history_length", type=int, default=8)

    ap.add_argument("--model_path", default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.8)
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs used by each vLLM tensor-parallel replica. This must not exceed the devices visible through CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--vllm_max_num_seqs", type=int, default=0, help="0 uses the actual rollout batch size of each vLLM replica. A positive override must be at least that replica's batch size.")
    ap.add_argument("--vllm_enforce_eager", type=int, choices=[0, 1], default=1, help="1 preserves eager execution; 0 enables vLLM CUDA Graph execution.")
    ap.add_argument("--max_model_len", type=int, default=10240)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--think_budget", type=int, default=3500)
    ap.add_argument("--temperature", type=float, default=1.0, help="Training-rollout sampling temperature. The default 1.0 preserves success/failure trajectory diversity.")
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--nowait", action="store_true")

    ap.add_argument("--skills_json", default="memory_data/alfworld/initial_skills.json")
    ap.add_argument("--retrieval_mode", default="embedding", choices=["embedding", "template"])
    ap.add_argument("--embedding_model_path", default=None)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--enable_hierarchy", type=int, default=1)
    ap.add_argument("--stable_cycles_l1", type=int, default=3)
    ap.add_argument("--stable_cycles_l2", type=int, default=5)
    ap.add_argument("--success_l1", type=float, default=0.7)
    ap.add_argument("--demote_threshold", type=float, default=0.3)
    ap.add_argument("--min_calls", type=int, default=10)

    ap.add_argument("--enable_coskill", type=int, default=1, help="Enable flat skill bullets. Cloud contrastive_distill produces dyn_ patches, and the edge prompt receives General, Task-specific, and Mistakes sections. Set 0 explicitly for an ablation.")
    ap.add_argument("--enable_skill_tree", type=int, default=1, help="Inject the agent skill tree into the edge-model prompt. Set 0 for the ablation baseline.")
    ap.add_argument("--enable_skill_tree_evolve", dest="enable_playbook_evolve", type=int, default=1, metavar="ENABLE_SKILL_TREE_EVOLVE")
    ap.add_argument("--enable_playbook_evolve", dest="enable_playbook_evolve", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--enable_failure_analysis", type=int, default=1)
    ap.add_argument("--enable_cloud_updates", type=int, default=1, help="0 keeps collecting raw traces/metrics but freezes the skill library. Used by fixed-artifact evaluation; default preserves the closed loop.")
    ap.add_argument("--max_new_skills", type=int, default=3)
    ap.add_argument("--skill_tree_evolve_min_samples", dest="playbook_evolve_min_samples", type=int, default=6, metavar="SKILL_TREE_EVOLVE_MIN_SAMPLES")
    ap.add_argument("--playbook_evolve_min_samples", dest="playbook_evolve_min_samples", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--coskill_debug", type=int, default=0)
    ap.add_argument("--required_tree_depth", type=int, default=0, help="Require a cloud-authored tree with exactly this many heading levels; 0 disables.")
    ap.add_argument("--tree_depth_repair_attempts", type=int, default=0, help="Same-evidence cloud repair attempts after a fixed-depth tree fails validation.")
    ap.add_argument("--tree_max_nodes", type=int, default=0, help="Hard maximum semantic heading nodes for cloud-authored trees; 0 disables.")
    ap.add_argument("--tree_max_chars", type=int, default=0, help="Hard maximum rendered characters for cloud-authored trees; 0 disables.")

    ap.add_argument("--capacity_watermark", type=int, default=50000)
    ap.add_argument("--perf_watermark", type=float, default=0.6)
    ap.add_argument("--min_samples", type=int, default=16)
    ap.add_argument("--loop_threshold", type=int, default=3)
    ap.add_argument("--trace_enable_loop_filter", type=int, default=1)
    ap.add_argument("--trace_enable_obs_delta", type=int, default=1)
    ap.add_argument("--trace_enable_prefix_tree", type=int, default=1)
    ap.add_argument("--trace_enable_consensus_prefix", type=int, default=1)
    ap.add_argument(
        "--trace_cloud_evidence_mode",
        choices=("tree_only", "flat"),
        default="tree_only",
        help=(
            "tree_only sends only the self-contained trajectory-tree codec to "
            "CloudAnalyzer; flat is reserved for compression-off ablations"
        ),
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log_trajectories", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0, help="Set 1 to restore the skill library and epoch/episode progress from skill_lib checkpoints plus summary_partial.json in the same outdir. The explicit opt-in prevents stale output state from overriding an intended fresh run.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    resume_state = _load_resume_state(args)
    enable_coskill = bool(args.enable_coskill)
    enable_skill_tree = bool(args.enable_skill_tree)
    enable_skill_tree_evolve = enable_skill_tree and bool(args.enable_playbook_evolve)
    if bool(args.enable_playbook_evolve) and not enable_skill_tree:
        print("[driver] enable_skill_tree=0, so skill tree evolution is disabled too")




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
        enable_playbook=enable_skill_tree,
    )


    traces_pool = TracesPool(
        capacity_watermark=args.capacity_watermark,
        perf_watermark=args.perf_watermark,
        min_samples=args.min_samples,
        loop_threshold=args.loop_threshold,
        output_dir=args.outdir,
        enable_loop_filter=bool(args.trace_enable_loop_filter),
        enable_obs_delta=bool(args.trace_enable_obs_delta),
        enable_prefix_tree=bool(args.trace_enable_prefix_tree),
        enable_consensus_prefix=bool(args.trace_enable_consensus_prefix),
        cloud_evidence_mode=args.trace_cloud_evidence_mode,
    )


    cloud_loop = CoSkillCloudLoop(
        output_dir=args.outdir,
        enable_coskill=enable_coskill,
        enable_playbook_evolve=enable_skill_tree_evolve,
        enable_failure_analysis=bool(args.enable_failure_analysis),
        max_new_skills=args.max_new_skills,
        playbook_evolve_min_samples=args.playbook_evolve_min_samples,
        coskill_debug=bool(args.coskill_debug),
        environment_name="ALFWorld",
        required_tree_depth=(args.required_tree_depth or None),
        tree_depth_repair_attempts=args.tree_depth_repair_attempts,
        tree_max_nodes=(args.tree_max_nodes or None),
        tree_max_chars=(args.tree_max_chars or None),
    )


















    if args.fixed_games_manifest:
        fixed_games, task_types, all_tids = _load_fixed_games_manifest(args.fixed_games_manifest)
        all_game_files = [g[0] for g in fixed_games]
        print(f"[driver] using fixed manifest: {args.fixed_games_manifest}")
    else:
        task_types = [t.strip() for t in args.task_types.split(",") if t.strip()]
        all_game_files = []
        all_tids = []
        for task_type in task_types:
            tid = _TASK_TYPE_TO_ID.get(task_type)
            if tid is None:
                print(f"[driver] unknown task_type {task_type}, skip")
                continue



            if args.num_games <= 0:
                games = find_games_by_type(task_type, split=args.split)
            else:
                games = find_games_by_type(task_type, split=args.split, sample_n=args.num_games if args.sample else None, sample_seed=args.sample_seed, limit=None if args.sample else args.num_games)
            if not games:
                print(f"[driver] no games for {task_type}, skip")
                continue
            game_files = [g[0] for g in games]
            print(f"[driver] {task_type}: {len(game_files)} games")
            all_game_files.extend(game_files)
            all_tids.append(tid)

    if not all_game_files:
        print("[driver] no games found for any requested task_type, exiting")
        return
    total_games = len(all_game_files)
    episodes_per_epoch = total_games * args.group_size
    print(f"[driver] combined pool: {total_games} games across {len(all_tids)} task_types x group_size={args.group_size} x epochs={args.epochs} = {episodes_per_epoch * args.epochs} episodes total")
    config = load_tw_config_types(all_tids, num_games=total_games)
    batch_rollout_size = max(1, int(args.batch_rollout_size or 1))
    data_parallel_workers = max(1, int(args.data_parallel_workers or 1))
    if int(args.vllm_max_num_seqs) < 0:
        raise ValueError("vllm_max_num_seqs must be >= 0")
    use_data_parallel = data_parallel_workers > 1
    env = None
    agent = None
    builder = None
    dp_in_queues = []
    dp_out_q = None
    dp_processes = []
    dp_worker_batch_sizes = []
    vllm_max_num_seqs_by_worker = []

    if use_data_parallel:
        if batch_rollout_size < data_parallel_workers:
            raise ValueError("batch_rollout_size must be >= data_parallel_workers")
        if args.rollout_worker_gpus:
            worker_gpus = [g.strip() for g in args.rollout_worker_gpus.split(",") if g.strip()]
        elif os.environ.get("CUDA_VISIBLE_DEVICES"):
            worker_gpus = [g.strip() for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
        else:
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    text=True,
                )
                worker_gpus = [g.strip() for g in raw.splitlines() if g.strip()]
            except Exception:
                worker_gpus = [str(i) for i in range(data_parallel_workers)]
        fixed_balanced = bool(args.fixed_games_manifest and batch_rollout_size == len(all_game_files) * int(args.group_size))
        if fixed_balanced:
            worker_plan = _fixed_manifest_dp_plan(all_game_files, int(args.group_size), data_parallel_workers)
            data_parallel_workers = len(worker_plan)
        else:
            dp_worker_batch_sizes = _split_int(batch_rollout_size, data_parallel_workers)
            worker_plan = [(all_game_files[wid::data_parallel_workers] or list(all_game_files), worker_bs) for wid, worker_bs in enumerate(dp_worker_batch_sizes)]
        if len(worker_gpus) < data_parallel_workers:
            raise ValueError(f"data_parallel_workers={data_parallel_workers} but only {len(worker_gpus)} GPU ids available: {worker_gpus}")
        worker_gpus = worker_gpus[:data_parallel_workers]
        dp_worker_batch_sizes = [batch for _, batch in worker_plan]
        vllm_max_num_seqs_by_worker = [int(args.vllm_max_num_seqs or worker_batch) for worker_batch in dp_worker_batch_sizes]
        for worker_id, (limit, worker_batch) in enumerate(zip(vllm_max_num_seqs_by_worker, dp_worker_batch_sizes)):
            if limit < worker_batch:
                raise ValueError(f"vllm_max_num_seqs={limit} is smaller than worker {worker_id}'s rollout batch={worker_batch}")
        replica_counts = {}
        worker_replica_offsets = []
        for worker_games, worker_bs in worker_plan:
            if fixed_balanced:
                per_game = worker_bs // len(worker_games)
                offsets = {}
                for game_file in worker_games:
                    game_id = _stable_game_id(game_file)
                    offsets[game_id] = replica_counts.get(game_id, 0)
                    replica_counts[game_id] = offsets[game_id] + per_game
                worker_replica_offsets.append(offsets)
            else:
                worker_replica_offsets.append({})
        if fixed_balanced and set(replica_counts.values()) != {int(args.group_size)}:
            raise RuntimeError(f"fixed manifest replica allocation is not balanced: {replica_counts}")

        ctx = mp.get_context("spawn")
        dp_out_q = ctx.Queue()
        args_dict = vars(args).copy()
        args_dict["tensor_parallel_size"] = 1
        for wid, (gpu_id, (worker_games, worker_bs), replica_offsets) in enumerate(zip(worker_gpus, worker_plan, worker_replica_offsets)):
            in_q = ctx.Queue(maxsize=2)
            proc = ctx.Process(
                target=_dp_rollout_worker,
                args=(wid, gpu_id, args_dict, worker_games, all_tids, worker_bs, replica_offsets, in_q, dp_out_q, resume_state["completed_groups"]),
                daemon=False,
            )





            parent_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                proc.start()
            finally:
                if parent_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_visible
            dp_in_queues.append(in_q)
            dp_processes.append(proc)
        print(f"[driver] data_parallel_workers={data_parallel_workers} gpus={worker_gpus} worker_batch_sizes={dp_worker_batch_sizes} fixed_manifest_balanced={fixed_balanced}; main process will aggregate trajectories and run cloud updates")
    else:



        dp_worker_batch_sizes = [batch_rollout_size]
        vllm_max_num_seqs = int(args.vllm_max_num_seqs or batch_rollout_size)
        if vllm_max_num_seqs < batch_rollout_size:
            raise ValueError(f"vllm_max_num_seqs={vllm_max_num_seqs} is smaller than the single replica rollout batch={batch_rollout_size}")
        vllm_max_num_seqs_by_worker = [vllm_max_num_seqs]
        if batch_rollout_size == 1:
            env = make_single_env(all_game_files, config, seed=args.seed)
        else:
            env = make_batch_env(all_game_files, config, batch_size=batch_rollout_size, seed=args.seed)
            print(f"[driver] batch_rollout_size={batch_rollout_size}: cloud updates are checked only after a full rollout group finishes")

        if resume_state["completed_groups"] > 0:



            _t0 = time.time()
            for _ in range(resume_state["completed_groups"]):
                env.reset()
            print(f"[driver][resume] skipped {resume_state['completed_groups']} reset() calls to align with prior run ({time.time() - _t0:.1f}s)")


        from agent_system.frozen_executor.vllm_agent import VLLMAgent

        agent = VLLMAgent(
            model_path=args.model_path,
            gpu_memory_utilization=args.gpu_mem_util,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=not args.no_thinking,
            seed=args.seed,
            no_wait=args.nowait,
            think_budget=args.think_budget,
            max_num_seqs=vllm_max_num_seqs,
            enforce_eager=bool(args.vllm_enforce_eager),
        )
        print(f"[driver] single vLLM replica batch={batch_rollout_size} TP={args.tensor_parallel_size} max_num_seqs={vllm_max_num_seqs} enforce_eager={bool(args.vllm_enforce_eager)}")



        builder = AlfWorldObsBuilder(mem_lib=skill_lib, with_skills=enable_coskill, top_k=args.top_k, history_length=args.history_length)
    print(f"[driver] retrieval_mode={args.retrieval_mode} with_skills={enable_coskill} enable_skill_tree={enable_skill_tree} enable_skill_tree_evolve={enable_skill_tree_evolve} enable_failure_analysis={bool(args.enable_failure_analysis)}")

    group_metrics_path = os.path.join(args.outdir, "group_metrics.jsonl")
    if resume_state["resume"]:
        _trim_uncheckpointed_group_metrics(
            group_metrics_path,
            int(resume_state["completed_groups"]),
        )
    trace_compression_metrics = _trace_compression_metric_fields(args)




    per_game = list(resume_state["per_game"])
    wins = resume_state["wins"]
    global_step = len(per_game)
    tt_stats = {}
    cloud_updates = int(resume_state.get("cloud_updates", 0) or 0)
    cloud_update_steps = list(resume_state.get("cloud_update_steps", []) or [])
    small_model_token_totals = dict(resume_state.get("small_model_tokens", {}))
    context_guard_totals = dict(
        resume_state.get(
            "context_guard",
            {"prompt_trims": 0, "trimmed_tokens": 0},
        )
    )
    small_model_token_accounting = resume_state.get("small_model_token_accounting", SMALL_MODEL_TOKEN_ACCOUNTING)
    large_model_token_offset = dict(resume_state.get("large_model_tokens", {}))
    small_model_token_totals_by_tt = {tt: dict(resume_state.get("small_model_tokens_by_tt", {}).get(tt, {"prompt": 0, "response": 0, "total": 0})) for tt in NORL_TASK_TYPES}
    large_model_token_offset_by_tt = {tt: dict(resume_state.get("large_model_tokens_by_tt", {}).get(tt, {"prompt": 0, "completion": 0, "total": 0})) for tt in NORL_TASK_TYPES}
    large_model_token_offset_mixed = dict(resume_state.get("large_model_tokens_mixed", {"prompt": 0, "completion": 0, "total": 0}))
    large_model_usage_offset = dict(resume_state.get("large_model_usage", {}) or {})
    large_model_usage_offset_by_tt = _canonicalize_count_breakdown(large_model_usage_offset.get("missing_calls_by_task_type", {}) or {})






    def _ingest_episode_result(epoch, ep_result):
        nonlocal global_step, wins
        global_step += 1
        won = ep_result["won"]
        used = ep_result["used"]
        raw_trace = ep_result["raw_trace"]
        tt_detected = _canonical_task_type(ep_result["task_type"])
        injected = ep_result["injected"]
        nval = ep_result["n_valid"]
        logrows = ep_result["logrows"]
        ep_tokens = ep_result.get("small_model_tokens") or {"prompt": 0, "response": 0, "total": 0}
        action_meta = raw_trace.get("meta") or {}
        n_strict_valid = int(action_meta.get("n_strict_valid_actions", 0) or 0)
        n_non_strict_valid = int(action_meta.get("n_non_strict_valid_actions", nval) or 0)
        n_salvaged = int(action_meta.get("n_salvaged_actions", 0) or 0)
        n_fallback = int(action_meta.get("n_fallback_actions", 0) or 0)
        pb_rec = ep_result.get("playbook_record")
        wins += int(won)

        traces_pool.add_trace(raw_trace)
        if hasattr(skill_lib, "record_usage"):
            skill_lib.record_usage(injected, success=won, task_type=tt_detected)
        if hasattr(skill_lib, "record_playbook_usage"):
            skill_lib.record_playbook_usage(tt_detected, success=won)

        ts = tt_stats.setdefault(tt_detected, {"episodes": 0, "wins": 0})
        ts["episodes"] += 1
        ts["wins"] += int(won)

        ep_record = {
            "epoch": epoch + 1,
            "detected_type": tt_detected,
            "won": bool(won),
            "used_steps": used,
            "valid_actions": nval,
            "step": global_step,
            "valid_action_ratio": round(nval / max(used, 1), 6),
            "non_strict_valid_actions": n_non_strict_valid,
            "non_strict_valid_action_ratio": round(n_non_strict_valid / max(used, 1), 6),
            "strict_valid_actions": n_strict_valid,
            "strict_valid_action_ratio": round(n_strict_valid / max(used, 1), 6),
            "salvaged_actions": n_salvaged,
            "fallback_actions": n_fallback,
            "cloud_round_used": cloud_updates,
            "skill_tree_enabled": enable_skill_tree,
            "skill_tree_evolve_enabled": enable_skill_tree_evolve,
            "skill_bullets_enabled": enable_coskill,
            "skill_ids_used": list(injected),
            "running_total_episodes": len(per_game) + 1,
            "running_total_wins": wins,
            "task_type_episodes": ts["episodes"],
            "task_type_wins": ts["wins"],
            "tokens_prompt": int(ep_tokens.get("prompt", 0) or 0),
            "tokens_response": int(ep_tokens.get("response", 0) or 0),
            "tokens_total": int(ep_tokens.get("total", 0) or 0),
        }
        per_game.append(ep_record)
        print(f"[driver] step={global_step} {tt_detected} won={won} steps={used}")

        if args.log_trajectories:
            _dump_episode(args.outdir, global_step, raw_trace["task"], tt_detected, won, used, logrows, pb_rec, injected)
        return ep_record, pb_rec

    def _large_model_token_usage():
        analyzer = getattr(cloud_loop, "cloud_analyzer", None)
        zero_by_tt = {tt: {"prompt": 0, "completion": 0, "total": 0} for tt in NORL_TASK_TYPES}
        if analyzer is None:
            return {
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "by_task_type": zero_by_tt,
                "mixed": {"prompt": 0, "completion": 0, "total": 0},
                "usage": {
                    "reported_calls": int(large_model_usage_offset.get("reported_calls", 0) or 0),
                    "missing_calls": int(large_model_usage_offset.get("missing_calls", 0) or 0),
                    "missing_calls_by_task_type": large_model_usage_offset_by_tt,
                    "missing_calls_mixed": int(large_model_usage_offset.get("missing_calls_mixed", 0) or 0),
                },
            }
        prompt = int(getattr(analyzer, "total_prompt_tokens", 0) or 0)
        completion = int(getattr(analyzer, "total_completion_tokens", 0) or 0)
        by_tt_prompt = _canonicalize_token_breakdown(
            {tt: {"prompt": value} for tt, value in (getattr(analyzer, "total_prompt_tokens_by_task_type", {}) or {}).items()},
            ("prompt",),
        )
        by_tt_completion = _canonicalize_token_breakdown(
            {tt: {"completion": value} for tt, value in (getattr(analyzer, "total_completion_tokens_by_task_type", {}) or {}).items()},
            ("completion",),
        )
        by_task_type = {}
        for tt in NORL_TASK_TYPES:
            p = int(large_model_token_offset_by_tt.get(tt, {}).get("prompt", 0) or 0) + int(by_tt_prompt[tt]["prompt"] or 0)
            c = int(large_model_token_offset_by_tt.get(tt, {}).get("completion", 0) or 0) + int(by_tt_completion[tt]["completion"] or 0)
            by_task_type[tt] = {"prompt": p, "completion": c, "total": p + c}
        mixed_prompt = int(large_model_token_offset_mixed.get("prompt", 0) or 0) + int(getattr(analyzer, "total_prompt_tokens_mixed", 0) or 0)
        mixed_completion = int(large_model_token_offset_mixed.get("completion", 0) or 0) + int(getattr(analyzer, "total_completion_tokens_mixed", 0) or 0)
        missing_by_tt = _canonicalize_count_breakdown(getattr(analyzer, "usage_missing_calls_by_task_type", {}) or {})
        for tt in NORL_TASK_TYPES:
            missing_by_tt[tt] += large_model_usage_offset_by_tt[tt]
        return {
            "prompt": int(large_model_token_offset.get("prompt", 0) or 0) + prompt,
            "completion": int(large_model_token_offset.get("completion", 0) or 0) + completion,
            "total": int(large_model_token_offset.get("total", 0) or 0) + prompt + completion,
            "by_task_type": by_task_type,
            "mixed": {
                "prompt": mixed_prompt,
                "completion": mixed_completion,
                "total": mixed_prompt + mixed_completion,
            },
            "usage": {
                "reported_calls": int(large_model_usage_offset.get("reported_calls", 0) or 0) + int(getattr(analyzer, "usage_reported_calls", 0) or 0),
                "missing_calls": int(large_model_usage_offset.get("missing_calls", 0) or 0) + int(getattr(analyzer, "usage_missing_calls", 0) or 0),
                "missing_calls_by_task_type": missing_by_tt,
                "missing_calls_mixed": int(large_model_usage_offset.get("missing_calls_mixed", 0) or 0) + int(getattr(analyzer, "usage_missing_calls_mixed", 0) or 0),
            },
        }

    def _context_guard_usage():
        if agent is not None:
            current = agent.get_context_guard_usage()
            return {key: int(context_guard_totals.get(key, 0) or 0) + int(current.get(key, 0) or 0) for key in context_guard_totals}
        return dict(context_guard_totals)

    def _write_group_metric(group_id, epoch, ingested, generated_count, fired, small_tokens, large_before, large_after, rollout_seconds, cloud_seconds, total_seconds):
        ""
        records = [record for record, _ in ingested]
        n = len(records)
        group_wins = sum(int(record["won"]) for record in records)
        lengths = [int(record["used_steps"]) for record in records]
        valid_actions = sum(int(record["valid_actions"]) for record in records)
        strict_valid_actions = sum(int(record["strict_valid_actions"]) for record in records)
        salvaged_actions = sum(int(record["salvaged_actions"]) for record in records)
        fallback_actions = sum(int(record["fallback_actions"]) for record in records)
        action_count = sum(lengths)
        action_count_cumulative = sum(int(row.get("used_steps", 0) or 0) for row in per_game)

        for key in ("prompt", "response", "total"):
            small_model_token_totals[key] += int(small_tokens.get(key, 0) or 0)

        small_by_tt = {tt: {"prompt": 0, "response": 0, "total": 0} for tt in NORL_TASK_TYPES}
        for record in records:
            tt = _canonical_task_type(record["detected_type"])
            bucket = small_by_tt[tt]
            bucket["prompt"] += int(record.get("tokens_prompt", 0) or 0)
            bucket["response"] += int(record.get("tokens_response", 0) or 0)
            bucket["total"] += int(record.get("tokens_total", 0) or 0)
        for tt in small_by_tt:
            totals = small_model_token_totals_by_tt.setdefault(tt, {"prompt": 0, "response": 0, "total": 0})
            for key in ("prompt", "response", "total"):
                totals[key] += small_by_tt[tt][key]

        small_by_tt_total = sum(bucket["total"] for bucket in small_by_tt.values())
        if small_by_tt_total != int(small_tokens.get("total", 0) or 0):
            raise RuntimeError(f"small-model token accounting mismatch: group total {small_tokens.get('total', 0)} != canonical task sum {small_by_tt_total}")

        large_delta = {
            "prompt": max(0, large_after["prompt"] - large_before["prompt"]),
            "completion": max(0, large_after["completion"] - large_before["completion"]),
        }
        large_delta["total"] = large_delta["prompt"] + large_delta["completion"]

        large_delta_by_tt = {}
        for tt in NORL_TASK_TYPES:
            p = max(0, large_after["by_task_type"][tt]["prompt"] - large_before["by_task_type"][tt]["prompt"])
            c = max(0, large_after["by_task_type"][tt]["completion"] - large_before["by_task_type"][tt]["completion"])
            large_delta_by_tt[tt] = {"prompt": p, "completion": c, "total": p + c}
        large_delta_mixed = {
            "prompt": max(0, large_after["mixed"]["prompt"] - large_before["mixed"]["prompt"]),
            "completion": max(0, large_after["mixed"]["completion"] - large_before["mixed"]["completion"]),
        }
        large_delta_mixed["total"] = large_delta_mixed["prompt"] + large_delta_mixed["completion"]
        large_usage_before = large_before.get("usage", {}) or {}
        large_usage_after = large_after.get("usage", {}) or {}
        large_missing_delta = max(0, int(large_usage_after.get("missing_calls", 0) or 0) - int(large_usage_before.get("missing_calls", 0) or 0))

        context_guard = _context_guard_usage()
        metrics = {
            "record/type": "train_update",
            "training/epoch": epoch + 1,
            "episode/count": n,
            "episode/generated_count": int(generated_count),
            "episode/wins": group_wins,
            "episode/success_rate": round(group_wins / max(n, 1), 6),
            "episode/wins_cumulative": wins,
            "episode/action_count": action_count,
            "episode/action_count_cumulative": action_count_cumulative,
            "episode/length/mean": round(sum(lengths) / max(n, 1), 6),
            "episode/length/max": max(lengths) if lengths else 0,
            "episode/length/min": min(lengths) if lengths else 0,
            "episode/valid_action_ratio": round(valid_actions / max(action_count, 1), 6),
            "episode/strict_valid_action_ratio": round(strict_valid_actions / max(action_count, 1), 6),
            "episode/non_strict_valid_action_ratio": round(valid_actions / max(action_count, 1), 6),
            "episode/salvaged_action_ratio": round(salvaged_actions / max(action_count, 1), 6),
            "episode/fallback_action_ratio": round(fallback_actions / max(action_count, 1), 6),
            "experiment/skill_tree_enabled": int(enable_skill_tree),
            "experiment/skill_tree_evolve_enabled": int(enable_skill_tree_evolve),
            "experiment/skill_bullets_enabled": int(enable_coskill),
            **trace_compression_metrics,
            "experiment/rl_enabled": 0,
            "experiment/tree_rl_internalize_enabled": 0,
            "experiment/cloud_round": cloud_updates,
            "experiment/max_model_len": int(args.max_model_len),
            "experiment/max_response_tokens": int(args.max_tokens),
            "experiment/max_prompt_tokens": int(args.max_model_len - args.max_tokens),
            "coskill/cloud_update_fired": bool(fired),
            "tokens/small_model/prompt": int(small_tokens.get("prompt", 0) or 0),
            "tokens/small_model/response": int(small_tokens.get("response", 0) or 0),
            "tokens/small_model/total": int(small_tokens.get("total", 0) or 0),
            "tokens/small_model/accounting": SMALL_MODEL_TOKEN_ACCOUNTING,
            "tokens/small_model/context_guard/prompt_trims_cumulative": (context_guard["prompt_trims"]),
            "tokens/small_model/context_guard/trimmed_tokens_cumulative": (context_guard["trimmed_tokens"]),
            "tokens/small_model/context_guard/any_prompt_trim": int(context_guard["prompt_trims"] > 0),
            "tokens/small_model/prompt_cumulative": small_model_token_totals["prompt"],
            "tokens/small_model/response_cumulative": small_model_token_totals["response"],
            "tokens/small_model/total_cumulative": small_model_token_totals["total"],
            "tokens/small_model/by_task_type/total_reconciled": small_by_tt_total,
            "tokens/small_model/by_task_type/reconciliation_error": (int(small_tokens.get("total", 0) or 0) - small_by_tt_total),
            "tokens/large_model/prompt": large_delta["prompt"],
            "tokens/large_model/completion": large_delta["completion"],
            "tokens/large_model/total": large_delta["total"],
            "tokens/large_model/accounting": "provider_api_usage",
            "tokens/large_model/prompt_cumulative": large_after["prompt"],
            "tokens/large_model/completion_cumulative": large_after["completion"],
            "tokens/large_model/total_cumulative": large_after["total"],
            "tokens/large_model/usage_missing_calls": large_missing_delta,
            "tokens/large_model/usage_missing_calls_cumulative": int(large_usage_after.get("missing_calls", 0) or 0),
            "tokens/large_model/usage_reported_calls_cumulative": int(large_usage_after.get("reported_calls", 0) or 0),
            "timing_s/rollout": round(float(rollout_seconds), 6),
            "timing_s/cloud_update": round(float(cloud_seconds), 6),
            "timing_s/group_total": round(float(total_seconds), 6),
            "perf/total_num_tokens": int(small_tokens.get("total", 0) or 0),
            "perf/throughput_episodes_per_second": round(n / max(rollout_seconds, 1e-9), 6),
            "perf/throughput_small_tokens_per_second": round(int(small_tokens.get("total", 0) or 0) / max(rollout_seconds, 1e-9), 6),
            "comparison/schema_version": 3,
            "comparison/method": "coskill",
            "comparison/benchmark": "alfworld",
            "comparison/rollout_accounting": "active_env_decisions",
            "comparison/timing_cloud_update_measured": 1,
            **cloud_loop.metrics(traces_pool, skill_lib),
        }
        by_type = {}
        for record in records:
            tt = record["detected_type"]
            stat = by_type.setdefault(tt, {"episodes": 0, "wins": 0})
            stat["episodes"] += 1
            stat["wins"] += int(record["won"])
        for tt in NORL_TASK_TYPES:
            stat = by_type.get(tt, {"episodes": 0, "wins": 0})
            cumulative = tt_stats.get(tt, {"episodes": 0, "wins": 0})
            prefix = f"episode/by_task_type/{tt}"
            metrics[f"{prefix}/count"] = stat["episodes"]
            metrics[f"{prefix}/wins"] = stat["wins"]
            metrics[f"{prefix}/success_rate"] = round(
                stat["wins"] / max(stat["episodes"], 1), 6)
            metrics[f"{prefix}/count_cumulative"] = cumulative["episodes"]
            metrics[f"{prefix}/wins_cumulative"] = cumulative["wins"]
            metrics[f"{prefix}/success_rate_cumulative"] = round(
                cumulative["wins"] / max(cumulative["episodes"], 1), 6)

        for tt in NORL_TASK_TYPES:
            metrics[f"tokens/small_model/by_task_type/{tt}/prompt"] = small_by_tt[tt]["prompt"]
            metrics[f"tokens/small_model/by_task_type/{tt}/response"] = small_by_tt[tt]["response"]
            metrics[f"tokens/small_model/by_task_type/{tt}/total"] = small_by_tt[tt]["total"]
            metrics[f"tokens/small_model/by_task_type/{tt}/total_cumulative"] = small_model_token_totals_by_tt[tt]["total"]
            metrics[f"tokens/large_model/by_task_type/{tt}/prompt"] = large_delta_by_tt[tt]["prompt"]
            metrics[f"tokens/large_model/by_task_type/{tt}/completion"] = large_delta_by_tt[tt]["completion"]
            metrics[f"tokens/large_model/by_task_type/{tt}/total"] = large_delta_by_tt[tt]["total"]
            metrics[f"tokens/large_model/by_task_type/{tt}/total_cumulative"] = large_after["by_task_type"][tt]["total"]
            metrics[f"tokens/large_model/by_task_type/{tt}/usage_missing_calls_cumulative"] = int((large_usage_after.get("missing_calls_by_task_type", {}) or {}).get(tt, 0) or 0)



        metrics["tokens/large_model/mixed/prompt"] = large_delta_mixed["prompt"]
        metrics["tokens/large_model/mixed/completion"] = large_delta_mixed["completion"]
        metrics["tokens/large_model/mixed/total"] = large_delta_mixed["total"]
        metrics["tokens/large_model/mixed/total_cumulative"] = large_after["mixed"]["total"]
        metrics["tokens/large_model/mixed/accounting"] = "provider_api_usage_mixed_task_types"
        metrics["tokens/large_model/mixed/usage_missing_calls_cumulative"] = int(large_usage_after.get("missing_calls_mixed", 0) or 0)
        large_attributed = (
            sum(large_delta_by_tt[tt]["total"] for tt in NORL_TASK_TYPES)
            + large_delta_mixed["total"]
        )
        metrics["tokens/large_model/attribution/total_reconciled"] = large_attributed
        metrics["tokens/large_model/attribution/reconciliation_error"] = (
            large_delta["total"] - large_attributed
        )

        canonical_record = {
            "step": group_id,
            "metrics": metrics,
        }
        _append_jsonl(group_metrics_path, canonical_record)
        print(
            f"[driver] group{group_id} metric: episodes={n} wins={group_wins} "
            f"success={100.0 * group_wins / max(n, 1):.1f}% "
            f"valid_action={100.0 * valid_actions / max(action_count, 1):.1f}% "
            f"strict_valid_action={100.0 * strict_valid_actions / max(action_count, 1):.1f}% "
            f"small_tokens={small_tokens.get('total', 0)} "
            f"large_tokens={large_delta['total']} rollout={rollout_seconds:.1f}s"
        )

    def _atomic_json_dump(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _skill_tree_snapshot():
        trees = {}
        if enable_skill_tree and hasattr(skill_lib, "task_playbooks"):
            for tt, rec in (getattr(skill_lib, "task_playbooks", {}) or {}).items():
                if isinstance(rec, dict):
                    trees[tt] = {
                        "version": rec.get("version", 0),
                        "level": rec.get("level"),
                        "n_nodes": len(rec.get("nodes") or {}),
                    }
        return trees

    def _build_summary(status="running", checkpoint_reason=None, completed_groups=0):
        n = len(per_game)
        trees = _skill_tree_snapshot()
        small_by_tt_total = sum(int(usage.get("total", 0) or 0) for usage in small_model_token_totals_by_tt.values())
        small_reconciliation_error = int(small_model_token_totals.get("total", 0) or 0) - small_by_tt_total
        return {
            "status": status,
            "checkpoint_reason": checkpoint_reason,
            "task_types": task_types,
            "epochs": args.epochs,
            "num_games_per_type": args.num_games,
            "group_size": args.group_size,
            "batch_rollout_size": batch_rollout_size,
            "data_parallel_workers": data_parallel_workers,
            "data_parallel_worker_batch_sizes": dp_worker_batch_sizes,
            "vllm_max_num_seqs": int(args.vllm_max_num_seqs or max(vllm_max_num_seqs_by_worker)),
            "vllm_max_num_seqs_by_worker": vllm_max_num_seqs_by_worker,
            "vllm_enforce_eager": bool(args.vllm_enforce_eager),
            "max_model_len": int(args.max_model_len),
            "max_response_tokens": int(args.max_tokens),
            "max_prompt_tokens": int(args.max_model_len - args.max_tokens),
            "checkpoint_every_groups": args.checkpoint_every_groups,
            "completed_rollout_groups": completed_groups,
            "fixed_games_manifest": args.fixed_games_manifest,
            "fixed_game_files": all_game_files if args.fixed_games_manifest else [],
            "skill_tree_enabled": enable_skill_tree,
            "skill_tree_evolve_enabled": enable_skill_tree_evolve,
            "skill_bullets_enabled": enable_coskill,
            "cloud_updates_enabled": bool(args.enable_cloud_updates),
            "required_tree_depth": int(args.required_tree_depth or 0),
            "tree_depth_repair_attempts": int(args.tree_depth_repair_attempts),
            "trace_compression": {
                "enable_loop_filter": bool(args.trace_enable_loop_filter),
                "enable_obs_delta": bool(args.trace_enable_obs_delta),
                "enable_prefix_tree": bool(args.trace_enable_prefix_tree),
                "enable_consensus_prefix": bool(args.trace_enable_consensus_prefix),
                "cloud_evidence_mode": args.trace_cloud_evidence_mode,
                "accounting": "chars_div_4",
            },
            "cloud_update_every": args.cloud_update_every,
            "cloud_update_steps": cloud_update_steps,
            "total_games_combined_pool": total_games,
            "total_episodes": n,
            "wins": wins,
            "success_rate": round(wins / max(n, 1), 4),
            "skill_tree_versions": {tt: rec["version"] for tt, rec in trees.items()},
            "skill_tree_nodes": {tt: rec["n_nodes"] for tt, rec in trees.items()},

            "skill_tree_version": max([rec["version"] for rec in trees.values()] or [0]),
            "token_usage": {
                "small_model": {
                    **dict(small_model_token_totals),
                    "accounting": small_model_token_accounting,
                    "by_task_type": {tt: dict(small_model_token_totals_by_tt[tt]) for tt in NORL_TASK_TYPES},
                    "by_task_type_total": small_by_tt_total,
                    "by_task_type_reconciliation_error": small_reconciliation_error,
                },
                "large_model": _large_model_token_usage(),
            },
            "context_guard": {
                **_context_guard_usage(),
                "protocol_valid": _context_guard_usage()["prompt_trims"] == 0,
            },
            "final_coskill_metrics": cloud_loop.metrics(traces_pool, skill_lib),
            "phase_stats": _phase_stats(per_game),
            "per_game": per_game,
        }

    def _save_progress_checkpoint(reason, completed_groups, epoch=None, ep_i=None):
        summary = _build_summary(
            status="running",
            checkpoint_reason=reason,
            completed_groups=completed_groups,
        )
        if epoch is not None:
            summary["current_epoch"] = epoch + 1
        if ep_i is not None:
            summary["next_episode_index_in_epoch"] = ep_i + 1

        skill_snapshot = None
        if hasattr(skill_lib, "save_skills"):
            save_dir = os.path.join(args.outdir, "skill_lib")
            os.makedirs(save_dir, exist_ok=True)
            skill_snapshot = os.path.join(save_dir, f"skills_checkpoint_step{global_step}.json")
            skill_lib.save_skills(skill_snapshot)
            skill_lib.save_skills(os.path.join(save_dir, "skills_latest_checkpoint.json"))
            summary["skill_lib_checkpoint"] = skill_snapshot

        _atomic_json_dump(summary, os.path.join(args.outdir, "summary_partial.json"))
        ckpt_path = os.path.join(args.outdir, "checkpoints", f"step{global_step:06d}.json")
        _atomic_json_dump(summary, ckpt_path)
        print(f"[driver] checkpoint saved at step={global_step} groups={completed_groups} reason={reason} summary={ckpt_path} skill_lib={skill_snapshot or '<none>'}")

    def _save_rollout_skill_snapshot():
        save_dir = os.path.join(args.outdir, "skill_lib")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"skills_rollout_step{global_step}.json")
        skill_lib.save_skills(path)
        skill_lib.save_skills(os.path.join(save_dir, "skills_latest_rollout.json"))
        return path

    def _run_data_parallel_group(group_id):
        skill_path = _save_rollout_skill_snapshot()
        for in_q in dp_in_queues:
            in_q.put({"group_id": group_id, "skill_path": skill_path})

        replies = []
        for _ in dp_in_queues:
            msg = dp_out_q.get()
            if msg.get("error"):
                raise RuntimeError(f"data-parallel rollout worker {msg.get('worker_id')} failed:\n{msg.get('error')}")
            replies.append(msg)
        replies.sort(key=lambda x: x["worker_id"])
        group_results = []
        small_tokens = {"prompt": 0, "response": 0, "total": 0}
        for msg in replies:
            group_results.extend(msg["results"])
            usage = msg.get("small_model_tokens") or {}
            for key in small_tokens:
                small_tokens[key] += int(usage.get(key, 0) or 0)
            guard_usage = msg.get("context_guard") or {}
            for key in context_guard_totals:
                context_guard_totals[key] += int(guard_usage.get(key, 0) or 0)
        return group_results, small_tokens

    def _shutdown_data_parallel_workers():
        for in_q in dp_in_queues:
            try:
                in_q.put(None)
            except Exception:
                pass
        for proc in dp_processes:
            try:
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass

    if use_data_parallel:
        atexit.register(_shutdown_data_parallel_workers)

    stop_early = False
    completed_groups = resume_state["completed_groups"]
    for epoch in range(resume_state["epoch0"], args.epochs):
        if stop_early:
            break

        ep_i = resume_state["ep_i0"] if epoch == resume_state["epoch0"] else 0
        while ep_i < episodes_per_epoch:
            if args.max_episodes > 0 and global_step >= args.max_episodes:
                print(f"[driver] max_episodes={args.max_episodes} reached, stopping early")
                stop_early = True
                break

            if use_data_parallel:
                remaining_epoch = episodes_per_epoch - ep_i
                remaining_cap = (args.max_episodes - global_step) if args.max_episodes > 0 else remaining_epoch
                n_to_count = min(batch_rollout_size, remaining_epoch, remaining_cap)
                if n_to_count <= 0:
                    stop_early = True
                    break

                group_id = completed_groups + 1
                print(f"[driver] dispatch dp group{group_id} epoch{epoch + 1} group_ep{ep_i + 1}-{ep_i + n_to_count}/{episodes_per_epoch} (global_step={global_step + 1})")
                group_started = time.time()
                group_results, small_tokens = _run_data_parallel_group(group_id)
                rollout_seconds = time.time() - group_started
                ingested = []
                for ep_result in group_results[:n_to_count]:
                    ingested.append(_ingest_episode_result(epoch, ep_result))
                ep_i += n_to_count



                force_reason = None
                if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                    force_reason = f"episode_interval_{args.cloud_update_every}"
                large_before = _large_model_token_usage()
                cloud_started = time.time()
                fired = cloud_loop.maybe_update(traces_pool, skill_lib, global_step, force_reason=force_reason) if args.enable_cloud_updates else False
                cloud_seconds = time.time() - cloud_started
                large_after = _large_model_token_usage()
                if fired:
                    cloud_updates += 1
                    cloud_update_steps.append(global_step)

                for j, (ep_record, _pb_used) in enumerate(ingested):
                    is_last = j == len(ingested) - 1
                    ep_record["cloud_update_fired_after_episode"] = bool(fired and is_last)

                _write_group_metric(
                    group_id,
                    epoch,
                    ingested,
                    len(group_results),
                    fired,
                    small_tokens,
                    large_before,
                    large_after,
                    rollout_seconds,
                    cloud_seconds,
                    time.time() - group_started,
                )

                completed_groups += 1
                if args.checkpoint_every_groups > 0 and completed_groups % args.checkpoint_every_groups == 0:
                    _save_progress_checkpoint(
                        reason=f"group_interval_{args.checkpoint_every_groups}",
                        completed_groups=completed_groups,
                        epoch=epoch,
                        ep_i=ep_i,
                    )
                continue

            if batch_rollout_size == 1:
                group_id = completed_groups + 1
                group_started = time.time()
                small_before = agent.get_token_usage()
                tag = f" epoch{epoch + 1} ep{ep_i + 1}/{episodes_per_epoch} (global_step={global_step + 1})"
                won, used, raw_trace, tt_detected, injected, nval, logrows = rollout_episode(env, agent, builder, args.max_steps, tag=tag, keep_logrows=bool(args.log_trajectories))
                rollout_seconds = time.time() - group_started
                small_tokens = _token_delta(agent.get_token_usage(), small_before)
                pb_rec = skill_lib.get_playbook_record(tt_detected) if enable_skill_tree and hasattr(skill_lib, "get_playbook_record") else None
                ep_result = {"won": won, "used": used, "raw_trace": raw_trace, "task_type": tt_detected, "injected": injected, "n_valid": nval, "logrows": logrows, "playbook_record": pb_rec, "small_model_tokens": small_tokens}
                ep_record, pb_used = _ingest_episode_result(epoch, ep_result)
                ep_i += 1

                force_reason = None
                if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                    force_reason = f"episode_interval_{args.cloud_update_every}"
                large_before = _large_model_token_usage()
                cloud_started = time.time()
                fired = cloud_loop.maybe_update(traces_pool, skill_lib, global_step, force_reason=force_reason) if args.enable_cloud_updates else False
                cloud_seconds = time.time() - cloud_started
                large_after = _large_model_token_usage()
                ep_record["cloud_update_fired_after_episode"] = bool(fired)
                if fired:
                    cloud_updates += 1
                    cloud_update_steps.append(global_step)
                _write_group_metric(
                    group_id,
                    epoch,
                    [(ep_record, pb_used)],
                    1,
                    fired,
                    small_tokens,
                    large_before,
                    large_after,
                    rollout_seconds,
                    cloud_seconds,
                    time.time() - group_started,
                )
                completed_groups += 1
                if args.checkpoint_every_groups > 0 and completed_groups % args.checkpoint_every_groups == 0:
                    _save_progress_checkpoint(
                        reason=f"group_interval_{args.checkpoint_every_groups}",
                        completed_groups=completed_groups,
                        epoch=epoch,
                        ep_i=ep_i,
                    )
                continue

            remaining_epoch = episodes_per_epoch - ep_i
            remaining_cap = (args.max_episodes - global_step) if args.max_episodes > 0 else remaining_epoch
            n_to_count = min(batch_rollout_size, remaining_epoch, remaining_cap)
            if n_to_count <= 0:
                stop_early = True
                break

            tag = f" epoch{epoch + 1} group_ep{ep_i + 1}-{ep_i + n_to_count}/{episodes_per_epoch} (global_step={global_step + 1})"
            group_id = completed_groups + 1
            group_started = time.time()
            small_before = agent.get_token_usage()
            group_results = rollout_batch_group(env, agent, skill_lib, args, batch_rollout_size, tag=tag)
            rollout_seconds = time.time() - group_started
            small_tokens = _token_delta(agent.get_token_usage(), small_before)
            ingested = []
            for ep_result in group_results[:n_to_count]:
                ingested.append(_ingest_episode_result(epoch, ep_result))
            ep_i += n_to_count



            force_reason = None
            if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                force_reason = f"episode_interval_{args.cloud_update_every}"
            large_before = _large_model_token_usage()
            cloud_started = time.time()
            fired = cloud_loop.maybe_update(traces_pool, skill_lib, global_step, force_reason=force_reason) if args.enable_cloud_updates else False
            cloud_seconds = time.time() - cloud_started
            large_after = _large_model_token_usage()
            if fired:
                cloud_updates += 1
                cloud_update_steps.append(global_step)

            for j, (ep_record, _pb_used) in enumerate(ingested):
                is_last = j == len(ingested) - 1
                ep_record["cloud_update_fired_after_episode"] = bool(fired and is_last)

            _write_group_metric(
                group_id,
                epoch,
                ingested,
                len(group_results),
                fired,
                small_tokens,
                large_before,
                large_after,
                rollout_seconds,
                cloud_seconds,
                time.time() - group_started,
            )

            completed_groups += 1
            if args.checkpoint_every_groups > 0 and completed_groups % args.checkpoint_every_groups == 0:
                _save_progress_checkpoint(
                    reason=f"group_interval_{args.checkpoint_every_groups}",
                    completed_groups=completed_groups,
                    epoch=epoch,
                    ep_i=ep_i,
                )


    summary = _build_summary(status="done", completed_groups=completed_groups)
    if hasattr(skill_lib, "save_skills"):
        save_dir = os.path.join(args.outdir, "skill_lib")
        os.makedirs(save_dir, exist_ok=True)
        final_skill_path = os.path.join(save_dir, f"skills_final_step{global_step}.json")
        skill_lib.save_skills(final_skill_path)
        skill_lib.save_skills(os.path.join(save_dir, "skills_latest_final.json"))
        summary["skill_lib_checkpoint"] = final_skill_path
    _atomic_json_dump(summary, os.path.join(args.outdir, "summary.json"))
    if use_data_parallel:
        _shutdown_data_parallel_workers()
    n = summary["total_episodes"]
    print(f"\n[driver] done. episodes={n} success_rate={summary['success_rate'] * 100:.1f}% skill_tree_versions={summary['skill_tree_versions']}")
    print(f"[driver] outputs under {args.outdir}/ (traces_pool, cloud_io, skill_lib, trajectories)")


def _append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _dump_episode(outdir, episode_idx, task, detected_task_type, won, used_steps, logrows, playbook_record, skill_ids_used=None):
    ""
    d = os.path.join(outdir, "trajectories")
    os.makedirs(d, exist_ok=True)
    status = "WIN" if won else "FAIL"
    base = f"ep{episode_idx:04d}_{detected_task_type}_{status}_{used_steps}steps"

    pb_meta = None
    if playbook_record:
        pb_meta = {"version": playbook_record.get("version"), "level": playbook_record.get("level"), "n_nodes": len(playbook_record.get("nodes") or {})}


    payload = {
        "episode": episode_idx,
        "task": task,
        "task_type": detected_task_type,
        "outcome": status,
        "used_steps": used_steps,
        "skill_tree_used": pb_meta,
        "skill_ids_used": list(skill_ids_used or []),
        "steps": logrows,
    }
    try:
        with open(os.path.join(d, base + "_episode.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[driver] episode json dump failed: {e}")


    try:
        lines = ["=" * 78, f" Episode #{episode_idx}  task_type={detected_task_type}  [{status} / {used_steps} steps]", "=" * 78, f" task: {task}", f" skill tree: v{pb_meta['version']} level={pb_meta['level']} ({pb_meta['n_nodes']} nodes)" if pb_meta else " skill tree: (none)", "=" * 78]
        for s in logrows:
            flag = "OK" if s["valid"] else "INVALID"
            forced = " [BUDGET-FORCED]" if s.get("forced") else ""
            lines.append("")
            lines.append(f"-- step {s['step']:>2} [{flag}]{forced} action={s['action']!r} source={s.get('execution_source', 'unknown')} reward={s.get('reward', 0)} won={s.get('won', False)}")
            lines.append(f"   obs: {_oneline(s.get('obs', ''))}")
        with open(os.path.join(d, base + "_trajectory.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[driver] episode trajectory.txt dump failed: {e}")


    try:
        lines = ["#" * 78, f"# Episode #{episode_idx}: complete edge-model prompt for every step", f"# task: {task}  |  {status} / {used_steps} steps", "#" * 78]
        for s in logrows:
            lines.append("")
            lines.append("/" * 78)
            lines.append(f"// step {s['step']}  action={s['action']!r}  valid_action={s['valid']}  strict_valid_action={s.get('strict_valid_action', s['valid'])}  source={s.get('execution_source', 'unknown')}")
            lines.append("/" * 78)
            lines.append(s.get("prompt", ""))
        with open(os.path.join(d, base + "_prompts.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[driver] episode prompts.txt dump failed: {e}")


def _oneline(text):
    if not text:
        return "(empty)"
    return " ".join(str(text).split())


def _phase_stats(per_game):
    ""
    phases = {}
    for row in per_game:
        key = str(row.get("cloud_round_used", 0))
        phase = phases.setdefault(key, {"episodes": 0, "wins": 0, "by_task_type": {}})
        phase["episodes"] += 1
        phase["wins"] += int(row.get("won", False))
        tt = row.get("detected_type", "unknown")
        ts = phase["by_task_type"].setdefault(tt, {"episodes": 0, "wins": 0})
        ts["episodes"] += 1
        ts["wins"] += int(row.get("won", False))
    for phase in phases.values():
        phase["success_rate"] = round(phase["wins"] / max(phase["episodes"], 1), 4)
        for ts in phase["by_task_type"].values():
            ts["success_rate"] = round(ts["wins"] / max(ts["episodes"], 1), 4)
    return phases


if __name__ == "__main__":
    main()
