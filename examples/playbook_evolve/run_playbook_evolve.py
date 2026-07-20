"""run_playbook_evolve.py — 无 verl/Ray/FSDP 的独立 Skill Tree 进化 driver。

冻结小模型（vLLM 直接加载，只一份）在 ALFWorld 上 rollout → 轨迹进 TracesPool →
水位线触发 CoSkillCloudLoop（失败诊断 + skill tree 从零生成/层次化细化 + 可选 skill 蒸馏）
→ 进化后的 agent skill tree 写回同一个 skill_lib，下一局 reset() 即注入。

所有运行条件对齐 examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh
（模型 / embedding 检索 / 环境 / 记忆分层 / 云端 / 水位线 / group_size 采样），
仅去掉 RL 权重训练（无 Ray、无 FSDP、无第二份模型、无反向传播 / checkpoint）。

复用件：mini_test 的 env_utils（进程内单环境）、agent_vllm（单次完整响应生成）、
prod_prompt.ProdObsBuilder（与 env_manager.build_text_obs 逐字节对齐，注入共享 skill_lib）、
run_generic 的解析/兜底工具；闭环三件套 + HierarchicalSkillLib + CoSkillCloudLoop。
"""
import os
import atexit
import hashlib
import json
import uuid
import argparse
import multiprocessing as mp
import subprocess
import time
import traceback

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.memory import TracesPool, HierarchicalSkillLib, CoSkillCloudLoop

from mini_test_pen_shelf.env_utils import (
    load_tw_config_types, find_games_by_type, make_single_env, make_batch_env, _TASK_TYPE_TO_ID,
)
from mini_test_pen_shelf.prod_prompt import ProdObsBuilder
from mini_test_pen_shelf.run_generic import extract_task, parse_model_output


SMALL_MODEL_TOKEN_ACCOUNTING = "vllm_request_tokens_single_pass"

# Canonical no-RL task_type vocabulary for the per-subtask token breakdown.
# Sourced from ALFWorld's own ground-truth traj_data.json field (the same
# vocabulary find_games_by_type/_TASK_TYPE_TO_ID key on) rather than
# ray_trainer.py's KNOWN_TASK_TYPES: the RL side only has decoded prompt text
# to work with and detects task_type via regex heuristic, but the no-RL
# driver already retrieves the exact ground-truth task_type per episode
# (ProdObsBuilder.retrieved["task_type"]), so it uses its own real vocabulary
# instead of forcing the RL side's heuristic taxonomy onto it. "unknown" is
# included since retrieval falls back to it when unset.
NORL_TASK_TYPES = tuple(_TASK_TYPE_TO_ID.keys()) + ("unknown",)


def _load_fixed_games_manifest(manifest_path, alfworld_data=None):
    """读取固定 game manifest，返回 ``(games, task_types, task_type_ids)``。

    game 路径可写绝对路径，也可写相对 ``$ALFWORLD_DATA`` 的路径。每项都会核对
    sibling ``traj_data.json`` 的 task_type 和 game 的 solvable 标记，防止 A/B
    因路径写错而悄悄跑到不同任务。
    """
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
            raise ValueError(
                f"manifest game #{i} expected {expected}, got {task_type}: {game_file}"
            )
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
    """``--resume 1`` 时从上一次留在 ``args.outdir`` 里的 checkpoint 恢复技能库 +
    进度；否则返回等价于当前"从零开始"行为的默认状态。

    技能库恢复优先用 ``skill_lib/skills_latest_checkpoint.json``（每次
    ``--checkpoint_every_groups`` 触发都会整份覆盖写，schema 与 ``--skills_json``
    种子文件完全一致，见 ``SkillsOnlyMemory.save_skills``/``__init__``），其次退回
    ``skills_latest_rollout.json``（每个 rollout group 开始前都会写一次，更新更频繁）。
    进度（epoch/episode 位置、累计胜局、per_game 历史）从 ``summary_partial.json``
    里同一批 checkpoint 写的字段读回。
    """
    state = {
        "resume": False,
        "skills_json_path": args.skills_json,
        "epoch0": 0,
        "ep_i0": 0,
        "completed_groups": 0,
        "wins": 0,
        "per_game": [],
        "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
        "small_model_token_accounting": SMALL_MODEL_TOKEN_ACCOUNTING,
        "large_model_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "small_model_tokens_by_tt": {
            tt: {"prompt": 0, "response": 0, "total": 0} for tt in NORL_TASK_TYPES},
        "large_model_tokens_by_tt": {
            tt: {"prompt": 0, "completion": 0, "total": 0} for tt in NORL_TASK_TYPES},
        "large_model_tokens_mixed": {"prompt": 0, "completion": 0, "total": 0},
        "cloud_updates": 0,
        "cloud_update_steps": [],
    }
    if not args.resume:
        return state

    summary_path = os.path.join(args.outdir, "summary_partial.json")
    if not os.path.isfile(summary_path):
        print(f"[driver][resume] --resume 1 但 {summary_path} 不存在，"
              "视为全新运行（从 --skills_json 种子文件、episode 0 开始）")
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
        print(f"[driver][resume] 未在 {skill_dir} 找到任何技能库 checkpoint，"
              f"技能库仍从种子文件 {args.skills_json} 加载（其余进度照常恢复）")

    state["resume"] = True
    state["epoch0"] = max(0, int(prev_summary.get("current_epoch", 1)) - 1)
    state["ep_i0"] = max(0, int(prev_summary.get("next_episode_index_in_epoch", 1)) - 1)
    state["completed_groups"] = int(prev_summary.get("completed_rollout_groups", 0) or 0)
    state["wins"] = int(prev_summary.get("wins", 0) or 0)
    state["per_game"] = list(prev_summary.get("per_game", []) or [])
    saved_small_tokens = (prev_summary.get("token_usage", {}) or {}).get("small_model", {}) or {}
    state["small_model_tokens"] = {
        key: int(saved_small_tokens.get(key, 0) or 0)
        for key in ("prompt", "response", "total")
    }
    previous_accounting = saved_small_tokens.get("accounting")
    if not previous_accounting and state["small_model_tokens"]["total"]:
        previous_accounting = "vllm_request_tokens_two_stage"
    if previous_accounting and previous_accounting != SMALL_MODEL_TOKEN_ACCOUNTING:
        state["small_model_token_accounting"] = (
            f"mixed:{previous_accounting}+{SMALL_MODEL_TOKEN_ACCOUNTING}")
    saved_large_tokens = (prev_summary.get("token_usage", {}) or {}).get("large_model", {}) or {}
    state["large_model_tokens"] = {
        key: int(saved_large_tokens.get(key, 0) or 0)
        for key in ("prompt", "completion", "total")
    }
    saved_small_by_tt = (saved_small_tokens.get("by_task_type", {}) or {})
    state["small_model_tokens_by_tt"] = {
        tt: {
            key: int((saved_small_by_tt.get(tt, {}) or {}).get(key, 0) or 0)
            for key in ("prompt", "response", "total")
        } for tt in NORL_TASK_TYPES
    }
    saved_large_by_tt = (saved_large_tokens.get("by_task_type", {}) or {})
    state["large_model_tokens_by_tt"] = {
        tt: {
            key: int((saved_large_by_tt.get(tt, {}) or {}).get(key, 0) or 0)
            for key in ("prompt", "completion", "total")
        } for tt in NORL_TASK_TYPES
    }
    saved_large_mixed = (saved_large_tokens.get("mixed", {}) or {})
    state["large_model_tokens_mixed"] = {
        key: int(saved_large_mixed.get(key, 0) or 0)
        for key in ("prompt", "completion", "total")
    }
    state["cloud_update_steps"] = list(prev_summary.get("cloud_update_steps", []) or [])
    state["cloud_updates"] = len(state["cloud_update_steps"])

    print(f"[driver][resume] 从 {summary_path} 恢复：epoch={state['epoch0']+1} "
          f"ep_i={state['ep_i0']} completed_groups={state['completed_groups']} "
          f"wins={state['wins']}/{len(state['per_game'])} "
          f"skill_lib<-{state['skills_json_path']}")
    print("[driver][resume] 注意：游戏顺序由 TextWorld 内部 shuffled_cycle 决定，"
          "跳过已完成局数是靠重建 env 后空转对应次数 reset() 对齐——同一 seed + 同一批 "
          "game_files 下可复现，但不是数学上绝对保证的精确重放。")
    return state


def rollout_episode(env, agent, builder, max_steps, tag="", keep_logrows=True):
    """跑一局（冻结模型）。返回 (won, steps_used, raw_trace, task_type, injected_ids, n_valid, logrows)。

    成功判定以环境 ``won`` 为权威（对所有 task_type 有效）。轨迹步保存模型实际看到的
    raw obs + 动作 + reward，供组 RawTrace 喂 TracesPool。

    ``tag`` 仅用于逐步进度打印：每个活跃环境步只发一次 vLLM 请求，直接生成完整
    ``<think>...<action>...`` 响应。长思考下单步仍可能较慢，因此保留逐步打点。
    不打点会让终端长时间零输出、看起来像卡死——所以每步打一行，而不是等整局完再打。
    """
    import time as _time
    obs_list, infos = env.reset()
    obs_text = obs_list[0]
    adm = infos["admissible_commands"][0]
    task = extract_task(obs_text)

    # 与主管线同款 prompt 构造器；reset 时在共享 skill_lib 上检索（拿到实时 skill tree）。
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
        # alfworld_projection 已经内置 admissible_commands 精确匹配 + salvage + 安全默认
        # 动作兜底，返回的 action 一定合法，不需要在这里再手工补救一次。
        actions, valids, action_details = alfworld_projection(
            [raw], [adm], return_details=True)
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
        print(f"  [rollout]{tag} step={step}/{max_steps} action={action!r} "
              f"valid={valid} forced={forced} won={won} ({_time.time()-_t0:.1f}s)")

        # RawTrace step: 模型这一步所基于的 raw obs + 采取的动作 + reward。
        steps.append({
            "step": step, "observation": obs_text, "action": action, "reward": reward,
            "valid_action": valid,
            "non_strict_valid_action": valid,
            "strict_valid_action": action_detail["strict_valid_action"],
            "execution_source": action_detail["execution_source"],
            "direct_admissible_action": action_detail["direct_admissible_action"],
        })
        if keep_logrows:
            logrows.append({"step": step, "prompt": prompt, "action": action, "valid": valid,
                            "valid_action": valid,
                            "non_strict_valid_action": valid,
                            "strict_valid_action": action_detail["strict_valid_action"],
                            "execution_source": action_detail["execution_source"],
                            "direct_admissible_action": action_detail["direct_admissible_action"],
                            "forced": bool(forced), "obs": nobs, "reward": reward, "won": won})

        builder.record(obs_text, action)
        obs_text, adm = nobs, nadm

        repeat = repeat + 1 if action == last_action else 0
        last_action = action
        if done or won:
            break
        # 安全网：同一动作连续无进展重复 ≥6 次判卡死，提前结束（与 run_generic 一致）。
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
            "skill_ids_used": injected_ids, "model_version": "frozen",
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


def rollout_batch_group(env, agent, skill_lib, args, batch_size, tag="",
                        fixed_replica_offsets=None):
    """Run one synchronous batch of ALFWorld episodes.

    A group is the unit that mimics one GRPO rollout batch: all active slots are
    reset together, prompts are generated together at every environment step,
    and CoSkill is allowed to update only after the whole group has finished.
    """
    import time as _time

    obs_list, infos = env.reset()
    # TextWorld returns info as a dict of lists for batch envs.
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
    builders = [
        ProdObsBuilder(mem_lib=skill_lib, with_skills=bool(args.enable_coskill),
                       top_k=args.top_k, history_length=args.history_length)
        for _ in range(batch_size)
    ]
    task_types = []
    injected_ids = []
    playbook_records = []
    for i, b in enumerate(builders):
        b.reset(tasks[i])
        tt = (b.retrieved or {}).get("task_type", "unknown")
        task_types.append(tt)
        injected_ids.append((b.retrieved or {}).get("injected_skill_ids", []) or [])
        pb_rec = skill_lib.get_playbook_record(tt) if args.enable_skill_tree and hasattr(
            skill_lib, "get_playbook_record") else None
        playbook_records.append(pb_rec)

    print(f"[rollout-batch]{tag} size={batch_size} tasks="
          f"{ {t: task_types.count(t) for t in sorted(set(task_types))} }")

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
    # Per-slot running token totals. act_batch_with_meta batches every active
    # slot's prompt into one vLLM generate() call per step, so a simple
    # before/after get_token_usage() bracket around the whole group can only
    # give a group total, not a per-episode one. agent.last_batch_request_tokens
    # is index-aligned with the `prompts`/`active` list passed to that call, so
    # it lets us attribute each step's exact tokens back to the owning slot.
    episode_tokens = [{"prompt": 0, "response": 0, "total": 0} for _ in range(batch_size)]

    for step in range(1, args.max_steps + 1):
        active = [i for i in range(batch_size) if not done[i]]
        if not active:
            break

        _t0 = _time.time()
        prompts = [
            builders[i].build(obs_list[i], adms[i], init=(step == 1))
            for i in active
        ]
        request_seeds = ([
            _fixed_request_seed(
                args.seed, fixed_seed_rows[i][0], fixed_seed_rows[i][1], step)
            for i in active
        ] if fixed_seed_rows is not None else None)
        seed_by_idx = ({i: request_seeds[j] for j, i in enumerate(active)}
                       if request_seeds is not None else {})
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
        # alfworld_projection 已经内置 admissible_commands 精确匹配 + salvage + 安全默认
        # 动作兜底，返回的 action 一定合法，不需要在这里再手工补救一次。
        actions, valids, action_details = alfworld_projection(
            raws, active_adms, return_details=True)

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
                # Inactive slots still need a placeholder action for the batch env.
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

            steps[i].append({
                "step": step, "observation": obs_list[i],
                "action": action, "reward": reward,
                "valid_action": valid_by_idx[i],
                "non_strict_valid_action": valid_by_idx[i],
                "strict_valid_action": action_detail["strict_valid_action"],
                "execution_source": action_detail["execution_source"],
                "direct_admissible_action": action_detail["direct_admissible_action"],
                "sampling_seed": seed_by_idx.get(i),
            })
            if keep_logrows:
                logrows[i].append({
                    "step": step, "prompt": prompts[active.index(i)],
                    "action": action, "valid": valid_by_idx[i],
                    "valid_action": valid_by_idx[i],
                    "non_strict_valid_action": valid_by_idx[i],
                    "strict_valid_action": action_detail["strict_valid_action"],
                    "execution_source": action_detail["execution_source"],
                    "direct_admissible_action": action_detail["direct_admissible_action"],
                    "sampling_seed": seed_by_idx.get(i),
                    "forced": forced_by_idx[i], "obs": nobs_list[i],
                    "reward": reward, "won": slot_won,
                })

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
        print(f"  [rollout-batch]{tag} step={step}/{args.max_steps} "
              f"active={len(active)} done={n_done}/{batch_size} won={n_won}/{batch_size} "
              f"({ _time.time()-_t0:.1f}s)")

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
                "skill_ids_used": injected_ids[i], "model_version": "frozen",
                "fixed_game_id": (fixed_seed_rows[i][0] if fixed_seed_rows else None),
                "fixed_replica_index": (fixed_seed_rows[i][1] if fixed_seed_rows else None),
                "n_valid_actions": n_valid[i],
                "valid_action_ratio": n_valid[i] / max(used[i], 1),
                "n_non_strict_valid_actions": n_valid[i],
                "non_strict_valid_action_ratio": n_valid[i] / max(used[i], 1),
                "n_strict_valid_actions": n_strict_valid[i],
                "strict_valid_action_ratio": n_strict_valid[i] / max(used[i], 1),
                "n_salvaged_actions": sum(
                    s.get("execution_source") == "salvaged" for s in steps[i]),
                "n_fallback_actions": sum(
                    s.get("execution_source") == "fallback" for s in steps[i]),
            },
        }
        episodes.append({
            "won": won[i], "used": used[i] or args.max_steps,
            "raw_trace": raw_trace, "task_type": task_types[i],
            "injected": injected_ids[i], "n_valid": n_valid[i],
            "logrows": logrows[i], "playbook_record": playbook_records[i],
            "small_model_tokens": episode_tokens[i],
        })
    return episodes


def _split_int(total: int, parts: int):
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _fixed_manifest_dp_plan(game_files, replicas_per_game: int, workers: int):
    """Build a hardware-independent fixed-manifest rollout assignment.

    Each manifest game must be evaluated exactly ``replicas_per_game`` times.
    For <= number-of-games workers, a worker receives whole games and a batch
    that is an exact multiple of its game count.  With more workers, games are
    split across single-game workers.  This lets 2/4/8-GPU runs execute the
    same trajectory multiset instead of changing task frequencies with DP.
    """
    games = list(game_files)
    if not games or replicas_per_game < 1 or workers < 1:
        raise ValueError("fixed manifest DP planning requires games, replicas and workers")
    workers = min(int(workers), len(games) * int(replicas_per_game))
    plan = []
    if workers <= len(games):
        assignments = [games[i::workers] for i in range(workers)]
        plan = [(assigned, len(assigned) * replicas_per_game)
                for assigned in assignments]
    else:
        worker_counts = _split_int(workers, len(games))
        for game, game_workers in zip(games, worker_counts):
            for chunk in _split_int(replicas_per_game, game_workers):
                plan.append(([game], chunk))
    assert len(plan) == workers
    assert sum(batch for _, batch in plan) == len(games) * replicas_per_game
    return plan


def _token_delta(after, before):
    """Subtract two ``{prompt,response,total}`` cumulative token snapshots."""
    prompt = max(0, int(after.get("prompt", 0)) - int(before.get("prompt", 0)))
    response = max(0, int(after.get("response", 0)) - int(before.get("response", 0)))
    return {"prompt": prompt, "response": response, "total": prompt + response}


def _dp_rollout_worker(worker_id, gpu_id, args_dict, game_files, task_type_ids,
                       fixed_batch_size, fixed_replica_offsets, in_q, out_q,
                       resume_skip_groups=0):
    """Persistent rollout worker for single-GPU data parallel inference.

    The worker owns one ALFWorld batch env and one vLLM engine pinned to one GPU.
    For every command it reloads the latest skill_lib snapshot, runs exactly one
    fixed-size rollout group, and returns raw episode records to the parent.  The
    parent is the only process that mutates TracesPool / cloud skill state.

    ``resume_skip_groups`` (--resume 1 续跑用)：本 worker 自己的 env 在之前那次运行里
    已经被 ``reset()`` 过这么多次（= 已完成的 rollout group 数，见 ``_load_resume_state``
    的调用点）。同一个 seed + 同一份 game_files 下，``env.reset()`` 推进的是 TextWorld
    内部确定性的 shuffled_cycle 迭代器，所以在真正开始收命令之前先空转这么多次 reset()
    （丢弃返回值），才能让本次运行接着上次的游戏序列继续，而不是从头重新发一遍已经跑过
    的局。这是尽力而为的对齐，不是数学上绝对保证的精确重放。
    """
    try:
        # ``spawn`` copied this one-GPU mask from the parent at ``start()``.
        # Do not rewrite it here: a second assignment can be interpreted in a
        # different CUDA visibility namespace by vLLM's EngineCore descendants
        # and remap worker 1 back onto GPU 0.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices != str(gpu_id):
            raise RuntimeError(
                f"worker {worker_id} expected CUDA_VISIBLE_DEVICES={gpu_id!r}, "
                f"got {visible_devices!r}; refusing unsafe vLLM launch")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        args = argparse.Namespace(**args_dict)
        args.tensor_parallel_size = 1

        config = load_tw_config_types(task_type_ids, num_games=len(game_files))
        env = make_batch_env(game_files, config, batch_size=fixed_batch_size,
                             seed=int(args.seed) + 1009 * (worker_id + 1))
        if resume_skip_groups > 0:
            _t0 = time.time()
            for _ in range(resume_skip_groups):
                env.reset()
            print(f"[dp-worker{worker_id}][resume] skipped {resume_skip_groups} reset() "
                  f"calls to align with prior run ({time.time()-_t0:.1f}s)")

        from mini_test_pen_shelf.agent_vllm import VLLMAgent
        required_max_num_seqs = fixed_batch_size
        vllm_max_num_seqs = int(args.vllm_max_num_seqs or required_max_num_seqs)
        if vllm_max_num_seqs < required_max_num_seqs:
            raise ValueError(
                f"vllm_max_num_seqs={vllm_max_num_seqs} is smaller than "
                f"worker {worker_id}'s rollout batch={required_max_num_seqs}")
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
        print(f"[dp-worker{worker_id}] ready gpu={gpu_id} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"batch={fixed_batch_size} "
              f"max_num_seqs={vllm_max_num_seqs} "
              f"enforce_eager={bool(args.vllm_enforce_eager)} "
              f"games={len(game_files)}")

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
                    env, agent, skill_lib, args, fixed_batch_size, tag=tag,
                    fixed_replica_offsets=fixed_replica_offsets,
                )
                token_usage = _token_delta(agent.get_token_usage(), tokens_before)
                out_q.put({
                    "worker_id": worker_id,
                    "group_id": group_id,
                    "results": results,
                    "small_model_tokens": token_usage,
                    "error": None,
                })
            except Exception:
                out_q.put({
                    "worker_id": worker_id,
                    "group_id": group_id,
                    "results": [],
                    "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
                    "error": traceback.format_exc(),
                })
    except Exception:
        out_q.put({
            "worker_id": worker_id,
            "group_id": None,
            "results": [],
            "small_model_tokens": {"prompt": 0, "response": 0, "total": 0},
            "error": traceback.format_exc(),
        })


def main():
    ap = argparse.ArgumentParser()
    # --- 环境 / 采样（对齐训练脚本）---
    ap.add_argument("--task_types",
                    default=("pick_and_place_simple,look_at_obj_in_light,"
                             "pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,"
                             "pick_cool_then_place_in_recep,pick_two_obj_and_place"),
                    help="逗号分隔的数据集 task_type；默认全 6 类（成功判定走 env won，对所有类型有效）")
    ap.add_argument("--fixed_games_manifest", default=None,
                    help="固定任务 JSON manifest；给定后忽略 task_types/num_games/sample，"
                         "OFF/ON 对照可据此复用完全相同的 game.tw-pddl")
    ap.add_argument("--num_games", type=int, default=-1,
                    help="每个 task_type 跑多少 game；<=0 表示【全部】（正式测试用全量数据，与训练脚本一致不抽样）")
    ap.add_argument("--group_size", type=int, default=6, help="每个 game rollout 次数（≈env.rollout.n）")
    ap.add_argument("--batch_rollout_size", type=int, default=1,
                    help=">1 时启用同步 batch rollout：一次 reset/step 多个 ALFWorld env，"
                         "vLLM 批量生成；云端更新只在整组完成后触发。"
                         "设 72 可近似对齐 GRPO: train_data_size=12 × group_size=6。")
    ap.add_argument("--data_parallel_workers", type=int, default=1,
                    help=">1 时启用多进程数据并行 rollout：每个 worker 绑定一张 GPU、"
                         "各跑一份 vLLM 和一部分 batch env；主进程统一汇总轨迹并触发云端更新。")
    ap.add_argument("--rollout_worker_gpus", default=None,
                    help="数据并行 worker 使用的 GPU 列表，如 '0,1'。默认从 CUDA_VISIBLE_DEVICES "
                         "或 nvidia-smi 自动取前 data_parallel_workers 张。")
    ap.add_argument("--sample", action="store_true", default=False,
                    help="仅当 num_games>0 时生效：跨物体均匀抽样 num_games 个；正式全量测试不需要")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1, help="整个 game 集合重复跑几轮（让 skill tree 多次进化）")
    ap.add_argument("--max_episodes", type=int, default=0,
                    help="总局数硬上限，跑够就提前结束（不管 epochs/游戏池还剩多少）。<=0 表示不限，"
                         "跑完 epochs×游戏池。用于小规模快速验证（如 --max_episodes 20）。")
    ap.add_argument("--cloud_update_every", type=int, default=0,
                    help="每 N 个 episode 在固定边界强制一次云端更新；<=0 只走双水位线。"
                         "用于 A/B 时消除轨迹长度导致的触发时点差异，生产运行保持 0。")
    ap.add_argument("--checkpoint_every_groups", type=int, default=0,
                    help="每 N 个 rollout group 保存一次轻量实验 checkpoint。只落盘 summary/skill_lib "
                         "快照，不强制云端更新；<=0 表示只在最终结束时保存。")
    ap.add_argument("--history_length", type=int, default=8,
                    help="WITH_MEMORY 模板携带的最近历史步数（obs+action）。训练脚本/mini_test 默认 2；"
                         "这里默认调大到 8，让 NO_HIS 记忆缺口更小，减少小模型因看不到早前状态而绕圈子。")
    # --- 模型 / vLLM（对齐训练脚本的 max_prompt/response）---
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.8)
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="vLLM tensor parallel GPU 数。双 A800 单机可设 2；"
                         "需不超过 CUDA_VISIBLE_DEVICES 中可见 GPU 数。")
    ap.add_argument("--vllm_max_num_seqs", type=int, default=0,
                    help="0 自动使用每个 vLLM replica 的实际 rollout batch；"
                         "正数覆盖值不得小于该 replica 的 batch。")
    ap.add_argument("--vllm_enforce_eager", type=int, choices=[0, 1], default=1,
                    help="1 保持 ALFWorld 既有 eager 执行；0 允许 vLLM CUDA Graph。")
    ap.add_argument("--max_model_len", type=int, default=10240)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--think_budget", type=int, default=3500)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="训练 rollout 采样温度默认 1.0（保证 success/failure 轨迹多样性）")
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--nowait", action="store_true")
    # --- 记忆 / 技能（对齐训练脚本）---
    ap.add_argument("--skills_json", default="memory_data/alfworld/claude_style_skills.json")
    ap.add_argument("--retrieval_mode", default="embedding", choices=["embedding", "template"])
    ap.add_argument("--embedding_model_path", default=None)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--enable_hierarchy", type=int, default=1)
    ap.add_argument("--stable_cycles_l1", type=int, default=3)
    ap.add_argument("--stable_cycles_l2", type=int, default=5)
    ap.add_argument("--success_l1", type=float, default=0.7)
    ap.add_argument("--demote_threshold", type=float, default=0.3)
    ap.add_argument("--min_calls", type=int, default=10)
    # --- 闭环开关（对齐训练脚本）---
    ap.add_argument("--enable_coskill", type=int, default=1,
                    help="是否启用扁平 skill bullets：云端 contrastive_distill 产 dyn_ 补丁，"
                         "端侧同时注入 General/Task-specific/Mistakes 三段。默认重新开启；"
                         "消融组显式传 0。")
    ap.add_argument("--enable_skill_tree", type=int, default=1,
                    help="是否把 agent skill tree 注入小模型 prompt。消融 baseline 设为 0。")
    ap.add_argument("--enable_skill_tree_evolve",
                    dest="enable_playbook_evolve", type=int, default=1,
                    metavar="ENABLE_SKILL_TREE_EVOLVE")
    ap.add_argument("--enable_playbook_evolve",
                    dest="enable_playbook_evolve", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--enable_failure_analysis", type=int, default=1)
    ap.add_argument("--enable_cloud_updates", type=int, default=1,
                    help="0 keeps collecting raw traces/metrics but freezes the skill library. "
                         "Used by fixed-artifact evaluation; default preserves the closed loop.")
    ap.add_argument("--max_new_skills", type=int, default=3)
    ap.add_argument("--skill_tree_evolve_min_samples",
                    dest="playbook_evolve_min_samples", type=int, default=6,
                    metavar="SKILL_TREE_EVOLVE_MIN_SAMPLES")
    ap.add_argument("--playbook_evolve_min_samples",
                    dest="playbook_evolve_min_samples", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--coskill_debug", type=int, default=0)
    ap.add_argument("--required_tree_depth", type=int, default=0,
                    help="Require a cloud-authored tree with exactly this many heading levels; 0 disables.")
    ap.add_argument("--tree_depth_repair_attempts", type=int, default=0,
                    help="Same-evidence cloud repair attempts after a fixed-depth tree fails validation.")
    # --- 轨迹池水位线（对齐训练脚本）---
    ap.add_argument("--capacity_watermark", type=int, default=50000)
    ap.add_argument("--perf_watermark", type=float, default=0.6)
    ap.add_argument("--min_samples", type=int, default=16)
    ap.add_argument("--loop_threshold", type=int, default=3)
    ap.add_argument("--trace_enable_loop_filter", type=int, default=1)
    ap.add_argument("--trace_enable_obs_delta", type=int, default=1)
    ap.add_argument("--trace_enable_prefix_tree", type=int, default=1)
    ap.add_argument("--trace_enable_consensus_prefix", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log_trajectories", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0,
                    help="1 则从同一个 --outdir 里上一次留下的 skill_lib checkpoint + "
                         "summary_partial.json 恢复技能库和 epoch/episode 进度，而不是从 "
                         "--skills_json 种子文件、episode 0 重新开始。显式 opt-in，避免误用"
                         "旧 outdir 的陈旧状态覆盖一次本想全新开始的运行。")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    resume_state = _load_resume_state(args)
    enable_coskill = bool(args.enable_coskill)
    enable_skill_tree = bool(args.enable_skill_tree)
    enable_skill_tree_evolve = enable_skill_tree and bool(args.enable_playbook_evolve)
    if bool(args.enable_playbook_evolve) and not enable_skill_tree:
        print("[driver] enable_skill_tree=0, so skill tree evolution is disabled too")

    # 1) 共享技能库（与训练脚本同参）。skill tree 进化写回它、rollout 从它检索，形成闭环。
    #    --resume 1 时 resume_state["skills_json_path"] 已指向上次的 checkpoint，否则
    #    等于 args.skills_json（种子文件），行为与不加 --resume 完全一致。
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

    # 2) 轨迹池（同水位线）。
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
    )

    # 3) 共享云端编排（trainer 也用同一个类）。
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
    )

    # 4) 建所有 task_type 的环境（在加载 vLLM/CUDA 之前！）。
    #    AlfredTWEnv.init_env() 用 asynchronous=True 会 fork 一个子进程跑游戏逻辑
    #    （textworld.gym.register_games）。若父进程已经初始化过 CUDA（vLLM 加载模型
    #    之后），此时再 fork 会导致子进程卡死——这是 CUDA 的经典 fork-safety 问题，
    #    与 mini_test_pen_shelf/run_generic.py 的顺序（先建 env 再建 VLLMAgent）保持
    #    一致即可避免。
    #
    #    单一混合 env，而非逐 task_type 顺序跑完再下一个（对齐生产 config_tw.yaml：
    #    task_types=[1..6] 全部混进同一个 AlfredTWEnv 的 game_files 池）。production
    #    的 reset() 顺序由 textworld 自己的 TextworldBatchGymEnv 决定：env.seed(seed)
    #    只在建环境时调用【一次】，内部把 game_files 整体 shuffle 一遍包成
    #    shuffled_cycle 迭代器——取完一整轮自动重新 shuffle 再继续（无限循环，reset()
    #    每次就是 next()）。所以只要把全部 task_type 的 game_files 合并成一个池子、
    #    只 seed 一次，接下来的 reset() 天然就是跨 task_type 随机混合，不会出现"先刷完
    #    一类再刷下一类"。（旧版按 task_type 分别建 env + 顺序嵌套循环，天然顺序刷完
    #    一类才换下一类；且旧版每局都重新 env.seed(seed+global_step)，等于每局都把
    #    cycle 迭代器重新打乱再只取第一个，反而破坏了 textworld 自带的"整轮循环"语义。）
    if args.fixed_games_manifest:
        fixed_games, task_types, all_tids = _load_fixed_games_manifest(
            args.fixed_games_manifest
        )
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
            # num_games<=0 -> ALL matching games (limit=None, sample_n=None); this is
            # the正式全量测试 path. num_games>0 -> either a diverse sample or a hard
            # limit, for quick smoke runs only.
            if args.num_games <= 0:
                games = find_games_by_type(task_type, split=args.split)
            else:
                games = find_games_by_type(task_type, split=args.split,
                                           sample_n=args.num_games if args.sample else None,
                                           sample_seed=args.sample_seed,
                                           limit=None if args.sample else args.num_games)
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
    print(f"[driver] combined pool: {total_games} games across {len(all_tids)} task_types "
          f"x group_size={args.group_size} x epochs={args.epochs} "
          f"= {episodes_per_epoch * args.epochs} episodes total")
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
        fixed_balanced = bool(
            args.fixed_games_manifest
            and batch_rollout_size == len(all_game_files) * int(args.group_size)
        )
        if fixed_balanced:
            worker_plan = _fixed_manifest_dp_plan(
                all_game_files, int(args.group_size), data_parallel_workers)
            data_parallel_workers = len(worker_plan)
        else:
            dp_worker_batch_sizes = _split_int(batch_rollout_size, data_parallel_workers)
            worker_plan = [
                (all_game_files[wid::data_parallel_workers] or list(all_game_files), worker_bs)
                for wid, worker_bs in enumerate(dp_worker_batch_sizes)
            ]
        if len(worker_gpus) < data_parallel_workers:
            raise ValueError(f"data_parallel_workers={data_parallel_workers} but only "
                             f"{len(worker_gpus)} GPU ids available: {worker_gpus}")
        worker_gpus = worker_gpus[:data_parallel_workers]
        dp_worker_batch_sizes = [batch for _, batch in worker_plan]
        vllm_max_num_seqs_by_worker = [
            int(args.vllm_max_num_seqs or worker_batch)
            for worker_batch in dp_worker_batch_sizes
        ]
        for worker_id, (limit, worker_batch) in enumerate(zip(
                vllm_max_num_seqs_by_worker, dp_worker_batch_sizes)):
            if limit < worker_batch:
                raise ValueError(
                    f"vllm_max_num_seqs={limit} is smaller than worker "
                    f"{worker_id}'s rollout batch={worker_batch}")
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
            raise RuntimeError(
                f"fixed manifest replica allocation is not balanced: {replica_counts}")

        ctx = mp.get_context("spawn")
        dp_out_q = ctx.Queue()
        args_dict = vars(args).copy()
        args_dict["tensor_parallel_size"] = 1
        for wid, (gpu_id, (worker_games, worker_bs), replica_offsets) in enumerate(
                zip(worker_gpus, worker_plan, worker_replica_offsets)):
            in_q = ctx.Queue(maxsize=2)
            proc = ctx.Process(
                target=_dp_rollout_worker,
                args=(wid, gpu_id, args_dict, worker_games, all_tids,
                      worker_bs, replica_offsets, in_q, dp_out_q,
                      resume_state["completed_groups"]),
                daemon=False,
            )
            # ``spawn`` captures environment variables before the target body
            # runs.  Apply a one-GPU mask before spawning so each vLLM
            # EngineCore inherits its assigned device rather than the driver's
            # whole ``0,1`` mask; otherwise both replicas can allocate their KV
            # cache on GPU 0 on some cluster configurations.
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
        print(f"[driver] data_parallel_workers={data_parallel_workers} "
              f"gpus={worker_gpus} worker_batch_sizes={dp_worker_batch_sizes} "
              f"fixed_manifest_balanced={fixed_balanced}; "
              "main process will aggregate trajectories and run cloud updates")
    else:
        # Keep topology metadata meaningful for the direct one-replica path;
        # older summaries exposed an empty worker-batch list even though this
        # process handled the complete rollout batch.
        dp_worker_batch_sizes = [batch_rollout_size]
        vllm_max_num_seqs = int(args.vllm_max_num_seqs or batch_rollout_size)
        if vllm_max_num_seqs < batch_rollout_size:
            raise ValueError(
                f"vllm_max_num_seqs={vllm_max_num_seqs} is smaller than "
                f"the single replica rollout batch={batch_rollout_size}")
        vllm_max_num_seqs_by_worker = [vllm_max_num_seqs]
        if batch_rollout_size == 1:
            env = make_single_env(all_game_files, config, seed=args.seed)
        else:
            env = make_batch_env(all_game_files, config, batch_size=batch_rollout_size, seed=args.seed)
            print(f"[driver] batch_rollout_size={batch_rollout_size}: cloud updates are checked "
                  "only after a full rollout group finishes")

        if resume_state["completed_groups"] > 0:
            # --resume 1：同一 seed + 同一份 all_game_files 下，env.reset() 推进的是
            # TextWorld 内部确定性的 shuffled_cycle 迭代器；空转跳过之前那次运行已经
            # reset() 过的次数，才能接着上次的游戏序列继续，而不是重新发一遍已跑过的局。
            _t0 = time.time()
            for _ in range(resume_state["completed_groups"]):
                env.reset()
            print(f"[driver][resume] skipped {resume_state['completed_groups']} reset() "
                  f"calls to align with prior run ({time.time()-_t0:.1f}s)")

        # 5) vLLM 冻结模型（只此一份）。放在所有 env 建好之后，避免上面的 fork-after-CUDA 问题。
        from mini_test_pen_shelf.agent_vllm import VLLMAgent
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
        print(f"[driver] single vLLM replica batch={batch_rollout_size} "
              f"TP={args.tensor_parallel_size} max_num_seqs={vllm_max_num_seqs} "
              f"enforce_eager={bool(args.vllm_enforce_eager)}")

        # 6) 与主管线逐字节对齐的 prompt 构造器，注入共享 skill_lib。
        #    with_skills 跟随 enable_coskill（训练脚本注入 general/task/mistakes bullets）。
        builder = ProdObsBuilder(mem_lib=skill_lib, with_skills=enable_coskill, top_k=args.top_k,
                                 history_length=args.history_length)
    print(f"[driver] retrieval_mode={args.retrieval_mode} with_skills={enable_coskill} "
          f"enable_skill_tree={enable_skill_tree} "
          f"enable_skill_tree_evolve={enable_skill_tree_evolve} "
          f"enable_failure_analysis={bool(args.enable_failure_analysis)}")

    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    group_metrics_path = os.path.join(args.outdir, "group_metrics.jsonl")
    # ``group_metrics.jsonl`` is the sole canonical group-level comparison log.
    # It already contains every ``comparison/*`` field below.  Older runs may
    # still have a duplicated comparison_metrics.jsonl; never delete it, but do
    # not create a second parallel metric stream for new/resumed runs.
    per_game = list(resume_state["per_game"])
    wins = resume_state["wins"]
    global_step = len(per_game)
    tt_stats = {}  # detected_type -> {"episodes":int, "wins":int}
    cloud_updates = int(resume_state.get("cloud_updates", 0) or 0)
    cloud_update_steps = list(resume_state.get("cloud_update_steps", []) or [])
    small_model_token_totals = dict(resume_state.get("small_model_tokens", {}))
    small_model_token_accounting = resume_state.get(
        "small_model_token_accounting", SMALL_MODEL_TOKEN_ACCOUNTING)
    large_model_token_offset = dict(resume_state.get("large_model_tokens", {}))
    small_model_token_totals_by_tt = {
        tt: dict(resume_state.get("small_model_tokens_by_tt", {}).get(
            tt, {"prompt": 0, "response": 0, "total": 0}))
        for tt in NORL_TASK_TYPES
    }
    large_model_token_offset_by_tt = {
        tt: dict(resume_state.get("large_model_tokens_by_tt", {}).get(
            tt, {"prompt": 0, "completion": 0, "total": 0}))
        for tt in NORL_TASK_TYPES
    }
    large_model_token_offset_mixed = dict(
        resume_state.get("large_model_tokens_mixed", {"prompt": 0, "completion": 0, "total": 0}))

    # env.seed() 只在 make_single_env() 建环境时调用过一次（见上方注释）：接下来的
    # reset() 全部靠 textworld 自带的 shuffled_cycle 自然推进 + 整轮重洗，不再逐局
    # 重新 seed —— 那样做等于每局都把循环打乱重来，反而丢失了"整轮不重复地过一遍"
    # 的语义，也是过去"轨迹全是同一 task_type"的根源（旧版按 task_type 分别建 env
    # 并顺序嵌套 for 循环，天然刷完一类才换下一类）。
    def _ingest_episode_result(epoch, ep_result):
        nonlocal global_step, wins
        global_step += 1
        won = ep_result["won"]
        used = ep_result["used"]
        raw_trace = ep_result["raw_trace"]
        tt_detected = ep_result["task_type"]
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

        ep_record = {"epoch": epoch + 1, "detected_type": tt_detected,
                     "won": bool(won), "used_steps": used,
                     "valid_actions": nval, "step": global_step,
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
                     "tokens_total": int(ep_tokens.get("total", 0) or 0)}
        per_game.append(ep_record)
        print(f"[driver] step={global_step} {tt_detected} won={won} steps={used}")

        if args.log_trajectories:
            _dump_episode(args.outdir, global_step, raw_trace["task"], tt_detected,
                          won, used, logrows, pb_rec, injected)
        return ep_record, pb_rec

    def _write_episode_metric(epoch, ep_record, pb_rec, fired):
        tt_detected = ep_record["detected_type"]
        tt_eps = ep_record["task_type_episodes"]
        tt_wins = ep_record["task_type_wins"]
        _append_jsonl(metrics_path, {
            "step": ep_record["step"],
            "metrics": {
                "training/epoch": epoch + 1,
                "episode/detected_type": tt_detected,
                "episode/won": bool(ep_record["won"]),
                "episode/length": ep_record["used_steps"],
                "episode/valid_actions": ep_record["valid_actions"],
                "episode/valid_action_ratio": ep_record["valid_action_ratio"],
                "episode/non_strict_valid_actions": ep_record["non_strict_valid_actions"],
                "episode/non_strict_valid_action_ratio": ep_record["non_strict_valid_action_ratio"],
                "episode/strict_valid_actions": ep_record["strict_valid_actions"],
                "episode/strict_valid_action_ratio": ep_record["strict_valid_action_ratio"],
                "episode/salvaged_actions": ep_record["salvaged_actions"],
                "episode/fallback_actions": ep_record["fallback_actions"],
                "experiment/skill_tree_enabled": int(enable_skill_tree),
                "experiment/skill_tree_evolve_enabled": int(enable_skill_tree_evolve),
                "experiment/skill_bullets_enabled": int(enable_coskill),
                "experiment/cloud_round_used": ep_record["cloud_round_used"],
                "coskill/cloud_update_fired": bool(fired),
                "episode/running_total_episodes": ep_record["running_total_episodes"],
                "episode/running_total_wins": ep_record["running_total_wins"],
                "episode/success_rate": round(
                    ep_record["running_total_wins"] / max(ep_record["running_total_episodes"], 1), 4),
                f"episode/{tt_detected}/episodes": tt_eps,
                f"episode/{tt_detected}/wins": tt_wins,
                f"episode/{tt_detected}/success_rate": round(tt_wins / max(tt_eps, 1), 4),
                "tokens/small_model/prompt": ep_record["tokens_prompt"],
                "tokens/small_model/response": ep_record["tokens_response"],
                "tokens/small_model/total": ep_record["tokens_total"],
                "tokens/small_model/accounting": SMALL_MODEL_TOKEN_ACCOUNTING,
                f"tokens/small_model/by_task_type/{tt_detected}/prompt": ep_record["tokens_prompt"],
                f"tokens/small_model/by_task_type/{tt_detected}/response": ep_record["tokens_response"],
                f"tokens/small_model/by_task_type/{tt_detected}/total": ep_record["tokens_total"],
                "skill_tree/version": pb_rec.get("version") if pb_rec else 0,
                "skill_tree/level": pb_rec.get("level") if pb_rec else None,
                "skill_tree/n_nodes": len(pb_rec.get("nodes") or {}) if pb_rec else 0,
                **cloud_loop.metrics(traces_pool, skill_lib),
            },
        })

    def _large_model_token_usage():
        analyzer = getattr(cloud_loop, "cloud_analyzer", None)
        zero_by_tt = {tt: {"prompt": 0, "completion": 0, "total": 0} for tt in NORL_TASK_TYPES}
        if analyzer is None:
            return {
                "prompt": 0, "completion": 0, "total": 0,
                "by_task_type": zero_by_tt,
                "mixed": {"prompt": 0, "completion": 0, "total": 0},
            }
        prompt = int(getattr(analyzer, "total_prompt_tokens", 0) or 0)
        completion = int(getattr(analyzer, "total_completion_tokens", 0) or 0)
        by_tt_prompt = getattr(analyzer, "total_prompt_tokens_by_task_type", {}) or {}
        by_tt_completion = getattr(analyzer, "total_completion_tokens_by_task_type", {}) or {}
        by_task_type = {}
        for tt in NORL_TASK_TYPES:
            p = int(large_model_token_offset_by_tt.get(tt, {}).get("prompt", 0) or 0) \
                + int(by_tt_prompt.get(tt, 0) or 0)
            c = int(large_model_token_offset_by_tt.get(tt, {}).get("completion", 0) or 0) \
                + int(by_tt_completion.get(tt, 0) or 0)
            by_task_type[tt] = {"prompt": p, "completion": c, "total": p + c}
        mixed_prompt = int(large_model_token_offset_mixed.get("prompt", 0) or 0) \
            + int(getattr(analyzer, "total_prompt_tokens_mixed", 0) or 0)
        mixed_completion = int(large_model_token_offset_mixed.get("completion", 0) or 0) \
            + int(getattr(analyzer, "total_completion_tokens_mixed", 0) or 0)
        return {
            "prompt": int(large_model_token_offset.get("prompt", 0) or 0) + prompt,
            "completion": int(large_model_token_offset.get("completion", 0) or 0) + completion,
            "total": int(large_model_token_offset.get("total", 0) or 0) + prompt + completion,
            "by_task_type": by_task_type,
            "mixed": {
                "prompt": mixed_prompt, "completion": mixed_completion,
                "total": mixed_prompt + mixed_completion,
            },
        }

    def _write_group_metric(group_id, epoch, ingested, generated_count, fired,
                            small_tokens, large_before, large_after,
                            rollout_seconds, cloud_seconds, total_seconds):
        """Write one GRPO-comparable aggregate row per rollout group."""
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
            tt = record["detected_type"]
            bucket = small_by_tt.setdefault(tt, {"prompt": 0, "response": 0, "total": 0})
            bucket["prompt"] += int(record.get("tokens_prompt", 0) or 0)
            bucket["response"] += int(record.get("tokens_response", 0) or 0)
            bucket["total"] += int(record.get("tokens_total", 0) or 0)
        for tt in small_by_tt:
            totals = small_model_token_totals_by_tt.setdefault(
                tt, {"prompt": 0, "response": 0, "total": 0})
            for key in ("prompt", "response", "total"):
                totals[key] += small_by_tt[tt][key]

        large_delta = {
            "prompt": max(0, large_after["prompt"] - large_before["prompt"]),
            "completion": max(0, large_after["completion"] - large_before["completion"]),
        }
        large_delta["total"] = large_delta["prompt"] + large_delta["completion"]

        large_delta_by_tt = {}
        for tt in NORL_TASK_TYPES:
            p = max(0, large_after["by_task_type"][tt]["prompt"]
                    - large_before["by_task_type"][tt]["prompt"])
            c = max(0, large_after["by_task_type"][tt]["completion"]
                    - large_before["by_task_type"][tt]["completion"])
            large_delta_by_tt[tt] = {"prompt": p, "completion": c, "total": p + c}
        large_delta_mixed = {
            "prompt": max(0, large_after["mixed"]["prompt"] - large_before["mixed"]["prompt"]),
            "completion": max(0, large_after["mixed"]["completion"] - large_before["mixed"]["completion"]),
        }
        large_delta_mixed["total"] = large_delta_mixed["prompt"] + large_delta_mixed["completion"]

        metrics = {
            # group_id is the no-RL equivalent of one GRPO training step.
            "training/group": group_id,
            "training/global_step": group_id,
            "training/epoch": epoch + 1,
            "rollout/global_episode_end": global_step,
            "episode/count": n,
            "episode/generated_count": int(generated_count),
            "episode/wins": group_wins,
            "episode/success_rate": round(group_wins / max(n, 1), 6),
            "episode/count_cumulative": global_step,
            "episode/wins_cumulative": wins,
            "episode/action_count": action_count,
            "episode/action_count_cumulative": action_count_cumulative,
            "episode/length/mean": round(sum(lengths) / max(n, 1), 6),
            "episode/length/max": max(lengths) if lengths else 0,
            "episode/length/min": min(lengths) if lengths else 0,
            "episode/valid_action_ratio": round(valid_actions / max(action_count, 1), 6),
            "episode/strict_valid_action_ratio": round(
                strict_valid_actions / max(action_count, 1), 6),
            # ALFWorld has no separate relaxed projection: its relaxed notion
            # is the parser-valid action count already reported above.
            "episode/relaxed_valid_action_ratio": round(
                valid_actions / max(action_count, 1), 6),
            "episode/non_strict_valid_action_ratio": round(
                valid_actions / max(action_count, 1), 6),
            "episode/salvaged_action_ratio": round(salvaged_actions / max(action_count, 1), 6),
            "episode/fallback_action_ratio": round(fallback_actions / max(action_count, 1), 6),
            "experiment/skill_tree_enabled": int(enable_skill_tree),
            "experiment/skill_tree_evolve_enabled": int(enable_skill_tree_evolve),
            "experiment/skill_bullets_enabled": int(enable_coskill),
            "experiment/rl_enabled": 0,
            "experiment/tree_rl_internalize_enabled": 0,
            "experiment/cloud_round": cloud_updates,
            "coskill/cloud_update_fired": bool(fired),
            "tokens/small_model/prompt": int(small_tokens.get("prompt", 0) or 0),
            "tokens/small_model/response": int(small_tokens.get("response", 0) or 0),
            "tokens/small_model/total": int(small_tokens.get("total", 0) or 0),
            "tokens/small_model/accounting": SMALL_MODEL_TOKEN_ACCOUNTING,
            "tokens/small_model/prompt_cumulative": small_model_token_totals["prompt"],
            "tokens/small_model/response_cumulative": small_model_token_totals["response"],
            "tokens/small_model/total_cumulative": small_model_token_totals["total"],
            "tokens/large_model/prompt": large_delta["prompt"],
            "tokens/large_model/completion": large_delta["completion"],
            "tokens/large_model/total": large_delta["total"],
            "tokens/large_model/accounting": "provider_api_usage",
            "tokens/large_model/prompt_cumulative": large_after["prompt"],
            "tokens/large_model/completion_cumulative": large_after["completion"],
            "tokens/large_model/total_cumulative": large_after["total"],
            "timing_s/rollout": round(float(rollout_seconds), 6),
            "timing_s/cloud_update": round(float(cloud_seconds), 6),
            "timing_s/group_total": round(float(total_seconds), 6),
            "perf/total_num_tokens": int(small_tokens.get("total", 0) or 0),
            "perf/throughput_episodes_per_second": round(n / max(rollout_seconds, 1e-9), 6),
            "perf/throughput_small_tokens_per_second": round(
                int(small_tokens.get("total", 0) or 0) / max(rollout_seconds, 1e-9), 6),
            "comparison/schema_version": 1,
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
        for tt, stat in sorted(by_type.items()):
            metrics[f"episode/{tt}/episodes"] = stat["episodes"]
            metrics[f"episode/{tt}/wins"] = stat["wins"]
            metrics[f"episode/{tt}/success_rate"] = round(
                stat["wins"] / max(stat["episodes"], 1), 6)

        for tt in NORL_TASK_TYPES:
            metrics[f"tokens/small_model/by_task_type/{tt}/prompt"] = small_by_tt[tt]["prompt"]
            metrics[f"tokens/small_model/by_task_type/{tt}/response"] = small_by_tt[tt]["response"]
            metrics[f"tokens/small_model/by_task_type/{tt}/total"] = small_by_tt[tt]["total"]
            metrics[f"tokens/small_model/by_task_type/{tt}/total_cumulative"] = \
                small_model_token_totals_by_tt[tt]["total"]
            metrics[f"tokens/large_model/by_task_type/{tt}/prompt"] = large_delta_by_tt[tt]["prompt"]
            metrics[f"tokens/large_model/by_task_type/{tt}/completion"] = large_delta_by_tt[tt]["completion"]
            metrics[f"tokens/large_model/by_task_type/{tt}/total"] = large_delta_by_tt[tt]["total"]
            metrics[f"tokens/large_model/by_task_type/{tt}/total_cumulative"] = \
                large_after["by_task_type"][tt]["total"]
        # contrastive_distill/diagnose_failures each mix every task_type into
        # one cloud call, so those tokens aren't attributable to a single
        # subtask - tracked honestly here instead of a fabricated split.
        metrics["tokens/large_model/mixed/prompt"] = large_delta_mixed["prompt"]
        metrics["tokens/large_model/mixed/completion"] = large_delta_mixed["completion"]
        metrics["tokens/large_model/mixed/total"] = large_delta_mixed["total"]
        metrics["tokens/large_model/mixed/total_cumulative"] = large_after["mixed"]["total"]
        metrics["tokens/large_model/mixed/accounting"] = "provider_api_usage_mixed_task_types"

        canonical_record = {
            "step": group_id,
            "global_episode_end": global_step,
            "metrics": metrics,
        }
        _append_jsonl(group_metrics_path, canonical_record)
        print(f"[driver] group{group_id} metric: episodes={n} wins={group_wins} "
              f"success={100.0 * group_wins / max(n, 1):.1f}% "
              f"valid_action={100.0 * valid_actions / max(action_count, 1):.1f}% "
              f"strict_valid_action={100.0 * strict_valid_actions / max(action_count, 1):.1f}% "
              f"small_tokens={small_tokens.get('total', 0)} "
              f"large_tokens={large_delta['total']} rollout={rollout_seconds:.1f}s")

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
        return {
            "status": status,
            "checkpoint_reason": checkpoint_reason,
            "task_types": task_types, "epochs": args.epochs,
            "num_games_per_type": args.num_games, "group_size": args.group_size,
            "batch_rollout_size": batch_rollout_size,
            "data_parallel_workers": data_parallel_workers,
            "data_parallel_worker_batch_sizes": dp_worker_batch_sizes,
            "vllm_max_num_seqs": int(args.vllm_max_num_seqs or max(
                vllm_max_num_seqs_by_worker)),
            "vllm_max_num_seqs_by_worker": vllm_max_num_seqs_by_worker,
            "vllm_enforce_eager": bool(args.vllm_enforce_eager),
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
                "accounting": "chars_div_4",
            },
            "cloud_update_every": args.cloud_update_every,
            "cloud_update_steps": cloud_update_steps,
            "total_games_combined_pool": total_games,
            "total_episodes": n, "wins": wins,
            "success_rate": round(wins / max(n, 1), 4),
            "skill_tree_versions": {tt: rec["version"] for tt, rec in trees.items()},
            "skill_tree_nodes": {tt: rec["n_nodes"] for tt, rec in trees.items()},
            # Backward-compatible scalar: max version among task trees.
            "skill_tree_version": max([rec["version"] for rec in trees.values()] or [0]),
            "token_usage": {
                "small_model": {
                    **dict(small_model_token_totals),
                    "accounting": small_model_token_accounting,
                    "by_task_type": {
                        tt: dict(small_model_token_totals_by_tt[tt]) for tt in NORL_TASK_TYPES
                    },
                },
                "large_model": _large_model_token_usage(),
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
        print(f"[driver] checkpoint saved at step={global_step} groups={completed_groups} "
              f"reason={reason} summary={ckpt_path} skill_lib={skill_snapshot or '<none>'}")

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
                raise RuntimeError(
                    f"data-parallel rollout worker {msg.get('worker_id')} failed:\n"
                    f"{msg.get('error')}"
                )
            replies.append(msg)
        replies.sort(key=lambda x: x["worker_id"])
        group_results = []
        small_tokens = {"prompt": 0, "response": 0, "total": 0}
        for msg in replies:
            group_results.extend(msg["results"])
            usage = msg.get("small_model_tokens") or {}
            for key in small_tokens:
                small_tokens[key] += int(usage.get(key, 0) or 0)
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
        # 只有恢复到的第一个 epoch 从中断的 episode 位置续跑，之后的 epoch 仍从 0 开始。
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
                print(f"[driver] dispatch dp group{group_id} epoch{epoch+1} "
                      f"group_ep{ep_i+1}-{ep_i+n_to_count}/{episodes_per_epoch} "
                      f"(global_step={global_step+1})")
                group_started = time.time()
                group_results, small_tokens = _run_data_parallel_group(group_id)
                rollout_seconds = time.time() - group_started
                ingested = []
                for ep_result in group_results[:n_to_count]:
                    ingested.append(_ingest_episode_result(epoch, ep_result))
                ep_i += n_to_count

                # Data-parallel mode: cloud update can happen only after all
                # workers finish the rollout group and the parent ingests it.
                force_reason = None
                if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                    force_reason = f"episode_interval_{args.cloud_update_every}"
                large_before = _large_model_token_usage()
                cloud_started = time.time()
                fired = (cloud_loop.maybe_update(
                    traces_pool, skill_lib, global_step, force_reason=force_reason
                ) if args.enable_cloud_updates else False)
                cloud_seconds = time.time() - cloud_started
                large_after = _large_model_token_usage()
                if fired:
                    cloud_updates += 1
                    cloud_update_steps.append(global_step)

                for j, (ep_record, pb_used) in enumerate(ingested):
                    is_last = (j == len(ingested) - 1)
                    ep_record["cloud_update_fired_after_episode"] = bool(fired and is_last)
                    _write_episode_metric(epoch, ep_record, pb_used, fired and is_last)

                _write_group_metric(
                    group_id, epoch, ingested, len(group_results), fired,
                    small_tokens, large_before, large_after,
                    rollout_seconds, cloud_seconds, time.time() - group_started,
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
                tag = f" epoch{epoch+1} ep{ep_i+1}/{episodes_per_epoch} (global_step={global_step+1})"
                won, used, raw_trace, tt_detected, injected, nval, logrows = \
                    rollout_episode(env, agent, builder, args.max_steps, tag=tag,
                                    keep_logrows=bool(args.log_trajectories))
                rollout_seconds = time.time() - group_started
                small_tokens = _token_delta(agent.get_token_usage(), small_before)
                pb_rec = skill_lib.get_playbook_record(tt_detected) if enable_skill_tree and hasattr(
                    skill_lib, "get_playbook_record") else None
                ep_result = {"won": won, "used": used, "raw_trace": raw_trace,
                             "task_type": tt_detected, "injected": injected,
                             "n_valid": nval, "logrows": logrows,
                             "playbook_record": pb_rec,
                             "small_model_tokens": small_tokens}
                ep_record, pb_used = _ingest_episode_result(epoch, ep_result)
                ep_i += 1

                force_reason = None
                if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                    force_reason = f"episode_interval_{args.cloud_update_every}"
                large_before = _large_model_token_usage()
                cloud_started = time.time()
                fired = (cloud_loop.maybe_update(
                    traces_pool, skill_lib, global_step, force_reason=force_reason
                ) if args.enable_cloud_updates else False)
                cloud_seconds = time.time() - cloud_started
                large_after = _large_model_token_usage()
                ep_record["cloud_update_fired_after_episode"] = bool(fired)
                if fired:
                    cloud_updates += 1
                    cloud_update_steps.append(global_step)
                _write_episode_metric(epoch, ep_record, pb_used, fired)
                _write_group_metric(
                    group_id, epoch, [(ep_record, pb_used)], 1, fired,
                    small_tokens, large_before, large_after,
                    rollout_seconds, cloud_seconds, time.time() - group_started,
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

            tag = f" epoch{epoch+1} group_ep{ep_i+1}-{ep_i+n_to_count}/{episodes_per_epoch} "\
                  f"(global_step={global_step+1})"
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

            # Batch mode: cloud update can happen only after the whole rollout group
            # has completed and all counted trajectories have been ingested.
            force_reason = None
            if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                force_reason = f"episode_interval_{args.cloud_update_every}"
            large_before = _large_model_token_usage()
            cloud_started = time.time()
            fired = (cloud_loop.maybe_update(
                traces_pool, skill_lib, global_step, force_reason=force_reason
            ) if args.enable_cloud_updates else False)
            cloud_seconds = time.time() - cloud_started
            large_after = _large_model_token_usage()
            if fired:
                cloud_updates += 1
                cloud_update_steps.append(global_step)

            for j, (ep_record, pb_used) in enumerate(ingested):
                is_last = (j == len(ingested) - 1)
                ep_record["cloud_update_fired_after_episode"] = bool(fired and is_last)
                _write_episode_metric(epoch, ep_record, pb_used, fired and is_last)

            _write_group_metric(
                group_id, epoch, ingested, len(group_results), fired,
                small_tokens, large_before, large_after,
                rollout_seconds, cloud_seconds, time.time() - group_started,
            )

            completed_groups += 1
            if args.checkpoint_every_groups > 0 and completed_groups % args.checkpoint_every_groups == 0:
                _save_progress_checkpoint(
                    reason=f"group_interval_{args.checkpoint_every_groups}",
                    completed_groups=completed_groups,
                    epoch=epoch,
                    ep_i=ep_i,
                )

    # 落盘 summary。
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
    print(f"\n[driver] done. episodes={n} success_rate={summary['success_rate']*100:.1f}% "
          f"skill_tree_versions={summary['skill_tree_versions']}")
    print(f"[driver] outputs under {args.outdir}/ (traces_pool, cloud_io, skill_lib, trajectories)")


def _append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _dump_episode(outdir, episode_idx, task, detected_task_type, won, used_steps,
                  logrows, playbook_record, skill_ids_used=None):
    """一局 = 一个文件（三种格式），含该局全部 step，而非按 step 拆成多个文件。

    文件名用 ``ep{idx}`` 而非 "step"，避免把"第几局"和"局内第几步"这两个概念混淆
    （旧版 ``traj_step{N}.json`` 里的 N 其实是局数，看起来却像是"某一步"）。

    产出三个文件（同一 basename）：
      - ``..._episode.json``  程序化读取：任务/结果/该局用到的 skill tree 版本/
        逐 step 结构化记录（prompt/action/valid/forced/obs/reward/won）。
      - ``..._trajectory.txt`` 人类可读：逐 step 的 动作/合法性/obs 摘要/reward，
        一眼看清整局怎么走的，不用啃 JSON。
      - ``..._prompts.txt``    该局【每一步】发给小模型的完整 prompt 原文（不再只存
        第 1 步），可核对 skill tree / skills 是否真的按预期出现在 prompt 里。
    """
    d = os.path.join(outdir, "trajectories")
    os.makedirs(d, exist_ok=True)
    status = "WIN" if won else "FAIL"
    base = f"ep{episode_idx:04d}_{detected_task_type}_{status}_{used_steps}steps"

    pb_meta = None
    if playbook_record:
        pb_meta = {"version": playbook_record.get("version"),
                   "level": playbook_record.get("level"),
                   "n_nodes": len(playbook_record.get("nodes") or {})}

    # ---- JSON ----
    payload = {
        "episode": episode_idx, "task": task, "task_type": detected_task_type,
        "outcome": status, "used_steps": used_steps,
        "skill_tree_used": pb_meta,
        "skill_ids_used": list(skill_ids_used or []),
        "steps": logrows,
    }
    try:
        with open(os.path.join(d, base + "_episode.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[driver] episode json dump failed: {e}")

    # ---- 人类可读轨迹 ----
    try:
        lines = ["=" * 78,
                 f" Episode #{episode_idx}  task_type={detected_task_type}  "
                 f"[{status} / {used_steps} steps]",
                 "=" * 78,
                 f" task: {task}",
                 f" skill tree: v{pb_meta['version']} level={pb_meta['level']} "
                 f"({pb_meta['n_nodes']} nodes)" if pb_meta else " skill tree: (none)",
                 "=" * 78]
        for s in logrows:
            flag = "OK" if s["valid"] else "INVALID"
            forced = " [BUDGET-FORCED]" if s.get("forced") else ""
            lines.append("")
            lines.append(f"-- step {s['step']:>2} [{flag}]{forced} action={s['action']!r} "
                         f"source={s.get('execution_source', 'unknown')} "
                         f"reward={s.get('reward', 0)} won={s.get('won', False)}")
            lines.append(f"   obs: {_oneline(s.get('obs', ''))}")
        with open(os.path.join(d, base + "_trajectory.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[driver] episode trajectory.txt dump failed: {e}")

    # ---- 每步完整 prompt 原文 ----
    try:
        lines = ["#" * 78, f"# Episode #{episode_idx} 每步发给小模型的完整 PROMPT 原文",
                 f"# task: {task}  |  {status} / {used_steps} steps", "#" * 78]
        for s in logrows:
            lines.append("")
            lines.append("/" * 78)
            lines.append(f"// step {s['step']}  action={s['action']!r}  "
                         f"valid_action={s['valid']}  "
                         f"strict_valid_action={s.get('strict_valid_action', s['valid'])}  "
                         f"source={s.get('execution_source', 'unknown')}")
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
    """按实际使用的云端版本轮次汇总，直接比较进化前/后的表现。"""
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
