"""run_playbook_evolve.py — 无 verl/Ray/FSDP 的独立 Skill Tree 进化 driver。

冻结小模型（vLLM 直接加载，只一份）在 ALFWorld 上 rollout → 轨迹进 TracesPool →
水位线触发 CoSkillCloudLoop（失败诊断 + skill tree 从零生成/层次化细化 + 可选 skill 蒸馏）
→ 进化后的 agent skill tree 写回同一个 skill_lib，下一局 reset() 即注入。

所有运行条件对齐 examples/grpo_trainer/run_alfworld_skills_train_update_lora.sh
（模型 / embedding 检索 / 环境 / 记忆分层 / 云端 / 水位线 / group_size 采样），
仅去掉 RL 权重训练（无 Ray、无 FSDP、无第二份模型、无反向传播 / checkpoint）。

复用件：mini_test 的 env_utils（进程内单环境）、agent_vllm（budget forcing）、
prod_prompt.ProdObsBuilder（与 env_manager.build_text_obs 逐字节对齐，注入共享 skill_lib）、
run_generic 的解析/兜底工具；闭环三件套 + HierarchicalSkillLib + CoSkillCloudLoop。
"""
import os
import json
import uuid
import argparse

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.memory import TracesPool, HierarchicalSkillLib, CoSkillCloudLoop

from mini_test_pen_shelf.env_utils import (
    load_tw_config_types, find_games_by_type, make_single_env, make_batch_env, _TASK_TYPE_TO_ID,
)
from mini_test_pen_shelf.prod_prompt import ProdObsBuilder
from mini_test_pen_shelf.run_generic import (
    extract_task, parse_model_output, salvage_action_from_back,
)


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


def rollout_episode(env, agent, builder, max_steps, tag="", keep_logrows=True):
    """跑一局（冻结模型）。返回 (won, steps_used, raw_trace, task_type, injected_ids, n_valid, logrows)。

    成功判定以环境 ``won`` 为权威（对所有 task_type 有效）。轨迹步保存模型实际看到的
    raw obs + 动作 + reward，供组 RawTrace 喂 TracesPool。

    ``tag`` 仅用于逐步进度打印：每步含 2 次 vLLM budget-forcing 生成，冻结模型下
    单步约 20-30s（enforce_eager + 长思考），max_steps=40 时一整局可达十几分钟。
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
    last_action = None
    repeat = 0

    for step in range(1, max_steps + 1):
        _t0 = _time.time()
        prompt = builder.build(obs_text, adm, init=(step == 1))
        raw, forced = agent.act_with_meta(prompt)
        _, _think = None, None
        actions, valids = alfworld_projection([raw], [adm])
        action = actions[0]
        valid = bool(valids[0])
        if not valid:
            sa, ok = salvage_action_from_back(raw, adm)
            if ok:
                action = sa
        if valid:
            n_valid += 1

        nobs_list, scores, dones, ninfos = env.step([action])
        nobs = nobs_list[0]
        nadm = ninfos["admissible_commands"][0]
        reward = float(scores[0]) if scores is not None else 0.0
        done = bool(dones[0])
        won = bool(ninfos.get("won", [False])[0])
        print(f"  [rollout]{tag} step={step}/{max_steps} action={action!r} "
              f"valid={valid} forced={forced} won={won} ({_time.time()-_t0:.1f}s)")

        # RawTrace step: 模型这一步所基于的 raw obs + 采取的动作 + reward。
        steps.append({"step": step, "observation": obs_text, "action": action, "reward": reward})
        if keep_logrows:
            logrows.append({"step": step, "prompt": prompt, "action": action, "valid": valid,
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
        "meta": {"skill_ids_used": injected_ids, "model_version": "frozen"},
    }
    return won, step, raw_trace, task_type, injected_ids, n_valid, logrows


def rollout_batch_group(env, agent, skill_lib, args, batch_size, tag=""):
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
    last_action = [None for _ in range(batch_size)]
    repeat = [0 for _ in range(batch_size)]

    keep_logrows = bool(args.log_trajectories)

    for step in range(1, args.max_steps + 1):
        active = [i for i in range(batch_size) if not done[i]]
        if not active:
            break

        _t0 = _time.time()
        prompts = [
            builders[i].build(obs_list[i], adms[i], init=(step == 1))
            for i in active
        ]
        raw_forced = agent.act_batch_with_meta(prompts)
        raws = [x[0] for x in raw_forced]
        forceds = [x[1] for x in raw_forced]
        active_adms = [adms[i] for i in active]
        actions, valids = alfworld_projection(raws, active_adms)

        full_actions = []
        action_by_idx = {}
        valid_by_idx = {}
        forced_by_idx = {}
        raw_by_idx = {}
        for local_i, i in enumerate(active):
            action = actions[local_i]
            valid = bool(valids[local_i])
            if not valid:
                sa, ok = salvage_action_from_back(raws[local_i], adms[i])
                if ok:
                    action = sa
            if valid:
                n_valid[i] += 1
            action_by_idx[i] = action
            valid_by_idx[i] = valid
            forced_by_idx[i] = bool(forceds[local_i])
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

            steps[i].append({
                "step": step, "observation": obs_list[i],
                "action": action, "reward": reward,
            })
            if keep_logrows:
                logrows[i].append({
                    "step": step, "prompt": prompts[active.index(i)],
                    "action": action, "valid": valid_by_idx[i],
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
            "meta": {"skill_ids_used": injected_ids[i], "model_version": "frozen"},
        }
        episodes.append({
            "won": won[i], "used": used[i] or args.max_steps,
            "raw_trace": raw_trace, "task_type": task_types[i],
            "injected": injected_ids[i], "n_valid": n_valid[i],
            "logrows": logrows[i], "playbook_record": playbook_records[i],
        })
    return episodes


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
    ap.add_argument("--history_length", type=int, default=8,
                    help="WITH_MEMORY 模板携带的最近历史步数（obs+action）。训练脚本/mini_test 默认 2；"
                         "这里默认调大到 8，让 NO_HIS 记忆缺口更小，减少小模型因看不到早前状态而绕圈子。")
    # --- 模型 / vLLM（对齐训练脚本的 max_prompt/response）---
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.8)
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
    ap.add_argument("--max_new_skills", type=int, default=3)
    ap.add_argument("--skill_tree_evolve_min_samples",
                    dest="playbook_evolve_min_samples", type=int, default=6,
                    metavar="SKILL_TREE_EVOLVE_MIN_SAMPLES")
    ap.add_argument("--playbook_evolve_min_samples",
                    dest="playbook_evolve_min_samples", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--coskill_debug", type=int, default=0)
    # --- 轨迹池水位线（对齐训练脚本）---
    ap.add_argument("--capacity_watermark", type=int, default=50000)
    ap.add_argument("--perf_watermark", type=float, default=0.6)
    ap.add_argument("--min_samples", type=int, default=16)
    ap.add_argument("--loop_threshold", type=int, default=3)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log_trajectories", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    enable_coskill = bool(args.enable_coskill)
    enable_skill_tree = bool(args.enable_skill_tree)
    enable_skill_tree_evolve = enable_skill_tree and bool(args.enable_playbook_evolve)
    if bool(args.enable_playbook_evolve) and not enable_skill_tree:
        print("[driver] enable_skill_tree=0, so skill tree evolution is disabled too")

    # 1) 共享技能库（与训练脚本同参）。skill tree 进化写回它、rollout 从它检索，形成闭环。
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
        enable_playbook=enable_skill_tree,
    )

    # 2) 轨迹池（同水位线）。
    traces_pool = TracesPool(
        capacity_watermark=args.capacity_watermark,
        perf_watermark=args.perf_watermark,
        min_samples=args.min_samples,
        loop_threshold=args.loop_threshold,
        output_dir=args.outdir,
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
    if batch_rollout_size == 1:
        env = make_single_env(all_game_files, config, seed=args.seed)
    else:
        env = make_batch_env(all_game_files, config, batch_size=batch_rollout_size, seed=args.seed)
        print(f"[driver] batch_rollout_size={batch_rollout_size}: cloud updates are checked "
              "only after a full rollout group finishes")

    # 5) vLLM 冻结模型（只此一份）。放在所有 env 建好之后，避免上面的 fork-after-CUDA 问题。
    from mini_test_pen_shelf.agent_vllm import VLLMAgent
    agent = VLLMAgent(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=not args.no_thinking,
        seed=args.seed,
        no_wait=args.nowait,
        think_budget=args.think_budget,
    )

    # 6) 与主管线逐字节对齐的 prompt 构造器，注入共享 skill_lib。
    #    with_skills 跟随 enable_coskill（训练脚本注入 general/task/mistakes bullets）。
    builder = ProdObsBuilder(mem_lib=skill_lib, with_skills=enable_coskill, top_k=args.top_k,
                             history_length=args.history_length)
    print(f"[driver] retrieval_mode={args.retrieval_mode} with_skills={enable_coskill} "
          f"enable_skill_tree={enable_skill_tree} "
          f"enable_skill_tree_evolve={enable_skill_tree_evolve} "
          f"enable_failure_analysis={bool(args.enable_failure_analysis)}")

    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    per_game = []
    wins = 0
    global_step = 0
    tt_stats = {}  # detected_type -> {"episodes":int, "wins":int}
    cloud_updates = 0
    cloud_update_steps = []

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
                     "cloud_round_used": cloud_updates,
                     "skill_tree_enabled": enable_skill_tree,
                     "skill_tree_evolve_enabled": enable_skill_tree_evolve,
                     "skill_bullets_enabled": enable_coskill,
                     "skill_ids_used": list(injected),
                     "running_total_episodes": len(per_game) + 1,
                     "running_total_wins": wins,
                     "task_type_episodes": ts["episodes"],
                     "task_type_wins": ts["wins"]}
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
                "skill_tree/version": pb_rec.get("version") if pb_rec else 0,
                "skill_tree/level": pb_rec.get("level") if pb_rec else None,
                "skill_tree/n_nodes": len(pb_rec.get("nodes") or {}) if pb_rec else 0,
                **cloud_loop.metrics(traces_pool, skill_lib),
            },
        })

    stop_early = False
    for epoch in range(args.epochs):
        if stop_early:
            break
        ep_i = 0
        while ep_i < episodes_per_epoch:
            if args.max_episodes > 0 and global_step >= args.max_episodes:
                print(f"[driver] max_episodes={args.max_episodes} reached, stopping early")
                stop_early = True
                break

            if batch_rollout_size == 1:
                tag = f" epoch{epoch+1} ep{ep_i+1}/{episodes_per_epoch} (global_step={global_step+1})"
                won, used, raw_trace, tt_detected, injected, nval, logrows = \
                    rollout_episode(env, agent, builder, args.max_steps, tag=tag,
                                    keep_logrows=bool(args.log_trajectories))
                pb_rec = skill_lib.get_playbook_record(tt_detected) if enable_skill_tree and hasattr(
                    skill_lib, "get_playbook_record") else None
                ep_result = {"won": won, "used": used, "raw_trace": raw_trace,
                             "task_type": tt_detected, "injected": injected,
                             "n_valid": nval, "logrows": logrows,
                             "playbook_record": pb_rec}
                ep_record, pb_used = _ingest_episode_result(epoch, ep_result)
                ep_i += 1

                force_reason = None
                if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                    force_reason = f"episode_interval_{args.cloud_update_every}"
                fired = cloud_loop.maybe_update(
                    traces_pool, skill_lib, global_step, force_reason=force_reason
                )
                ep_record["cloud_update_fired_after_episode"] = bool(fired)
                if fired:
                    cloud_updates += 1
                    cloud_update_steps.append(global_step)
                _write_episode_metric(epoch, ep_record, pb_used, fired)
                continue

            remaining_epoch = episodes_per_epoch - ep_i
            remaining_cap = (args.max_episodes - global_step) if args.max_episodes > 0 else remaining_epoch
            n_to_count = min(batch_rollout_size, remaining_epoch, remaining_cap)
            if n_to_count <= 0:
                stop_early = True
                break

            tag = f" epoch{epoch+1} group_ep{ep_i+1}-{ep_i+n_to_count}/{episodes_per_epoch} "\
                  f"(global_step={global_step+1})"
            group_results = rollout_batch_group(env, agent, skill_lib, args, batch_rollout_size, tag=tag)
            ingested = []
            for ep_result in group_results[:n_to_count]:
                ingested.append(_ingest_episode_result(epoch, ep_result))
            ep_i += n_to_count

            # Batch mode: cloud update can happen only after the whole rollout group
            # has completed and all counted trajectories have been ingested.
            force_reason = None
            if args.cloud_update_every > 0 and global_step % args.cloud_update_every == 0:
                force_reason = f"episode_interval_{args.cloud_update_every}"
            fired = cloud_loop.maybe_update(
                traces_pool, skill_lib, global_step, force_reason=force_reason
            )
            if fired:
                cloud_updates += 1
                cloud_update_steps.append(global_step)

            for j, (ep_record, pb_used) in enumerate(ingested):
                is_last = (j == len(ingested) - 1)
                ep_record["cloud_update_fired_after_episode"] = bool(fired and is_last)
                _write_episode_metric(epoch, ep_record, pb_used, fired and is_last)

    # 落盘 summary。
    n = len(per_game)
    final_trees = {}
    if enable_skill_tree and hasattr(skill_lib, "task_playbooks"):
        for tt, rec in (getattr(skill_lib, "task_playbooks", {}) or {}).items():
            if isinstance(rec, dict):
                final_trees[tt] = {
                    "version": rec.get("version", 0),
                    "level": rec.get("level"),
                    "n_nodes": len(rec.get("nodes") or {}),
                }
    summary = {
        "task_types": task_types, "epochs": args.epochs,
        "num_games_per_type": args.num_games, "group_size": args.group_size,
        "fixed_games_manifest": args.fixed_games_manifest,
        "fixed_game_files": all_game_files if args.fixed_games_manifest else [],
        "skill_tree_enabled": enable_skill_tree,
        "skill_tree_evolve_enabled": enable_skill_tree_evolve,
        "skill_bullets_enabled": enable_coskill,
        "cloud_update_every": args.cloud_update_every,
        "cloud_update_steps": cloud_update_steps,
        "total_games_combined_pool": total_games,
        "total_episodes": n, "wins": wins,
        "success_rate": round(wins / max(n, 1), 4),
        "skill_tree_versions": {tt: rec["version"] for tt, rec in final_trees.items()},
        "skill_tree_nodes": {tt: rec["n_nodes"] for tt, rec in final_trees.items()},
        # Backward-compatible scalar: max version among task trees.
        "skill_tree_version": max([rec["version"] for rec in final_trees.values()] or [0]),
        "final_coskill_metrics": cloud_loop.metrics(traces_pool, skill_lib),
        "phase_stats": _phase_stats(per_game),
        "per_game": per_game,
    }
    json.dump(summary, open(os.path.join(args.outdir, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
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
            lines.append(f"// step {s['step']}  action={s['action']!r}  valid={s['valid']}")
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
