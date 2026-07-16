"""run_generic.py — 通用 ALFWorld mini 测试：支持任意 task_type 的 A/B 测试。

支持三类:
  1. pick_and_place_simple  (--mode generic)   任意 object -> receptacle
  2. pick_two_obj_and_place (--mode pick_two)   两个同类 object -> receptacle
  3. pen→shelf 仍可用原 run_mini_test.py

逐步注入 [TARGET]/[INVENTORY]/[PROGRESS]/[ALREADY SEARCHED]，弥补 NO_HIS 无记忆。
每局落盘完整轨迹 + 逐步 prompt，run 结束写 summary.json，供 compare_ab.py 对比。

用法:
  python -m mini_test_pen_shelf.run_generic --mode generic --num_games 10 \
      --max_steps 30 --strategy --outdir mini_test_pen_shelf/output_pp_strategy
  python -m mini_test_pen_shelf.run_generic --mode pick_two --num_games 1 \
      --repeats 5 --max_steps 40 --strategy --outdir .../output_p2_strategy
"""
import re
import os
import json
import argparse
from collections import Counter

from agent_system.environments.env_package.alfworld.projection import alfworld_projection

from mini_test_pen_shelf.env_utils import (
    load_tw_config_types,
    find_games_by_type,
    extract_task_target,
    make_single_env,
)
from mini_test_pen_shelf import report as R
from mini_test_pen_shelf.prod_prompt import ProdObsBuilder


MODE_TASK_TYPE = {
    "generic": "pick_and_place_simple",
    "pick_two": "pick_two_obj_and_place",
}
MODE_TASK_ID = {"generic": 1, "pick_two": 6}


def extract_task(obs_text):
    start = obs_text.find("Your task is to: ")
    if start != -1:
        return obs_text[start + len("Your task is to: "):].strip()
    return "(未找到任务描述)"


def parse_task_string(task, fallback_obj="object", fallback_recep="receptacle",
                      fallback_count=1):
    """从环境真实任务串解析 (object, receptacle, count)。
    任务串就是环境判定 won 的依据，比预解析的 traj 更可靠——之前用 traj 因 env.reset
    取游戏顺序与抽样顺序错配，导致注入的 [TARGET] 与模型实际所玩游戏不符、死循环。
    支持:
      "put a/some <OBJ> in/on the <RECEP>"
      "put two <OBJ> in/on the <RECEP>" / "find two <OBJ> and put them in <RECEP>"
    """
    t = task.lower()
    count = 2 if re.search(r"\btwo\b", t) else fallback_count
    # 物体：put/find 后、in/on/and 前的名词
    obj = fallback_obj
    recep = fallback_recep
    m = re.search(r"(?:put|find|move|place)\s+(?:a |some |two |the )?"
                  r"([a-z]+?)s?\b.*?\b(?:in|on|into|onto|to)\s+(?:the |a )?([a-z]+)", t)
    if m:
        obj = m.group(1).strip()
        recep = m.group(2).strip()
    return obj, recep, count


def parse_model_output(raw):
    think, action = "", ""
    mt = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    if mt:
        think = mt.group(1).strip()
    ma = re.search(r"<action>(.*?)</action>", raw, re.DOTALL | re.IGNORECASE)
    if ma:
        action = ma.group(1).strip()
    return think, action


def track_recep(action):
    m = re.search(r"(?:go to|open|examine|close)\s+(.+)", action.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else None


def run_one_game(env, agent, traj, mode, max_steps, run_idx, builder, outdir):
    # traj 仅作 fallback；真正目标以环境 reset 后的任务串为准（避免顺序错配）
    tgt = extract_task_target(traj)
    fb_obj = tgt["object"] or "object"
    fb_recep = tgt["parent"] or "receptacle"
    fb_count = tgt["count"]

    obs_list, infos = env.reset()
    obs_text = obs_list[0]
    adm = infos["admissible_commands"][0]
    task = extract_task(obs_text)

    # 从环境真实任务串解析目标（权威来源，env 据此判 won）
    target_obj, target_recep, need_count = parse_task_string(
        task, fb_obj, fb_recep, fb_count)
    # pick_two 模式恒为 2
    if mode == "pick_two":
        need_count = 2

    R.print_game_header(run_idx, task,
                        {"object_target": target_obj, "parent_target": target_recep,
                         "pen_locations": []}, "")

    # 与主管线一致的 prompt 构造器（playbook+history，可选 bullets）；每局重置历史与检索。
    builder.reset(task)

    logger = None
    if outdir:
        from mini_test_pen_shelf.trajectory_logger import TrajectoryLogger
        logger = TrajectoryLogger(outdir, run_idx, task, target_obj,
                                  {"pen_locations": [], "object_target": target_obj,
                                   "parent_target": target_recep})

    holding = None
    searched = set()
    closed_pending = set()  # 到达过但还没打开的关闭容器（cabinet/drawer/fridge）
    placed_ids = set()      # 已放进目标容器的【具体实例】名（如 'soapbottle 1'），去重计数
    failed_actions = set()  # 导致 "Nothing happens" 的无效动作，注入 prompt 阻止重复
    last_action = None
    repeat_count = 0        # 同一动作连续重复次数
    won = False
    step = 0
    n_valid = 0
    n_truncated = 0         # thinking 撞 think_budget 被预算强制收尾的步数
    n_salvaged = 0          # 强制收尾后仍需从后往前匹配救回动作的步数
    prev_adm = set(adm)

    for step in range(1, max_steps + 1):
        # 与主管线 build_text_obs 同款：首步 NO_HIS(+playbook)，其后 WITH_MEMORY+history。
        prompt = builder.build(obs_text, adm, init=(step == 1))
        raw, forced = agent.act_with_meta(prompt)
        think, _ = parse_model_output(raw)
        # alfworld_projection 现在自己就会做 admissible_commands 精确匹配 + salvage +
        # 安全默认动作兜底，返回的 action 已经保证合法，不需要在这里再手工补救一次。
        actions, valids, action_details = alfworld_projection(
            [raw], [adm], return_details=True
        )
        action = actions[0]
        valid = bool(valids[0])
        # ``valid`` is now the SkillRL-compatible non-strict protocol score;
        # it intentionally does not say whether the executable action needed a
        # salvage/default fallback.  Keep the mini-test recovery statistic tied
        # to the projection's execution source instead.
        salvaged = action_details[0]["execution_source"] != "direct"
        if forced:
            n_truncated += 1
            tag = "✅直接执行" if not salvaged else "⚠已救回(salvage/默认)"
            print(f"  ⏱ [预算强制] Step {step} thinking 到 think_budget 被强制收尾 → {tag}: {action!r}")
        if valid:
            n_valid += 1
        if salvaged:
            n_salvaged += 1
        recep = track_recep(action)

        nobs_list, scores, dones, ninfos = env.step([action])
        nobs_text = nobs_list[0]
        nadm = ninfos["admissible_commands"][0]
        done = bool(dones[0])
        won = bool(ninfos.get("won", [False])[0])

        # 无效动作检测：环境回 "Nothing happens" 说明该动作虽格式合法但环境拒绝
        # （如 go to 一个不在 admissible 的 id）。记入 failed_actions 注入 prompt，
        # 并统计连续重复，防止像 game_09 那样 26 次重复同一无效动作卡死。
        if re.search(r"nothing happens", nobs_text, re.IGNORECASE):
            failed_actions.add(action)
        if action == last_action:
            repeat_count += 1
        else:
            repeat_count = 0
        last_action = action

        # 手持/放置追踪
        mpick = re.search(r"you pick up the (.+?) from the (\w+)", nobs_text, re.IGNORECASE)
        if mpick:
            holding = mpick.group(1).strip().lower()
            picked_from = mpick.group(2).strip().lower()
            # 若把已放进目标容器的实例又取回来，撤销其已放置计数
            if target_recep in picked_from and holding in placed_ids:
                placed_ids.discard(holding)
        mplace = re.search(r"you (?:move|put) the (.+?) (?:to|in|on) the (\w+)",
                           nobs_text, re.IGNORECASE)
        if mplace:
            placed_obj = mplace.group(1).strip().lower()   # 如 'soapbottle 1'
            placed_dst = mplace.group(2).strip().lower()
            holding = None
            if target_obj in placed_obj and target_recep in placed_dst:
                # 按【具体实例名】去重计数，避免同一个物体放两次被算成两个
                placed_ids.add(placed_obj)
                if len(placed_ids) >= need_count:
                    done = True

        # 标记搜过且无目标物的容器
        if not holding and recep:
            recep_norm = recep.strip().lower()
            contents_visible = re.search(
                r"(?:you open the .+?\. .+? is open|on the .+?, you see|in it, you see|"
                r"arrive at .+?\. on the)", nobs_text, re.IGNORECASE)
            is_closed = re.search(r"\bis closed\b", nobs_text, re.IGNORECASE)
            sees_target = re.search(rf"\b{re.escape(target_obj)}\b", nobs_text, re.IGNORECASE)
            if contents_visible and not is_closed and not sees_target:
                searched.add(recep_norm)
            # 到达一个【关闭】容器：内容未知，记入 closed_pending 提示模型「先 open 再判断」，
            # 避免反复 go to 同一个没开的柜子（cabinet/drawer/fridge）空耗步数。
            if is_closed:
                closed_pending.add(recep_norm)
            # 一旦打开（不再 closed），从 pending 移除
            if recep_norm in closed_pending and re.search(r"is open\b", nobs_text, re.IGNORECASE):
                closed_pending.discard(recep_norm)

        new_adm = set(nadm)
        R.print_step(step=step, think=think, action=action, valid=valid,
                     obs=nobs_text, added=new_adm - prev_adm,
                     removed=prev_adm - new_adm, sighting=None, won=won)
        if logger:
            logger.log_step(step=step, prompt=prompt, raw=raw, think=think,
                            action=action, valid=valid, obs=nobs_text,
                            holding=holding, searched=searched, found_here=None,
                            won=won, reward=(scores[0] if scores is not None else None),
                            truncated=forced, salvaged=salvaged)

        # 记入历史（本步动作所基于的 raw obs + 动作），供下一步 WITH_MEMORY 拼 history。
        builder.record(obs_text, action)
        obs_text, adm, prev_adm = nobs_text, nadm, new_adm
        if done:
            break
        # 安全网：同一动作连续重复 ≥6 次仍无进展（如 game_09 的 26 次 go to shelf 4），
        # 判定卡死，提前结束本局，避免白白耗满 max_steps。
        if repeat_count >= 6:
            print(f"  🔁 Step {step} 动作 {action!r} 连续重复 {repeat_count+1} 次仍无效，判定卡死，提前结束")
            break

    R.print_game_footer(won, set(), step)
    if n_truncated:
        print(f"  ⏱ 本局 {n_truncated} 步思考到预算被强制收尾"
              f"（其中 {n_salvaged} 步动作阶段后仍靠从后往前匹配救回）")
    if logger:
        logger.flush(won, step)
        print(f"  📁 轨迹与 prompt 已写入: {logger.outdir}")
    return won, step, n_valid, task, target_obj, target_recep, n_truncated, n_salvaged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["generic", "pick_two"], required=True)
    ap.add_argument("--num_games", type=int, default=10)
    ap.add_argument("--sample", action="store_true",
                    help="从全部命中游戏里跨物体均匀抽 num_games 个（避免取前 N 个全是同物体过拟合）")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=30)
    # prompt 与主管线对齐：只允许注入云端运行中生成的 playbook、history 默认 2、skill-bullets 默认关。
    ap.add_argument("--with_skills", action="store_true",
                    help="注入 general/task/mistakes 三类 bullet 技能（默认关，先不加 skills）")
    ap.add_argument("--no_playbook", action="store_true",
                    help="关闭结构化 playbook（默认开）")
    ap.add_argument("--history_length", type=int, default=2,
                    help="最近历史步数，与主管线默认一致(=2)")
    ap.add_argument("--skills_json", default=None,
                    help="技能库 JSON 路径（默认用 memory_data/alfworld/claude_style_skills.json）")
    ap.add_argument("--strategy", action="store_true",
                    help="[deprecated] 旧 A/B 开关，已无效（playbook 默认开）；保留以兼容旧脚本")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--nowait", action="store_true",
                    help="开启 NoWait（默认关闭，抑制 Wait/Hmm 等回溯词）")
    ap.add_argument("--think_budget", type=int, default=3500,
                    help="思考预算 token 数：到此还没 </think> 就强制收尾出 action")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    task_type = MODE_TASK_TYPE[args.mode]
    if args.sample:
        games = find_games_by_type(task_type, split=args.split,
                                   sample_n=args.num_games, sample_seed=args.sample_seed)
    else:
        games = find_games_by_type(task_type, split=args.split, limit=args.num_games)
    if not games:
        print(f"没有找到 {task_type} 游戏。")
        return
    game_files = [g[0] for g in games]
    config = load_tw_config_types([MODE_TASK_ID[args.mode]], num_games=len(game_files))
    env = make_single_env(game_files, config, seed=args.seed)

    from mini_test_pen_shelf.agent_vllm import VLLMAgent
    agent = VLLMAgent(model_path=args.model_path,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len, temperature=args.temperature,
                      enable_thinking=not args.no_thinking, seed=args.seed,
                      no_wait=args.nowait, think_budget=args.think_budget)

    # 与主管线同款 prompt 构造器（复用 SkillsOnlyMemory + SimpleMemory + 模板）。
    builder = ProdObsBuilder(skills_json_path=args.skills_json,
                             history_length=args.history_length,
                             with_skills=args.with_skills,
                             enable_playbook=not args.no_playbook)
    print(f"[prod_prompt] playbook={'on' if not args.no_playbook else 'off'} "
          "handwritten_seed=off "
          f"skills(bullets)={'on' if args.with_skills else 'off'} "
          f"history_length={args.history_length}")

    per_game = []
    wins = 0
    run_idx = 0
    tot_trunc = 0
    tot_salv = 0
    for i, (gf, traj) in enumerate(games):
        for rep in range(args.repeats):
            run_idx += 1
            if args.repeats > 1:
                env.seed(args.seed + run_idx)
            won, used, nval, task, tobj, trec, ntrunc, nsalv = run_one_game(
                env, agent, traj, args.mode, args.max_steps, run_idx,
                builder, args.outdir)
            wins += int(won)
            tot_trunc += ntrunc
            tot_salv += nsalv
            per_game.append({"game_idx": run_idx, "base_game": i + 1, "repeat": rep + 1,
                             "task": task, "target": tobj, "receptacle": trec,
                             "won": bool(won), "used_steps": used,
                             "valid_actions": nval,
                             "valid_rate": round(nval / max(used, 1), 4),
                             "truncated_steps": ntrunc, "salvaged_steps": nsalv})

    n = len(per_game)
    R.print_final_summary(n, wins, Counter())
    win_g = [g for g in per_game if g["won"]]
    tot_steps = sum(g["used_steps"] for g in per_game)
    summary = {"with_skills": bool(args.with_skills),
               "playbook": (not args.no_playbook),
               "handwritten_seed_playbook": False,
               "history_length": args.history_length, "mode": args.mode,
               "task_type": task_type, "split": args.split,
               "num_base_games": len(games), "repeats": args.repeats,
               "num_games": n, "max_steps": args.max_steps, "wins": wins,
               "success_rate": round(wins / max(n, 1), 4),
               "avg_steps_all": round(tot_steps / max(n, 1), 2),
               "avg_steps_win": (round(sum(g["used_steps"] for g in win_g) / len(win_g), 2)
                                 if win_g else None),
               "avg_valid_rate": round(sum(g["valid_rate"] for g in per_game) / max(n, 1), 4),
               "truncated_steps_total": tot_trunc,
               "salvaged_steps_total": tot_salv,
               "truncation_rate": round(tot_trunc / max(tot_steps, 1), 4),
               "per_game": per_game}
    spath = os.path.join(args.outdir, "summary.json")
    json.dump(summary, open(spath, "w"), ensure_ascii=False, indent=2)
    print(f"\n  📊 run 汇总: {spath}")
    print(f"     mode={args.mode} playbook={summary['playbook']} with_skills={summary['with_skills']} "
          f"成功率={summary['success_rate']*100:.1f}% "
          f"平均步数={summary['avg_steps_all']} 合法率={summary['avg_valid_rate']*100:.1f}%")
    print(f"     thinking 截断 {tot_trunc} 步 / 救回 {tot_salv} 步 "
          f"(截断率 {summary['truncation_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
