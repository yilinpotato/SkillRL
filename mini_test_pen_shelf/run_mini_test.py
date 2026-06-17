"""
run_mini_test.py — 迷你测试主程序

只跑 pen→shelf 任务，逐步详细打印环境变化，并统计 pen 实际出现的位置。

流程（单环境串行，history 关闭）：
  1. find_pen_shelf_games() 筛选游戏
  2. 对每个游戏：reset -> 看 pen 真值位置 -> 用 Qwen3-4B 一步步决策
  3. 每步打印：模型 think / action / 是否合法 / 新 observation / admissible 变化
  4. 汇总 pen 被发现的位置 + 成功率

用法:
  export ALFWORLD_DATA=/path/to/alfworld
  export MODEL_PATH=/path/to/Qwen3-4B
  python -m mini_test_pen_shelf.run_mini_test --num_games 3 --max_steps 30
"""
import re
import argparse
from collections import Counter

from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE_NO_HIS
from agent_system.environments.env_package.alfworld.projection import alfworld_projection

from mini_test_pen_shelf.env_utils import (
    load_tw_config,
    find_pen_shelf_games,
    extract_pen_ground_truth,
    make_single_env,
)
from mini_test_pen_shelf import report as R


def build_obs_prompt(obs_text, admissible_commands, use_strategy=False,
                     holding=None, target="pen", searched=None, found_here=None):
    """与 env_manager.build_text_obs(init=True) 等价：用 NO_HIS 模板。
    use_strategy=True 时在最前面拼上 pen→shelf 的自然语言策略 playbook。
    holding:  当前手里拿着的物体名（如 'pen 1'），None 表示空手。
    target:   任务真正要搬的物体（'pen' 或 'pencil'）。
    searched: 已经搜过且没有目标物的容器集合（如 {'drawer 1','drawer 2'}）。
    found_here: 当前位置看到的目标物名（如 'pen 1'），None 表示没看到。
        NO_HIS 模板无历史，手持/已搜过/眼前所见都必须显式注入，模型才不会绕圈。"""
    reformatted = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")
    lines = [f"[TARGET] You must put a '{target}' on a shelf. Only a {target} counts."]
    if holding:
        lines.append(
            f"[INVENTORY] You ARE holding {holding}. "
            f"Go to a shelf and place it with 'move {holding} to shelf <id>' "
            f"(copy the exact action from the list). Do not search anything else.")
    else:
        lines.append(f"[INVENTORY] Your hands are EMPTY. Keep searching for the {target}.")
        if found_here:
            lines.append(
                f"[HERE] You can see {found_here} at your current spot. "
                f"Take it now: 'take {found_here} from <recep> <id>'.")
        if searched:
            lines.append(
                "[ALREADY SEARCHED — do NOT revisit these, they are empty]: "
                + ", ".join(sorted(searched)) + ".")
    inv = "\n".join(lines)
    obs_with_inv = f"{obs_text}\n{inv}"
    if use_strategy:
        from mini_test_pen_shelf.strategy import build_strategy_prompt
        return build_strategy_prompt(
            ALFWORLD_TEMPLATE_NO_HIS, obs_with_inv, reformatted
        )
    return ALFWORLD_TEMPLATE_NO_HIS.format(
        current_observation=obs_with_inv,
        admissible_actions=reformatted,
    )


def extract_task(obs_text):
    start = obs_text.find("Your task is to: ")
    if start != -1:
        return obs_text[start + len("Your task is to: "):].strip()
    return "(未找到任务描述)"


# 从一段 observation 文本里找出现的 pen 所在容器（模型探索视角的「发现」）
_SEE_PEN = re.compile(
    r"(?:on|in)\s+the\s+([a-z]+)\s+\d*[,.]?[^.]*?\bpe(?:n|ncil)\b", re.IGNORECASE
)
_FOUND_HERE = re.compile(r"\bpe(?:n|ncil)\b", re.IGNORECASE)


def detect_pen_sighting(obs_text, current_receptacle):
    """
    若 observation 提到 pen/pencil，且我们正打开/查看某个 receptacle，
    则记录「在该 receptacle 看到了 pen」。返回容器类型或 None。
    """
    if not _FOUND_HERE.search(obs_text):
        return None
    if current_receptacle:
        # 归一化容器类型: "drawer 2" -> "drawer"
        return re.split(r"[\s\d]", current_receptacle.strip(), maxsplit=1)[0].lower()
    return None


def track_current_receptacle(action):
    """从动作里推断当前正在交互的 receptacle，如 'go to drawer 2' / 'open drawer 2'。"""
    m = re.search(r"(?:go to|open|examine|close)\s+(.+)", action.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_model_output(raw):
    """拆出 think 和 action 文本，便于打印。"""
    think = ""
    action = ""
    mt = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    if mt:
        think = mt.group(1).strip()
    ma = re.search(r"<action>(.*?)</action>", raw, re.DOTALL | re.IGNORECASE)
    if ma:
        action = ma.group(1).strip()
    return think, action


def run_one_game(env, agent, game_file, traj, max_steps, game_idx,
                 use_strategy=True, outdir=None):
    """跑单个 pen→shelf 游戏，返回 (won, discovered_pen_locations:set)。"""
    # 真值位置（零成本，从 game 文件解析）
    gt = extract_pen_ground_truth(game_file, traj)

    obs_list, infos = env.reset()
    obs_text = obs_list[0]
    adm = infos["admissible_commands"][0]
    task = extract_task(obs_text)

    R.print_game_header(game_idx, task, gt, game_file)

    discovered = set()
    current_recep = None
    prev_adm = set(adm)
    won = False
    step = 0
    n_valid = 0      # 本局合法动作数（统计合法率）
    holding = None   # 当前手里的物体；NO_HIS 模板无历史，需手动维护后注入 prompt
    searched = set() # 已搜过且无目标物的容器（"drawer 1"...），防止绕圈
    # 目标物体：以**环境实际任务串**为准（won 就是按它判定的），
    # 而非 game 文件的 object_target —— 二者偶尔不一致（如 game 标 pen 但
    # 环境任务是 "put some pencil on shelf"），信任后者才不会拿错物体死循环。
    tl = task.lower()
    if "pencil" in tl:
        target = "pencil"
    elif "pen" in tl:
        target = "pen"
    else:
        target = (gt.get("object_target") or "pen").strip().lower()
        if target not in ("pen", "pencil"):
            target = "pen"

    # 轨迹/prompt 落盘
    logger = None
    if outdir:
        from mini_test_pen_shelf.trajectory_logger import TrajectoryLogger
        logger = TrajectoryLogger(outdir, game_idx, task, target, gt)

    for step in range(1, max_steps + 1):
        # 当前位置是否看得到目标物（注入 [HERE] 提示，让模型立刻 take）
        found_here = None
        if not holding:
            mfh = re.search(rf"\b({target} \d+)\b", obs_text, re.IGNORECASE)
            if mfh:
                found_here = mfh.group(1).lower()
        prompt = build_obs_prompt(obs_text, adm, use_strategy=use_strategy,
                                  holding=holding, target=target,
                                  searched=searched, found_here=found_here)
        raw = agent.act(prompt)
        think, action_text = parse_model_output(raw)

        # 用项目的 projection 解析+校验动作（同时拿到合法标记）
        actions, valids = alfworld_projection([raw], [adm])
        action = actions[0]
        valid = bool(valids[0])
        if valid:
            n_valid += 1

        # 推断正在交互的 receptacle（用于 pen 发现归因）
        recep = track_current_receptacle(action)
        if recep:
            current_recep = recep

        # 环境前进一步
        nobs_list, scores, dones, ninfos = env.step([action])
        nobs_text = nobs_list[0]
        nadm = ninfos["admissible_commands"][0]
        done = bool(dones[0])
        won = bool(ninfos.get("won", [False])[0])

        # 更新手持状态（从 observation 文本推断）
        mpick = re.search(r"you pick up the (.+?) from", nobs_text, re.IGNORECASE)
        if mpick:
            holding = mpick.group(1).strip().lower()
        # 放置：本环境用 "You move the X to the Y" 也兼容 "put ... in/on"
        mplace = re.search(
            r"you (?:move|put) the (.+?) (?:to|in|on) the (\w+)", nobs_text, re.IGNORECASE)
        if mplace:
            holding = None
            # 目标物放到 shelf 上后就收尾（done=True），避免「放上去→won 未置位→
            # 被告知继续找→取回重放」的死循环。won 仍以环境判定为准，不伪造成功：
            # 若该局目标其实是 box/bowl 或拿错了物体，env 不给 won，如实记为失败。
            placed_obj = mplace.group(1).strip().lower()
            placed_dst = mplace.group(2).strip().lower()
            if target in placed_obj and placed_dst == "shelf":
                done = True

        # 标记「已搜过且没有目标物」的容器：仅当确实看到了内容物（打开后的容器，
        # 或台面类 sidetable/dresser/shelf 到达即可见）且其中无目标物时记入 searched。
        # 注意：到达一个「closed」的抽屉时内容物未知，绝不能标记为已搜。
        if not holding and recep:
            recep_norm = recep.strip().lower()
            contents_visible = re.search(
                r"(?:you open the .+?\. .+? is open|on the .+?, you see|in it, you see|"
                r"arrive at .+?\. on the)", nobs_text, re.IGNORECASE)
            is_closed = re.search(r"\bis closed\b", nobs_text, re.IGNORECASE)
            sees_target = re.search(rf"\b{target}\b", nobs_text, re.IGNORECASE)
            if contents_visible and not is_closed and not sees_target:
                searched.add(recep_norm)

        # pen 发现归因
        sighting = detect_pen_sighting(nobs_text, current_recep)
        if sighting:
            discovered.add(sighting)

        # admissible 变化
        new_adm = set(nadm)
        added = new_adm - prev_adm
        removed = prev_adm - new_adm

        R.print_step(
            step=step,
            think=think,
            action=action,
            valid=valid,
            obs=nobs_text,
            added=added,
            removed=removed,
            sighting=sighting,
            won=won,
        )

        if logger:
            logger.log_step(
                step=step, prompt=prompt, raw=raw, think=think,
                action=action, valid=valid, obs=nobs_text,
                holding=holding, searched=searched, found_here=found_here,
                won=won, reward=(scores[0] if scores is not None else None),
            )

        obs_text, adm, prev_adm = nobs_text, nadm, new_adm
        if done:
            break

    R.print_game_footer(won, discovered, step)
    if logger:
        logger.flush(won, step)
        print(f"  📁 轨迹与 prompt 已写入: {logger.outdir}")
    return won, discovered, step, n_valid, task, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_games", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--strategy", action="store_true",
                    help="在 prompt 前拼上 pen→shelf 自然语言策略 playbook")
    ap.add_argument("--split", default="train",
                    choices=["train", "valid_seen", "valid_unseen"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_thinking", action="store_true",
                    help="关闭 Qwen3 thinking 模式（更快但更弱）")
    ap.add_argument("--outdir", default=None,
                    help="把每局完整轨迹和逐步 prompt 落盘到该文件夹"
                         "（默认 mini_test_pen_shelf/output）")
    ap.add_argument("--repeats", type=int, default=1,
                    help="把同一个游戏重复跑 N 次（温度采样使每次轨迹不同），"
                         "用于统计同一任务上的成功率")
    args = ap.parse_args()

    # 输出目录：默认放在本包下的 output/
    outdir = args.outdir
    if outdir is None:
        import os as _os
        outdir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "output")

    # 1. 筛选 pen→shelf 游戏
    games = find_pen_shelf_games(split=args.split, limit=args.num_games)
    if not games:
        print("没有找到 pen→shelf 游戏，检查 ALFWORLD_DATA / split。")
        return
    game_files = [g[0] for g in games]

    # 2. 构造单环境（纯文本，无 GPU）
    config = load_tw_config(num_games=len(game_files))
    env = make_single_env(game_files, config, seed=args.seed)

    # 3. 加载模型（vLLM）
    from mini_test_pen_shelf.agent_vllm import VLLMAgent
    agent = VLLMAgent(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        enable_thinking=not args.no_thinking,
        seed=args.seed,
    )

    # 4. 逐游戏跑（每个游戏可重复 N 次，温度采样使每次不同）
    total_loc = Counter()
    wins = 0
    per_game = []          # 每次 run 的聚合指标，用于 summary.json 和 A/B 对比
    run_idx = 0
    for i, (game_file, traj) in enumerate(games):
        for rep in range(args.repeats):
            run_idx += 1
            # 每次重复换 seed 重置环境，制造布局/初始状态差异
            if args.repeats > 1:
                env.seed(args.seed + run_idx)
            won, discovered, used_steps, n_valid, task, target = run_one_game(
                env, agent, game_file, traj, args.max_steps, run_idx,
                use_strategy=args.strategy, outdir=outdir,
            )
            wins += int(won)
            for loc in discovered:
                total_loc[loc] += 1
            per_game.append({
                "game_idx": run_idx,
                "base_game": i + 1,
                "repeat": rep + 1,
                "task": task,
                "target": target,
                "won": bool(won),
                "used_steps": used_steps,
                "valid_actions": n_valid,
                "valid_rate": round(n_valid / max(used_steps, 1), 4),
                "discovered_pen_locations": sorted(discovered),
            })

    n = len(per_game)
    R.print_final_summary(n, wins, total_loc)

    # 5. 写 run 级别 summary.json（A/B 对比脚本读取它）
    import os as _os
    import json as _json
    win_games = [g for g in per_game if g["won"]]
    summary = {
        "strategy": bool(args.strategy),
        "split": args.split,
        "num_base_games": len(games),
        "repeats": args.repeats,
        "num_games": n,
        "max_steps": args.max_steps,
        "wins": wins,
        "success_rate": round(wins / max(n, 1), 4),
        "avg_steps_all": round(sum(g["used_steps"] for g in per_game) / max(n, 1), 2),
        "avg_steps_win": (round(sum(g["used_steps"] for g in win_games) / len(win_games), 2)
                          if win_games else None),
        "avg_valid_rate": round(sum(g["valid_rate"] for g in per_game) / max(n, 1), 4),
        "per_game": per_game,
    }
    spath = _os.path.join(outdir, "summary.json")
    with open(spath, "w") as f:
        _json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  📊 run 汇总已写入: {spath}")
    print(f"     strategy={summary['strategy']}  成功率={summary['success_rate']*100:.1f}%  "
          f"平均步数(全部)={summary['avg_steps_all']}  合法率={summary['avg_valid_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
