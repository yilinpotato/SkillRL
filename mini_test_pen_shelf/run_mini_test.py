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


def build_obs_prompt(obs_text, admissible_commands):
    """与 env_manager.build_text_obs(init=True) 等价：用 NO_HIS 模板。"""
    reformatted = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")
    return ALFWORLD_TEMPLATE_NO_HIS.format(
        current_observation=obs_text,
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


def run_one_game(env, agent, game_file, traj, max_steps, game_idx):
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

    for step in range(1, max_steps + 1):
        prompt = build_obs_prompt(obs_text, adm)
        raw = agent.act(prompt)
        think, action_text = parse_model_output(raw)

        # 用项目的 projection 解析+校验动作（同时拿到合法标记）
        actions, valids = alfworld_projection([raw], [adm])
        action = actions[0]
        valid = bool(valids[0])

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

        obs_text, adm, prev_adm = nobs_text, nadm, new_adm
        if done:
            break

    R.print_game_footer(won, discovered, step)
    return won, discovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_games", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--split", default="train",
                    choices=["train", "valid_seen", "valid_unseen"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_thinking", action="store_true",
                    help="关闭 Qwen3 thinking 模式（更快但更弱）")
    args = ap.parse_args()

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

    # 4. 逐游戏跑
    total_loc = Counter()
    wins = 0
    for i, (game_file, traj) in enumerate(games):
        won, discovered = run_one_game(
            env, agent, game_file, traj, args.max_steps, i + 1
        )
        wins += int(won)
        for loc in discovered:
            total_loc[loc] += 1

    R.print_final_summary(len(games), wins, total_loc)


if __name__ == "__main__":
    main()
