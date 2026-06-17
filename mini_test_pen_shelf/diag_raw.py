"""diag_raw.py — 抓一条模型原始输出，确认 Thinking 模型到底吐什么格式。"""
import os


def main():
    from mini_test_pen_shelf.env_utils import load_tw_config, find_pen_shelf_games, make_single_env
    from mini_test_pen_shelf.run_mini_test import build_obs_prompt

    games = find_pen_shelf_games(split="train", limit=1)
    game_files = [g[0] for g in games]
    config = load_tw_config(num_games=len(game_files))
    env = make_single_env(game_files, config, seed=0)
    obs_list, infos = env.reset()
    obs_text = obs_list[0]
    adm = infos["admissible_commands"][0]
    prompt = build_obs_prompt(obs_text, adm)

    from mini_test_pen_shelf.agent_vllm import VLLMAgent
    agent = VLLMAgent(
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.55")),
        max_model_len=8192,
        max_tokens=int(os.environ.get("MAX_TOKENS", "2048")),
        temperature=0.4,
    )

    built = agent._build_prompt(prompt)
    print("\n##### CHAT-TEMPLATE 拼接后末尾 300 字: #####")
    print(repr(built[-300:]))

    raw = agent.act(prompt)
    print("\n##### 模型原始输出 (len=%d 字符): #####" % len(raw))
    print("--- 开头 400 ---")
    print(repr(raw[:400]))
    print("--- 结尾 400 ---")
    print(repr(raw[-400:]))
    print("\n##### 标签检测 #####")
    print("含 <think>:", "<think>" in raw, " 含 </think>:", "</think>" in raw)
    print("含 <action>:", "<action>" in raw, " 含 </action>:", "</action>" in raw)


if __name__ == "__main__":
    main()
