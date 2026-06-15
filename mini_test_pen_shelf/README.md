# Mini Test — "put some pen on shelf"

一个**自包含**的迷你测试环境,只为回答两个问题：

1. **pen 通常出现在哪些位置？**（哪些 receptacle 里能找到 pen / pencil）
2. **环境随动作发生哪些变化？**（每一步 observation / admissible actions 怎么变）

它**绕过** verl / Ray / GRPO / LoRA / skill-memory 整套训练管线，直接调用最底层的纯文本环境
`AlfredTWEnv`（TextWorld 变体，**零渲染、零 GPU 占用**），只筛选出
`pick_and_place_simple` 且目标是 **pen → shelf** 的游戏来跑。

消耗极小，3090 可快速运行。模型部分用 vLLM 加载 **Qwen3-4B**（bf16 约 8–9GB 显存）。

---

## 目录文件

| 文件 | 作用 | 是否用 GPU |
|------|------|-----------|
| `env_utils.py` | 加载 `AlfredTWEnv`、筛选 pen→shelf 游戏、解析 game 文件拿 pen **真值位置** | 否 |
| `inspect_pen_locations.py` | 只统计 pen 真值位置分布，**完全不加载模型** | 否 |
| `agent_vllm.py` | vLLM 封装 Qwen3-4B，按 prompt 模板产出 `<think>/<action>` | 是 |
| `run_mini_test.py` | 主程序：跑 N 个 pen→shelf 游戏，详细打印每步环境变化 + pen 位置统计 | 是 |
| `run.sh` | 一键脚本，设好环境变量后调用上面两个 | — |

---

## 用法

### 0. 前置（在 3090 机器上）
```bash
export ALFWORLD_DATA=/path/to/alfworld        # 已下载好的 json_2.1.1 + game.tw-pddl
export MODEL_PATH=/path/to/Qwen3-4B           # 本地模型目录
```

### 1. 先做零成本探查（不开模型，秒级）
看 pen 在数据集里到底初始出现在哪些 receptacle：
```bash
python -m mini_test_pen_shelf.inspect_pen_locations
```

### 2. 跑带模型的完整迷你 rollout
```bash
bash mini_test_pen_shelf/run.sh
# 或
python -m mini_test_pen_shelf.run_mini_test --num_games 3 --max_steps 30
```

输出包含：
- 每个游戏的任务描述、pen 真值初始位置
- 每一步：模型 think、选择的 action、是否合法、环境新 observation、admissible actions 变化（新增/消失）
- 结尾：本次跑到的所有 pen 出现位置汇总 + 成功率

> 全程单环境串行、`history_length` 关闭、`AlfredTWEnv` 纯文本，显存只由 vLLM 4B 决定。
