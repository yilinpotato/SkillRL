#!/usr/bin/env bash
# run.sh — 一键跑 pen→shelf 迷你测试
#
# 用法:
#   export ALFWORLD_DATA=/path/to/alfworld     # 已下载好的数据
#   export MODEL_PATH=/path/to/Qwen3-4B        # 本地模型目录
#   bash mini_test_pen_shelf/run.sh
#
# 可选参数会透传给 run_mini_test.py，例如:
#   bash mini_test_pen_shelf/run.sh --num_games 5 --max_steps 25
set -e

# 离线，避免联网拉模型/数据
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

: "${ALFWORLD_DATA:?请先 export ALFWORLD_DATA=/path/to/alfworld}"
: "${MODEL_PATH:?请先 export MODEL_PATH=/path/to/Qwen3-4B}"

# 切到项目根目录（本脚本的上一级），保证 `python -m` 能找到包
cd "$(dirname "$0")/.."

echo "================================================================"
echo " ALFWORLD_DATA = $ALFWORLD_DATA"
echo " MODEL_PATH    = $MODEL_PATH"
echo "================================================================"

# 第 0 步：零成本探查 pen 真值位置（不开模型）
echo ""
echo ">>> [0/1] 零成本探查 pen 真值位置分布 (不加载模型)"
python -m mini_test_pen_shelf.inspect_pen_locations --split train --limit 100 || true

# 第 1 步：带模型的完整迷你 rollout
echo ""
echo ">>> [1/1] vLLM + Qwen3-4B 跑 pen→shelf rollout"
python -m mini_test_pen_shelf.run_mini_test \
    --num_games "${NUM_GAMES:-3}" \
    --max_steps "${MAX_STEPS:-30}" \
    --gpu_mem_util "${GPU_MEM_UTIL:-0.55}" \
    "$@"
