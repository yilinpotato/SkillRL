#!/usr/bin/env bash
# run_ab.sh — A/B 对比：有 strategy template vs 无 template，全部 13 个 pen→shelf 游戏。
#
# 两臂并行：A 臂(有策略)用 GPU 0，B 臂(无策略)用 GPU 1，各自归档到独立文件夹，
# 跑完自动生成对比报告。
#
# 用法:
#   export MODEL_PATH=/path/to/Qwen3-4B-Thinking
#   export ALFWORLD_DATA=/path/to/alfworld
#   bash mini_test_pen_shelf/run_ab.sh                 # 默认 13 局、40 步
#   NUM_GAMES=13 MAX_STEPS=40 bash mini_test_pen_shelf/run_ab.sh
set -e

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"

cd "$(dirname "$0")/.."
ROOT="mini_test_pen_shelf"
OUT_A="$ROOT/output_strategy"     # A: 有 template
OUT_B="$ROOT/output_baseline"     # B: 无 template
OUT_AB="$ROOT/output_ab"          # 对比报告归档
mkdir -p "$OUT_A" "$OUT_B" "$OUT_AB"

NUM_GAMES="${NUM_GAMES:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
REPEATS="${REPEATS:-8}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

echo "================================================================"
echo " A/B 对比  |  base_games=$NUM_GAMES  repeats=$REPEATS  max_steps=$MAX_STEPS"
echo " A(有template) -> GPU $GPU_A -> $OUT_A"
echo " B(无template) -> GPU $GPU_B -> $OUT_B"
echo "================================================================"

# A 臂：带策略
CUDA_VISIBLE_DEVICES=$GPU_A python -m mini_test_pen_shelf.run_mini_test \
    --num_games "$NUM_GAMES" --max_steps "$MAX_STEPS" --repeats "$REPEATS" \
    --strategy --outdir "$OUT_A" \
    > "$OUT_A/run.log" 2>&1 &
PID_A=$!

# B 臂：无策略（baseline）—— 仍保留 inventory/searched 注入，仅去掉 playbook
CUDA_VISIBLE_DEVICES=$GPU_B python -m mini_test_pen_shelf.run_mini_test \
    --num_games "$NUM_GAMES" --max_steps "$MAX_STEPS" --repeats "$REPEATS" \
    --outdir "$OUT_B" \
    > "$OUT_B/run.log" 2>&1 &
PID_B=$!

echo "A 臂 PID=$PID_A , B 臂 PID=$PID_B ，等待两臂完成..."
wait $PID_A; echo "A 臂完成 (exit=$?)"
wait $PID_B; echo "B 臂完成 (exit=$?)"

# 生成对比报告
python -m mini_test_pen_shelf.compare_ab \
    --with_strategy "$OUT_A/summary.json" \
    --no_strategy   "$OUT_B/summary.json" \
    --archive       "$OUT_AB/ab_report"
