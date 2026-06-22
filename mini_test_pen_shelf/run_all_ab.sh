#!/usr/bin/env bash
# run_all_ab.sh — 三组 A/B 测试。每组 strategy / baseline 两臂，分别归档+出对比报告。
#
#   组1: pen→shelf            (已由 run_ab.sh 跑过，这里跳过；如需重跑设 RUN_G1=1)
#   组2: pick_and_place_simple 前 N 个不同游戏，测跨物体/容器泛化
#   组3: pick_two_obj_and_place 单个任务跑 R 次重复
#
# 两臂默认并行：strategy 用 GPU_A，baseline 用 GPU_B。
# 单卡模式：设 SINGLE_GPU=1（或只给一张卡），两臂在同一张卡上【顺序】跑，避免 OOM。
#
# 用法:
#   export MODEL_PATH=...  ALFWORLD_DATA=...
#   bash mini_test_pen_shelf/run_all_ab.sh                      # 双卡并行
#   SINGLE_GPU=1 GPU_A=0 bash mini_test_pen_shelf/run_all_ab.sh # 单卡(0)顺序
set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
cd "$(dirname "$0")/.."
ROOT="mini_test_pen_shelf"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"
SINGLE_GPU="${SINGLE_GPU:-0}"   # 1 = 单卡顺序跑两臂
# 单卡 24G 跑不下 util=0.8 的双开，给 run_generic 透传一个更稳的显存比例。
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"   # 本次运行时间戳，用于备份旧结果的目录名

PP_GAMES="${PP_GAMES:-10}"     # 组2 抽取游戏数（跨物体均匀抽）
PP_STEPS="${PP_STEPS:-30}"
P2_GAMES="${P2_GAMES:-10}"     # 组3 抽取的不同 pick_two 子任务数
P2_STEPS="${P2_STEPS:-40}"

run_pair () {  # $1=mode $2=extra_args $3=tag
  local mode="$1" extra="$2" tag="$3"
  local outA="$ROOT/output_${tag}_strategy" outB="$ROOT/output_${tag}_baseline"
  local outAB="$ROOT/output_ab"
  # 重跑前：把已存在的旧结果整体备份到 backups/<tag>_<时间戳>/，再清空重来，
  # 避免新旧轨迹文件（文件名带结果/步数）混在一起。BACKUP=0 可关闭备份直接清空。
  local bkroot="$ROOT/backups/${tag}_${RUN_STAMP}"
  for d in "$outA" "$outB"; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
      if [ "${BACKUP:-1}" = "1" ]; then
        mkdir -p "$bkroot"; mv "$d" "$bkroot/"; echo "[$tag] 旧结果已备份: $bkroot/$(basename "$d")"
      else
        rm -rf "$d"; echo "[$tag] 旧结果已清空: $d"
      fi
    fi
  done
  mkdir -p "$outA" "$outB" "$outAB"
  if [ "$SINGLE_GPU" = "1" ]; then
    echo ">>> [$tag] mode=$mode  单卡顺序 GPU$GPU_A  (先 strategy 后 baseline)"
    CUDA_VISIBLE_DEVICES=$GPU_A python -m mini_test_pen_shelf.run_generic \
        --mode "$mode" $extra --gpu_mem_util "$GPU_MEM_UTIL" --strategy \
        --outdir "$outA" > "$outA/run.log" 2>&1
    echo "[$tag] A(strategy) done ($?)"
    CUDA_VISIBLE_DEVICES=$GPU_A python -m mini_test_pen_shelf.run_generic \
        --mode "$mode" $extra --gpu_mem_util "$GPU_MEM_UTIL" \
        --outdir "$outB" > "$outB/run.log" 2>&1
    echo "[$tag] B(baseline) done ($?)"
  else
    echo ">>> [$tag] mode=$mode  A(strategy)->GPU$GPU_A  B(baseline)->GPU$GPU_B"
    CUDA_VISIBLE_DEVICES=$GPU_A python -m mini_test_pen_shelf.run_generic \
        --mode "$mode" $extra --strategy --outdir "$outA" > "$outA/run.log" 2>&1 &
    local pa=$!
    CUDA_VISIBLE_DEVICES=$GPU_B python -m mini_test_pen_shelf.run_generic \
        --mode "$mode" $extra --outdir "$outB" > "$outB/run.log" 2>&1 &
    local pb=$!
    wait $pa; echo "[$tag] A done ($?)"; wait $pb; echo "[$tag] B done ($?)"
  fi
  python -m mini_test_pen_shelf.compare_ab \
      --with_strategy "$outA/summary.json" --no_strategy "$outB/summary.json" \
      --archive "$outAB/${tag}_report" || echo "[$tag] compare 失败(可能某臂没出 summary)"
}

# 组2: pick_and_place_simple 泛化 —— 从全部命中里跨物体均匀抽 PP_GAMES 个（避免过拟合）
run_pair generic "--num_games $PP_GAMES --sample --max_steps $PP_STEPS" "pickplace"

# 组3: pick_two_obj_and_place —— 同样跨物体抽 P2_GAMES 个不同子任务做 A/B
run_pair pick_two "--num_games $P2_GAMES --sample --max_steps $P2_STEPS" "picktwo"

echo "全部完成。报告在 $ROOT/output_ab/"
