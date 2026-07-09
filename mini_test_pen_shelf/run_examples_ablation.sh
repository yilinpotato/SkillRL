#!/usr/bin/env bash
# run_examples_ablation.sh — 单卡(默认GPU1)顺序跑，做「playbook 去掉 few-shot 示例」的消融。
#
# 复用已有结果: pickplace playbook-ON+examples (90%) / playbook-OFF (70%)。
# 本脚本补齐 4 个 run：
#   1) picktwo  playbook-ON +examples   (上次被中断，重跑)
#   2) picktwo  playbook-OFF            (基线)
#   3) pickplace playbook-ON -examples  (消融)
#   4) picktwo  playbook-ON -examples   (消融)
# 然后出 3 份对比报告。两臂都 skills-off；--sample_seed 0 保证各条件用同一批游戏可比。
#
# 用法: GPU=1 bash mini_test_pen_shelf/run_examples_ablation.sh
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
cd "$(dirname "$0")/.."
ROOT="mini_test_pen_shelf"
GPU="${GPU:-1}"          # 只用 GPU1（0 号有人在用，绝不碰）
MEM="${MEM:-0.8}"        # 单卡单进程，可给高一点显存比例
STAMP="$(date +%Y%m%d_%H%M%S)"

run () {  # $1=mode $2=max_steps $3=extra_flags $4=outdir
  local out="$4"
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    mkdir -p "$ROOT/backups/ablation_$STAMP"; mv "$out" "$ROOT/backups/ablation_$STAMP/"
  fi
  mkdir -p "$out"
  echo ">>> [$(date +%H:%M:%S)] $(basename "$out")  mode=$1 flags='$3'"
  CUDA_VISIBLE_DEVICES=$GPU python -m mini_test_pen_shelf.run_generic \
      --mode "$1" --num_games 10 --sample --sample_seed 0 --max_steps "$2" \
      --gpu_mem_util "$MEM" $3 --outdir "$out" > "$out/run.log" 2>&1
  echo "    done(exit=$?) -> $out/run.log"
}

run pick_two 40 ""                       "$ROOT/output_picktwo_strategy"
run pick_two 40 "--no_playbook"          "$ROOT/output_picktwo_baseline"
run generic  30 "--no_playbook_examples" "$ROOT/output_pickplace_strategy_noex"
run pick_two 40 "--no_playbook_examples" "$ROOT/output_picktwo_strategy_noex"

CMP="$ROOT/output_ab"; mkdir -p "$CMP"
echo ">>> 生成对比报告..."
# picktwo: playbook on vs off
python -m mini_test_pen_shelf.compare_ab \
  --with_strategy "$ROOT/output_picktwo_strategy/summary.json" \
  --no_strategy   "$ROOT/output_picktwo_baseline/summary.json" \
  --archive "$CMP/picktwo_report" || true
# pickplace: examples on vs off (both playbook-on)
python -m mini_test_pen_shelf.compare_ab \
  --with_strategy "$ROOT/output_pickplace_strategy/summary.json" \
  --no_strategy   "$ROOT/output_pickplace_strategy_noex/summary.json" \
  --archive "$CMP/pickplace_examples_report" || true
# picktwo: examples on vs off (both playbook-on)
python -m mini_test_pen_shelf.compare_ab \
  --with_strategy "$ROOT/output_picktwo_strategy/summary.json" \
  --no_strategy   "$ROOT/output_picktwo_strategy_noex/summary.json" \
  --archive "$CMP/picktwo_examples_report" || true
echo ">>> ALL DONE  $(date +%H:%M:%S)"
