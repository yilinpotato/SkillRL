#!/usr/bin/env bash
# run_picktwo_newpb.sh — 用「新 pick_two playbook(含 一次只拿一个/放完继续找 提示)」全量重跑
# picktwo 两条臂 (on+examples / -examples)，10局/40步，单卡 GPU1。
#   1) 先等旧-playbook 的 arm1 (output_picktwo_strategy) 跑完，移到 _oldpb 留参考
#   2) 新 playbook: picktwo on+examples  -> output_picktwo_strategy
#   3) 新 playbook: picktwo -examples    -> output_picktwo_strategy_noex
#   4) 出对比报告 picktwo_examples_report (examples on vs off, 新playbook)
# 用法: GPU=1 bash mini_test_pen_shelf/run_picktwo_newpb.sh
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
cd "$(dirname "$0")/.."
ROOT="mini_test_pen_shelf"; GPU="${GPU:-1}"; MEM="${MEM:-0.8}"; STAMP="$(date +%Y%m%d_%H%M%S)"

echo ">>> [$(date +%H:%M:%S)] 等旧-playbook arm1 跑完 (output_picktwo_strategy/summary.json) ..."
while [ ! -f "$ROOT/output_picktwo_strategy/summary.json" ]; do sleep 30; done
sleep 15   # 让 arm1 python 完全退出并释放显存
echo ">>> [$(date +%H:%M:%S)] arm1 完成，移到 output_picktwo_strategy_oldpb 作参考"
rm -rf "$ROOT/output_picktwo_strategy_oldpb"
mv "$ROOT/output_picktwo_strategy" "$ROOT/output_picktwo_strategy_oldpb"

run () {  # $1=flags $2=outdir
  local out="$2"
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    mkdir -p "$ROOT/backups/newpb_$STAMP"; mv "$out" "$ROOT/backups/newpb_$STAMP/"; fi
  mkdir -p "$out"
  echo ">>> [$(date +%H:%M:%S)] picktwo $(basename "$out") flags='$1'"
  CUDA_VISIBLE_DEVICES=$GPU python -m mini_test_pen_shelf.run_generic \
     --mode pick_two --num_games 10 --sample --sample_seed 0 --max_steps 40 \
     --gpu_mem_util "$MEM" $1 --outdir "$out" > "$out/run.log" 2>&1
  echo "    done(exit=$?) -> $out/run.log"
}

run ""                       "$ROOT/output_picktwo_strategy"        # 新playbook, on+examples
run "--no_playbook_examples" "$ROOT/output_picktwo_strategy_noex"   # 新playbook, -examples

CMP="$ROOT/output_ab"; mkdir -p "$CMP"
python -m mini_test_pen_shelf.compare_ab \
  --with_strategy "$ROOT/output_picktwo_strategy/summary.json" \
  --no_strategy   "$ROOT/output_picktwo_strategy_noex/summary.json" \
  --archive "$CMP/picktwo_examples_report" || true
echo ">>> ALL DONE $(date +%H:%M:%S)"
