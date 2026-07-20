#!/usr/bin/env bash
# One-GPU, one-update diagnostic for the real CoSkill Tree-RL execution path.
# It is intentionally NOT an experiment launcher: its small batch, short
# context, low vLLM cache and CPU offload make it unsuitable for reporting.
set -euo pipefail

BENCHMARK="${1:-alfworld}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
case "$BENCHMARK" in
    alfworld|webshop) ;;
    *) echo "Usage: $0 {alfworld|webshop} [Hydra overrides...]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# The caller must expose exactly one GPU (for example Docker --gpus device=1).
# Keep the diagnostic output separate so it cannot be resumed as a real run.
export TREE_RL_SMOKE_TEST=1
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-2}"
export GROUP_SIZE="${GROUP_SIZE:-2}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-2}"
# Reuse the fixed image corpus; only the batches above are intentionally tiny.
export PREPARED_TRAIN_DATA_SIZE="${PREPARED_TRAIN_DATA_SIZE:-12}"
export PREPARED_VAL_DATA_SIZE="${PREPARED_VAL_DATA_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:-999}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export MAX_STEPS="${MAX_STEPS:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-3072}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-1}"
export REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-1}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export TREE_RL_MIN_UPDATES="${TREE_RL_MIN_UPDATES:-1}"
export TREE_RL_MIN_TRAIN_EPISODES="${TREE_RL_MIN_TRAIN_EPISODES:-2}"
export TREE_RL_MIN_PROBE_EPISODES="${TREE_RL_MIN_PROBE_EPISODES:-2}"
export TREE_RL_STATE_SAVE_FREQ="${TREE_RL_STATE_SAVE_FREQ:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-coskill_tree_rl_smoke_${BENCHMARK}}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${OUTPUT_ROOT:-/outputs}/smoke/$EXPERIMENT_NAME}"

if [[ "${COSKILL_CONTAINER:-0}" == "1" ]]; then
    # A 24 GiB card cannot hold the 4B FSDP actor and a separate vLLM engine
    # concurrently.  Fail before Ray initializes rather than leaving a long,
    # misleading vLLM OOM trace.  The diagnostic remains end-to-end on such a
    # card when an explicitly mounted small model (for example Qwen3-0.6B) is
    # supplied via MODEL_PATH.
    read -r visible_gpu_memory_gib <<< "$(python3 - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(0)
print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))
PY
)"
    if [[ "${visible_gpu_memory_gib:-0}" -lt 40 && "$MODEL_PATH" == *"4B"* ]]; then
        cat >&2 <<EOF
Single-GPU Tree-RL smoke with a 4B model needs at least 40 GiB of VRAM.
This ${visible_gpu_memory_gib} GiB GPU cannot co-reside the FSDP actor and vLLM.
For a real one-GPU pipeline check, mount Qwen3-0.6B and set:
  MODEL_PATH=/models/Qwen3-0.6B
Formal 4B Tree-RL experiments must still use 2, 4, or 8 GPUs.
EOF
        exit 2
    fi
fi

echo "Starting one-GPU Tree-RL smoke test: benchmark=$BENCHMARK output=$RL_OUTPUT_DIR"
exec bash "$SCRIPT_DIR/run_coskill_tree_rl.sh" "$BENCHMARK" "$@"
