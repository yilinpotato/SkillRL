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

echo "Starting one-GPU Tree-RL smoke test: benchmark=$BENCHMARK output=$RL_OUTPUT_DIR"
exec bash "$SCRIPT_DIR/run_coskill_tree_rl.sh" "$BENCHMARK" "$@"
