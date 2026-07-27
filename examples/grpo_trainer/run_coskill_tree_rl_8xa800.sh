#!/usr/bin/env bash
# Fast eight-A800 entrypoint for CoSkill ALFWorld Tree-RL.
#
# Default eight-GPU throughput profile:
#   - 16 ALFWorld goals x group_size=6 = 96 rollouts per train step
#   - PPO mini=48, per-rank mini=6, per-rank micro=2
#   - max_steps=40, val_data_size=32 and all model/reward/tree settings unchanged
# Set TREE_RL_8GPU_THROUGHPUT_MODE=0 for the historical exact-contract profile:
#   - 12 x 6 = 72 rollouts, PPO mini=72, per-rank micro=1
#
# The delegated launcher performs a mandatory *real* DeepSeek API probe on
# every direct start before Ray/vLLM allocate GPUs.  Do not set a bypass flag.
#
# Usage after the scheduler/container exposes exactly eight A800s:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash examples/grpo_trainer/run_coskill_tree_rl_8xa800.sh
# Optional root/leaf selection: TREE_RL_ORDER=leaf ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    detected="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$detected" != "8" ]]; then
        echo "Eight-A800 launcher requires exactly 8 scheduler-visible GPUs; found $detected." >&2
        echo "Set CUDA_VISIBLE_DEVICES to the allocated eight physical GPUs before starting." >&2
        exit 2
    fi
else
    visible_count="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
    if [[ "$visible_count" != "8" ]]; then
        echo "Eight-A800 launcher requires exactly 8 CUDA_VISIBLE_DEVICES entries; got $CUDA_VISIBLE_DEVICES." >&2
        exit 2
    fi
fi

export TREE_RL_USE_ALL_8=1
export TREE_RL_8GPU_THROUGHPUT_MODE="${TREE_RL_8GPU_THROUGHPUT_MODE:-1}"
export N_GPUS_PER_NODE=8
export GROUP_SIZE=6
export VAL_DATA_SIZE=32
if [[ "$TREE_RL_8GPU_THROUGHPUT_MODE" == "1" ]]; then
    export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-48}"
    export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
    export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
    export REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-4}"
    export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
else
    export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-12}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-72}"
    export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
fi

# The main launcher refuses a disabled/non-real cloud check.  Pin these values
# here as an auditable eight-GPU contract as well.
export CLOUD_BOOTSTRAP_CHECK=1
export CLOUD_BOOTSTRAP_PROBE=1

exec bash "$SCRIPT_DIR/run_coskill_tree_rl.sh" alfworld "$@"
