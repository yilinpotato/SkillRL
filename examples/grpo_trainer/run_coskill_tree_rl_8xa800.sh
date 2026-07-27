#!/usr/bin/env bash
# Fast formal eight-A800 entrypoint for CoSkill ALFWorld Tree-RL.
#
# Contract deliberately retained from the regular launcher:
#   - 12 ALFWorld task instances x group_size=6 = 72 rollouts per train step
#   - max_steps=40, train_data_size=12, group_size=6, val_data_size=32
#   - same model, prompt/response limits, reward, skill-tree and cloud settings
#
# Necessary eight-rank geometry change:
#   36 cannot divide evenly across eight FSDP ranks.  We therefore use one
#   72-sample PPO mini-batch (9 samples/rank, micro-batch=1) instead of two
#   36-sample mini-batches.  This is the fastest stable eight-rank geometry,
#   but its optimizer-update geometry differs from formal 2/4-GPU curves.
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
export N_GPUS_PER_NODE=8
# 72 rollouts are fixed.  These values are explicit so a scheduler environment
# cannot silently alter the experiment contract.
export TRAIN_DATA_SIZE=12
export GROUP_SIZE=6
export VAL_DATA_SIZE=32
export PPO_MINI_BATCH_SIZE=72
export PPO_MICRO_BATCH_SIZE_PER_GPU=1

# The main launcher refuses a disabled/non-real cloud check.  Pin these values
# here as an auditable eight-GPU contract as well.
export CLOUD_BOOTSTRAP_CHECK=1
export CLOUD_BOOTSTRAP_PROBE=1

exec bash "$SCRIPT_DIR/run_coskill_tree_rl.sh" alfworld "$@"
