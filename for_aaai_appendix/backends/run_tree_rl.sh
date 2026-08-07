#!/usr/bin/env bash

set -euo pipefail

BENCHMARK="${1:-}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
case "$BENCHMARK" in
    alfworld|webshop) ;;
    *)
        echo "Usage: $0 {alfworld|webshop} [Hydra overrides...]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE
cd "$PROJECT_ROOT"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export RAY_IGNORE_HTTP_PROXY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTHONUNBUFFERED=1
export COSKILL_QUIET_LOGS="${COSKILL_QUIET_LOGS:-1}"
if [[ "$COSKILL_QUIET_LOGS" == "1" ]]; then
    export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
    export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
    export TQDM_DISABLE="${TQDM_DISABLE:-1}"
fi
unset PYTORCH_CUDA_ALLOC_CONF

source "$PROJECT_ROOT/scripts/configure_vllm_acceleration.sh"

export CLOUD_BOOTSTRAP_CHECK="${CLOUD_BOOTSTRAP_CHECK:-1}"
export CLOUD_BOOTSTRAP_PROBE="${CLOUD_BOOTSTRAP_PROBE:-1}"
[[ "$CLOUD_BOOTSTRAP_CHECK" == "1" ]] || { echo "CLOUD_BOOTSTRAP_CHECK=1 is required for Tree-RL." >&2; exit 1; }
[[ "$CLOUD_BOOTSTRAP_PROBE" == "1" ]] || { echo "CLOUD_BOOTSTRAP_PROBE=1 (a real API request) is required for Tree-RL." >&2; exit 1; }
CLOUD_SKILLS_JSON="${SKILLS_JSON:-memory_data/${BENCHMARK}/initial_skills.json}"
if [[ "${COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE:-0}" == "1" ]]; then
    echo "Cloud probe already passed in the container entrypoint for this startup."
else
    echo "Checking cloud bootstrap (real probe=1) before environment/CUDA/Ray/vLLM setup..."
    python3 scripts/check_cloud_bootstrap.py \
        --environment "$BENCHMARK" \
        --skills-json "$CLOUD_SKILLS_JSON" \
        --probe
fi

IS_CONTAINER=0
if [[ "${COSKILL_CONTAINER:-0}" == "1" || -f /.dockerenv || -f /run/.containerenv ]]; then
    IS_CONTAINER=1
fi

if [[ "$IS_CONTAINER" == "1" ]]; then
    RUN_ENV="Docker container"
    export CACHE_ROOT="${CACHE_ROOT:-/models/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-/outputs}"
    export ALFWORLD_DATA="${ALFWORLD_DATA:-/opt/data/alfworld}"
    export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data}"
    DEFAULT_RAY_NUM_CPUS="$(nproc)"
elif [[ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
    RUN_ENV="scheduled host"
    export CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
    export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
    DEFAULT_RAY_NUM_CPUS="$(nproc)"
else
    if [[ "${COSKILL_ALLOW_LOCAL_MULTI_GPU:-0}" != "1" ]]; then
        echo "CoSkill tree RL needs an isolated 2/4-GPU allocation." >&2
        echo "The shared local machine remains protected; use Docker/scheduler or explicitly set COSKILL_ALLOW_LOCAL_MULTI_GPU=1." >&2
        exit 1
    fi
    RUN_ENV="isolated local multi-GPU"
    export CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
    export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
    DEFAULT_RAY_NUM_CPUS="$(nproc)"
fi

TREE_RL_SMOKE_TEST="${TREE_RL_SMOKE_TEST:-0}"
if [[ "$TREE_RL_SMOKE_TEST" != "0" && "$TREE_RL_SMOKE_TEST" != "1" ]]; then
    echo "TREE_RL_SMOKE_TEST must be 0 (formal run) or 1 (single-GPU diagnostic)." >&2
    exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DETECTED_GPU_COUNT="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$DETECTED_GPU_COUNT" -le 0 ]]; then
        echo "No CUDA GPUs are visible inside this runtime." >&2
        exit 1
    fi
    DETECTED_GPUS="$(awk -v n="$DETECTED_GPU_COUNT" 'BEGIN {for (i=0;i<n;i++) printf "%s%d", (i?",":""), i}')"
    export CUDA_VISIBLE_DEVICES="$DETECTED_GPUS"
fi
ALLOCATED_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
ALLOCATED_NUM_GPUS="$(tr ',' '\n' <<<"$ALLOCATED_CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
TREE_RL_GPU_SLOT="${TREE_RL_GPU_SLOT:-0}"
TREE_RL_USE_ALL_8="${TREE_RL_USE_ALL_8:-0}"
TREE_RL_8GPU_THROUGHPUT_MODE="${TREE_RL_8GPU_THROUGHPUT_MODE:-1}"
TREE_RL_ALLOW_SINGLE_GPU="${TREE_RL_ALLOW_SINGLE_GPU:-0}"
if [[ "$TREE_RL_USE_ALL_8" != "0" && "$TREE_RL_USE_ALL_8" != "1" ]]; then
    echo "TREE_RL_USE_ALL_8 must be 0 (two four-GPU slots) or 1 (one eight-GPU experiment)." >&2
    exit 1
fi
if [[ "$TREE_RL_8GPU_THROUGHPUT_MODE" != "0" && "$TREE_RL_8GPU_THROUGHPUT_MODE" != "1" ]]; then
    echo "TREE_RL_8GPU_THROUGHPUT_MODE must be 0 (72 rows) or 1 (16x6=96 rows)." >&2
    exit 1
fi
if [[ "$TREE_RL_ALLOW_SINGLE_GPU" != "0" && "$TREE_RL_ALLOW_SINGLE_GPU" != "1" ]]; then
    echo "TREE_RL_ALLOW_SINGLE_GPU must be 0 or 1." >&2
    exit 1
fi
if [[ "$ALLOCATED_NUM_GPUS" == "8" ]]; then
    if [[ "$TREE_RL_USE_ALL_8" == "1" ]]; then
        echo "8-GPU all-in-one mode enabled: ranks=$CUDA_VISIBLE_DEVICES."
        if [[ "$TREE_RL_8GPU_THROUGHPUT_MODE" == "1" ]]; then
            echo "8-GPU throughput profile enabled: 16 goals x 6 = 96 rollouts/update."
            echo "This is a non-comparable sampling/optimizer variant; do not merge its learning curve with 72-rollout results."
        else
            echo "8-GPU exact-contract profile enabled: 12 goals x 6 = 72 rollouts/update."
        fi
    else
        if [[ "$TREE_RL_GPU_SLOT" != "0" && "$TREE_RL_GPU_SLOT" != "1" ]]; then
            echo "With 8 visible GPUs, TREE_RL_GPU_SLOT must be 0 or 1." >&2
            exit 1
        fi
        IFS=',' read -r -a GPU_ARRAY <<<"$ALLOCATED_CUDA_VISIBLE_DEVICES"
        GPU_OFFSET=$((TREE_RL_GPU_SLOT * 4))
        SELECTED_GPU_ARRAY=("${GPU_ARRAY[@]:GPU_OFFSET:4}")
        export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SELECTED_GPU_ARRAY[*]}")"
        echo "8-GPU allocation detected: this experiment uses slot $TREE_RL_GPU_SLOT ($CUDA_VISIBLE_DEVICES)."
        echo "Run a second independent task with TREE_RL_GPU_SLOT=$((1 - TREE_RL_GPU_SLOT)) to use the other four GPUs without changing PPO geometry."
    fi
elif [[ "$TREE_RL_SMOKE_TEST" == "1" && "$ALLOCATED_NUM_GPUS" == "1" ]]; then
    echo "Single-GPU Tree-RL smoke mode enabled; this is not a formal experiment."
elif [[ "$TREE_RL_ALLOW_SINGLE_GPU" == "1" && "$ALLOCATED_NUM_GPUS" == "1" ]]; then
    echo "Single-GPU formal Tree-RL enabled: rollout and validation remain the standard 72/32 contract."
elif [[ "$ALLOCATED_NUM_GPUS" != "2" && "$ALLOCATED_NUM_GPUS" != "4" ]]; then
    echo "Need 2, 4, or 8 visible GPUs (or one GPU with TREE_RL_ALLOW_SINGLE_GPU=1); CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES ($ALLOCATED_NUM_GPUS GPUs)." >&2
    exit 1
fi
NUM_GPUS="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
if [[ "$TREE_RL_SMOKE_TEST" == "1" && "$NUM_GPUS" == "1" ]]; then
    :
elif [[ "$TREE_RL_ALLOW_SINGLE_GPU" == "1" && "$NUM_GPUS" == "1" ]]; then
    :
elif [[ "$TREE_RL_USE_ALL_8" == "1" && "$NUM_GPUS" == "8" ]]; then
    :
elif [[ "$NUM_GPUS" != "2" && "$NUM_GPUS" != "4" ]]; then
    echo "Need exactly 2 or 4 visible GPUs (or one with TREE_RL_ALLOW_SINGLE_GPU=1, or eight with TREE_RL_USE_ALL_8=1); CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)." >&2
    exit 1
fi
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$NUM_GPUS}"
if [[ "$N_GPUS_PER_NODE" != "$NUM_GPUS" ]]; then
    echo "N_GPUS_PER_NODE must equal the allocated visible GPU count ($NUM_GPUS)." >&2
    exit 1
fi

TREE_RL_ORDER="${TREE_RL_ORDER:-root}"
if [[ "$TREE_RL_ORDER" != "root" && "$TREE_RL_ORDER" != "leaf" ]]; then
    echo "TREE_RL_ORDER must be root or leaf, got: $TREE_RL_ORDER" >&2
    exit 1
fi

if [[ "$TREE_RL_SMOKE_TEST" == "1" ]]; then
    DEFAULT_TRAIN_DATA_SIZE=2
    DEFAULT_GROUP_SIZE=2
    DEFAULT_VAL_DATA_SIZE=2
else
    if [[ "$TREE_RL_USE_ALL_8" == "1" && "$NUM_GPUS" == "8" && "$TREE_RL_8GPU_THROUGHPUT_MODE" == "1" ]]; then
        DEFAULT_TRAIN_DATA_SIZE=16
    else
        DEFAULT_TRAIN_DATA_SIZE=12
    fi
    DEFAULT_GROUP_SIZE=6
    DEFAULT_VAL_DATA_SIZE=32
fi
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-$DEFAULT_TRAIN_DATA_SIZE}"
GROUP_SIZE="${GROUP_SIZE:-$DEFAULT_GROUP_SIZE}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-$DEFAULT_VAL_DATA_SIZE}"
ROLLOUTS_PER_STEP=$((TRAIN_DATA_SIZE * GROUP_SIZE))
if (( ROLLOUTS_PER_STEP % NUM_GPUS != 0 )); then
    echo "Expanded rollout batch TRAIN_DATA_SIZE×GROUP_SIZE=${ROLLOUTS_PER_STEP} must be divisible by $NUM_GPUS Ray/FSDP ranks." >&2
    exit 1
fi
if (( VAL_DATA_SIZE % NUM_GPUS != 0 )); then
    echo "VAL_DATA_SIZE=$VAL_DATA_SIZE must be divisible by $NUM_GPUS Ray/FSDP ranks." >&2
    exit 1
fi
if [[ "$TREE_RL_SMOKE_TEST" == "1" ]]; then
    DEFAULT_PPO_MINI_BATCH=4
elif (( NUM_GPUS == 8 && ROLLOUTS_PER_STEP % 48 == 0 )); then
    DEFAULT_PPO_MINI_BATCH=48
elif (( NUM_GPUS == 8 && ROLLOUTS_PER_STEP == 72 )); then
    DEFAULT_PPO_MINI_BATCH=72
else
    DEFAULT_PPO_MINI_BATCH=36
fi

GPU_PROFILE="$(python3 - <<'PY'
import torch
print("; ".join(
    f"cuda:{i}={torch.cuda.get_device_properties(i).name}/"
    f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.1f}GiB"
    for i in range(torch.cuda.device_count())
))
PY
)"
MIN_GPU_MEMORY_GIB="$(python3 - <<'PY'
import torch
print(int(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30))
PY
)"
if [[ "$TREE_RL_SMOKE_TEST" != "1" && "$NUM_GPUS" == "1" && "$MIN_GPU_MEMORY_GIB" -lt 80 ]]; then
    echo "Formal single-GPU Tree-RL requires at least 80 GiB VRAM; detected ${MIN_GPU_MEMORY_GIB} GiB." >&2
    echo "Use 2/4 GPUs, or run the explicitly non-comparable TREE_RL_SMOKE_TEST=1 diagnostic." >&2
    exit 1
fi
GPU_FAMILY="$(python3 - <<'PY'
import re
import torch

name = torch.cuda.get_device_properties(0).name.lower()
for candidate in ("a800", "a100", "h20", "h100", "h200"):
    if candidate in name:
        print(candidate)
        break
else:
    print(re.sub(r"[^a-z0-9]+", "-", name).strip("-") or "gpu")
PY
)"
if (( NUM_GPUS == 8 && ROLLOUTS_PER_STEP >= 96 && MIN_GPU_MEMORY_GIB >= 72 )); then
    DEFAULT_PPO_MICRO_BATCH_PER_GPU=2
elif (( NUM_GPUS >= 4 || MIN_GPU_MEMORY_GIB < 60 )); then
    DEFAULT_PPO_MICRO_BATCH_PER_GPU=1
else
    DEFAULT_PPO_MICRO_BATCH_PER_GPU=2
fi
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$DEFAULT_PPO_MINI_BATCH}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_PPO_MICRO_BATCH_PER_GPU}"
if (( PPO_MINI_BATCH_SIZE <= 0 || PPO_MICRO_BATCH_SIZE_PER_GPU <= 0 )); then
    echo "PPO_MINI_BATCH_SIZE and PPO_MICRO_BATCH_SIZE_PER_GPU must be positive." >&2
    exit 1
fi
if (( ROLLOUTS_PER_STEP % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "ROLLOUTS_PER_STEP=$ROLLOUTS_PER_STEP must be divisible by PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE." >&2
    exit 1
fi
if (( PPO_MINI_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE must be divisible by $NUM_GPUS Ray/FSDP ranks." >&2
    exit 1
fi
PPO_MINI_BATCH_PER_GPU=$((PPO_MINI_BATCH_SIZE / NUM_GPUS))
if (( PPO_MINI_BATCH_PER_GPU % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
    echo "Per-rank PPO mini-batch=$PPO_MINI_BATCH_PER_GPU must be divisible by PPO_MICRO_BATCH_SIZE_PER_GPU=$PPO_MICRO_BATCH_SIZE_PER_GPU." >&2
    exit 1
fi
PPO_GRAD_ACCUM_STEPS=$((PPO_MINI_BATCH_PER_GPU / PPO_MICRO_BATCH_SIZE_PER_GPU))
PPO_GLOBAL_MICRO_BATCH=$((NUM_GPUS * PPO_MICRO_BATCH_SIZE_PER_GPU))

if [[ "$TREE_RL_SMOKE_TEST" == "1" ]]; then
    DEFAULT_MAX_RESPONSE_LENGTH=512
    DEFAULT_ALFWORLD_PROMPT_LENGTH=2048
    DEFAULT_WEBSHOP_PROMPT_LENGTH=2048
    DEFAULT_ALFWORLD_STEPS=4
    DEFAULT_WEBSHOP_STEPS=4
else
    DEFAULT_MAX_RESPONSE_LENGTH=4096
    DEFAULT_ALFWORLD_PROMPT_LENGTH=6144
    DEFAULT_WEBSHOP_PROMPT_LENGTH=8192
    DEFAULT_ALFWORLD_STEPS=40
    DEFAULT_WEBSHOP_STEPS=15
fi
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$DEFAULT_MAX_RESPONSE_LENGTH}"
if [[ "$BENCHMARK" == "alfworld" ]]; then
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$DEFAULT_ALFWORLD_PROMPT_LENGTH}"
    MAX_STEPS="${MAX_STEPS:-$DEFAULT_ALFWORLD_STEPS}"
    SKILLS_JSON="${SKILLS_JSON:-memory_data/alfworld/initial_skills.json}"
    ENV_NAME="alfworld/AlfredTWEnv"
    RETRIEVAL_MODE="${RETRIEVAL_MODE:-template}"
    PROJECT_NAME="verl_agent_alfworld"
else
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$DEFAULT_WEBSHOP_PROMPT_LENGTH}"
    MAX_STEPS="${MAX_STEPS:-$DEFAULT_WEBSHOP_STEPS}"
    SKILLS_JSON="${SKILLS_JSON:-memory_data/webshop/initial_skills.json}"
    ENV_NAME="Webshop"
    RETRIEVAL_MODE="${RETRIEVAL_MODE:-template}"
    PROJECT_NAME="verl_agent_webshop"
fi

if [[ "$IS_CONTAINER" == "1" ]]; then
    export MODEL_PATH="${MODEL_PATH:-/models/Qwen3-4B-Thinking-2507}"
else
    export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
fi
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"
export ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
if [[ "$TREE_RL_SMOKE_TEST" == "1" ]]; then
    DEFAULT_TOTAL_TRAINING_STEPS=1
    DEFAULT_SAVE_FREQ=1
    DEFAULT_TEST_FREQ=999
    DEFAULT_VAL_BEFORE_TRAIN=False
    DEFAULT_LORA_RANK=8
    DEFAULT_LORA_ALPHA=16
    DEFAULT_LOG_PROB_MICRO_BATCH=1
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.45
    DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS=3072
    DEFAULT_VLLM_MAX_NUM_SEQS=4
    DEFAULT_ACTOR_PARAM_OFFLOAD=True
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=True
else
    DEFAULT_TRAIN_ROLLOUT_BUDGET=7200
    if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
        DEFAULT_TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS"
    else
        if (( DEFAULT_TRAIN_ROLLOUT_BUDGET % ROLLOUTS_PER_STEP != 0 )); then
            echo "Formal rollout budget $DEFAULT_TRAIN_ROLLOUT_BUDGET is not divisible by rollouts-per-step $ROLLOUTS_PER_STEP." >&2
            echo "Set TOTAL_TRAINING_STEPS explicitly for this custom geometry." >&2
            exit 1
        fi
        DEFAULT_TOTAL_TRAINING_STEPS=$((DEFAULT_TRAIN_ROLLOUT_BUDGET / ROLLOUTS_PER_STEP))
    fi
    DEFAULT_SAVE_FREQ=10
    DEFAULT_TEST_FREQ=10
    DEFAULT_VAL_BEFORE_TRAIN=True
    DEFAULT_LORA_RANK=32
    DEFAULT_LORA_ALPHA=64
    DEFAULT_LOG_PROB_MICRO_BATCH=1
    for candidate in 4 3 2 1; do
        if (( ROLLOUTS_PER_STEP % (candidate * NUM_GPUS) == 0 )); then
            DEFAULT_LOG_PROB_MICRO_BATCH=$candidate
            break
        fi
    done
    DEFAULT_ACTOR_PARAM_OFFLOAD=False
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=False
    if (( MIN_GPU_MEMORY_GIB < 48 )); then
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.65
    elif (( MIN_GPU_MEMORY_GIB < 72 )); then
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.72
    else
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.8
    fi
    if (( NUM_GPUS == 8 && ROLLOUTS_PER_STEP >= 96 )); then
        DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS=32768
    else
        DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS=16384
    fi
    DEFAULT_VLLM_MAX_NUM_SEQS=256
fi
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$DEFAULT_TOTAL_TRAINING_STEPS}"
export SAVE_FREQ="${SAVE_FREQ:-$DEFAULT_SAVE_FREQ}"
export TEST_FREQ="${TEST_FREQ:-$DEFAULT_TEST_FREQ}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-$DEFAULT_VAL_BEFORE_TRAIN}"
TRAINING_ROLLOUTS_TOTAL=$((TOTAL_TRAINING_STEPS * ROLLOUTS_PER_STEP))
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export LORA_RANK="${LORA_RANK:-$DEFAULT_LORA_RANK}"
export LORA_ALPHA="${LORA_ALPHA:-$DEFAULT_LORA_ALPHA}"
export LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-$DEFAULT_LOG_PROB_MICRO_BATCH}"
export REF_LOG_PROB_MICRO_BATCH_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-$DEFAULT_LOG_PROB_MICRO_BATCH}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$DEFAULT_VLLM_MAX_NUM_SEQS}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-$DEFAULT_ACTOR_PARAM_OFFLOAD}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-$DEFAULT_ACTOR_OPTIMIZER_OFFLOAD}"
export BATCH_TOKENIZE_OBSERVATIONS="${BATCH_TOKENIZE_OBSERVATIONS:-True}"
case "$BATCH_TOKENIZE_OBSERVATIONS" in
    True|False) ;;
    *)
        echo "BATCH_TOKENIZE_OBSERVATIONS must be True or False, got: $BATCH_TOKENIZE_OBSERVATIONS" >&2
        exit 1
        ;;
esac

gcd_int() {
    local left="$1" right="$2" remainder
    while (( right != 0 )); do
        remainder=$((left % right))
        left=$right
        right=$remainder
    done
    printf '%s\n' "$left"
}
lcm_int() {
    local left="$1" right="$2" divisor
    divisor="$(gcd_int "$left" "$right")"
    printf '%s\n' "$((left / divisor * right))"
}
ROLLOUT_LOGPROB_GLOBAL_MICRO=$((LOG_PROB_MICRO_BATCH_PER_GPU * NUM_GPUS))
REF_LOGPROB_GLOBAL_MICRO=$((REF_LOG_PROB_MICRO_BATCH_PER_GPU * NUM_GPUS))
BATCH_ADJUST_DIVISOR="$(lcm_int "$PPO_GLOBAL_MICRO_BATCH" "$ROLLOUT_LOGPROB_GLOBAL_MICRO")"
BATCH_ADJUST_DIVISOR="$(lcm_int "$BATCH_ADJUST_DIVISOR" "$REF_LOGPROB_GLOBAL_MICRO")"
if (( ROLLOUTS_PER_STEP % BATCH_ADJUST_DIVISOR != 0 )); then
    echo "Expanded rollout geometry is incompatible with the requested micro-batches: " >&2
    echo "rollouts=$ROLLOUTS_PER_STEP actor_global_micro=$PPO_GLOBAL_MICRO_BATCH rollout_logprob_global_micro=$ROLLOUT_LOGPROB_GLOBAL_MICRO ref_logprob_global_micro=$REF_LOGPROB_GLOBAL_MICRO lcm=$BATCH_ADJUST_DIVISOR." >&2
    echo "It would require adjust_batch to copy individual trajectories and alter GRPO group weights. Set LOG_PROB_MICRO_BATCH_PER_GPU and REF_LOG_PROB_MICRO_BATCH_PER_GPU to values whose global micro-batches divide $ROLLOUTS_PER_STEP." >&2
    exit 1
fi
export COMPACT_FINISHED_TRAJECTORIES="${COMPACT_FINISHED_TRAJECTORIES:-True}"
case "$COMPACT_FINISHED_TRAJECTORIES" in
    True|False) ;;
    *)
        echo "COMPACT_FINISHED_TRAJECTORIES must be True or False, got: $COMPACT_FINISHED_TRAJECTORIES" >&2
        exit 1
        ;;
esac
export WEBSHOP_PROMPT_CHAR_LIMIT="${WEBSHOP_PROMPT_CHAR_LIMIT:-24000}"

export TREE_RL_MIN_UPDATES="${TREE_RL_MIN_UPDATES:-5}"
export TREE_RL_MIN_TRAIN_EPISODES="${TREE_RL_MIN_TRAIN_EPISODES:-24}"
export TREE_RL_TRAIN_SUCCESS_THRESHOLD="${TREE_RL_TRAIN_SUCCESS_THRESHOLD:-0.7}"
export TREE_RL_MIN_PROBE_EPISODES="${TREE_RL_MIN_PROBE_EPISODES:-24}"
export TREE_RL_PROBE_SUCCESS_THRESHOLD="${TREE_RL_PROBE_SUCCESS_THRESHOLD:-0.7}"
export TREE_RL_STATE_SAVE_FREQ="${TREE_RL_STATE_SAVE_FREQ:-1}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_coskill_tree_rl_${TREE_RL_ORDER}_${NUM_GPUS}x${GPU_FAMILY}}"
OUTPUT_DIR="${RL_OUTPUT_DIR:-${OUTPUT_DIR:-$OUTPUT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR" "$DATA_ROOT/text"
export JSONL_METRICS_DIR="$OUTPUT_DIR"

RESUME_SKILL_TREE_STATE="${RESUME_SKILL_TREE_STATE:-auto}"
TREE_STATE_PATH="$OUTPUT_DIR/skill_lib/skills_tree_rl_latest.json"
case "$RESUME_SKILL_TREE_STATE" in
    auto)
        if [[ -f "$TREE_STATE_PATH" ]]; then
            SKILLS_JSON="$TREE_STATE_PATH"
            echo "Resuming progressive skill tree from: $SKILLS_JSON"
        fi
        ;;
    disable) ;;
    *)
        echo "RESUME_SKILL_TREE_STATE must be auto or disable." >&2
        exit 1
        ;;
esac

if [[ "$BENCHMARK" == "webshop" ]]; then
    if [[ -z "${WEBSHOP_DATA_DIR:-}" ]]; then
        export WEBSHOP_DATA_DIR="$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data"
    fi
    if [[ -z "${WEBSHOP_DATA_DIR:-}" ]]; then
        echo "WebShop assets missing. Set WEBSHOP_DATA_DIR to the populated data directory." >&2
        exit 1
    fi
fi

echo "CoSkill tree RL: benchmark=$BENCHMARK environment=$RUN_ENV"
echo "GPU allocation: allocated=$ALLOCATED_CUDA_VISIBLE_DEVICES selected=$CUDA_VISIBLE_DEVICES ranks=$NUM_GPUS slot=$TREE_RL_GPU_SLOT all_8=$TREE_RL_USE_ALL_8 throughput_8=$TREE_RL_8GPU_THROUGHPUT_MODE formal_single=$TREE_RL_ALLOW_SINGLE_GPU"
echo "GPU profile: $GPU_PROFILE (minimum ${MIN_GPU_MEMORY_GIB}GiB; vLLM utilization=$VLLM_GPU_MEMORY_UTILIZATION)"
echo "vLLM topology: DP=$NUM_GPUS TP=1 PP=1 (unchanged)"
echo "Active trajectory compaction: $COMPACT_FINISHED_TRAJECTORIES (only completed rows are excluded from future vLLM calls)"
echo "GRPO rollout: train_data_size=$TRAIN_DATA_SIZE group_size=$GROUP_SIZE total=$ROLLOUTS_PER_STEP"
if (( ROLLOUTS_PER_STEP != 72 )); then
    echo "WARNING: this run uses $ROLLOUTS_PER_STEP rollouts/update and is not a 72-rollout learning-curve replicate."
fi
if [[ "$TREE_RL_SMOKE_TEST" == "1" ]]; then
    echo "WARNING: smoke mode uses tiny non-comparable rollout/length/offload settings in an isolated output directory."
fi
echo "PPO geometry: global_mini=$PPO_MINI_BATCH_SIZE per_rank_mini=$PPO_MINI_BATCH_PER_GPU micro_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU global_micro=$PPO_GLOBAL_MICRO_BATCH accumulation=$PPO_GRAD_ACCUM_STEPS"
echo "Log-prob geometry: rollout_micro_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU ref_micro_per_gpu=$REF_LOG_PROB_MICRO_BATCH_PER_GPU batch_adjust_divisor=$BATCH_ADJUST_DIVISOR (expanded batch requires no copied rows)"
echo "Rollout preprocessing: batch_tokenize_observations=$BATCH_TOKENIZE_OBSERVATIONS vllm_max_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
echo "Training budget: steps=$TOTAL_TRAINING_STEPS rollouts_per_step=$ROLLOUTS_PER_STEP total_train_rollouts=$TRAINING_ROLLOUTS_TOTAL"
echo "Validation: val_data_size=$VAL_DATA_SIZE test_freq=$TEST_FREQ val_before_train=$VAL_BEFORE_TRAIN"
echo "Tree curriculum: order=$TREE_RL_ORDER train>=${TREE_RL_MIN_TRAIN_EPISODES}@${TREE_RL_TRAIN_SUCCESS_THRESHOLD} probe>=${TREE_RL_MIN_PROBE_EPISODES}@${TREE_RL_PROBE_SUCCESS_THRESHOLD}"
echo "Output: $OUTPUT_DIR"

PREPARE_DATA="${PREPARE_DATA:-auto}"
case "$PREPARE_DATA" in auto|always) ;; *) echo "PREPARE_DATA must be auto or always" >&2; exit 1;; esac
if [[ "$TREE_RL_SMOKE_TEST" == "1" ]]; then
    DEFAULT_PREPARED_TRAIN_DATA_SIZE=12
    DEFAULT_PREPARED_VAL_DATA_SIZE=32
else
    DEFAULT_PREPARED_TRAIN_DATA_SIZE="$TRAIN_DATA_SIZE"
    DEFAULT_PREPARED_VAL_DATA_SIZE="$VAL_DATA_SIZE"
fi
PREPARED_TRAIN_DATA_SIZE="${PREPARED_TRAIN_DATA_SIZE:-$DEFAULT_PREPARED_TRAIN_DATA_SIZE}"
PREPARED_VAL_DATA_SIZE="${PREPARED_VAL_DATA_SIZE:-$DEFAULT_PREPARED_VAL_DATA_SIZE}"
if (( PREPARED_TRAIN_DATA_SIZE < TRAIN_DATA_SIZE || PREPARED_VAL_DATA_SIZE < VAL_DATA_SIZE )); then
    echo "Prepared parquet rows must cover train/val batch sizes: prepared=$PREPARED_TRAIN_DATA_SIZE/$PREPARED_VAL_DATA_SIZE batch=$TRAIN_DATA_SIZE/$VAL_DATA_SIZE." >&2
    exit 1
fi
REUSE_PREPARED_DATA=0
if [[ "$PREPARE_DATA" == "auto" && -f "$DATA_ROOT/text/train.parquet" && -f "$DATA_ROOT/text/test.parquet" ]]; then
    if python3 - "$DATA_ROOT/text/train.parquet" "$PREPARED_TRAIN_DATA_SIZE" \
            "$DATA_ROOT/text/test.parquet" "$PREPARED_VAL_DATA_SIZE" <<'PY'
import sys
import pyarrow.parquet as pq

for path, expected in ((sys.argv[1], int(sys.argv[2])), (sys.argv[3], int(sys.argv[4]))):
    if pq.ParquetFile(path).metadata.num_rows != expected:
        raise SystemExit(1)
PY
    then
        REUSE_PREPARED_DATA=1
        echo "Reusing prepared text parquet rows: $PREPARED_TRAIN_DATA_SIZE/$PREPARED_VAL_DATA_SIZE (train/val batch: $TRAIN_DATA_SIZE/$VAL_DATA_SIZE)"
    fi
fi
if [[ "$REUSE_PREPARED_DATA" == "0" ]]; then
    if [[ "$IS_CONTAINER" == "1" && "${ALLOW_CONTAINER_DATA_DOWNLOAD:-0}" != "1" ]]; then
        echo "Container parquet is missing/mismatched; refusing an implicit dataset download." >&2
        echo "Rebuild the image or set ALLOW_CONTAINER_DATA_DOWNLOAD=1 with network access." >&2
        exit 1
    fi
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$DATA_ROOT" \
        --train_data_size "$PREPARED_TRAIN_DATA_SIZE" \
        --val_data_size "$PREPARED_VAL_DATA_SIZE"
fi

ppo_args=(
    algorithm.adv_estimator=grpo
    "data.train_files=$DATA_ROOT/text/train.parquet"
    "data.val_files=$DATA_ROOT/text/test.parquet"
    "data.train_batch_size=$TRAIN_DATA_SIZE"
    "data.val_batch_size=$VAL_DATA_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=left
    data.return_raw_chat=True

    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_PER_GPU"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    "actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$VLLM_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
    algorithm.use_kl_in_reward=False

    "env.env_name=$ENV_NAME"
    env.seed=0
    "env.max_steps=$MAX_STEPS"
    env.history_length=8
    "++env.webshop.prompt_char_limit=$WEBSHOP_PROMPT_CHAR_LIMIT"
    "env.rollout.n=$GROUP_SIZE"
    "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
    +env.use_skills_only_memory=True
    "+env.compact_finished_trajectories=$COMPACT_FINISHED_TRAJECTORIES"
    "+env.batch_tokenize_observations=$BATCH_TOKENIZE_OBSERVATIONS"
    "+env.skills_only_memory.skills_json_path=$SKILLS_JSON"
    "+env.skills_only_memory.retrieval_mode=$RETRIEVAL_MODE"
    +env.skills_only_memory.top_k=6
    +env.skills_only_memory.enable_dynamic_update=True
    +env.skills_only_memory.update_skills_from_train=True
    +env.skills_only_memory.skill_update_freq=1
    +env.skills_only_memory.enable_coskill=True
    +env.skills_only_memory.enable_hierarchy=True
    +env.skills_only_memory.enable_playbook=True
    +env.skills_only_memory.enable_playbook_evolve=True
    +env.skills_only_memory.enable_failure_analysis=True
    +env.skills_only_memory.playbook_evolve_min_samples=6
    +env.skills_only_memory.max_new_skills=3
    +env.skills_only_memory.stable_cycles_l1=3
    +env.skills_only_memory.stable_cycles_l2=5
    +env.skills_only_memory.success_l1=0.7
    +env.skills_only_memory.demote_threshold=0.3
    +env.skills_only_memory.min_calls=10
    +env.skills_only_memory.enable_internalize=False
    +env.skills_only_memory.enable_tree_rl_internalize=True
    "+env.skills_only_memory.tree_rl_order=$TREE_RL_ORDER"
    "+env.skills_only_memory.tree_rl_min_updates=$TREE_RL_MIN_UPDATES"
    "+env.skills_only_memory.tree_rl_min_train_episodes=$TREE_RL_MIN_TRAIN_EPISODES"
    "+env.skills_only_memory.tree_rl_train_success_threshold=$TREE_RL_TRAIN_SUCCESS_THRESHOLD"
    "+env.skills_only_memory.tree_rl_min_probe_episodes=$TREE_RL_MIN_PROBE_EPISODES"
    "+env.skills_only_memory.tree_rl_probe_success_threshold=$TREE_RL_PROBE_SUCCESS_THRESHOLD"
    "+env.skills_only_memory.tree_rl_state_save_freq=$TREE_RL_STATE_SAVE_FREQ"
    +env.dump_raw_trajectories=False
    +env.traces_pool.capacity_watermark=50000
    +env.traces_pool.perf_watermark=0.6
    +env.traces_pool.min_samples=16
    +env.traces_pool.loop_threshold=3
    +env.traces_pool.enable_loop_filter=true
    +env.traces_pool.enable_obs_delta=true
    +env.traces_pool.enable_prefix_tree=true
    +env.traces_pool.enable_consensus_prefix=true
    +env.traces_pool.cloud_evidence_mode=tree_only

    trainer.critic_warmup=0
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.default_local_dir=$OUTPUT_DIR"
    "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.ray_wait_register_center_timeout=1200
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    "trainer.total_epochs=$TOTAL_TRAINING_STEPS"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN"
)

ppo_args+=("trainer.logger=['console','jsonl']")

if [[ "$BENCHMARK" == "webshop" ]]; then
    ppo_args+=(
        ++env.webshop.use_small=True
        ++env.webshop.human_goals=False
    )
fi
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    ppo_args+=("ray_init.num_cpus=$RAY_NUM_CPUS")
else
    echo "Using scheduler-provided Ray cluster: RAY_ADDRESS=$RAY_ADDRESS"
fi

{
    echo "timestamp=$(date -Is)"
    for key in BENCHMARK RUN_ENV TREE_RL_SMOKE_TEST TREE_RL_ALLOW_SINGLE_GPU ALLOCATED_CUDA_VISIBLE_DEVICES ALLOCATED_NUM_GPUS CUDA_VISIBLE_DEVICES NUM_GPUS TREE_RL_GPU_SLOT TREE_RL_USE_ALL_8 TREE_RL_8GPU_THROUGHPUT_MODE GPU_PROFILE GPU_FAMILY MIN_GPU_MEMORY_GIB VLLM_GPU_MEMORY_UTILIZATION VLLM_ATTENTION_BACKEND VLLM_USE_FLASHINFER_SAMPLER COSKILL_ENABLE_FLASHINFER_SAMPLER COSKILL_FLASHINFER_OVERLAY N_GPUS_PER_NODE MODEL_PATH DATA_ROOT OUTPUT_ROOT OUTPUT_DIR TRAIN_DATA_SIZE GROUP_SIZE VAL_DATA_SIZE PREPARED_TRAIN_DATA_SIZE PREPARED_VAL_DATA_SIZE TOTAL_TRAINING_STEPS TRAINING_ROLLOUTS_TOTAL PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU PPO_MINI_BATCH_PER_GPU PPO_GLOBAL_MICRO_BATCH PPO_GRAD_ACCUM_STEPS LOG_PROB_MICRO_BATCH_PER_GPU REF_LOG_PROB_MICRO_BATCH_PER_GPU ROLLOUT_LOGPROB_GLOBAL_MICRO REF_LOGPROB_GLOBAL_MICRO BATCH_ADJUST_DIVISOR MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_STEPS WEBSHOP_PROMPT_CHAR_LIMIT TEST_FREQ VAL_BEFORE_TRAIN ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD TREE_RL_ORDER TREE_RL_MIN_UPDATES TREE_RL_MIN_TRAIN_EPISODES TREE_RL_TRAIN_SUCCESS_THRESHOLD TREE_RL_MIN_PROBE_EPISODES TREE_RL_PROBE_SUCCESS_THRESHOLD TREE_RL_STATE_SAVE_FREQ COMPACT_FINISHED_TRAJECTORIES BATCH_TOKENIZE_OBSERVATIONS CLOUD_BOOTSTRAP_CHECK CLOUD_BOOTSTRAP_PROBE RESUME_SKILL_TREE_STATE PREPARE_DATA REUSE_PREPARED_DATA; do
        echo "$key=${!key-}"
    done
    echo "ROLLOUTS_PER_STEP=$ROLLOUTS_PER_STEP"
    [[ "$BENCHMARK" == "webshop" ]] && echo "WEBSHOP_DATA_DIR=$WEBSHOP_DATA_DIR"
} > "$OUTPUT_DIR/run_config.env"
printf '%s\n' "${ppo_args[@]}" > "$OUTPUT_DIR/ppo_args.txt"

TEE_ARGS=()
if [[ -f "$OUTPUT_DIR/training.log" ]]; then
    TEE_ARGS=(-a)
    printf '\n===== CoSkill tree RL resume: %s =====\n' "$(date -Is)" >> "$OUTPUT_DIR/training.log"
fi
python3 -u -m verl.trainer.main_ppo "${ppo_args[@]}" "$@" 2>&1 | tee "${TEE_ARGS[@]}" "$OUTPUT_DIR/training.log"
