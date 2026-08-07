#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

IS_CONTAINER=0
if [[ "${COSKILL_CONTAINER:-0}" == "1" || -f /.dockerenv || -f /run/.containerenv ]]; then
    IS_CONTAINER=1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DETECTED_GPU_COUNT="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$DETECTED_GPU_COUNT" -le 0 ]]; then
        echo "No CUDA GPUs are visible." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$(awk -v n="$DETECTED_GPU_COUNT" 'BEGIN {for (i=0;i<n;i++) printf "%s%d", (i?",":""), i}')"
fi
NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
VLLM_PARALLEL_TOPOLOGY="${VLLM_PARALLEL_TOPOLOGY:-auto}"
case "$VLLM_PARALLEL_TOPOLOGY" in
    auto)
        DEFAULT_DP="$NUM_VISIBLE_GPUS"
        DEFAULT_TP=1
        DEFAULT_PP=1
        ;;
    dp2_tp2)
        if [ "$NUM_VISIBLE_GPUS" -lt 4 ]; then
            echo "VLLM_PARALLEL_TOPOLOGY=dp2_tp2 requires four visible GPUs." >&2
            exit 1
        fi
        DEFAULT_DP=2
        DEFAULT_TP=2
        DEFAULT_PP=1
        ;;
    dp4)
        if [ "$NUM_VISIBLE_GPUS" -lt 4 ]; then
            echo "VLLM_PARALLEL_TOPOLOGY=dp4 requires four visible GPUs." >&2
            exit 1
        fi
        DEFAULT_DP=4
        DEFAULT_TP=1
        DEFAULT_PP=1
        ;;
    manual)
        DEFAULT_DP="$NUM_VISIBLE_GPUS"
        DEFAULT_TP=1
        DEFAULT_PP=1
        ;;
    *)
        echo "Unknown VLLM_PARALLEL_TOPOLOGY=$VLLM_PARALLEL_TOPOLOGY (use auto, dp2_tp2, dp4, or manual)." >&2
        exit 1
        ;;
esac
DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$DEFAULT_DP}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TP}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-$DEFAULT_PP}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
REQUIRED_GPUS=$((DATA_PARALLEL_WORKERS * TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
if [ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]; then
    echo "Need $REQUIRED_GPUS visible GPUs for DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE; only $NUM_VISIBLE_GPUS are visible." >&2
    exit 1
fi
ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-$CUDA_VISIBLE_DEVICES}"
if [[ "$IS_CONTAINER" == "1" || "$NUM_VISIBLE_GPUS" -gt 1 ]]; then
    DEFAULT_VLLM_ENFORCE_EAGER=0
else
    DEFAULT_VLLM_ENFORCE_EAGER=1
fi
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-$DEFAULT_VLLM_ENFORCE_EAGER}"
if [ "$VLLM_ENFORCE_EAGER" != "0" ] && [ "$VLLM_ENFORCE_EAGER" != "1" ]; then
    echo "VLLM_ENFORCE_EAGER must be 0 or 1." >&2
    exit 1
fi

source "$PROJECT_ROOT/scripts/configure_vllm_acceleration.sh"

if [[ "$IS_CONTAINER" == "1" ]]; then
    RUN_ENV="Docker container"
    export CACHE_ROOT="${CACHE_ROOT:-/models/.cache}"
    export DATA_ROOT="${DATA_ROOT:-/opt/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-/outputs}"
else
    RUN_ENV="host"
    export CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
    export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/verl-agent}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
fi

export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

if [ -z "${WEBSHOP_DATA_DIR:-}" ]; then
    WEBSHOP_DATA_DIR="$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data"
fi
if [ -z "${WEBSHOP_DATA_DIR:-}" ]; then
    echo "WebShop assets not found. Set WEBSHOP_DATA_DIR to the data directory beside a populated search_engine/indexes directory." >&2
    exit 1
fi
export WEBSHOP_DATA_DIR

PROJECT_NAME="${PROJECT_NAME:-verl_agent_webshop}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_skill_tree_evolve_norl_coskill_standard}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-12}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-6}"
TOTAL_GROUPS="${TOTAL_GROUPS:-100}"
MAX_EPISODES="${MAX_EPISODES:-7200}"
VALIDATION_EVERY_GROUPS="${VALIDATION_EVERY_GROUPS:-5}"
VALIDATION_BEFORE_TRAIN="${VALIDATION_BEFORE_TRAIN:-1}"
VALIDATION_TEMPERATURE="${VALIDATION_TEMPERATURE:-0.4}"
VALIDATION_SEED="${VALIDATION_SEED:-1000}"
CHECKPOINT_EVERY_GROUPS="${CHECKPOINT_EVERY_GROUPS:-2}"
RESUME="${RESUME:-0}"
if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
    echo "RESUME must be 0 (new run) or 1 (continue from summary_partial.json)." >&2
    exit 1
fi
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-0}"
PROMPT_CHAR_LIMIT="${PROMPT_CHAR_LIMIT:-24000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THINK_BUDGET="${THINK_BUDGET:-3840}"
ACTION_BUDGET="${ACTION_BUDGET:-256}"
if [ "$THINK_BUDGET" -le 0 ] || [ "$ACTION_BUDGET" -le 0 ]; then
    echo "Legacy THINK_BUDGET and ACTION_BUDGET metadata must remain positive" >&2
    exit 1
fi

echo "Run environment: $RUN_ENV"
echo "CACHE_ROOT: $CACHE_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "vLLM topology: $VLLM_PARALLEL_TOPOLOGY (DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE; GPUs=$REQUIRED_GPUS)"
echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS:-0} (0 means each worker's actual rollout batch size)"
echo "vLLM enforce_eager: $VLLM_ENFORCE_EAGER (0 enables CUDA Graphs after warm-up)"
echo "vLLM FlashInfer sampler: $VLLM_USE_FLASHINFER_SAMPLER"
echo "WebShop data: $WEBSHOP_DATA_DIR"
echo "Rollout standard: train=$TRAIN_DATA_SIZE val=$VAL_DATA_SIZE group_size=$GROUP_SIZE groups=$TOTAL_GROUPS max_episodes=$MAX_EPISODES"
echo "Held-out validation: split=[0,500) goals=$VAL_DATA_SIZE every=$VALIDATION_EVERY_GROUPS groups before_train=$VALIDATION_BEFORE_TRAIN temp=$VALIDATION_TEMPERATURE"
echo "Resume: $RESUME (requires checkpoint-consistent summary_partial.json in OUTPUT_DIR)"
echo "Token standard: prompt<=8192, one response<=$MAX_TOKENS (legacy think=$THINK_BUDGET action=$ACTION_BUDGET)"
echo "Outputs: $OUTPUT_DIR"

if [[ "${COSKILL_LAUNCHER_DRY_RUN:-0}" == "1" ]]; then
    echo "CoSkill no-RL launcher dry run passed: benchmark=webshop env=$RUN_ENV GPUs=$NUM_VISIBLE_GPUS DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE DATA_ROOT=$DATA_ROOT"
    exit 0
fi

TEE_ARGS=()
if [ "$RESUME" = "1" ]; then
    TEE_ARGS=(-a)
    printf '\n===== CoSkill WebShop resume: %s =====\n' "$(date -Is)" >> "$OUTPUT_DIR/driver.log"
fi

python3 -u -m examples.playbook_evolve.run_webshop_evolve \
    --outdir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --webshop_file_path "$WEBSHOP_DATA_DIR/items_shuffle_1000.json" \
    --webshop_attr_path "$WEBSHOP_DATA_DIR/items_ins_v2_1000.json" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE" \
    --validation_every_groups "$VALIDATION_EVERY_GROUPS" \
    --validation_before_train "$VALIDATION_BEFORE_TRAIN" \
    --validation_temperature "$VALIDATION_TEMPERATURE" \
    --validation_seed "$VALIDATION_SEED" \
    --group_size "$GROUP_SIZE" \
    --total_groups "$TOTAL_GROUPS" \
    --max_episodes "$MAX_EPISODES" \
    --max_steps 15 \
    --seed 0 \
    --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
    --rollout_worker_gpus "$ROLLOUT_WORKER_GPUS" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
    --checkpoint_every_groups "$CHECKPOINT_EVERY_GROUPS" \
    --resume "$RESUME" \
    --history_length 8 \
    --prompt_char_limit "$PROMPT_CHAR_LIMIT" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    --think_budget "$THINK_BUDGET" \
    --action_budget "$ACTION_BUDGET" \
    --temperature 1.0 \
    --gpu_mem_util 0.8 \
    --skills_json memory_data/webshop/initial_skills.json \
    --retrieval_mode template \
    --top_k 6 \
    --enable_hierarchy 1 \
    --stable_cycles_l1 3 \
    --stable_cycles_l2 5 \
    --success_l1 0.7 \
    --demote_threshold 0.3 \
    --min_calls 10 \
    --enable_coskill 1 \
    --enable_skill_tree 1 \
    --enable_skill_tree_evolve 1 \
    --enable_failure_analysis 1 \
    --max_new_skills 3 \
    --skill_tree_evolve_min_samples 6 \
    --capacity_watermark 50000 \
    --perf_watermark 0.6 \
    --min_samples 16 \
    --loop_threshold 3 \
    --trace_enable_loop_filter 1 \
    --trace_enable_obs_delta 1 \
    --trace_enable_prefix_tree 1 \
    --trace_enable_consensus_prefix 1 \
    --trace_cloud_evidence_mode tree_only \
    --log_trajectories "$LOG_TRAJECTORIES" \
    "$@" 2>&1 | tee "${TEE_ARGS[@]}" "$OUTPUT_DIR/driver.log"
