#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE

CLOUD_BOOTSTRAP_PROBE="${CLOUD_BOOTSTRAP_PROBE:-1}"
if [[ "$CLOUD_BOOTSTRAP_PROBE" != "0" && "$CLOUD_BOOTSTRAP_PROBE" != "1" ]]; then
    echo "CLOUD_BOOTSTRAP_PROBE must be 0 or 1." >&2
    exit 1
fi

CLOUD_SKILLS_JSON="memory_data/alfworld/initial_skills.json"
for ((cloud_arg_index = 1; cloud_arg_index <= $#; cloud_arg_index++)); do
    cloud_arg="${!cloud_arg_index}"
    case "$cloud_arg" in
        --skills_json)
            next_cloud_arg_index=$((cloud_arg_index + 1))
            if [[ "$next_cloud_arg_index" -gt "$#" ]]; then
                echo "--skills_json requires a path." >&2
                exit 1
            fi
            CLOUD_SKILLS_JSON="${!next_cloud_arg_index}"
            cloud_arg_index=$next_cloud_arg_index
            ;;
        --skills_json=*)
            CLOUD_SKILLS_JSON="${cloud_arg#--skills_json=}"
            ;;
    esac
done
unset cloud_arg cloud_arg_index next_cloud_arg_index

if [[ "${COSKILL_LAUNCHER_DRY_RUN:-0}" != "1" && "${COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE:-0}" != "1" ]]; then
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        echo "DEEPSEEK_API_KEY is required before a CoSkill no-RL rollout." >&2
        echo "Export DEEPSEEK_API_KEY or set it in ${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}." >&2
        exit 2
    fi
    cloud_check_args=(
        --environment alfworld
        --skills-json "$CLOUD_SKILLS_JSON"
    )
    if [[ "$CLOUD_BOOTSTRAP_PROBE" == "1" ]]; then
        cloud_check_args+=(--probe)
    fi
    echo "Checking cloud API before CUDA/vLLM allocation (probe=$CLOUD_BOOTSTRAP_PROBE)..."
    python3 "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py" "${cloud_check_args[@]}"
    unset cloud_check_args
elif [[ "${COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE:-0}" == "1" ]]; then
    echo "Cloud API was already validated by the container entrypoint."
else
    echo "Skipping cloud API preflight for COSKILL_LAUNCHER_DRY_RUN=1."
fi
unset CLOUD_SKILLS_JSON COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID

COSKILL_ONE_GPU="${COSKILL_ONE_GPU:-0}"
if [[ "$COSKILL_ONE_GPU" != "0" && "$COSKILL_ONE_GPU" != "1" ]]; then
    echo "COSKILL_ONE_GPU must be 0 or 1." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

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
if [[ "$COSKILL_ONE_GPU" == "1" ]]; then
    SELECTED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
    if [[ -z "$SELECTED_GPU" ]]; then
        echo "COSKILL_ONE_GPU=1 but CUDA_VISIBLE_DEVICES is empty." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
    NUM_VISIBLE_GPUS=1
    DATA_PARALLEL_WORKERS=1
    ROLLOUT_WORKER_GPUS="$SELECTED_GPU"
    TENSOR_PARALLEL_SIZE=1
else
    DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$NUM_VISIBLE_GPUS}"
    ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-$CUDA_VISIBLE_DEVICES}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
fi

if ! [[ "$DATA_PARALLEL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DATA_PARALLEL_WORKERS must be a positive integer." >&2
    exit 1
fi
if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE must be a positive integer." >&2
    exit 1
fi
if [[ "$DATA_PARALLEL_WORKERS" -gt 1 && "$TENSOR_PARALLEL_SIZE" -ne 1 ]]; then
    echo "ALFWorld data-parallel workers each own one vLLM replica; use TENSOR_PARALLEL_SIZE=1 when DATA_PARALLEL_WORKERS>1." >&2
    exit 1
fi
if [[ "$DATA_PARALLEL_WORKERS" -gt 1 ]]; then
    REQUIRED_GPUS="$DATA_PARALLEL_WORKERS"
else
    REQUIRED_GPUS="$TENSOR_PARALLEL_SIZE"
fi
if [[ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]]; then
    echo "Need $REQUIRED_GPUS visible GPU(s) for DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE; only $NUM_VISIBLE_GPUS visible." >&2
    exit 1
fi

VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
if ! [[ "$VLLM_MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then
    echo "VLLM_MAX_NUM_SEQS must be a non-negative integer." >&2
    exit 1
fi
if [[ "$IS_CONTAINER" == "1" || "$NUM_VISIBLE_GPUS" -gt 1 ]]; then
    DEFAULT_VLLM_ENFORCE_EAGER=0
else
    DEFAULT_VLLM_ENFORCE_EAGER=1
fi
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-$DEFAULT_VLLM_ENFORCE_EAGER}"
if [[ "$VLLM_ENFORCE_EAGER" != "0" && "$VLLM_ENFORCE_EAGER" != "1" ]]; then
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
echo "Run environment detected: $RUN_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "data_parallel_workers: $DATA_PARALLEL_WORKERS"
echo "rollout_worker_gpus: ${ROLLOUT_WORKER_GPUS:-<auto>}"
echo "vLLM tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
echo "single-GPU mode: $COSKILL_ONE_GPU"
echo "vLLM enforce_eager: $VLLM_ENFORCE_EAGER"
echo "vLLM FlashInfer sampler: $VLLM_USE_FLASHINFER_SAMPLER"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"

export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

PROJECT_NAME="${PROJECT_NAME:-verl_agent_alfworld}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_skill_tree_evolve_norl_v8}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "$OUTPUT_DIR"
echo "All run outputs will be saved to: $OUTPUT_DIR"

MAX_EPISODES="${MAX_EPISODES:-7200}"
BATCH_ROLLOUT_SIZE="${BATCH_ROLLOUT_SIZE:-72}"
if ! [[ "$BATCH_ROLLOUT_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_ROLLOUT_SIZE must be a positive integer." >&2
    exit 1
fi
MAX_WORKER_BATCH=$(((BATCH_ROLLOUT_SIZE + DATA_PARALLEL_WORKERS - 1) / DATA_PARALLEL_WORKERS))
if [[ "$VLLM_MAX_NUM_SEQS" -gt 0 && "$VLLM_MAX_NUM_SEQS" -lt "$MAX_WORKER_BATCH" ]]; then
    echo "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS is smaller than the largest per-replica rollout batch=$MAX_WORKER_BATCH." >&2
    exit 1
fi
EFFECTIVE_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
if [[ "$EFFECTIVE_VLLM_MAX_NUM_SEQS" == "0" ]]; then
    EFFECTIVE_VLLM_MAX_NUM_SEQS="$MAX_WORKER_BATCH"
fi
echo "vLLM max_num_seqs: $VLLM_MAX_NUM_SEQS (effective largest replica limit: $EFFECTIVE_VLLM_MAX_NUM_SEQS)"
CHECKPOINT_EVERY_GROUPS="${CHECKPOINT_EVERY_GROUPS:-2}"
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-0}"

if [[ "${COSKILL_LAUNCHER_DRY_RUN:-0}" == "1" ]]; then
    echo "CoSkill no-RL launcher dry run passed: benchmark=alfworld env=$RUN_ENV GPUs=$NUM_VISIBLE_GPUS DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE batch=$BATCH_ROLLOUT_SIZE max_num_seqs=$EFFECTIVE_VLLM_MAX_NUM_SEQS DATA_ROOT=$DATA_ROOT"
    exit 0
fi

python3 -u -m examples.playbook_evolve.run_playbook_evolve \
    --outdir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --task_types "pick_and_place_simple,look_at_obj_in_light,pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep,pick_two_obj_and_place" \
    --num_games -1 \
    --group_size 6 \
    --split train \
    --max_steps 40 \
    --seed 0 \
    --epochs 1 \
    --max_episodes "$MAX_EPISODES" \
    --batch_rollout_size "$BATCH_ROLLOUT_SIZE" \
    --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
    --rollout_worker_gpus "$ROLLOUT_WORKER_GPUS" \
    --checkpoint_every_groups "$CHECKPOINT_EVERY_GROUPS" \
    --history_length 8 \
    --max_model_len 10240 \
    --max_tokens 4096 \
    --think_budget 3500 \
    --temperature 1.0 \
    --gpu_mem_util 0.8 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
    --skills_json memory_data/alfworld/initial_skills.json \
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
    "$@" 2>&1 | tee "$OUTPUT_DIR/driver.log"
