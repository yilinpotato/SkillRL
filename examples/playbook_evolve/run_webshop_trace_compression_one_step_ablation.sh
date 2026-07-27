#!/usr/bin/env bash
# WebShop-only shared 72-rollout full-trajectory training-step compression
# ablation. One immutable 12-goal x 6-replica capture is reused by the normal
# CoSkill trajectory-tree arm and the all trace transforms off arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/load_private_env.sh"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "DEEPSEEK_API_KEY is required; export it or put it in $PROJECT_ROOT/.env." >&2
    exit 2
fi

IS_CONTAINER=0
if [[ "${COSKILL_CONTAINER:-0}" == "1" || -f /.dockerenv || -f /run/.containerenv ]]; then
    IS_CONTAINER=1
fi

if [[ "$IS_CONTAINER" == "1" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        DETECTED_GPU_COUNT="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
        if [[ "$DETECTED_GPU_COUNT" -le 0 ]]; then
            echo "No CUDA GPUs are visible inside this container." >&2
            exit 2
        fi
        export CUDA_VISIBLE_DEVICES="$(
            awk -v n="$DETECTED_GPU_COUNT" \
                'BEGIN {for (i=0;i<n;i++) printf "%s%d", (i?",":""), i}'
        )"
    fi
    export CACHE_ROOT="${CACHE_ROOT:-/models/.cache}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-/outputs}"
elif [[ -d /GLOBALFS/hit_wxia_1 ]]; then
    export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
    # Respect a scheduler allocation. The safe project fallback is physical
    # GPU 1; users can request two cards with CUDA_VISIBLE_DEVICES=0,1.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
else
    export CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "1" ]]; then
        echo "Local shared-server execution only permits CUDA_VISIBLE_DEVICES=1." >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES=1
fi

GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF{n++} END{print n+0}')"
DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$GPU_COUNT}"
if [[ "$DATA_PARALLEL_WORKERS" != "$GPU_COUNT" ]]; then
    echo "DATA_PARALLEL_WORKERS must equal visible GPU count for shared capture." >&2
    exit 2
fi
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
REQUIRED_GPUS=$((DATA_PARALLEL_WORKERS * TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
if [[ "$REQUIRED_GPUS" -gt "$GPU_COUNT" ]]; then
    echo "Need $REQUIRED_GPUS GPUs for DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE; only $GPU_COUNT visible." >&2
    exit 2
fi

if [[ -z "${WEBSHOP_DATA_DIR:-}" ]]; then
    for candidate in \
        "$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data" \
        "$(dirname "$PROJECT_ROOT")/Skill0/agent_system/environments/env_package/webshop/webshop/data"
    do
        if [[ -f "$candidate/items_shuffle_1000.json" \
            && -f "$candidate/items_ins_v2_1000.json" \
            && -f "$candidate/items_human_ins.json" \
            && -d "$(dirname "$candidate")/search_engine/indexes" ]]; then
            WEBSHOP_DATA_DIR="$candidate"
            break
        fi
    done
fi
if [[ -z "${WEBSHOP_DATA_DIR:-}" ]]; then
    echo "WebShop assets not found. Set WEBSHOP_DATA_DIR to the populated data directory." >&2
    exit 2
fi
export WEBSHOP_DATA_DIR

export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
# Both arms receive exactly the same expanded evidence-rendering budget.
export COSKILL_CLOUD_EVIDENCE_MULTIPLIER="${COSKILL_CLOUD_EVIDENCE_MULTIPLIER:-10}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

SKILLS_JSON="${SKILLS_JSON:-memory_data/webshop/claude_style_skills.json}"
echo "Checking cloud API before CUDA/model setup (probe=${CLOUD_BOOTSTRAP_PROBE:-1})..."
probe_args=(--environment webshop --skills-json "$SKILLS_JSON")
if [[ "${CLOUD_BOOTSTRAP_PROBE:-1}" == "1" ]]; then
    probe_args+=(--probe)
fi
python3 "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py" "${probe_args[@]}"

AB_ROOT="${AB_ROOT:-$OUTPUT_ROOT/webshop_train_step_trace_compression_ablation/$(date +%Y%m%d_%H%M%S)}"
echo "[webshop-train-step-trace-ablation] GPUs=$CUDA_VISIBLE_DEVICES DP=$DATA_PARALLEL_WORKERS TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE root=$AB_ROOT"
echo "[webshop-train-step-trace-ablation] shared capture: 12 distinct train goals x 6 replicas = 72 full rollouts x max_steps=15"
echo "[webshop-train-step-trace-ablation] cloud arms: normal tree codec / all trace transforms off"
echo "[webshop-train-step-trace-ablation] WebShop data=$WEBSHOP_DATA_DIR"
echo "[webshop-train-step-trace-ablation] cloud evidence multiplier=$COSKILL_CLOUD_EVIDENCE_MULTIPLIER (both arms)"

if [[ "${COSKILL_LAUNCHER_DRY_RUN:-0}" == "1" ]]; then
    echo "WebShop trace-compression launcher dry run passed."
    exit 0
fi

exec python3 -u -m examples.playbook_evolve.webshop_trace_compression_one_step_ablation \
    --root "$AB_ROOT" \
    --webshop_file_path "$WEBSHOP_DATA_DIR/items_shuffle_1000.json" \
    --webshop_attr_path "$WEBSHOP_DATA_DIR/items_ins_v2_1000.json" \
    --model_path "$MODEL_PATH" \
    --skills_json "$SKILLS_JSON" \
    --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
    --rollout_worker_gpus "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-0}" \
    --gpu_mem_util "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
    --prompt_char_limit "${PROMPT_CHAR_LIMIT:-24000}" \
    --max_model_len "${MAX_MODEL_LEN:-12288}" \
    --max_tokens "${MAX_TOKENS:-4096}" \
    --think_budget "${THINK_BUDGET:-3840}" \
    --action_budget "${ACTION_BUDGET:-256}" \
    "$@"
