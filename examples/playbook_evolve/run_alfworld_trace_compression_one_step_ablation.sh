#!/usr/bin/env bash
# ALFWorld-only, shared 72-rollout full-trajectory training-step compression ablation.
# It captures raw trajectories once and reuses their exact SHA-256 across the
# normal CoSkill compression arm and the all-transforms-off arm.
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

if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
    export CACHE_ROOT="${CACHE_ROOT:-/GLOBALFS/hit_wxia_1/.cache}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
    # Respect an allocation supplied by the scheduler.  The fallback is the
    # project-wide permitted single-card default, never an implicit GPU 0.
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
[[ "$DATA_PARALLEL_WORKERS" == "$GPU_COUNT" ]] || {
    echo "DATA_PARALLEL_WORKERS must equal the visible GPU count for this capture." >&2
    exit 2
}

export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
# Both arms use this same expanded rendering budget.  It exposes the token
# effect of full trajectory compression without changing normal CoSkill runs.
export COSKILL_CLOUD_EVIDENCE_MULTIPLIER="${COSKILL_CLOUD_EVIDENCE_MULTIPLIER:-10}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

SKILLS_JSON="${SKILLS_JSON:-memory_data/alfworld/claude_style_skills.json}"
echo "Checking cloud API before CUDA/model setup (probe=${CLOUD_BOOTSTRAP_PROBE:-1})..."
probe_args=(--environment alfworld --skills-json "$SKILLS_JSON")
if [[ "${CLOUD_BOOTSTRAP_PROBE:-1}" == "1" ]]; then
    probe_args+=(--probe)
fi
python3 "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py" "${probe_args[@]}"

AB_ROOT="${AB_ROOT:-$OUTPUT_ROOT/alfworld_train_step_trace_compression_ablation/$(date +%Y%m%d_%H%M%S)}"
echo "[train-step-trace-ablation] GPUs=$CUDA_VISIBLE_DEVICES DP=$DATA_PARALLEL_WORKERS root=$AB_ROOT"
echo "[train-step-trace-ablation] shared capture: 6 task types x 12 rollouts x max_steps=40; cloud arms: on/off"
echo "[train-step-trace-ablation] cloud evidence multiplier=$COSKILL_CLOUD_EVIDENCE_MULTIPLIER (both arms)"

exec python -u -m examples.playbook_evolve.trace_compression_one_step_ablation \
    --root "$AB_ROOT" --alfworld_data "$ALFWORLD_DATA" --model_path "$MODEL_PATH" \
    --skills_json "$SKILLS_JSON" --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
    --rollout_worker_gpus "$CUDA_VISIBLE_DEVICES" \
    --gpu_mem_util "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" "$@"
