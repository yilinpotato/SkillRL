#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/CoSkill}"
cd "$PROJECT_ROOT"

TASK="${1:-preflight}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

ensure_model() {
    if [[ -f "$MODEL_PATH/config.json" ]] && compgen -G "$MODEL_PATH/*.safetensors" >/dev/null; then
        echo "Model ready: $MODEL_PATH"
        return
    fi
    if [[ "${MODEL_AUTO_DOWNLOAD:-1}" != "1" ]]; then
        echo "Model not found at $MODEL_PATH and MODEL_AUTO_DOWNLOAD is disabled." >&2
        exit 1
    fi
    echo "Downloading ${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-4B-Thinking-2507} from ModelScope to $MODEL_PATH"
    mkdir -p "$MODEL_PATH"
    download_args=(
        --model "${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-4B-Thinking-2507}"
        --local_dir "$MODEL_PATH"
        --max-workers "${MODELSCOPE_MAX_WORKERS:-8}"
    )
    if [[ -n "${MODELSCOPE_MODEL_REVISION:-}" ]]; then
        download_args+=(--revision "$MODELSCOPE_MODEL_REVISION")
    fi
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 modelscope download "${download_args[@]}"
}

gpu_preflight() {
    python - <<'PY'
import torch

n = torch.cuda.device_count()
if n not in (2, 4, 8):
    raise SystemExit(f"CoSkill Tree-RL requires 2, 4, or 8 visible GPUs, got {n}")
print(f"Visible CUDA GPUs: {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {props.name} {props.total_memory / 2**30:.1f} GiB")
if n == 8:
    print("An individual experiment will use one four-GPU slot; set TREE_RL_GPU_SLOT=0 or 1.")
PY
}

data_preflight() {
    python docker/coskill/preflight.py
}

case "$TASK" in
    shell)
        exec bash "$@"
        ;;
    preflight)
        ensure_model
        gpu_preflight
        data_preflight
        if [[ "${CLOUD_BOOTSTRAP_PROBE:-0}" == "1" ]]; then
            python scripts/check_cloud_bootstrap.py \
                --environment "${PREFLIGHT_BENCHMARK:-alfworld}" \
                --skills-json "memory_data/${PREFLIGHT_BENCHMARK:-alfworld}/claude_style_skills.json" \
                --probe
        fi
        echo "Preflight passed. Choose alfworld-root, alfworld-leaf, webshop-root, or webshop-leaf."
        ;;
    alfworld-root|alfworld-leaf|webshop-root|webshop-leaf)
        ensure_model
        gpu_preflight
        data_preflight
        BENCHMARK="${TASK%%-*}"
        TREE_RL_ORDER="${TASK##*-}"
        export COSKILL_CONTAINER=1 TREE_RL_ORDER
        exec bash examples/grpo_trainer/run_coskill_tree_rl.sh "$BENCHMARK" "$@"
        ;;
    *)
        echo "Unknown task '$TASK'. Use preflight, shell, alfworld-root, alfworld-leaf, webshop-root, or webshop-leaf." >&2
        exit 2
        ;;
esac
