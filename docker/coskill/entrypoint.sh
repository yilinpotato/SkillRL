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
    local mode="${1:-tree_rl}"
    python - "$mode" <<'PY'
import os
import sys

import torch

n = torch.cuda.device_count()
mode = sys.argv[1]
allow_single = os.environ.get("PREFLIGHT_ALLOW_SINGLE_GPU", "0") == "1"
if mode == "rollout":
    if n < 1:
        raise SystemExit("No CUDA GPUs are visible for no-RL rollout.")
    print(f"Visible CUDA GPUs for no-RL rollout: {n}")
elif n == 1 and (allow_single or mode == "smoke"):
    if mode == "smoke":
        print("Visible CUDA GPUs: 1 (diagnostic smoke only; formal Tree-RL still needs 2, 4, or 8 GPUs)")
    else:
        print("Visible CUDA GPUs: 1 (preflight-only mode; formal Tree-RL still needs 2, 4, or 8 GPUs)")
elif n not in (2, 4, 8):
    raise SystemExit(f"CoSkill Tree-RL requires 2, 4, or 8 visible GPUs, got {n}")
else:
    print(f"Visible CUDA GPUs: {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {props.name} {props.total_memory / 2**30:.1f} GiB")
if n == 8:
    print("Default: one four-GPU slot (TREE_RL_GPU_SLOT=0 or 1). Opt-in: TREE_RL_USE_ALL_8=1 uses all 8 GPUs but changes PPO mini-batch geometry.")
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
        echo "Preflight passed. Choose alfworld-root, alfworld-leaf, webshop-root, webshop-leaf, alfworld-smoke, webshop-smoke, alfworld-norl, webshop-norl, or alfworld-ablation."
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
    alfworld-smoke|webshop-smoke)
        ensure_model
        gpu_preflight smoke
        data_preflight
        BENCHMARK="${TASK%%-*}"
        export COSKILL_CONTAINER=1
        exec bash examples/grpo_trainer/run_coskill_tree_rl_smoke.sh "$BENCHMARK" "$@"
        ;;
    alfworld-norl|webshop-norl)
        ensure_model
        gpu_preflight rollout
        data_preflight
        BENCHMARK="${TASK%%-*}"
        export COSKILL_CONTAINER=1
        if [[ "$BENCHMARK" == "alfworld" ]]; then
            exec bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh "$@"
        fi
        exec bash examples/playbook_evolve/run_webshop_playbook_evolve_norl.sh "$@"
        ;;
    alfworld-ablation)
        ensure_model
        data_preflight
        # The fixed-trajectory protocol has its own runner and its own
        # rollout-count contract.  It is deliberately not folded into GRPO.
        export COSKILL_CONTAINER=1
        export AB_ROOT="${AB_ROOT:-$OUTPUT_ROOT/alfworld_fixed_trajectory_ablation}"
        exec bash examples/playbook_evolve/run_alfworld_fixed_trajectory_ablation.sh "$@"
        ;;
    *)
        echo "Unknown task '$TASK'. Use preflight, shell, alfworld-root, alfworld-leaf, webshop-root, webshop-leaf, alfworld-smoke, webshop-smoke, alfworld-norl, webshop-norl, or alfworld-ablation." >&2
        exit 2
        ;;
esac
