#!/usr/bin/env bash
# Qwen3-0.6B model-scale arm.  It delegates to the canonical SkillRL launcher
# so every training, retrieval, dynamic-skill, and token-budget setting stays
# identical to the 4B arm; only MODEL_PATH and the output name differ.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
        export MODEL_PATH="/GLOBALFS/hit_wxia_1/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    else
        export MODEL_PATH="${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    fi
fi

export PROJECT_NAME="${PROJECT_NAME:-verl_agent_alfworld}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_0.6b_skills_dynamic_lora_v2}"

exec bash "$SCRIPT_DIR/run_alfworld_skills_train_update_lora.sh" "$@"
