#!/usr/bin/env bash
# Qwen3-0.6B model-scale arm.  All canonical SkillRL settings are inherited;
# only the local edge-model path and isolated output directory change.
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

export PROJECT_NAME="${PROJECT_NAME:-verl_agent_webshop}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0.6b_webshop_skillRL_v3}"

exec bash "$SCRIPT_DIR/run_webshop_skills.sh" "$@"
