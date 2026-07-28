#!/usr/bin/env bash
# CoSkill model-scale arm: identical to the standard frozen WebShop CoSkill
# loop, except for the edge model.  It pins the old no-RL external skill-tree
# evolution design so model scale is the only experimental variable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

unset rl
export RL=0

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
        export MODEL_PATH="/GLOBALFS/hit_wxia_1/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B"
    else
        export MODEL_PATH="${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B"
    fi
fi

export PROJECT_NAME="${PROJECT_NAME:-verl_agent_webshop}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-1.7b_skill_tree_evolve_norl_coskill_standard}"

exec bash "$SCRIPT_DIR/run_webshop_playbook_evolve_norl.sh" "$@"
