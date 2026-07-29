#!/usr/bin/env bash
# CoSkill model-scale arm: identical to the standard frozen ALFWorld CoSkill
# loop, except for the edge model.  It deliberately pins the old no-RL design
# (external skill-tree evolution; no Ray/GRPO/LoRA) for a clean scale study.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# This experiment is not the optional Tree-RL curriculum exposed by the base
# launcher.  Ignore an inherited lower-case shell variable as well.
unset rl
export RL=0

# Preserve explicit MODEL_PATH overrides (e.g. a mounted checkpoint).  The
# default follows the existing ModelScope cache layout used by the 4B launcher.
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
        export MODEL_PATH="/GLOBALFS/hit_wxia_1/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    else
        export MODEL_PATH="${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    fi
fi

export PROJECT_NAME="${PROJECT_NAME:-verl_agent_alfworld}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-0.6b_skill_tree_evolve_norl_v8}"

exec bash "$SCRIPT_DIR/run_alfworld_playbook_evolve_norl.sh" "$@"
