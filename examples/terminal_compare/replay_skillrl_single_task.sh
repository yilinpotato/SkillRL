#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/smoke_outputs/alfworld_terminal_dual}"
SKILLRL_SKILLS="${SKILLRL_SKILLS:-$WORKSPACE_ROOT/skillRL alfworld/updated_skills_step50.json}"
SPEEDUP="${SPEEDUP:-100}"
TASK_INDEX="${TASK_INDEX:-1}"
GPU="${GPU:-1}"
GPU_NAME="${GPU_NAME:-NVIDIA GeForce RTX 3090}"

cd "$PROJECT_ROOT"
exec "${PYTHON:-python}" -u examples/terminal_compare/replay_alfworld_single_task.py \
    --run-dir "$OUTPUT_ROOT/skillrl" \
    --label SKILLRL \
    --skill-source-label "SkillRL flat skills" \
    --skill-source "$SKILLRL_SKILLS" \
    --gpu-index "$GPU" \
    --gpu-name "$GPU_NAME" \
    --task-index "$TASK_INDEX" \
    --speedup "$SPEEDUP" \
    "$@"
