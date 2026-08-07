#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/smoke_outputs/alfworld_terminal_dual}"
COSKILL_SKILLS="${COSKILL_SKILLS:-$WORKSPACE_ROOT/coskill-skilltree noRL alfworld fix/skill_lib/skills_checkpoint_step3600.json}"
SPEEDUP="${SPEEDUP:-100}"
GPU="${GPU:-0}"
GPU_NAME="${GPU_NAME:-NVIDIA GeForce RTX 3090}"

cd "$PROJECT_ROOT"
exec "${PYTHON:-python}" -u examples/terminal_compare/replay_alfworld_serial.py \
    --run-dir "$OUTPUT_ROOT/coskill" \
    --label COSKILL \
    --skill-source-label "CoSkill tree" \
    --skill-source "$COSKILL_SKILLS" \
    --gpu-index "$GPU" \
    --gpu-name "$GPU_NAME" \
    --speedup "$SPEEDUP" \
    "$@"
