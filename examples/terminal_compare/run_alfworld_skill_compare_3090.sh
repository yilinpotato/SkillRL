#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
MODEL_PATH="${MODEL_PATH:-$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
COSKILL_SKILLS="${COSKILL_SKILLS:-$WORKSPACE_ROOT/coskill-skilltree noRL alfworld fix/skill_lib/skills_checkpoint_step3600.json}"
SKILLRL_SKILLS="${SKILLRL_SKILLS:-$WORKSPACE_ROOT/skillRL alfworld/updated_skills_step50.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/smoke_outputs/alfworld_terminal_skill_compare}"
GPU="${GPU:-0}"
ARM="${ARM:-all}"

cd "$PROJECT_ROOT"
exec "${PYTHON:-python}" -u examples/terminal_compare/alfworld_skill_compare.py \
    --project-root "$PROJECT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --alfworld-data "$ALFWORLD_DATA" \
    --model-path "$MODEL_PATH" \
    --coskill-skills "$COSKILL_SKILLS" \
    --skillrl-skills "$SKILLRL_SKILLS" \
    --arm "$ARM" \
    --gpu "$GPU" \
    --tasks-per-type 4 \
    "$@"
