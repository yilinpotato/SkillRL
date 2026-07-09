#!/usr/bin/env bash
# Two-arm controlled evolution test: first bullets OFF, then bullets ON.
# Each phase contains exactly ROLLOUTS_PER_TASK rollouts of each fixed game.
# The cloud update is forced at the phase boundary, so phase 2 evaluates the
# first evolved context under identical update timing in both arms.
set -euo pipefail

cd "$(dirname "$0")/../.."
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
AB_ROOT="${AB_ROOT:-$PWD/skillrl_outputs/fixed_two_tasks_ab/$STAMP}"
MANIFEST="${MANIFEST:-$PWD/examples/playbook_evolve/fixed_two_tasks.json}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-embedding}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-6}"
MAX_STEPS="${MAX_STEPS:-40}"
PHASE_EPISODES=$((2 * ROLLOUTS_PER_TASK))
TOTAL_EPISODES=$((2 * PHASE_EPISODES))
mkdir -p "$AB_ROOT"

run_arm() {
    local name="$1"
    local bullets="$2"
    echo ">>> Running $name (enable_coskill=$bullets) -> $AB_ROOT/$name"
    OUTPUT_DIR="$AB_ROOT/$name" \
      bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh \
        --fixed_games_manifest "$MANIFEST" \
        --group_size "$ROLLOUTS_PER_TASK" \
        --epochs 2 \
        --max_episodes "$TOTAL_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --cloud_update_every "$PHASE_EPISODES" \
        --capacity_watermark 1000000000 \
        --perf_watermark 1.1 \
        --min_samples 999999 \
        --skill_tree_evolve_min_samples "$ROLLOUTS_PER_TASK" \
        --retrieval_mode "$RETRIEVAL_MODE" \
        --enable_coskill "$bullets"
}

# Requested order: establish skill-tree-only evolution first, then add bullets.
run_arm bullets_off 0
run_arm bullets_on 1

python3 \
  examples/playbook_evolve/compare_fixed_two_tasks.py \
  --off "$AB_ROOT/bullets_off/summary.json" \
  --on "$AB_ROOT/bullets_on/summary.json" \
  --out "$AB_ROOT/comparison.md"

echo ">>> Done: $AB_ROOT/comparison.md"
