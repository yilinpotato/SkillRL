#!/usr/bin/env bash
# Multi-goal ablation: harder/wider than fixed_two_tasks.
#
# Default arms:
#   none            : no skill tree, no flat skill-patch bullets
#   tree_only       : agent skill tree only
#   patch_only      : flat skill-patch bullets only
#   tree_plus_patch : skill tree + contrastively distilled flat skill-patch bullets
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
AB_ROOT="${AB_ROOT:-$PWD/skillrl_outputs/goal_sweep_ab/$STAMP}"
TASK_TYPES="${TASK_TYPES:-pick_and_place_simple,look_at_obj_in_light,pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep,pick_two_obj_and_place}"

# A practical default: 6 goal types × 2 diverse games/type × 1 rollout/game × 3 epochs
# = 36 episodes per arm, with one cloud update after each epoch. Four arms by
# default = 144 episodes total.
# Raise NUM_GAMES_PER_TYPE / ROLLOUTS_PER_GAME / EPOCHS for the full paper run.
NUM_GAMES_PER_TYPE="${NUM_GAMES_PER_TYPE:-2}"
ROLLOUTS_PER_GAME="${ROLLOUTS_PER_GAME:-1}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-40}"
SAMPLE_SEED="${SAMPLE_SEED:-7}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-template}"

N_TASK_TYPES=$(python3 - <<PY
print(len([x for x in "${TASK_TYPES}".split(",") if x.strip()]))
PY
)
EPISODES_PER_EPOCH=$((N_TASK_TYPES * NUM_GAMES_PER_TYPE * ROLLOUTS_PER_GAME))
TOTAL_EPISODES=$((EPISODES_PER_EPOCH * EPOCHS))
# Per-task skill trees evolve from samples of their own task type, so this
# threshold is per task type rather than per mixed epoch.
TREE_MIN_SAMPLES="${TREE_MIN_SAMPLES:-$((NUM_GAMES_PER_TYPE * ROLLOUTS_PER_GAME))}"

mkdir -p "$AB_ROOT"
echo ">>> goal_sweep root: $AB_ROOT"
echo ">>> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ">>> task_types=$TASK_TYPES"
echo ">>> episodes_per_epoch=$EPISODES_PER_EPOCH total_episodes_per_arm=$TOTAL_EPISODES"

run_arm() {
    local name="$1"
    local enable_tree="$2"
    local enable_tree_evolve="$3"
    local enable_patches="$4"
    echo ">>> Running $name (tree=$enable_tree tree_evolve=$enable_tree_evolve patches=$enable_patches) -> $AB_ROOT/$name"
    OUTPUT_DIR="$AB_ROOT/$name" \
      bash examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh \
        --task_types "$TASK_TYPES" \
        --num_games "$NUM_GAMES_PER_TYPE" \
        --sample \
        --sample_seed "$SAMPLE_SEED" \
        --group_size "$ROLLOUTS_PER_GAME" \
        --epochs "$EPOCHS" \
        --max_episodes "$TOTAL_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --cloud_update_every "$EPISODES_PER_EPOCH" \
        --capacity_watermark 1000000000 \
        --perf_watermark 1.1 \
        --min_samples 999999 \
        --skill_tree_evolve_min_samples "$TREE_MIN_SAMPLES" \
        --retrieval_mode "$RETRIEVAL_MODE" \
        --enable_skill_tree "$enable_tree" \
        --enable_skill_tree_evolve "$enable_tree_evolve" \
        --enable_coskill "$enable_patches"
}

run_arm none 0 0 0
run_arm tree_only 1 1 0
run_arm patch_only 0 0 1
run_arm tree_plus_patch 1 1 1

python3 examples/playbook_evolve/compare_goal_sweep_ab.py \
  --root "$AB_ROOT" \
  --baseline none \
  --out "$AB_ROOT/comparison.md"

echo ">>> Done: $AB_ROOT/comparison.md"
