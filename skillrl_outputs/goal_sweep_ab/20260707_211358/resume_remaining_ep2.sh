#!/usr/bin/env bash
set -euo pipefail

cd /data2/myl/CoSkill
set +u
source /data2/myl/miniconda3/etc/profile.d/conda.sh
conda activate skillRL
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo ">>> python: $(command -v python3)"
echo ">>> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

AB_ROOT=/data2/myl/CoSkill/skillrl_outputs/goal_sweep_ab/20260707_211358
TASK_TYPES="pick_and_place_simple,look_at_obj_in_light,pick_clean_then_place_in_recep,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep,pick_two_obj_and_place"
NUM_GAMES_PER_TYPE=2
ROLLOUTS_PER_GAME=1
EPOCHS=2
MAX_STEPS=40
SAMPLE_SEED=7
RETRIEVAL_MODE=embedding
EPISODES_PER_EPOCH=12
TOTAL_EPISODES=24
TREE_MIN_SAMPLES=12

run_arm() {
  local name="$1"
  local enable_tree="$2"
  local enable_tree_evolve="$3"
  local enable_patches="$4"
  echo ">>> [$(date)] Running $name (tree=$enable_tree tree_evolve=$enable_tree_evolve patches=$enable_patches) -> $AB_ROOT/$name"
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

run_arm tree_only 1 1 0
run_arm patch_only 0 0 1
run_arm tree_plus_patch 1 1 1

python3 examples/playbook_evolve/compare_goal_sweep_ab.py \
  --root "$AB_ROOT" \
  --baseline none \
  --out "$AB_ROOT/comparison.md"

echo ">>> [$(date)] Done: $AB_ROOT/comparison.md"
