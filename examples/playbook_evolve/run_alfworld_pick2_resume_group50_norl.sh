#!/usr/bin/env bash
# Resume the *state* of the published ALFWorld CoSkill no-RL run at the end of
# rollout group 50, then re-sample only the pick_two_obj_and_place slice for
# virtual groups 51--100. Each original six-task group contains 12 pick2
# episodes, so this fresh suffix has 50 x 12 = 600 episodes. The companion
# merger replaces only those original pick2 rows and retains all other tasks.
#
# Pinned source experiment:
#   yilinpotato/SkillRL@65ee21b0d0cd9976e2097d469cc548f66dad286f
#   outputs/alfworld_full_20260721/checkpoints/step003600.json
#   outputs/alfworld_full_20260721/skill_lib/skills_checkpoint_step3600.json
#
# Normal launch (requires exactly two visible GPUs, as in the source run).
# Keep this launcher anywhere convenient, but point PROJECT_ROOT at a clean
# checkout of the pinned commit, not at a later development worktree:
#   PROJECT_ROOT=/path/to/clean/SkillRL-final-results \
#   CUDA_VISIBLE_DEVICES=2,3 \
#   bash /path/to/run_alfworld_pick2_resume_group50_norl.sh
#
# Safe continuation after this pick2 suffix is interrupted:
#   PROJECT_ROOT=/path/to/clean/SkillRL-final-results \
#   CUDA_VISIBLE_DEVICES=2,3 TARGET_RESUME=1 OUTPUT_DIR=/path/to/the/new/output \
#   bash /path/to/run_alfworld_pick2_resume_group50_norl.sh

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "This launcher accepts no positional/driver arguments. Use the documented environment variables only." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

# Do not enable xtrace: .env normally contains the cloud API key.
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE

readonly SOURCE_BRANCH="codex/alfworld-final-results-20260724"
readonly SOURCE_COMMIT="65ee21b0d0cd9976e2097d469cc548f66dad286f"
readonly SOURCE_DRIVER_SHA256="e710ed615e434efeb08bc7a6aeb203d54227f160f7be02c56124205a0a15e822"
readonly SOURCE_SUMMARY_SHA256="87e84c5ea45c1dab42597269a8c56faaba2f7504b4dedcdc0ad191e59c02e281"
readonly SOURCE_SKILLS_SHA256="abb2393dab2c83ba79898907de9053299458ee94aa08934c2dbb3dd63b447065"
readonly RESUME_GROUP=50
readonly EPISODES_PER_GROUP=72
readonly SOURCE_STEP=$((RESUME_GROUP * EPISODES_PER_GROUP))
readonly TARGET_GROUPS=50
readonly PICK2_EPISODES_PER_VIRTUAL_GROUP=12
readonly TARGET_EPISODES=$((TARGET_GROUPS * PICK2_EPISODES_PER_VIRTUAL_GROUP))
# The pick2-only pool is smaller than 600 episodes on some ALFWorld versions;
# max_episodes is the hard stop and epochs merely permits deterministic cycles.
readonly TARGET_EPOCHS=100
readonly PICK2_TASK_TYPE="pick_two_obj_and_place"

PICK2_DRY_RUN="${PICK2_DRY_RUN:-0}"
PICK2_STRICT_SOURCE_CHECK="${PICK2_STRICT_SOURCE_CHECK:-1}"
TARGET_RESUME="${TARGET_RESUME:-0}"
if [[ "$PICK2_DRY_RUN" != "0" && "$PICK2_DRY_RUN" != "1" ]] || \
   [[ "$PICK2_STRICT_SOURCE_CHECK" != "0" && "$PICK2_STRICT_SOURCE_CHECK" != "1" ]] || \
   [[ "$TARGET_RESUME" != "0" && "$TARGET_RESUME" != "1" ]]; then
    echo "PICK2_DRY_RUN, PICK2_STRICT_SOURCE_CHECK, and TARGET_RESUME must be 0 or 1." >&2
    exit 2
fi
if [[ "$PICK2_STRICT_SOURCE_CHECK" == "0" && "$PICK2_DRY_RUN" != "1" ]]; then
    echo "Refusing a real run with PICK2_STRICT_SOURCE_CHECK=0. It is allowed only for a dry-run syntax check." >&2
    exit 2
fi

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$PROJECT_ROOT/outputs/alfworld_full_20260721}"
SOURCE_SUMMARY="${SOURCE_SUMMARY:-$SOURCE_RUN_DIR/checkpoints/step$(printf '%06d' "$SOURCE_STEP").json}"
SOURCE_SKILLS="${SOURCE_SKILLS:-$SOURCE_RUN_DIR/skill_lib/skills_checkpoint_step${SOURCE_STEP}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/alfworld_pick2_resume_group50_norl}"

if [[ "$PICK2_STRICT_SOURCE_CHECK" == "1" ]]; then
    [[ -f "$PROJECT_ROOT/examples/playbook_evolve/run_playbook_evolve.py" ]] || {
        echo "Missing no-RL driver under PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
    actual_driver_sha="$(sha256sum "$PROJECT_ROOT/examples/playbook_evolve/run_playbook_evolve.py" | awk '{print $1}')"
    [[ "$actual_driver_sha" == "$SOURCE_DRIVER_SHA256" ]] || {
        echo "Driver SHA mismatch: expected $SOURCE_DRIVER_SHA256, got $actual_driver_sha." >&2
        echo "The original no-RL driver must remain byte-identical; do not mix later driver changes." >&2
        exit 1
    }
    if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        actual_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
        git -C "$PROJECT_ROOT" diff --quiet -- . || {
            echo "Tracked source changes detected at $actual_commit. Commit or remove them before a comparable rerun." >&2
            exit 1
        }
    fi
fi

[[ -f "$SOURCE_SUMMARY" ]] || { echo "Missing source checkpoint summary: $SOURCE_SUMMARY" >&2; exit 1; }
[[ -f "$SOURCE_SKILLS" ]] || { echo "Missing source skill checkpoint: $SOURCE_SKILLS" >&2; exit 1; }
SOURCE_SUMMARY="$(cd "$(dirname "$SOURCE_SUMMARY")" && pwd)/$(basename "$SOURCE_SUMMARY")"
SOURCE_SKILLS="$(cd "$(dirname "$SOURCE_SKILLS")" && pwd)/$(basename "$SOURCE_SKILLS")"
actual_summary_sha="$(sha256sum "$SOURCE_SUMMARY" | awk '{print $1}')"
actual_skills_sha="$(sha256sum "$SOURCE_SKILLS" | awk '{print $1}')"
[[ "$actual_summary_sha" == "$SOURCE_SUMMARY_SHA256" ]] || {
    echo "Source summary SHA mismatch: expected $SOURCE_SUMMARY_SHA256, got $actual_summary_sha." >&2; exit 1; }
[[ "$actual_skills_sha" == "$SOURCE_SKILLS_SHA256" ]] || {
    echo "Source skill SHA mismatch: expected $SOURCE_SKILLS_SHA256, got $actual_skills_sha." >&2; exit 1; }

# Verify both the checkpoint boundary and the fact that no unexported trace
# bucket must be restored.  A source checkpoint at any other group is rejected.
python3 - "$SOURCE_SUMMARY" "$SOURCE_SKILLS" "$SOURCE_STEP" <<'PY'
import json
import sys

summary_path, skills_path, step_text = sys.argv[1:]
source_step = int(step_text)
with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)
with open(skills_path, encoding="utf-8") as f:
    skills = json.load(f)

expected_tasks = [
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
]
checks = {
    "status": "running",
    "checkpoint_reason": "group_interval_2",
    "completed_rollout_groups": 50,
    "current_epoch": 1,
    "next_episode_index_in_epoch": source_step + 1,
    "total_episodes": source_step,
    "group_size": 6,
    "batch_rollout_size": 72,
    "data_parallel_workers": 2,
    "checkpoint_every_groups": 2,
    "skill_bullets_enabled": True,
    "skill_tree_enabled": True,
    "skill_tree_evolve_enabled": True,
    "cloud_updates_enabled": True,
    "cloud_update_every": 0,
}
for key, expected in checks.items():
    actual = summary.get(key)
    if actual != expected:
        raise SystemExit(f"Source checkpoint mismatch for {key}: expected {expected!r}, got {actual!r}")
if summary.get("task_types") != expected_tasks:
    raise SystemExit("Source checkpoint task_types is not the published six-task protocol")
if summary.get("data_parallel_worker_batch_sizes") != [36, 36]:
    raise SystemExit("Source checkpoint does not have the original 2 x 36 data-parallel topology")
pool = summary.get("final_coskill_metrics") or {}
if pool.get("coskill/pool/pending_added") != 0 or pool.get("coskill/pool/pending_tokens") != 0:
    raise SystemExit("Source checkpoint has a non-empty pending trace pool; this launcher cannot safely isolate it")
if source_step not in (summary.get("cloud_update_steps") or []):
    raise SystemExit("Source checkpoint was not saved immediately after its group-50 cloud-update boundary")

playbooks = skills.get("playbooks") or skills.get("skill_trees") or skills.get("playbook_records") or {}
expected_playbooks = [
    "pick_and_place", "clean", "heat", "cool",
    "look_at_obj_in_light", "pick_two_obj_and_place",
]
missing = [task for task in expected_playbooks if not isinstance(playbooks.get(task), dict)]
if missing:
    raise SystemExit(f"Source skill checkpoint is missing the published skill trees: {missing}")
# This is an additional identity check on the known group-50 snapshot, not a
# identity check on the group-50 state; the pick2 tree is the only tree used
# during this targeted suffix.
pick2 = playbooks["pick_two_obj_and_place"]
if pick2.get("version") != 36 or pick2.get("level") != 3:
    raise SystemExit("Source skill checkpoint is not the published group-50 tree snapshot")
print("[pick2-resume] source checkpoint validated: group=50 step=3600, pool pending=0, pick2 tree=v36/L3")
PY

# The source used two data-parallel workers. The targeted 12-episode pick2
# slice necessarily uses 6+6 requests per worker rather than the source's
# mixed-task 36+36 requests; all model and generation settings remain pinned.
# CUDA Graph only changes vLLM execution scheduling; it does not change the
# prompts, seeds, sampling parameters, environment sequence, or update
# boundaries.  Use eager mode only as an explicit debugging fallback.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must explicitly name exactly two allocated GPUs (for example 2,3)." >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#gpu_ids[@]}" -ne 2 ]] || [[ -z "${gpu_ids[0]}" || -z "${gpu_ids[1]}" ]]; then
    echo "Expected exactly two visible GPUs to reproduce the original 2 x 36 topology; got '$CUDA_VISIBLE_DEVICES'." >&2
    exit 2
fi
DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-2}"
ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-$CUDA_VISIBLE_DEVICES}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
if [[ "$DATA_PARALLEL_WORKERS" != "2" || "$ROLLOUT_WORKER_GPUS" != "$CUDA_VISIBLE_DEVICES" || \
      ( "$VLLM_ENFORCE_EAGER" != "0" && "$VLLM_ENFORCE_EAGER" != "1" ) || \
      "$VLLM_MAX_NUM_SEQS" != "0" || "$TENSOR_PARALLEL_SIZE" != "1" ]]; then
    echo "Topology/runtime mismatch. Required: DP=2, worker GPUs=$CUDA_VISIBLE_DEVICES, TP=1, VLLM_ENFORCE_EAGER=0 or 1, max_num_seqs=0." >&2
    exit 2
fi

# Keep the published repair protocol on vLLM's native top-k/top-p sampler.
# CUDA Graph remains enabled through VLLM_ENFORCE_EAGER=0, independently of
# the sampler implementation.
export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ -d /GLOBALFS/hit_wxia_1 ]]; then
    DEFAULT_CACHE_ROOT="/GLOBALFS/hit_wxia_1/.cache"
else
    DEFAULT_CACHE_ROOT="$HOME/.cache"
fi
CACHE_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export ALFWORLD_DATA MODEL_PATH
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1

export SKILL_UPDATER_BACKEND="${SKILL_UPDATER_BACKEND:-deepseek}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
[[ "$SKILL_UPDATER_BACKEND" == "deepseek" && "$DEEPSEEK_MODEL" == "deepseek-v4-flash" ]] || {
    echo "Cloud backend/model must remain deepseek / deepseek-v4-flash for this reproduction." >&2; exit 2; }
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
    echo "DEEPSEEK_API_KEY is absent. Put it in $PROJECT_ROOT/.env or export it before launching." >&2; exit 2; }

if [[ "$TARGET_RESUME" == "0" ]]; then
    if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
        echo "Refusing to mix results: OUTPUT_DIR already contains files: $OUTPUT_DIR" >&2
        echo "Choose a new OUTPUT_DIR, or set TARGET_RESUME=1 only to resume this same pick2 suffix." >&2
        exit 1
    fi
else
    [[ -f "$OUTPUT_DIR/summary_partial.json" ]] || {
        echo "TARGET_RESUME=1 requires $OUTPUT_DIR/summary_partial.json" >&2; exit 1; }
    python3 - "$OUTPUT_DIR/summary_partial.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    state = json.load(f)
if state.get("task_types") != ["pick_two_obj_and_place"]:
    raise SystemExit("TARGET_RESUME output is not this pick_two_obj_and_place suffix")
if state.get("batch_rollout_size") != 12 or state.get("group_size") != 12:
    raise SystemExit("TARGET_RESUME output has incompatible rollout geometry")
PY
fi

if [[ "$PICK2_DRY_RUN" == "1" ]]; then
    echo "[pick2-resume] dry run passed"
    echo "  source summary: $SOURCE_SUMMARY"
    echo "  source skills:  $SOURCE_SKILLS"
    echo "  target output:  $OUTPUT_DIR"
    echo "  protocol: pick2 replacement suffix, 50 virtual groups x 12 = $TARGET_EPISODES episodes, DP=2 x 6"
    exit 0
fi

[[ -d "$ALFWORLD_DATA" ]] || { echo "ALFWORLD_DATA does not exist: $ALFWORLD_DATA" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
[[ -d "$ALFWORLD_DATA/json_2.1.1/train" ]] || {
    echo "ALFWORLD_DATA is not the expected ALFWorld json_2.1.1 dataset: $ALFWORLD_DATA" >&2; exit 1; }
[[ "$(basename "$MODEL_PATH")" == "Qwen3-4B-Thinking-2507" ]] || {
    echo "MODEL_PATH must point to Qwen3-4B-Thinking-2507, got: $MODEL_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

# This provenance file states that this is a targeted replacement ledger, not
# an independent six-task run. The merger will retain the original five other
# task slices while replacing pick2 results for virtual groups 51--100.
python3 - "$OUTPUT_DIR/resume_provenance.json" "$SOURCE_SUMMARY" "$SOURCE_SKILLS" <<'PY'
import hashlib, json, os, sys
out, summary, skills = sys.argv[1:]
payload = {
    "kind": "pick2_replacement_suffix_from_published_group50_state",
    "source": {
        "branch": "codex/alfworld-final-results-20260724",
        "commit": "65ee21b0d0cd9976e2097d469cc548f66dad286f",
        "checkpoint_summary": os.path.abspath(summary),
        "checkpoint_skills": os.path.abspath(skills),
        "checkpoint_group": 50,
        "checkpoint_episode": 3600,
        "source_pool_pending_added": 0,
        "source_pool_pending_tokens": 0,
        "summary_sha256": hashlib.sha256(open(summary, "rb").read()).hexdigest(),
        "skills_sha256": hashlib.sha256(open(skills, "rb").read()).hexdigest(),
    },
    "target": {
        "task_types": ["pick_two_obj_and_place"],
        "fresh_metric_ledger": True,
        "target_virtual_groups": 50,
        "target_episodes": 600,
        "episodes_per_virtual_group": 12,
        "epochs_cycle_allowance": 100,
        "batch_rollout_size": 12,
        "group_size": 12,
        "data_parallel_workers": 2,
        "worker_batch_sizes": [6, 6],
        "metric_rebase_required": True,
        "metric_rebase": {
            "replace_original_pick2_groups": [51, 100],
            "replace_original_pick2_episodes": 600,
            "expected_merged_six_task_episodes": 7200,
        },
    },
    "resume_semantics": (
        "Do not pass --resume against the source output. The source skill "
        "state is loaded as --skills_json into a fresh pick2 suffix; "
        "the source trace pool was empty immediately after the group-50 cloud update."
    ),
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-coskill-alfworld}"
WANDB_NAME="${WANDB_NAME:-alfworld_pick2_resume_group50_norl}"

echo "[pick2-resume] source: group=$RESUME_GROUP / step=$SOURCE_STEP; target: $TARGET_GROUPS virtual groups / $TARGET_EPISODES pick2 episodes"
echo "[pick2-resume] this replaces only the 12 pick2 trajectories in each original six-task group 51--100."
echo "[pick2-resume] GPUs=$CUDA_VISIBLE_DEVICES DP=$DATA_PARALLEL_WORKERS worker_batch=6+6 eager=$VLLM_ENFORCE_EAGER flashinfer=$VLLM_USE_FLASHINFER_SAMPLER output=$OUTPUT_DIR"

python3 -u -m examples.playbook_evolve.run_playbook_evolve \
    --outdir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --task_types "$PICK2_TASK_TYPE" \
    --num_games -1 \
    --group_size 12 \
    --split train \
    --max_steps 40 \
    --seed 0 \
    --epochs "$TARGET_EPOCHS" \
    --max_episodes "$TARGET_EPISODES" \
    --batch_rollout_size 12 \
    --data_parallel_workers 2 \
    --rollout_worker_gpus "$ROLLOUT_WORKER_GPUS" \
    --checkpoint_every_groups 2 \
    --resume "$TARGET_RESUME" \
    --wandb "$WANDB_ENABLED" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --history_length 8 \
    --max_model_len 10240 \
    --max_tokens 4096 \
    --think_budget 3500 \
    --temperature 1.0 \
    --gpu_mem_util 0.8 \
    --vllm_max_num_seqs 0 \
    --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
    --tensor_parallel_size 1 \
    --skills_json "$SOURCE_SKILLS" \
    --retrieval_mode template \
    --top_k 6 \
    --enable_hierarchy 1 \
    --stable_cycles_l1 3 \
    --stable_cycles_l2 5 \
    --success_l1 0.7 \
    --demote_threshold 0.3 \
    --min_calls 10 \
    --enable_coskill 1 \
    --enable_skill_tree 1 \
    --enable_skill_tree_evolve 1 \
    --enable_failure_analysis 1 \
    --max_new_skills 3 \
    --skill_tree_evolve_min_samples 6 \
    --capacity_watermark 50000 \
    --perf_watermark 0.6 \
    --min_samples 16 \
    --loop_threshold 3 \
    --trace_enable_loop_filter 1 \
    --trace_enable_obs_delta 1 \
    --trace_enable_prefix_tree 1 \
    --trace_enable_consensus_prefix 1 \
    --cloud_update_every 0 \
    --enable_cloud_updates 1 \
    --log_trajectories 0 \
    2>&1 | tee "$OUTPUT_DIR/driver.log"
