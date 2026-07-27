#!/usr/bin/env bash
# Run disjoint V4 validation arms as two independent one-A800 evaluators.
# This is intentionally arm-parallel, not DP=2 within one rollout group.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="$SCRIPT_DIR/run_alfworld_skill_tree_depth_v4_extend_validation.sh"

: "${SOURCE_V4_ROOT:?Set SOURCE_V4_ROOT to the completed frozen V4 run.}"
: "${AB_ROOT:?Set AB_ROOT to the prepared/current validation-extension root.}"

GPU_LIST="${V4_PARALLEL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ "${#GPUS[@]}" != 2 || -z "${GPUS[0]}" || -z "${GPUS[1]}" ]]; then
  echo "Set V4_PARALLEL_GPUS (or CUDA_VISIBLE_DEVICES) to exactly two GPUs, e.g. 0,1." >&2
  exit 2
fi
if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "The two independent evaluators must use different GPUs." >&2
  exit 2
fi

GPU0_ARMS="${V4_GPU0_ARMS:-skill_level_l0,skill_level_l2,skill_level_l4}"
GPU1_ARMS="${V4_GPU1_ARMS:-skill_level_l1,skill_level_l3,skill_level_l5}"
LOG_DIR="$AB_ROOT/parallel_arm_logs"
mkdir -p "$LOG_DIR"

declare -A SEEN_ARMS=()
IFS=',' read -r -a ASSIGNED_ARMS <<<"$GPU0_ARMS,$GPU1_ARMS"
for arm in "${ASSIGNED_ARMS[@]}"; do
  case "$arm" in
    skill_level_l[0-5]) ;;
    *)
      echo "Invalid arm assignment: $arm" >&2
      exit 2
      ;;
  esac
  if [[ -n "${SEEN_ARMS[$arm]:-}" ]]; then
    echo "Arm assigned more than once: $arm" >&2
    exit 2
  fi
  SEEN_ARMS["$arm"]=1
done
if [[ "${#SEEN_ARMS[@]}" != 6 ]]; then
  echo "GPU arm assignments must cover skill_level_l0 through skill_level_l5 exactly once." >&2
  exit 2
fi

echo "[v4-arm-parallel] root=$AB_ROOT"
echo "[v4-arm-parallel] gpu=${GPUS[0]} arms=$GPU0_ARMS"
echo "[v4-arm-parallel] gpu=${GPUS[1]} arms=$GPU1_ARMS"
echo "[v4-arm-parallel] preparing/validating shared immutable inputs once"

V4_EXTENSION_CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
V4_EXTENSION_SOURCE_ROOT="$SOURCE_V4_ROOT" \
V4_EXTENSION_ROOT="$AB_ROOT" \
DATA_PARALLEL_WORKERS=1 \
bash "$BASE_LAUNCHER" --phase prepare

PID0=""
PID1=""
terminate_workers() {
  [[ -z "$PID0" ]] || kill "$PID0" 2>/dev/null || true
  [[ -z "$PID1" ]] || kill "$PID1" 2>/dev/null || true
}
trap terminate_workers INT TERM

run_worker() {
  local gpu="$1"
  local arms="$2"
  local log="$3"
  V4_EXTENSION_CUDA_VISIBLE_DEVICES="$gpu" \
  V4_EXTENSION_SOURCE_ROOT="$SOURCE_V4_ROOT" \
  V4_EXTENSION_ROOT="$AB_ROOT" \
  DATA_PARALLEL_WORKERS=1 \
  V4_RESUME=1 \
  bash "$BASE_LAUNCHER" \
    --phase evaluate \
    --arms "$arms" \
    --resume 1 \
    >"$log" 2>&1
}

run_worker "${GPUS[0]}" "$GPU0_ARMS" "$LOG_DIR/gpu_${GPUS[0]}.log" &
PID0=$!
run_worker "${GPUS[1]}" "$GPU1_ARMS" "$LOG_DIR/gpu_${GPUS[1]}.log" &
PID1=$!
printf '%s\n' "$PID0" >"$LOG_DIR/gpu_${GPUS[0]}.pid"
printf '%s\n' "$PID1" >"$LOG_DIR/gpu_${GPUS[1]}.pid"
echo "[v4-arm-parallel] workers started: $PID0 $PID1"

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e
if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "[v4-arm-parallel] worker failure: gpu0_status=$STATUS0 gpu1_status=$STATUS1" >&2
  echo "[v4-arm-parallel] inspect $LOG_DIR; completed arm checkpoints are preserved." >&2
  exit 1
fi

echo "[v4-arm-parallel] all arm evaluators completed; writing one global summary"
V4_EXTENSION_CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
V4_EXTENSION_SOURCE_ROOT="$SOURCE_V4_ROOT" \
V4_EXTENSION_ROOT="$AB_ROOT" \
DATA_PARALLEL_WORKERS=1 \
bash "$BASE_LAUNCHER" --phase summary
echo "[v4-arm-parallel] done: $AB_ROOT/summary.json"
