#!/usr/bin/env bash
# Strict local one-GPU container test for both independent ALFWorld ablations.
# It uses the real model, ALFWorld environment, vLLM, cloud backend, artifact
# generation, evaluation, resume-safe summaries, and metric writers, but keeps
# the smoke workload bounded. Formal launch defaults remain unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-coskill-alfworld-ablation:l0-l5-trace-off-20260722}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}}"
TEST_ID="${TEST_ID:-$(date +%Y%m%d_%H%M%S)}"
HOST_OUTPUT_ROOT="${HOST_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/docker_alfworld_ablation_single_gpu/$TEST_ID}"
HOST_CACHE_ROOT="${HOST_CACHE_ROOT:-$PROJECT_ROOT/.docker-cache}"

if [[ ! -f "$HOST_MODEL_PATH/config.json" ]]; then
  echo "Complete model not found: HOST_MODEL_PATH=$HOST_MODEL_PATH" >&2
  exit 2
fi
GPU1_NAME="$(nvidia-smi --id=1 --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
[[ -n "$GPU1_NAME" ]] || { echo "Physical GPU 1 is unavailable." >&2; exit 2; }
active="$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)"
[[ -z "$active" ]] || {
  echo "Strict local test requires idle physical GPU 1; occupied by PID(s): $active" >&2
  exit 2
}

wait_for_gpu1_cleanup() {
  local deadline=$((SECONDS + 180))
  local pids
  while :; do
    pids="$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)"
    [[ -z "$pids" ]] && return 0
    (( SECONDS < deadline )) || {
      echo "GPU 1 did not release after the previous container; PID(s): $pids" >&2
      return 2
    }
    sleep 5
  done
}

mkdir -p "$HOST_OUTPUT_ROOT" "$HOST_CACHE_ROOT"
export IMAGE_NAME HOST_MODEL_PATH HOST_OUTPUT_ROOT HOST_CACHE_ROOT CUDA_VISIBLE_DEVICES=1

echo "[single-gpu-test] image=$IMAGE_NAME gpu1=$GPU1_NAME outputs=$HOST_OUTPUT_ROOT"

# Real cloud reachability before loading the model.
bash "$SCRIPT_DIR/run_alfworld_ablation_container.sh" cloud-check --probe

# Complete L0-L5 pipeline at bounded smoke scale. Every phase is real; only
# rollout multiplicity, generation length, max steps, and repair budget shrink.
export AB_ROOT=/workspace/outputs/skill_level_smoke
bash "$SCRIPT_DIR/run_alfworld_ablation_container.sh" skill-level \
  --phase all \
  --rollouts_per_type 1 \
  --eval_groups_per_level 1 \
  --tree_generation_attempts 2 \
  --max_steps 1 \
  --gpu_mem_util 0.72 \
  --vllm_enforce_eager 1 \
  --driver_arg=--max_model_len \
  --driver_arg=4096 \
  --driver_arg=--max_tokens \
  --driver_arg=64 \
  --driver_arg=--think_budget \
  --driver_arg=32

# Complete compression-off no-RL entry at one-episode smoke scale.
unset AB_ROOT
wait_for_gpu1_cleanup
export TRACE_OUTPUT_DIR=/workspace/outputs/trace_compression_off_smoke
export MAX_EPISODES=1 BATCH_ROLLOUT_SIZE=1 VLLM_MAX_NUM_SEQS=1 VLLM_ENFORCE_EAGER=1
bash "$SCRIPT_DIR/run_alfworld_ablation_container.sh" trace-compression-off \
  --max_steps 1 \
  --max_model_len 4096 \
  --max_tokens 64 \
  --think_budget 32 \
  --gpu_mem_util 0.72

python - "$HOST_OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
skill = root / "skill_level_smoke"
trace = root / "trace_compression_off_smoke"
arm_rows = [json.loads(x) for x in (skill / "metrics.jsonl").read_text().splitlines() if x]
task_rows = [json.loads(x) for x in (skill / "metrics_by_task.jsonl").read_text().splitlines() if x]
assert [x["arm"] for x in arm_rows] == [f"skill_level_l{i}" for i in range(6)]
assert len(task_rows) == 36
group = json.loads((trace / "group_metrics.jsonl").read_text().splitlines()[-1])["metrics"]
assert group["experiment/trace_compression/condition"] == "all_off"
for name in ("enable_loop_filter", "enable_obs_delta", "enable_prefix_tree", "enable_consensus_prefix"):
    assert group[f"experiment/trace_compression/{name}"] == 0
summary = json.loads((trace / "summary.json").read_text())
assert summary["total_episodes"] == 1
print(json.dumps({
    "status": "ok", "skill_arm_rows": len(arm_rows),
    "skill_task_rows": len(task_rows), "trace_episodes": summary["total_episodes"],
    "trace_condition": group["experiment/trace_compression/condition"],
}, ensure_ascii=False))
PY

echo "[single-gpu-test] PASS outputs=$HOST_OUTPUT_ROOT"
