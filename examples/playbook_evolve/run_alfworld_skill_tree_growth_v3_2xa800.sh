#!/usr/bin/env bash
# Two A800 V3 launcher.  DP=2 only accelerates each isolated arm; it never shares skills across arms.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "Set CUDA_VISIBLE_DEVICES to the two scheduler-assigned GPUs (for example 2,3)." >&2; exit 2;
}
[[ "$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF{n++} END{print n+0}')" == "2" ]] || {
  echo "Expected exactly two GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 2;
}
export DATA_PARALLEL_WORKERS=2 COSKILL_FORCE_ACCELERATOR=1
exec bash "$SCRIPT_DIR/run_alfworld_skill_tree_growth_v3.sh" "$@"
