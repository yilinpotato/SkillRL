#!/usr/bin/env bash
# One A800 V3 launcher.  Set CUDA_VISIBLE_DEVICES to the scheduler allocation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
[[ "$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF{n++} END{print n+0}')" == "1" ]] || {
  echo "Expected exactly one GPU, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 2;
}
export DATA_PARALLEL_WORKERS=1 COSKILL_FORCE_ACCELERATOR=1
exec bash "$SCRIPT_DIR/run_alfworld_skill_tree_growth_v3.sh" "$@"
