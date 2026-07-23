#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_RAW_TRACES="${EXTERNAL_RAW_TRACES:-__MISSING_EXTERNAL_RAW_TRACES__}"
exec bash "$SCRIPT_DIR/run_alfworld_fixed_trajectory_ablation_1xa800.sh" \
  --external_raw_traces "$EXTERNAL_RAW_TRACES" \
  --eval_games_per_type "${EVAL_GAMES_PER_TYPE:-5}" \
  "$@"
