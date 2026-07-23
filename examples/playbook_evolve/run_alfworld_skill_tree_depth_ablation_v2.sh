#!/usr/bin/env bash
# V2: derive L1-L5 from a single canonical L5 tree built from imported traces.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the cloud preflight in the delegated launcher first.  A sentinel then
# makes Python issue the precise missing-corpus error without falling back to
# the old self-bootstrap protocol.
EXTERNAL_RAW_TRACES="${EXTERNAL_RAW_TRACES:-__MISSING_EXTERNAL_RAW_TRACES__}"
EVAL_GAMES_PER_TYPE="${EVAL_GAMES_PER_TYPE:-5}"
exec bash "$SCRIPT_DIR/run_alfworld_fixed_trajectory_ablation.sh" \
  --external_raw_traces "$EXTERNAL_RAW_TRACES" \
  --eval_games_per_type "$EVAL_GAMES_PER_TYPE" \
  "$@"
