#!/usr/bin/env bash
# Preferred name for the independent L0-L5 skill-level experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_alfworld_fixed_trajectory_ablation.sh" "$@"
