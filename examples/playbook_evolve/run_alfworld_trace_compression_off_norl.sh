#!/usr/bin/env bash
# One-arm trace-compression ablation. The existing standard CoSkill no-RL run is
# the compression-on control; this launcher changes only the four trace payload
# transformations and inherits every other setting from the production no-RL
# launcher (100 groups x 72 rollouts by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b_coskill_norl_trace_compression_off}"

exec bash "$SCRIPT_DIR/run_alfworld_playbook_evolve_norl.sh" \
  "$@" \
  --trace_enable_loop_filter 0 \
  --trace_enable_obs_delta 0 \
  --trace_enable_prefix_tree 0 \
  --trace_enable_consensus_prefix 0
