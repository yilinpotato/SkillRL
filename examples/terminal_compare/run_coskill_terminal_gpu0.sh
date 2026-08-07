#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM=coskill
export GPU="${GPU:-0}"
exec "$SCRIPT_DIR/run_alfworld_skill_compare_3090.sh" "$@"
