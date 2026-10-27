#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARISON_ROOT="${COMPARISON_ROOT:-$ROOT/tools/experiment_control}"
exec python "$COMPARISON_ROOT/comparison/launch.py" --repo "$ROOT" "$@"
