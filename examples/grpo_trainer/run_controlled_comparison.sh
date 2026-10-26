#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${COMPARISON_ROOT:?Set the train-alfworld-data checkout containing comparison/launch.py}"
exec python "$COMPARISON_ROOT/comparison/launch.py" --repo "$ROOT" --method skillrl "$@"
