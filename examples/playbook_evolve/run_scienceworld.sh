#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -n "${SCIENCEWORLD_ENV_FILE:-}" ]]; then
    set -a
    source "$SCIENCEWORLD_ENV_FILE"
    set +a
fi
exec python -m examples.playbook_evolve.run_scienceworld "$@"
