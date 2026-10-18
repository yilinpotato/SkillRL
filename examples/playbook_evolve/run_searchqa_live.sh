#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -n "${SEARCHQA_ENV_FILE:-}" ]]; then
    set -a
    source "$SEARCHQA_ENV_FILE"
    set +a
fi
exec python -m examples.playbook_evolve.run_searchqa_live "$@"
