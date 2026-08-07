#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash run.sh --benchmark {alfworld|webshop} [options] [-- Hydra overrides]

Options:
  --benchmark NAME       alfworld or webshop
  --rl {0|1}             frozen CoSkill (0, default) or Tree-RL (1)
  --tree-order MODE      root or leaf (default: root)
  --prepare              prepare assets before launch
  --print-command        print the resolved command and exit
  -h, --help             show this message
EOF
}

BENCHMARK=""
RL_MODE=0
TREE_ORDER=root
PREPARE=0
PRINT_COMMAND=0
EXTRA=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --benchmark)
            [[ "$#" -ge 2 ]] || { echo "--benchmark requires a value" >&2; exit 2; }
            BENCHMARK="$2"
            shift 2
            ;;
        --benchmark=*)
            BENCHMARK="${1#*=}"
            shift
            ;;
        --rl)
            [[ "$#" -ge 2 ]] || { echo "--rl requires 0 or 1" >&2; exit 2; }
            RL_MODE="$2"
            shift 2
            ;;
        --rl=*)
            RL_MODE="${1#*=}"
            shift
            ;;
        --tree-order)
            [[ "$#" -ge 2 ]] || { echo "--tree-order requires root or leaf" >&2; exit 2; }
            TREE_ORDER="$2"
            shift 2
            ;;
        --tree-order=*)
            TREE_ORDER="${1#*=}"
            shift
            ;;
        --prepare)
            PREPARE=1
            shift
            ;;
        --print-command)
            PRINT_COMMAND=1
            shift
            ;;
        --)
            shift
            EXTRA+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA+=("$1")
            shift
            ;;
    esac
done

case "$BENCHMARK" in
    alfworld|webshop) ;;
    *)
        echo "--benchmark must be alfworld or webshop" >&2
        exit 2
        ;;
esac
[[ "$RL_MODE" == "0" || "$RL_MODE" == "1" ]] || {
    echo "--rl must be 0 or 1" >&2
    exit 2
}
[[ "$TREE_ORDER" == "root" || "$TREE_ORDER" == "leaf" ]] || {
    echo "--tree-order must be root or leaf" >&2
    exit 2
}

if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

export PROJECT_ROOT="$ROOT"
export CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data/verl-agent}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$ROOT/data/alfworld}"
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$ROOT/agent_system/environments/env_package/webshop/webshop/data}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export TREE_RL_ORDER="$TREE_ORDER"

if [[ "$PREPARE" == "1" ]]; then
    bash "$ROOT/prepare.sh"
fi

if [[ "$RL_MODE" == "1" ]]; then
    COMMAND=(bash "$ROOT/backends/run_tree_rl.sh" "$BENCHMARK")
elif [[ "$BENCHMARK" == "alfworld" ]]; then
    COMMAND=(bash "$ROOT/backends/run_norl_alfworld.sh")
else
    COMMAND=(bash "$ROOT/backends/run_norl_webshop.sh")
fi
COMMAND+=("${EXTRA[@]}")

if [[ "$PRINT_COMMAND" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
