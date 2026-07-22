#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-coskill-alfworld-fixed-ablation:skillrl-cu128}"
INCLUDE_ALFWORLD="${INCLUDE_ALFWORLD:-1}"
INCLUDE_MODEL="${INCLUDE_MODEL:-0}"
DEFAULT_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
ALFWORLD_SOURCE="${ALFWORLD_SOURCE:-${ALFWORLD_DATA:-$DEFAULT_CACHE_ROOT/alfworld}}"
MODEL_SOURCE="${MODEL_SOURCE:-${MODEL_PATH:-$DEFAULT_CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}}"
BUILD_PROXY="${BUILD_PROXY:-}"
USE_PACKED_ENV="${USE_PACKED_ENV:-1}"
PACKED_ENV_ARCHIVE="${PACKED_ENV_ARCHIVE:-$PROJECT_ROOT/.docker-assets/skillRL-conda.tar.gz}"
BASE_IMAGE="${BASE_IMAGE:-}"

PROXY_ARGS=()
if [[ -n "$BUILD_PROXY" ]]; then
  PROXY_ARGS+=(
    --build-arg "HTTP_PROXY=$BUILD_PROXY"
    --build-arg "HTTPS_PROXY=$BUILD_PROXY"
    --build-arg "ALL_PROXY=$BUILD_PROXY"
    --build-arg "http_proxy=$BUILD_PROXY"
    --build-arg "https_proxy=$BUILD_PROXY"
    --build-arg "all_proxy=$BUILD_PROXY"
  )
fi

EMPTY_CONTEXT="$(mktemp -d)"
cleanup() { rm -rf -- "$EMPTY_CONTEXT"; }
trap cleanup EXIT
ALFWORLD_CONTEXT="$EMPTY_CONTEXT"
MODEL_CONTEXT="$EMPTY_CONTEXT"

if [[ -z "$BASE_IMAGE" && "$INCLUDE_ALFWORLD" == "1" ]]; then
  if [[ -z "$ALFWORLD_SOURCE" || ! -d "$ALFWORLD_SOURCE/json_2.1.1" ]]; then
    echo "Set ALFWORLD_SOURCE (or ALFWORLD_DATA) to an ALFWorld directory containing json_2.1.1." >&2
    exit 2
  fi
  ALFWORLD_CONTEXT="$ALFWORLD_SOURCE"
else
  ALFWORLD_CONTEXT="$EMPTY_CONTEXT"
fi

if [[ -z "$BASE_IMAGE" && "$INCLUDE_MODEL" == "1" ]]; then
  if [[ -z "$MODEL_SOURCE" || ! -f "$MODEL_SOURCE/config.json" ]]; then
    echo "Set MODEL_SOURCE (or MODEL_PATH) to a complete ModelScope model snapshot." >&2
    exit 2
  fi
  MODEL_CONTEXT="$MODEL_SOURCE"
else
  MODEL_CONTEXT="$EMPTY_CONTEXT"
fi

SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
DOCKERFILE="$PROJECT_ROOT/docker/Dockerfile.alfworld-ablation"
ENV_CONTEXT_ARGS=()
if [[ -n "$BASE_IMAGE" ]]; then
  DOCKERFILE="$PROJECT_ROOT/docker/Dockerfile.alfworld-ablation-overlay"
  ENV_CONTEXT_ARGS=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
elif [[ "$USE_PACKED_ENV" == "1" ]]; then
  if [[ ! -f "$PACKED_ENV_ARCHIVE" ]]; then
    echo "Packed environment not found: $PACKED_ENV_ARCHIVE" >&2
    echo "Create it with: conda-pack -n skillRL --ignore-editable-packages -o $PACKED_ENV_ARCHIVE" >&2
    exit 2
  fi
  if [[ "$(basename "$PACKED_ENV_ARCHIVE")" != "skillRL-conda.tar.gz" ]]; then
    echo "PACKED_ENV_ARCHIVE must be named skillRL-conda.tar.gz." >&2
    exit 2
  fi
  if [[ ! -f "$(dirname "$PACKED_ENV_ARCHIVE")/faiss-overlay.tar.gz" ]]; then
    echo "faiss-overlay.tar.gz is missing beside PACKED_ENV_ARCHIVE." >&2
    echo "See docker/alfworld-ablation/README.md for the packaging command." >&2
    exit 2
  fi
  DOCKERFILE="$PROJECT_ROOT/docker/Dockerfile.alfworld-ablation-packed"
  ENV_CONTEXT_ARGS=(--build-context "conda_env=$(dirname "$PACKED_ENV_ARCHIVE")")
fi

docker buildx build --load \
  "${PROXY_ARGS[@]}" \
  "${ENV_CONTEXT_ARGS[@]}" \
  --build-context "alfworld=$ALFWORLD_CONTEXT" \
  --build-context "model=$MODEL_CONTEXT" \
  --build-arg "SOURCE_COMMIT=$SOURCE_COMMIT" \
  -f "$DOCKERFILE" \
  -t "$IMAGE_NAME" "$PROJECT_ROOT"

echo "Built $IMAGE_NAME (ALFWorld=$INCLUDE_ALFWORLD, model=$INCLUDE_MODEL, base=${BASE_IMAGE:-none})"
if [[ -n "${EXPORT_TAR:-}" ]]; then
  mkdir -p "$(dirname "$EXPORT_TAR")"
  if [[ "$EXPORT_TAR" == *.gz ]]; then
    docker save "$IMAGE_NAME" | gzip -1 > "$EXPORT_TAR"
  else
    docker save --output "$EXPORT_TAR" "$IMAGE_NAME"
  fi
  echo "Exported $IMAGE_NAME to $EXPORT_TAR"
fi
