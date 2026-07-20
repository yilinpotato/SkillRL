#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSET_DIR="$SCRIPT_DIR/assets"
IMAGE_TAG="${IMAGE_TAG:-coskill:skillrl-cu128-data}"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04}"
UBUNTU_APT_MIRROR="${UBUNTU_APT_MIRROR:-https://mirrors.aliyun.com/ubuntu}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-skillRL}"
ALFWORLD_SOURCE="${ALFWORLD_SOURCE:-$HOME/.cache/alfworld}"
WEBSHOP_DATA_SOURCE="${WEBSHOP_DATA_SOURCE:-$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop/data}"
WEBSHOP_INDEX_SOURCE="${WEBSHOP_INDEX_SOURCE:-$(dirname "$PROJECT_ROOT")/Skill0/agent_system/environments/env_package/webshop/webshop/search_engine/indexes}"
PREPARED_DATA_SOURCE="${PREPARED_DATA_SOURCE:-$PROJECT_ROOT/skillrl_data/verl-agent}"
INCLUDE_MODEL="${INCLUDE_MODEL:-0}"
MODEL_SOURCE="${MODEL_SOURCE:-$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
DOCKER_COMMAND=(docker)
if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    DOCKER_COMMAND=(sudo docker)
fi

mkdir -p "$ASSET_DIR"

for path in \
    "$ALFWORLD_SOURCE/json_2.1.1" \
    "$ALFWORLD_SOURCE/logic" \
    "$WEBSHOP_DATA_SOURCE/items_shuffle_1000.json" \
    "$WEBSHOP_DATA_SOURCE/items_ins_v2_1000.json" \
    "$WEBSHOP_DATA_SOURCE/items_human_ins.json" \
    "$WEBSHOP_INDEX_SOURCE" \
    "$PREPARED_DATA_SOURCE/text/train.parquet" \
    "$PREPARED_DATA_SOURCE/text/test.parquet"
do
    if [[ ! -e "$path" ]]; then
        echo "Required Docker asset is missing: $path" >&2
        exit 1
    fi
done

# The launcher and preflight both rely on the fixed 12/32 prepared parquet
# split.  Validate it while the source environment is still available rather
# than allowing a misleading missing-asset failure on the target server.
conda run -n "$CONDA_ENV_NAME" python - "$PREPARED_DATA_SOURCE" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq

root = Path(sys.argv[1])
for name, expected in (("train.parquet", 12), ("test.parquet", 32)):
    path = root / "text" / name
    actual = pq.ParquetFile(path).metadata.num_rows
    if actual != expected:
        raise SystemExit(f"{path} has {actual} rows; expected {expected}")
PY

echo "[1/5] Packing the exact Conda environment: $CONDA_ENV_NAME"
conda pack -n "$CONDA_ENV_NAME" --ignore-editable-packages \
    --force -o "$ASSET_DIR/skillRL.tar.gz"

# The environment has a pip faiss-cpu upgrade over an older conda Faiss
# package.  Store the runtime files separately so Docker can restore the
# currently importable package after conda-unpack.
SITE_PACKAGES="$(conda run -n "$CONDA_ENV_NAME" python -c 'import site; print(site.getsitepackages()[0])')"
FAISS_ENTRIES=(faiss faiss-1.9.0-py3.10.egg-info faiss_cpu-1.13.2.dist-info faiss_cpu.libs)
for entry in "${FAISS_ENTRIES[@]}"; do
    if [[ ! -e "$SITE_PACKAGES/$entry" ]]; then
        echo "Required Faiss runtime entry is missing: $SITE_PACKAGES/$entry" >&2
        exit 1
    fi
done
tar -czf "$ASSET_DIR/faiss-runtime-overlay.tar.gz" \
    -C "$SITE_PACKAGES" "${FAISS_ENTRIES[@]}"

echo "[2/5] Packing ALFWorld text-game data (json_2.1.1 + logic)"
tar -czf "$ASSET_DIR/alfworld-data.tar.gz" \
    -C "$(dirname "$ALFWORLD_SOURCE")" \
    "$(basename "$ALFWORLD_SOURCE")/json_2.1.1" \
    "$(basename "$ALFWORLD_SOURCE")/logic"

echo "[3/5] Packing fixed prepared GRPO parquet data (train=12, test=32)"
tar -czf "$ASSET_DIR/prepared-verl-data.tar.gz" \
    -C "$(dirname "$PREPARED_DATA_SOURCE")" \
    "$(basename "$PREPARED_DATA_SOURCE")/text/train.parquet" \
    "$(basename "$PREPARED_DATA_SOURCE")/text/test.parquet"

echo "[4/5] Packing the 1,000-product WebShop split and matching Lucene index"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
mkdir -p "$STAGE_DIR/data" "$STAGE_DIR/search_engine/indexes"
cp "$WEBSHOP_DATA_SOURCE/items_shuffle_1000.json" "$STAGE_DIR/data/"
cp "$WEBSHOP_DATA_SOURCE/items_ins_v2_1000.json" "$STAGE_DIR/data/"
cp "$WEBSHOP_DATA_SOURCE/items_human_ins.json" "$STAGE_DIR/data/"
cp -a "$WEBSHOP_INDEX_SOURCE/." "$STAGE_DIR/search_engine/indexes/"
tar -czf "$ASSET_DIR/webshop-small-data.tar.gz" -C "$STAGE_DIR" data search_engine

if [[ "$INCLUDE_MODEL" == "1" ]]; then
    if [[ ! -f "$MODEL_SOURCE/config.json" ]]; then
        echo "MODEL_SOURCE is not a complete model directory: $MODEL_SOURCE" >&2
        exit 1
    fi
    echo "Packing Qwen weights into the optional offline layer"
    tar -czf "$ASSET_DIR/model-optional.tar.gz" \
        -C "$(dirname "$MODEL_SOURCE")" "$(basename "$MODEL_SOURCE")"
else
    EMPTY_MODEL_DIR="$(mktemp -d)"
    tar -czf "$ASSET_DIR/model-optional.tar.gz" -C "$EMPTY_MODEL_DIR" .
    rm -rf "$EMPTY_MODEL_DIR"
fi

echo "[5/5] Building $IMAGE_TAG"
echo "CUDA base image: $CUDA_BASE_IMAGE"
echo "Ubuntu apt mirror: $UBUNTU_APT_MIRROR"
echo "Docker build network: $DOCKER_BUILD_NETWORK"
DOCKER_BUILDKIT=1 "${DOCKER_COMMAND[@]}" build --progress=plain \
    --network "$DOCKER_BUILD_NETWORK" \
    --build-arg "CUDA_BASE_IMAGE=$CUDA_BASE_IMAGE" \
    --build-arg "UBUNTU_APT_MIRROR=$UBUNTU_APT_MIRROR" \
    --build-context "coskill_source=$PROJECT_ROOT" \
    --build-context "coskill_assets=$ASSET_DIR" \
    -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_TAG" "$SCRIPT_DIR"
echo "Built $IMAGE_TAG"
