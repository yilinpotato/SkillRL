#!/usr/bin/env bash
# Build a thin main Tree-RL image on top of the pushed ALFWorld ablation base.
# It stages the WebShop archive and prepared GRPO parquet contract; the large
# skillRL and ALFWorld layers are inherited from BASE_IMAGE and deduplicated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSET_DIR="$SCRIPT_DIR/assets"
IMAGE_TAG="${IMAGE_TAG:-coskill:skillrl-cu128-data-shared}"
BASE_IMAGE="${BASE_IMAGE:-coskill-alfworld-fixed-ablation:skillrl-cu128}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-skillRL}"
UBUNTU_APT_MIRROR="${UBUNTU_APT_MIRROR:-https://mirrors.aliyun.com/ubuntu}"
PREPARED_DATA_SOURCE="${PREPARED_DATA_SOURCE:-$PROJECT_ROOT/skillrl_data/verl-agent}"
DOCKER_COMMAND=(docker)
if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    DOCKER_COMMAND=(sudo docker)
fi

WEB_ASSET="$ASSET_DIR/webshop-small-data.tar.gz"
if [[ ! -s "$WEB_ASSET" ]]; then
    echo "Missing WebShop build asset: $WEB_ASSET" >&2
    echo "Run docker/coskill/build_image.sh once, or create this archive first." >&2
    exit 1
fi
for path in "$PREPARED_DATA_SOURCE/text/train.parquet" "$PREPARED_DATA_SOURCE/text/test.parquet"; do
    if [[ ! -s "$path" ]]; then
        echo "Required prepared GRPO parquet is missing: $path" >&2
        exit 1
    fi
done

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
cp "$WEB_ASSET" "$STAGE_DIR/webshop-small-data.tar.gz"

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
tar -czf "$STAGE_DIR/prepared-verl-data.tar.gz" \
    -C "$(dirname "$PREPARED_DATA_SOURCE")" \
    "$(basename "$PREPARED_DATA_SOURCE")/text/train.parquet" \
    "$(basename "$PREPARED_DATA_SOURCE")/text/test.parquet"

# Match the current standalone main image's Faiss files exactly.  conda-pack
# can restore the old conda Faiss package after its pip upgrade in a different
# order, so this small overlay is intentional and has no network dependency.
SITE_PACKAGES="$(conda run -n "$CONDA_ENV_NAME" python -c 'import site; print(site.getsitepackages()[0])')"
FAISS_ENTRIES=(faiss faiss-1.9.0-py3.10.egg-info faiss_cpu-1.13.2.dist-info faiss_cpu.libs)
for entry in "${FAISS_ENTRIES[@]}"; do
    if [[ ! -e "$SITE_PACKAGES/$entry" ]]; then
        echo "Required Faiss runtime entry is missing: $SITE_PACKAGES/$entry" >&2
        exit 1
    fi
done
tar -czf "$STAGE_DIR/faiss-runtime-overlay.tar.gz" \
    -C "$SITE_PACKAGES" "${FAISS_ENTRIES[@]}"

echo "Building thin main image: $IMAGE_TAG"
echo "Shared ablation base: $BASE_IMAGE"
echo "Ubuntu APT mirror: $UBUNTU_APT_MIRROR"
DOCKER_BUILDKIT=1 "${DOCKER_COMMAND[@]}" build --progress=plain \
    --network "$DOCKER_BUILD_NETWORK" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "UBUNTU_APT_MIRROR=$UBUNTU_APT_MIRROR" \
    --build-context "coskill_source=$PROJECT_ROOT" \
    --build-context "webshop_assets=$STAGE_DIR" \
    --build-context "runtime_assets=$STAGE_DIR" \
    -f "$SCRIPT_DIR/Dockerfile.from_ablation" -t "$IMAGE_TAG" "$SCRIPT_DIR"
echo "Built $IMAGE_TAG"
