#!/usr/bin/env bash
# Build a thin main Tree-RL image on top of the pushed ALFWorld ablation base.
# It stages only the WebShop archive: the large skillRL and ALFWorld layers are
# inherited from BASE_IMAGE and therefore deduplicated by the registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSET_DIR="$SCRIPT_DIR/assets"
IMAGE_TAG="${IMAGE_TAG:-coskill:skillrl-cu128-data-shared}"
BASE_IMAGE="${BASE_IMAGE:-coskill-alfworld-fixed-ablation:skillrl-cu128}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
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

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
cp "$WEB_ASSET" "$STAGE_DIR/webshop-small-data.tar.gz"

echo "Building thin main image: $IMAGE_TAG"
echo "Shared ablation base: $BASE_IMAGE"
DOCKER_BUILDKIT=1 "${DOCKER_COMMAND[@]}" build --progress=plain \
    --network "$DOCKER_BUILD_NETWORK" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-context "coskill_source=$PROJECT_ROOT" \
    --build-context "webshop_assets=$STAGE_DIR" \
    -f "$SCRIPT_DIR/Dockerfile.from_ablation" -t "$IMAGE_TAG" "$SCRIPT_DIR"
echo "Built $IMAGE_TAG"
