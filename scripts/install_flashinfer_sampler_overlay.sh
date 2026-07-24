#!/usr/bin/env bash
# Install FlashInfer outside the shared Conda environment.
#
# vLLM 0.11 only needs FlashInfer's sampling module.  Installing it into a
# target overlay keeps the existing torch/vLLM lock untouched and can be
# mounted into Docker containers.  The overlay is about 0.5 GB because current
# FlashInfer packages CUDA helper libraries with the sampler.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TARGET="${COSKILL_FLASHINFER_OVERLAY:-$PROJECT_ROOT/.cache/flashinfer-cu128}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.15.post1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "PYTHON_BIN is unavailable: $PYTHON_BIN" >&2
    exit 2
fi

CUDA_VERSION="$($PYTHON_BIN -c 'import torch; print(torch.version.cuda or "")')"
if [[ "$CUDA_VERSION" != 12.8* && "${FLASHINFER_ALLOW_UNTESTED_CUDA:-0}" != "1" ]]; then
    echo "This tested overlay targets torch CUDA 12.8, found '$CUDA_VERSION'." >&2
    echo "Set FLASHINFER_ALLOW_UNTESTED_CUDA=1 only after a separate smoke test." >&2
    exit 2
fi

mkdir -p "$TARGET"
echo "Installing FlashInfer sampler overlay in: $TARGET"
"$PYTHON_BIN" -m pip install --no-cache-dir --upgrade --target "$TARGET" \
    "flashinfer-python==$FLASHINFER_VERSION" \
    'apache-tvm-ffi==0.1.12' \
    'cuda-python==13.3.1' \
    'cuda-tile==1.5.0' \
    'nccl4py==0.3.1' \
    'nvidia-cudnn-frontend==1.26.0' \
    'nvidia-cutlass-dsl==4.6.1' \
    'nvidia-ml-py==13.610.43' \
    'tabulate==0.10.0'

PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import flashinfer
import flashinfer.sampling
print(f"FlashInfer sampler import passed: {flashinfer.__version__}")
PY

cat <<EOF
Installed without changing the shared Python environment.
For a new run, export:
  COSKILL_FLASHINFER_OVERLAY=$TARGET
  COSKILL_ENABLE_FLASHINFER_SAMPLER=1
Then verify one engine log contains:
  Using FlashInfer for top-p & top-k sampling.
EOF
