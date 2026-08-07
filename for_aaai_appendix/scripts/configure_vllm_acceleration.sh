#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this helper from a CoSkill launcher; do not execute it directly." >&2
    exit 2
fi

COSKILL_ENABLE_FLASHINFER_SAMPLER="${COSKILL_ENABLE_FLASHINFER_SAMPLER:-0}"
if [[ "$COSKILL_ENABLE_FLASHINFER_SAMPLER" != "0" && "$COSKILL_ENABLE_FLASHINFER_SAMPLER" != "1" ]]; then
    echo "COSKILL_ENABLE_FLASHINFER_SAMPLER must be 0 or 1." >&2
    return 2
fi

if [[ "$COSKILL_ENABLE_FLASHINFER_SAMPLER" == "0" ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "vLLM sampler: PyTorch native (FlashInfer opt-in is disabled)"
    return 0
fi

if [[ -n "${COSKILL_FLASHINFER_OVERLAY:-}" ]]; then
    if [[ ! -d "$COSKILL_FLASHINFER_OVERLAY" ]]; then
        echo "COSKILL_FLASHINFER_OVERLAY does not exist: $COSKILL_FLASHINFER_OVERLAY" >&2
        return 2
    fi
    export PYTHONPATH="$COSKILL_FLASHINFER_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! python3 -c 'import flashinfer, flashinfer.sampling; print(flashinfer.__version__)' >/dev/null 2>&1; then
    cat >&2 <<'EOF'
FlashInfer sampler was requested but flashinfer.sampling cannot be imported.
Run scripts/install_flashinfer_sampler_overlay.sh once, then set both:
  COSKILL_ENABLE_FLASHINFER_SAMPLER=1
  COSKILL_FLASHINFER_OVERLAY=/path/to/flashinfer-cu128
EOF
    return 2
fi

export VLLM_USE_FLASHINFER_SAMPLER=1
echo "vLLM sampler: FlashInfer (explicit opt-in; verify EngineCore log contains 'Using FlashInfer')"
