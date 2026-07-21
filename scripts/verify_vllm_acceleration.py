#!/usr/bin/env python3
"""Read-only vLLM acceleration probe for a single visible GPU.

Run it as a real file (not ``python -``): vLLM V1 starts EngineCore with the
``spawn`` multiprocessing method, which cannot re-import a stdin program.
The probe uses one deterministic request and prints the capabilities needed to
audit CUDA Graph and optional FlashInfer sampler activation in EngineCore logs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.45)
    parser.add_argument(
        "--enforce-eager",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 verifies CUDA Graph; 1 is a compatibility control.",
    )
    args = parser.parse_args()

    import torch
    import vllm

    flashinfer_available = importlib.util.find_spec("flashinfer") is not None
    print(
        "[acceleration-probe] "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"vllm={vllm.__version__} flashinfer_available={flashinfer_available} "
        f"VLLM_USE_FLASHINFER_SAMPLER={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')} "
        f"VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')} "
        f"enforce_eager={bool(args.enforce_eager)}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; expose exactly one GPU via CUDA_VISIBLE_DEVICES.")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=bool(args.enforce_eager),
        seed=17,
    )
    output = llm.generate(
        ["Reply with exactly: ok."],
        SamplingParams(temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )[0].outputs[0].text
    print(f"[acceleration-probe] generation_ok={output!r}", flush=True)


if __name__ == "__main__":
    main()
