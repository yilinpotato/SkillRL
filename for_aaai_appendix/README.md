# CoSkill Artifact

This directory contains the code and documentation required to run the
ALFWorld and WebShop experiments reported for CoSkill.

## Contents

- `run.sh`: common entry point for frozen inference and Tree-RL.
- `prepare.sh`: model and benchmark asset preparation.
- `backends/`: benchmark-specific launch implementations.
- `agent_system/`, `verl/`: CoSkill and GRPO runtime code.
- `memory_data/`: initial skill-library files.
- `prompts/`: prompt appendix and prompt provenance.
- [`appendix/SKILL_TREE_EXAMPLES.md`](appendix/SKILL_TREE_EXAMPLES.md):
  representative learned skill trees in a human-readable format.
- `tools/preflight.py`: non-GPU asset and configuration checks.

Generated outputs, model weights, benchmark downloads, API credentials, and
checkpoints are excluded from version control.

The released training path is Qwen3 text-only with FSDP, Ray, GRPO, and vLLM.
Unused Megatron, alternative model-family, multimodal, GiGPO, cold-start
generation, and external tracking backends are not included. Metrics are
written to the console and `group_metrics.jsonl`; field semantics are
defined in `METRICS.md`.

## Installation

Python 3.10 and a CUDA environment compatible with vLLM are recommended.

```bash
conda activate skillRL
python -m pip install -e ".[vllm]"
python -m pip install -r requirements-artifact.txt
```

Java 11 is required to build the WebShop Lucene index.

## Asset preparation

The following command downloads Qwen3-4B-Thinking-2507, ALFWorld text-game
data, and the 1,000-product WebShop split, then builds the WebShop index:

```bash
bash prepare.sh
```

Each asset can be prepared separately:

```bash
bash prepare.sh --assets model
bash prepare.sh --assets alfworld
bash prepare.sh --assets webshop
bash prepare.sh --assets parquet
```

All default paths are contained in this directory. `CACHE_ROOT`, `MODEL_PATH`,
`ALFWORLD_DATA`, `WEBSHOP_DATA_DIR`, `DATA_ROOT`, and `OUTPUT_ROOT` may be
overridden for shared storage.

## Running

Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`.

```bash
# Frozen CoSkill
bash run.sh --benchmark alfworld --rl 0
bash run.sh --benchmark webshop --rl 0

# Progressive skill-tree internalization
bash run.sh --benchmark alfworld --rl 1 --tree-order root
bash run.sh --benchmark webshop --rl 1 --tree-order leaf
```

The common defaults are 12 training tasks, group size 6, 72 rollouts per
update, 32 validation tasks, history length 8, response limit 4,096 tokens,
40 ALFWorld steps, and 15 WebShop steps. Environment variables and Hydra
overrides remain available for controlled ablations.

Inspect a resolved command without starting a run:

```bash
bash run.sh --benchmark alfworld --rl 1 --tree-order leaf --print-command
```

Validate downloaded assets. The repository does not contain benchmark data or
generated parquet files:

```bash
python tools/preflight.py --benchmark all --allow-missing-parquet
```

Before publishing the anonymous repository, run:

```bash
python tools/check_metrics_schema.py
python tools/check_release.py
```

The release check rejects embedded credentials, machine-specific absolute
paths, and authoring-tool traces in both file names and text files.

Prompt text and source locations are documented in `prompts/PROMPTS.md` and
`prompts/PROVENANCE.md`. Representative learned skill trees and their
selection evidence are provided in
[`appendix/SKILL_TREE_EXAMPLES.md`](appendix/SKILL_TREE_EXAMPLES.md).
