# SearchQA live-search

Current method settings: [shared no-RL method v2](NEW_BENCHMARK_METHOD.md).

This optional no-RL backend uses SearchQA questions with **live web search**, not
the dataset's stored snippets. Report it as `searchqa_live`, not the original
SearchQA reading-comprehension protocol. Tree-RL is not implemented for this backend.

The frozen executor chooses `search[query]`, `open[result number]`, or
`answer[short answer]`. Each response uses one `<think>...</think>` block followed
by one `<action>...</action>` block. Only the question is visible initially.
Reference answers stay in the evaluator; training provides success/failure to the
existing trajectory pool and cloud skill loop. Validation and test trajectories
never enter the pool or update skill usage statistics.

## Setup

Run from the repository root in the existing Python environment:

```bash
python -m pip install datasets huggingface_hub openai
python tools/prepare_searchqa.py
```

Preparation resolves and records the Hugging Face dataset revision. This mirror
is the **MRQA SearchQA version**, with train and validation only. We deterministically
reserve approximately 5% of source train for learning validation using the first
8 hex digits of SHA256(key) modulo 100 < 5. Source validation is held out as test.
These are project splits, not the original SearchQA official three-way split.
Preparation strips stored contexts and writes
`data/searchqa/{train,validation,test}.jsonl`. The source dataset is downloaded and cached before selection; --limit caps exported examples, not download size. For a small initial export:

```bash
python tools/prepare_searchqa.py --output-dir data/searchqa-small --limit 32
```

Alternatively provide JSONL with `id`, `question`, and `answers` (list of strings).
Duplicate IDs, empty answers and train/validation question overlap are rejected.

Add these values to the existing `.env` (do not commit credentials):

```bash
TAVILY_API_KEY=your-search-service-key
EDGE_API_BASE=http://127.0.0.1:8000/v1
EDGE_MODEL=Qwen/Qwen3-4B-Thinking-2507
EDGE_API_KEY=local
```

The executor uses an OpenAI-compatible chat server, including a separately running
vLLM server. It does not load or stop GPU processes. In an allocated, idle GPU slot,
start the local model separately, for example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3-4B-Thinking-2507 \
  --served-model-name Qwen/Qwen3-4B-Thinking-2507 --port 8000
```

Cloud learning uses the existing `DEEPSEEK_API_KEY`, `DEEPSEEK_API_BASE`, and
`DEEPSEEK_MODEL` settings (or the existing Azure backend). Select an available
cloud model explicitly for your account. `--no-cloud` runs a fixed-skill baseline.
Tavily search/extract charges are separate from model tokens. Search-generated
answers are disabled; the executor must select its own answer from evidence.

## Run

Validate data without network or GPU use:

```bash
bash examples/playbook_evolve/run_searchqa_live.sh \
  --data data/searchqa/train.jsonl --validation-data data/searchqa/validation.jsonl \
  --output-dir outputs/searchqa-check --check-only
```

Start with a small run before a full experiment:

```bash
bash examples/playbook_evolve/run_searchqa_live.sh \
  --data data/searchqa/train.jsonl --validation-data data/searchqa/validation.jsonl \
  --output-dir outputs/searchqa-live-pilot --groups 2 --batch-size 16 \
  --validation-size 4 --validation-every 1
```

Evaluate a saved skill library separately:

```bash
bash examples/playbook_evolve/run_searchqa_live.sh \
  --mode eval --data data/searchqa/test.jsonl \
  --skills outputs/searchqa-live-pilot/skills_latest.json \
  --output-dir outputs/searchqa-live-test --limit 100 --groups 100
```

Defaults: seed 0, 100 groups, up to 72 distinct questions per group, 15 actions,
8 searches, 8 interaction-history pairs, 4096 output tokens per decision, and
12000 characters per tool observation. Questions are shuffled deterministically
and used once per run; the last group can be smaller. Execution is sequential
through the chat endpoint. This is not GRPO's 12-by-6 sampling protocol.
Validation runs before training, every 5 groups, and at the final group.
Cloud updates use existing watermarks, not a forced update every group.
Outputs must be empty. There is no exact resume; loading `--skills` starts a new
run and does not restore the previous pool, lifecycle counters or search state.

## Outputs and limitations

- `run_config.json`: selected IDs, protocol, data hash and non-secret configuration.
- `group_metrics.jsonl`: normalized answer EM, token F1, action validity, steps,
  model tokens, search calls/credits and timing. Scores are fractions in [0, 1].
- `validation_metrics.jsonl`: validation-only metrics with separate token totals.
- `train_trajectories.jsonl` / `eval_trajectories.jsonl`: actions and observations.
- `search_audit.jsonl`: timestamped search/extract requests and responses.
- `skills_latest.json`, `skill_lib/`, `cloud_io/`: existing CoSkill skill artifacts.
- `completed.json`: written only after the run finishes successfully.

Provider errors stop the run rather than being counted as agent failures. Partial
artifacts remain available. Missing token/credit usage is explicitly counted and
must not be interpreted as zero cost. Live sources change over time; audit logs
preserve returned evidence but do not make future searches deterministic.

The default search excludes `j-archive.com` and `j-archive.org`, where many source
questions originate. This reduces direct answer lookup but does not guarantee
that mirrors or benchmark contamination are absent. Use the same provider,
budgets, exclusions and collection window for all compared methods. The EM/F1
normalization is SQuAD-style; it does not imply official SearchQA leaderboard parity.

Tests: `python -m unittest discover -s tests -p 'test_searchqa_live.py'`.

Sources: [SearchQA dataset](https://huggingface.co/datasets/lucadiliello/searchqa),
[Tavily search](https://docs.tavily.com/documentation/api-reference/endpoint/search),
[Tavily extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract).

## Server configuration

Use `SEARCHQA_ENV_FILE` to source your private search configuration with the standalone launcher. Cloud and executor settings can be exported separately.
