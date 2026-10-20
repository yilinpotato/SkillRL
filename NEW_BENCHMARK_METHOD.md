# ScienceWorld and live SearchQA: shared no-RL method v2

Only these two backends import `agent_system/method_protocol.py`. ALFWorld,
WebShop, their launchers, shared memory classes and RL behavior are unchanged.
The existing batch/sampling schedule is unchanged; this is not a 12-by-6 rollout migration.

## Explicit shared settings

| Setting | Default |
| --- | --- |
| learning temperature / seed | 1.0 / 0 |
| dev and standalone evaluation temperature / seed | 0.4 / 1000 |
| history | last 8 observations and actually executed actions |
| output budget | 4096 tokens per decision |
| input budget | 8192 tokens using the selected tokenizer and chat template |
| retrieval | template; general top-k 6; task-specific top-k 6; existing mistakes limit 5 |
| hierarchy / skill tree | enabled |
| minimum skill calls | 10 (legacy launcher setting, not class default 20) |
| stable cycles L1 / L2 | 3 / 5 |
| promotion success / demotion threshold | 0.7 / 0.3 |
| pool capacity / failure watermark / minimum samples | 50000 / 0.6 / 16 |
| evidence compression | loop filter, observation delta, prefix and consensus trees; tree_only |
| new skills / minimum tree samples | 3 / 6 |

CLI flags expose sampling, budget, retrieval limits, lifecycle thresholds and cloud
thresholds. `run_config.json` records effective arguments and the method manifest.
Initial skills remain empty unless `--skills` supplies an audited domain-specific
file; no new seed answers or successful trajectories are inserted. Initial files
are copied into the run directory and supplied skill files are hashed.

For an offline Qwen server, pass `--tokenizer-path "$MODEL_PATH"`. The `EDGE_TOKENIZER_PATH` environment variable is also supported. Otherwise the
model ID is used as the Hugging Face tokenizer ID. For API-only models without an
accessible tokenizer, explicitly select and disclose a reference tokenizer. The
input cap is measured by that tokenizer, not claimed to equal a remote provider's
billing tokens. Actual token accounting always uses provider usage; omissions
remain visible. No model weights or GPU are needed to load a tokenizer.

Only complete oldest observation/action pairs may be removed to fit the input
budget. The task, skills and current observation are preserved. If those alone
exceed the cap, execution fails before sending a request instead of silently
truncating evidence. Step usage records include budget tokens and dropped pairs.
Raw reasoning remains in trajectory logs but never returns as interaction history.

## Action contract

Both prompts request a think block followed by one action block. Execution may
recover a single complete action tag outside a completed think block, or an
action-only reply. Extra explanatory text is a format violation but need not
block a uniquely specified action. Unclosed thinking, multiple action tags and
incomplete action tags are rejected. Never search within thinking for a candidate
command and never invent a fallback action.

`parsed_action`, `executed_action`, `execution_source` (direct/recovered/none),
`format_valid` and environment feedback are recorded separately. A rejected
command consumes a decision but is recorded as no execution. ScienceWorld still
submits parsed commands to the simulator and separately diagnoses membership in
its valid-action list; that list is not a substitute for simulator feedback.
SearchQA validates its verbs, search budget and previously returned URL numbers.
No repeat-action early stopping is introduced in either backend.

## Environment contracts retained

ScienceWorld stays at 1.2.3, official train/dev/test variations, no simplifications
by default, no gold paths, 100 decisions and up to 80 action examples. Success is
score 100; raw scores and normalized task scores remain separate. Action examples
are an action-assisted protocol choice, not a claim about all ScienceWorld papers.

SearchQA stays live-search QA with question-only input, evaluator-only references,
15 decisions, at most 8 searches, existing Tavily settings and 12000-character
observations. EM/F1 and search usage are retained. This is a custom interactive
protocol, not the original fixed-snippet SearchQA evaluation.

## Learning, evaluation and audit

The cloud trigger is checked at the existing group boundary; the pool thresholds
are explicit. The cloud receives cumulative completed TRAIN episode count as its
progress index. An update becomes available to subsequent episodes. Validation
never enters the pool or changes usage/lifecycle statistics; before/after skill
and tree fingerprints enforce this. Each validation uses the same evaluation seed
and temperature. Train/eval model usage and SearchQA tool usage accumulate separately.

`metrics.jsonl` records episodes; `group_metrics.jsonl` retains legacy fields and
adds common action ratios, task score, sampling settings, protocol and progress.
Action ratios use pooled decision counts, not the average of per-episode ratios.
Cloud prompt/response/total and missing-usage counts have delta and cumulative
fields. Old SearchQA `cloud/updated` remains an alias of `coskill/updated`.
`step_events.jsonl` immediately persists returned decisions in both environments.

`summary_partial.json` records completed progress and the latest skill fingerprint.
It is an audit snapshot, NOT an exact-resume checkpoint: pool/RNG/environment state
is not restored. Existing nonempty output directories are still rejected, and
`--skills` starts a new run. Provider errors fail the run, with no completed marker;
they are not scored as benchmark failures. No automatic API retries are added.

## Compatibility and validation

Prior strict-parser Qwen smoke results remain v1 evidence and must not be silently
merged with v2 results. This change is a method-contract migration, not measured
proof of improved task success. Targeted tests cover history, recovery, token
budgeting, sampling, lifecycle configuration, cumulative progress and evaluation
isolation. Legacy ALFWorld/WebShop action and prompt tests remain unchanged.

Verified for this migration: 39 targeted and legacy contract tests passed. A real
local Qwen tokenizer test dropped 6 old pairs and fit within its requested cap;
a scripted two-decision ScienceWorld 1.2.3 run executed recovered `look around`
actions. The latter tests simulator plumbing, not real-model task performance.
No GPU inference or paid model/search API experiment was performed for v2.
