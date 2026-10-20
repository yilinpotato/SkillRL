# ScienceWorld for CoSkill

Current method settings: [shared no-RL method v2](NEW_BENCHMARK_METHOD.md).

Frozen-model, multi-turn ScienceWorld execution uses the existing CoSkill trajectory
pool, cloud analysis and task-specific skill trees. This backend is no-RL; it does
not implement ScienceWorld GRPO or Tree-RL. SearchQA's chat transport is reused,
but neither Tavily nor any search service is required.

## Installation and environment check

Use Python 3.10+ and Java 8+. Install the pinned release with its bundled simulator:

```bash
python -m pip install scienceworld==1.2.3
bash examples/playbook_evolve/run_scienceworld.sh \
  --env-smoke --per-task 1 --tasks boil melt chemistry-mix \
  --output-dir outputs/scienceworld-env-check
```

The smoke command resets real tasks in official train/dev/test splits and performs
`look around`. It calls no model or cloud APIs and measures environment availability,
not agent capability. Use an empty output directory for each invocation.

## Model experiments

As with SearchQA, supply a running OpenAI-compatible executor (for example local vLLM):
`EDGE_API_BASE`, `EDGE_MODEL`, and optional `EDGE_API_KEY`. Configure the existing
cloud analyzer through `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`, or the Azure settings.
The standalone launcher optionally sources `SCIENCEWORLD_ENV_FILE`. It never starts
or stops GPU processes. Follow the server's GPU allocation rules before starting vLLM.

```bash
# Small frozen-model learning run with independent dev evaluation
bash examples/playbook_evolve/run_scienceworld.sh \
  --tasks boil melt chemistry-mix --per-task 4 --groups 2 --batch-size 6 \
  --validation-size 3 --validation-every 1 \
  --output-dir outputs/scienceworld-pilot

# Evaluate saved skills on official test variations
bash examples/playbook_evolve/run_scienceworld.sh \
  --mode eval --split test --tasks boil melt chemistry-mix --per-task 2 \
  --skills outputs/scienceworld-pilot/skills_latest.json \
  --output-dir outputs/scienceworld-test
```

Use `--no-cloud` for a fixed-skill baseline. Clouds update only when the existing
pool watermarks fire; a very small pilot may not trigger learning. For a full run,
omit `--tasks` and `--per-task` to consider all 30 task types and their variations.
Selected instances are deterministically shuffled and consumed once, with up to
100 groups of 72 instances by default. The last group can be smaller; group count
is capped by available data and `--groups`. This is not GRPO's repeated sampling.

## Protocol and outputs

- Pinned `scienceworld==1.2.3`; defaults: no simplifications, seed 0, 100 model
  decisions, history of 8 interactions, 4096 generated tokens per decision.
- `env.load(..., generateGoldPath=False)` is always used. No gold path, object tree,
  hidden world state or goal-progress oracle is queried or supplied to the model.
- Task description, observation, visible surroundings, inventory, and up to 80
  current valid-action examples are supplied, sampled round-robin across action verbs. This is an explicitly action-assisted
  protocol; the examples are partial, not a restriction on accepted actions.
- Native Qwen responses with a template-prefilled opening think tag are normalized
  only when a closing think tag and following action are present. Truncated reasoning
  is not converted into an action.
- Exact `<think>...</think><action>literal environment command</action>` format.
  Action tags quoted inside reasoning are not executed; only the single action
  after the closing think tag is accepted. Extra prose outside those blocks is format-invalid; v2 may still execute a unique action.
  V2 separately diagnoses format and recovers a unique complete action outside reasoning.
  Unrecoverable responses consume a decision without advancing the simulator.
  Syntactically valid commands are sent to the simulator; action validity records
  exact membership in its pre-action valid list and is a diagnostic, not a reward.
- Official variation IDs from train/dev/test are checked for overlap and saved in
  `run_config.json` with version, simplifications, seed, and model settings.
- Training success requires score 100. Partial progress and negative terminal
  failures remain failures. `episode/score` keeps the raw -100..100 score;
  the trajectory pool receives nonnegative score / 100 as `task_score`.
- Dev evaluation occurs before training, every 5 groups, and at the last group.
  Dev/test trajectories never enter the learning pool or skill usage statistics.
- `group_metrics.jsonl`: success, score, steps, format/action validity, per-task
  success/score, edge/cloud API tokens, usage omissions, and wall time.
- `validation_metrics.jsonl`: independent dev metrics and token accounting.
- `step_events.jsonl` saves every returned model step immediately for interruption diagnosis.
- Split-specific trajectory JSONL, `skills_latest.json`, `skill_lib/` and
  `cloud_io/` retain the auditable CoSkill evidence chain.
- `completed.json` is written only on successful completion; provider errors stop
  the run instead of being counted as model failures. No exact resume is provided;
  `--skills` starts a new run rather than restoring pool/lifecycle counters.

ScienceWorld reset itself performs a look action. We record model decisions separately
from simulator moves. Compare methods with matching versions, simplifications,
variations, observation/action-assistance settings and budgets. Later ScienceWorld
versions may change simulator behavior; this integration does not claim score parity
with a different release or protocol.

Tests: `python -m unittest discover -s tests -p 'test_scienceworld_adapter.py'`.
Reference: [official ScienceWorld repository](https://github.com/allenai/ScienceWorld).

## Local Qwen integration smoke (2026-10-19)

A real RTX 3090 GPU 0 run used Qwen3-4B-Thinking-2507 through vLLM 0.11.0,
ScienceWorld 1.2.3, seed 0, no simplifications, an empty skill library, official
test variations, 15 decisions per task, and 4096 output tokens per decision.
The server used a 24576-token context, eager mode, and one concurrent sequence.

| Task | Variation | Score | Success | Decisions |
| --- | --- | --- | --- | --- |
| chemistry-mix | 27 | 8 | No | 15 |
| boil | 24 | 0 | No | 15 |

The final run completed normally in 649.55 seconds; 2/30 decisions met the output
format and valid-action contracts. Provider-reported usage was 162708 prompt and
25476 completion tokens (188184 total), with no missing usage calls. Most invalid
responses contained prose or additional reasoning tags outside the required
blocks. Model planning also failed to complete either task. This is integration
smoke evidence, not a formal benchmark or evidence of CoSkill learning gains;
evaluation did not update skills or call the cloud learner.

The run exposed and fixed template-prefilled Qwen think-tag handling and action
tags quoted inside reasoning. Valid-action examples now cover different verbs
instead of taking only a lexicographic prefix. Regression validation: 20 targeted
tests passed; Ruff and git whitespace checks passed. Interrupted diagnostic runs
are separate from these final results. Private run artifacts are retained on the
execution server under `coskill-workspace/results/scienceworld-qwen-gpu0-20261019/eval-final`.
The dedicated model service was stopped after the test; GPU 0 returned to 36 MiB
and GPU 1 remained at 19281 MiB.
