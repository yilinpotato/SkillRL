# ALFWorld Terminal Comparison

This directory separates measurement from presentation.

## Run the real evaluation

Run each arm in its own terminal. These commands execute the model and ALFWorld environment and retain the original per-task records.

```bash
conda activate skillRL
OUTPUT_ROOT="$PWD/smoke_outputs/alfworld_terminal_dual_v2" \
  bash examples/terminal_compare/run_coskill_terminal_gpu0.sh
```

```bash
conda activate skillRL
OUTPUT_ROOT="$PWD/smoke_outputs/alfworld_terminal_dual_v2" \
  bash examples/terminal_compare/run_skillrl_terminal_gpu1.sh
```

Each arm retains `summary.json`, `metrics.jsonl`, `group_metrics.jsonl`, `traces_pool/raw_traces.jsonl`, `driver.log`, and `terminal_run_summary.json`. The evaluation uses four fixed tasks from each of the six ALFWorld task types, with a 40-step limit.

## Replay the measured results

After an arm has completed, replay its real task results as an accelerated serial terminal display:

```bash
OUTPUT_ROOT="$PWD/smoke_outputs/alfworld_terminal_dual_v2" SPEEDUP=100 \
  bash examples/terminal_compare/replay_coskill_terminal.sh
```

```bash
OUTPUT_ROOT="$PWD/smoke_outputs/alfworld_terminal_dual_v2" SPEEDUP=100 \
  bash examples/terminal_compare/replay_skillrl_terminal.sh
```

The replay does not load a model or use a GPU. It preserves task order, task type, task text, step count, success, and the measured total wall time. Because the real evaluation is batched, exact per-task wall times are unavailable. The replay therefore distributes the measured total time in proportion to each task's actual small-model token count and divides every interval by `SPEEDUP`. For example, an estimated 300-second task appears after 3 seconds at `SPEEDUP=100`, while the final line still reports the measured unscaled total time.

The timing derivation and preserved per-task token fields are written to `serial_replay_timeline.json`. Use `--no-wait` only to validate terminal formatting.

The replay headers recover the split, task count, seed, step limit, batch size, and CUDA Graph setting from `comparison_manifest.json`. Override the displayed original device when needed with `GPU=0` and `GPU_NAME="NVIDIA GeForce RTX 3090"`; replay itself never allocates that device.

## Replay every action from one task

Select a one-based task index and replay every recorded environment action:

```bash
OUTPUT_ROOT="$PWD/smoke_outputs/alfworld_terminal_dual_v2" \
TASK_INDEX=1 SPEEDUP=100 \
  bash examples/terminal_compare/replay_coskill_single_task.sh
```

Use `replay_skillrl_single_task.sh` for the matching SkillRL task. Each terminal line shows only the environment step and action. The generated `single_task_replay_NN.json` preserves the full observation and original step metadata. Exact per-step generation times were not recorded by the batched evaluation, so the selected task's token-weighted estimated duration is divided evenly across its recorded environment steps.
