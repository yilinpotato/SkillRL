# Metrics Schema

All JSONL records use the envelope:

```json
{"step": 1, "metrics": {"record/type": "train_update"}}
```

The current comparison schema is version 3.

Do not append schema-v3 records to an existing schema-v1/v2 output directory.
Keep completed legacy outputs read-only, or migrate them before resuming.
For a resumed training checkpoint, point the new run at a fresh output
directory and record the source checkpoint in the run manifest.

`group_metrics.jsonl` is the only JSONL metric stream and the primary
plotting input. It contains one
`train_update` record per frozen rollout group or RL optimizer update.
Validation fields are merged into the corresponding update record. When
validation runs before training, step 0 contains one standalone
`validation` record.

`summary_partial.json` and `summary.json` are checkpoint and final summaries,
not additional time-series sources.

The top-level `step` is the only update identifier.
`training/group`, `training/global_step`, `global_episode_end`, and
`episode/count_cumulative` are not emitted. Current-group rollout count is
available as `episode/count`; completed progress remains available in the
checkpoint summaries.

## Namespaces

- `episode/*`: outcome, score, length, action validity, and rollout counts.
- `episode/by_task_type/<type>/*`: current-update and cumulative category
  counts, wins, and success rates.
- `validation/*`: held-out metrics.
- `tokens/small_model/*`: actual tokenizer-visible vLLM or actor requests.
- `tokens/large_model/*`: provider-reported cloud API usage.
- `coskill/*`: pool, skill lifecycle, tree evolution, and timing state.
- `timing_s/*`, `perf/*`: wall-clock and throughput measurements.
- `experiment/*`, `parallel/*`, `comparison/*`: run contract and hardware
  metadata.

`perf/total_num_tokens` is retained as the SkillRL-compatible alias of the
current update's `tokens/small_model/total`. `episode/valid_action_ratio`
uses the executable-action criterion employed by the reward path and is
therefore intentionally equal to `episode/non_strict_valid_action_ratio`.
`episode/strict_valid_action_ratio` separately measures exact
`<think>...</think><action>...</action>` compliance. The former
`relaxed_valid_action_ratio` alias is not emitted.

## Task taxonomies

ALFWorld uses six categories:
`pick_and_place`, `pick_two_obj_and_place`, `look_at_obj_in_light`, `clean`,
`heat`, and `cool`. WebShop uses the CoSkill analysis taxonomy:
`apparel`, `footwear`, `home_decor`, `electronics`, `accessories`,
`beauty_health`, and `other`. These WebShop labels are deterministic
goal-text groupings, not native benchmark classes and do not inspect target
product metadata. Frozen and Tree-RL paths share
`agent_system/task_taxonomy.py`. Word-boundary matching prevents collisions
such as `laptop` being classified as apparel because it contains `top`.

Per-task small-model totals include reconciliation fields. A nonzero
`tokens/small_model/by_task_type/reconciliation_error` indicates a logging
bug and should stop downstream analysis.
