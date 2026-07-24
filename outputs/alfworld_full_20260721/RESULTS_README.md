# ALFWorld CoSkill no-RL final results

This directory contains the final artifacts for the 100-group / 7,200-episode
ALFWorld CoSkill no-RL run completed on 2026-07-24.

## Fixed experiment design

- rollout batch: 72
- group size: 6
- task allocation design: 6 / 12 / 32
- data parallel: 2 GPUs
- tensor parallel: 1
- pipeline parallel: 1
- maximum environment steps: 40
- seed: 0

## Canonical result

`group_metrics.jsonl` is the canonical group-level result. It contains exactly
100 rows with the same 188-field metric schema for every row. Group 100 contains
72 episodes, 52 wins, and a success rate of 0.722222.

The process was interrupted after all 72 terminal rollouts completed but before
the group-100 metric writer ran. Outcome, task, action, validity, and timing
fields were recovered from the final 72 persisted raw traces. Full model
reasoning text existed only in worker memory, so group-100 small-model tokens
were estimated from group 99's measured per-task token-per-action rates:

- prompt: 3,659,758
- response: 3,805,150
- total: 7,464,908

Group 100 is terminal and therefore records no subsequent cloud/skill update:
`coskill/cloud_update_fired=false` and group large-model token deltas are zero.
The field names and types remain identical to groups 1-99 for compatibility
with the unified analysis scripts.

`summary_partial.json` and `checkpoints/step007056.json` represent the latest
formal resumable checkpoint (group 98), not the terminal aggregate. Repair
backups and local W&B runtime directories are intentionally excluded from Git.
