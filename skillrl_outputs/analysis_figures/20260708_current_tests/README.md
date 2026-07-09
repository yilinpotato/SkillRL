# Current CoSkill Test Figures

Generated from existing metrics under:

- `skillrl_outputs/fixed_two_tasks_ab/20260706_163012`
- `skillrl_outputs/goal_sweep_ab/20260707_211358`

## Files

- `fixed_two_test_detailed.png`: fixed two-task A/B smoke test.
- `goal_sweep_none_test_detailed.png`: goal-sweep `none` baseline, including fair first-24 slice and 49-episode partial run.
- `goal_sweep_ablation_progress.png`: current completion status for four goal-sweep arms.
- `summary.csv`: numeric table used by the figures.

## Main numbers

- Fixed two-task `bullets_off`: 24/24 = 100.0%
- Fixed two-task `bullets_on`: 24/24 = 100.0%
- Goal sweep `none@24`: 14/24 = 58.3%
- Goal sweep `none@49`: 30/49 = 61.2%

Note: use `none@24` as the fair baseline when comparing against 24-episode arms.
