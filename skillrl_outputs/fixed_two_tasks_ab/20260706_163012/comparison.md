# Fixed two-task CoSkill A/B

Both arms use the same pick-one and pick-two game instances. Round 0 is before the first cloud update; round 1 is after one playbook/skill evolution update.

| Arm | Round | Episodes | Wins | Success | pick_one | pick_two | Cloud patches | Playbook updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bullets OFF | 0 | 12 | 12 | 100.0% | 100.0% | 100.0% | 0 | 3 |
| bullets OFF | 1 | 12 | 12 | 100.0% | 100.0% | 100.0% | 0 | 3 |
| bullets ON | 0 | 12 | 12 | 100.0% | 100.0% | 100.0% | 4 | 2 |
| bullets ON | 1 | 12 | 12 | 100.0% | 100.0% | 100.0% | 4 | 2 |

Post-update bullets ON − OFF: **+0.0 percentage points**.

Inspect each arm's `cloud_io/`, `trajectories/`, and `metrics.jsonl` before attributing a small-sample difference to the bullets.
