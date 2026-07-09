# Fixed two-task bullets A/B

This test holds the ALFWorld instances and cloud-update boundary constant while changing only the flat skill-bullets component.

- `fixed_two_tasks.json` selects one pick-one and one pick-two task. Both use AlarmClock → Desk in FloorPlan307.
- `bullets_off` evolves only the hierarchical playbook tree.
- `bullets_on` also runs `contrastive_distill`, ingests `dyn_` patches, and injects flat bullets.
- By default, each phase has three rollouts per task. Phase 1 is pre-update; phase 2 evaluates the first evolved context. The update after phase 2 tests history-aware evolution and is preserved in `cloud_io/`.

Run from the repository root:

```bash
GPU=1 bash examples/playbook_evolve/run_fixed_two_tasks_ab.sh
```

The regular driver defaults to embedding retrieval. For a faster pipeline smoke test that avoids loading the embedding model:

```bash
GPU=1 RETRIEVAL_MODE=template bash examples/playbook_evolve/run_fixed_two_tasks_ab.sh
```

For the stronger six-rollout-per-task comparison aligned with the main run's group size:

```bash
GPU=1 ROLLOUTS_PER_TASK=6 bash examples/playbook_evolve/run_fixed_two_tasks_ab.sh
```

Set `DEEPSEEK_API_KEY` first. Results are written under `skillrl_outputs/fixed_two_tasks_ab/<timestamp>/`, including `comparison.md`, per-arm summaries, every SLM prompt, cloud prompts, evolved playbooks, and metrics.
