# Code Map

## Entry points

`run.sh` is the public entry point. `--benchmark` selects ALFWorld or WebShop,
and `--rl` selects the frozen executor or Tree-RL. The scripts in `backends/`
contain hardware detection and benchmark-specific launch arguments.

## Online CoSkill loop

- `agent_system/memory/traces_pool.py`: loop filtering, observation deltas,
  prefix-tree merging, trigger checks, and cloud-batch projection.
- `agent_system/memory/coskill_loop.py`: cloud-update orchestration.
- `agent_system/memory/cloud_analyzer.py`: contrastive distillation, failure
  diagnosis, skill-tree evolution, and cloud token accounting.
- `agent_system/memory/hierarchical_skill_lib.py`: skill storage, retrieval,
  usage attribution, and lifecycle state.
- `agent_system/memory/skills_only_memory.py`: task classification, prompt
  rendering, skill-tree state, and Tree-RL curriculum.
- `agent_system/memory/playbook_tree.py`: skill-tree parsing and node
  visibility.
- `agent_system/task_taxonomy.py`: shared ALFWorld and WebShop task
  classification for retrieval and metrics.

## Frozen executor

- `examples/playbook_evolve/run_playbook_evolve.py`: ALFWorld experiment loop.
- `examples/playbook_evolve/run_webshop_evolve.py`: WebShop experiment loop.
- `agent_system/frozen_executor/vllm_agent.py`: Qwen/vLLM generation.
- `agent_system/frozen_executor/alfworld_prompt.py`: ALFWorld prompt assembly.
- `agent_system/frozen_executor/webshop_prompt.py`: WebShop prompt assembly.
- `agent_system/frozen_executor/alfworld_runtime.py`: ALFWorld environment
  loading and task selection.
- `agent_system/frozen_executor/webshop_runtime.py`: WebShop environment
  loading and observation handling.

## Tree-RL

- `verl/trainer/main_ppo.py`: Ray training entry.
- `verl/trainer/ppo/ray_trainer.py`: GRPO, validation, CoSkill updates, and
  progressive tree internalization.
- `agent_system/multi_turn_rollout/rollout_loop.py`: batched multi-step
  environment interaction.
- `agent_system/environments/env_manager.py`: benchmark prompt and environment
  management for Ray workers.

## Benchmark adapters

- `agent_system/environments/env_package/alfworld/`
- `agent_system/environments/env_package/webshop/`
- `agent_system/environments/prompts/`

Only ALFWorld and WebShop adapters are included. Search, Sokoban, AppWorld,
GymCards, GiGPO, cold-start generation, Megatron model implementations,
multimodal model adapters, unrelated trainers, tests, output files, and
ablation launchers are excluded.
Tree-RL uses the Hugging Face Qwen3 text model through FSDP; frozen execution
uses the same model through vLLM.
