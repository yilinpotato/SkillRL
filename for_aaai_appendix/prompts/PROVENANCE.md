# Prompt Source Map

## Edge prompts

The ALFWorld base templates are defined in
`agent_system/environments/prompts/alfworld.py`. The WebShop base templates are
defined in `agent_system/environments/prompts/webshop.py`.

For frozen CoSkill runs, `agent_system/frozen_executor/alfworld_prompt.py` and
`agent_system/frozen_executor/webshop_prompt.py` insert the task, current
observation, admissible actions, recent interaction history, and retrieved
skills. Tree-RL uses the equivalent assembly logic in
`agent_system/environments/env_manager.py`.

`agent_system/memory/skills_only_memory.py::format_for_prompt()` renders the
retrieved context in this order:

1. learned skill tree;
2. general principles;
3. task-relevant skills;
4. mistakes to avoid.

`agent_system/frozen_executor/vllm_agent.py::_build_prompt()` applies the tokenizer-owned
Qwen chat template after the repository prompt has been rendered.

On the initial ALFWorld step, the current implementation prepends the retrieved
skill tree to the no-history template. On the initial WebShop step, it inserts
the complete formatted retrieval block. This benchmark-specific difference is
preserved in the released code and prompt appendix.

## Cloud prompts

The online cloud loop uses three prompt builders in
`agent_system/memory/cloud_analyzer.py`:

- `_build_contrastive_prompt()`;
- `_build_diagnose_prompt()`;
- `_build_evolve_prompt()`.

`_domain_context()` supplies the benchmark-specific action and success
contract. Cloud requests use one user message and no additional repository
system prompt. Complete runtime prompts and responses are written under
`OUTPUT_DIR/cloud_io/`.
