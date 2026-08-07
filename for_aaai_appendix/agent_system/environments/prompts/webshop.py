# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Dict, List, Sequence, Tuple







DEFAULT_WEBSHOP_PROMPT_CHAR_LIMIT = 24000


def format_webshop_history(
    records: Sequence[Dict[str, Any]],
    *,
    first_step: int,
) -> str:
    ""
    lines: List[str] = []
    for offset, record in enumerate(records):
        step_num = int(first_step) + offset
        lines.append(
            f"[Observation {step_num}: '{record.get('text_obs', '')}', "
            f"Action {step_num}: '{record.get('action', '')}']"
        )
    return "\n".join(lines)


def fit_webshop_history_to_char_limit(
    records: Sequence[Dict[str, Any]],
    *,
    first_step: int,
    render_prompt: Callable[[str, int], str],
    char_limit: int = DEFAULT_WEBSHOP_PROMPT_CHAR_LIMIT,
) -> Tuple[str, int, int, bool]:
    ""
    try:
        limit = int(char_limit)
    except (TypeError, ValueError):
        limit = DEFAULT_WEBSHOP_PROMPT_CHAR_LIMIT
    limit = max(1, limit)

    recent = list(records)
    final_prompt = ""
    for dropped in range(len(recent) + 1):
        kept = recent[dropped:]
        history = format_webshop_history(kept, first_step=int(first_step) + dropped)
        final_prompt = render_prompt(history, len(kept))
        if len(final_prompt) <= limit:
            return final_prompt, len(kept), dropped, False



    return final_prompt, 0, len(recent), True


WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
First reason about the current situation inside exactly one <think>...</think> block. Then return exactly one action block in this form: <action>click[...]</action> or <action>search[...]</action>. Do not output a bare action or any text after </action>.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
First reason about the current situation inside exactly one <think>...</think> block. Then return exactly one action block in this form: <action>click[...]</action> or <action>search[...]</action>. Do not output a bare action or any text after </action>.
"""

WEBSHOP_TEMPLATE_WITH_MEMORY = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
First reason about the current situation inside exactly one <think>...</think> block. Then return exactly one action block in this form: <action>click[...]</action> or <action>search[...]</action>. Do not output a bare action or any text after </action>.
"""
