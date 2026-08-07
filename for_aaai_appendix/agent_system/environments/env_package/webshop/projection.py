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

import re
from typing import List

_EXECUTABLE_ACTION = re.compile(r"\b(?:search|click)\s*\[[^\]\r\n]*\]", re.IGNORECASE)


def _recover_executable_action(raw: str, block_text: str = ""):
    ""
    block_match = _EXECUTABLE_ACTION.fullmatch(block_text.strip()) if block_text else None
    if block_match:
        return block_match.group(0).lower(), "direct"
    matches = list(_EXECUTABLE_ACTION.finditer(raw or ""))
    if matches:
        return matches[-1].group(0).lower(), "salvaged"


    return "search[]", "fallback"


def webshop_projection(actions: List[str], return_details: bool = False):
    ""

    valids = [0] * len(actions)
    details = []

    for i in range(len(actions)):
        original_str = actions[i]
        lowered = actions[i].lower()


        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag, start_idx + len(start_tag))
        has_action_block = start_idx != -1 and end_idx != -1 and start_idx < end_idx
        block_text = ""
        if has_action_block:
            block_text = lowered[start_idx + len(start_tag):end_idx].strip()
        actions[i], execution_source = _recover_executable_action(
            lowered, block_text=block_text)
        recovered_from_model = execution_source != "fallback"




        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        action_start_idx = original_str.find("<action>")
        strict_valid_action = has_action_block and execution_source == "direct" and (
            original_str.count("<think>") == 1
            and original_str.count("</think>") == 1
            and original_str.count("<action>") == 1
            and original_str.count("</action>") == 1
            and think_start_idx != -1
            and think_end_idx != -1
            and think_start_idx < think_end_idx < action_start_idx < original_str.find("</action>")
        )


        contains_cjk = bool(re.search(r'[\u4e00-\u9fff]', original_str))
        if contains_cjk:
            strict_valid_action = False




        valids[i] = int(recovered_from_model)
        details.append({
            "valid_action": bool(recovered_from_model),
            "strict_valid_action": bool(strict_valid_action),
            "execution_source": execution_source,
            "has_action_block": bool(has_action_block),
            "has_think_block": think_start_idx != -1 and think_end_idx != -1,
            "contains_cjk": contains_cjk,
        })

    if return_details:
        return actions, valids, details
    return actions, valids
