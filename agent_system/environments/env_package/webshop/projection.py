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

from typing import List
import re

def webshop_projection(actions: List[str], return_details: bool = False):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>...</think><action>search[...]</action>
        <think>...</think><action>click[...]</action>

    Both blocks are mandatory, and the completed reasoning block must precede
    the action block.  This matches the ALFWorld action protocol and the
    WebShop SFT/rollout contract.
    """

    valids = [0] * len(actions)
    details = []

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        has_action_block = start_idx != -1 and end_idx != -1 and start_idx < end_idx
        if has_action_block:
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            actions[i] = extracted_action
        else:
            # Preserve historical malformed-output handling; the environment
            # decides whether this suffix is executable.
            actions[i] = actions[i][-20:]

        # Require one completed thinking block before the action.  Do this on
        # the original case-preserving string because the tags form part of
        # the model output protocol.
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        action_start_idx = original_str.find("<action>")
        strict_valid_action = has_action_block and (
            original_str.count("<think>") == 1
            and original_str.count("</think>") == 1
            and original_str.count("<action>") == 1
            and original_str.count("</action>") == 1
            and think_start_idx != -1
            and think_end_idx != -1
            and think_start_idx < think_end_idx < action_start_idx < original_str.find("</action>")
        )

        # check if contains any Chinese characters
        contains_cjk = bool(re.search(r'[\u4e00-\u9fff]', original_str))
        if contains_cjk:
            strict_valid_action = False

        # valid_action is the historical, non-strict action-block metric;
        # strict_valid_action is the complete dual-block protocol metric.
        valids[i] = int(strict_valid_action)
        details.append({
            "valid_action": bool(has_action_block),
            "strict_valid_action": bool(strict_valid_action),
            "execution_source": "direct" if has_action_block else "malformed",
            "has_action_block": bool(has_action_block),
            "has_think_block": think_start_idx != -1 and think_end_idx != -1,
            "contains_cjk": contains_cjk,
        })

    if return_details:
        return actions, valids, details
    return actions, valids
