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
from typing import List, Tuple


def salvage_action_from_back(raw: str, adm: List[str]) -> Tuple[str, bool]:
    ""
    low = raw.lower()
    best = None
    for cmd in adm:
        if cmd == "help":
            continue
        pos = low.rfind(cmd.lower())
        if pos != -1 and (best is None or pos > best[0]):
            best = (pos, cmd)
    if best:
        return best[1], True
    return "", False


def alfworld_projection(actions: List[str], action_pools: List[List[str]],
                        return_details: bool = False):
    ""

    valids = [0] * len(actions)
    details = []

    for i in range(len(actions)):
        original_str = actions[i]
        pool = action_pools[i] if i < len(action_pools) else []
        lowered_pool = {cmd.strip().lower(): cmd for cmd in pool}
        lowered = actions[i].lower()


        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag)
        extracted_action = None
        try:
            if start_idx != -1 and end_idx != -1:

                extracted_action = lowered[start_idx + len(start_tag):end_idx].strip().lower()
        except Exception:
            extracted_action = None

        format_valid = extracted_action is not None



        matched_directly = extracted_action is not None and extracted_action in lowered_pool
        execution_source = "direct"
        if matched_directly:
            actions[i] = lowered_pool[extracted_action]
            recovered_from_model = True
        else:
            salvaged, ok = salvage_action_from_back(original_str, pool)
            if ok:
                actions[i] = salvaged
                execution_source = "salvaged"
                recovered_from_model = True
            else:
                actions[i] = "look" if "look" in pool else (pool[0] if pool else "look")
                execution_source = "fallback"
                recovered_from_model = False





        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        has_think_block = think_start_idx != -1 and think_end_idx != -1

        contains_cjk = bool(re.search(r'[\u4e00-\u9fff]', original_str))
        relaxed_protocol_valid = bool(
            format_valid and think_end_idx != -1 and not contains_cjk)
        non_strict_valid = bool(recovered_from_model or relaxed_protocol_valid)
        valids[i] = int(non_strict_valid)



        strict_valid_action = bool(format_valid and matched_directly) and (
            not contains_cjk
            and original_str.count("<think>") == 1
            and original_str.count("</think>") == 1
            and original_str.count("<action>") == 1
            and original_str.count("</action>") == 1
            and think_start_idx < think_end_idx < start_idx < end_idx
        )
        details.append({
            "valid_action": non_strict_valid,
            "strict_valid_action": strict_valid_action,
            "execution_source": execution_source,
            "has_action_block": bool(format_valid),
            "direct_admissible_action": bool(matched_directly),
            "has_think_block": has_think_block,
            "contains_cjk": contains_cjk,
        })

    if return_details:
        return actions, valids, details
    return actions, valids
