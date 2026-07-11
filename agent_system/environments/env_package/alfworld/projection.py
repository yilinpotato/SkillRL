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

from typing import List, Tuple
import re


def salvage_action_from_back(raw: str, adm: List[str]) -> Tuple[str, bool]:
    """thinking \u88ab\u622a\u65ad\u3001\u6ca1\u6709\u53ef\u7528 <action> \u6807\u7b7e\uff08\u6216\u6807\u7b7e\u5185\u5bb9\u5bf9\u4e0d\u4e0a admissible_commands\uff09
    \u65f6\u7684\u515c\u5e95\uff1a\u4ece\u6a21\u578b\u539f\u59cb\u6587\u672c\u3010\u672b\u5c3e\u5f80\u524d\u3011\u626b\uff0c\u627e\u6700\u540e\u4e00\u4e2a\u51fa\u73b0\u5728 admissible \u5217\u8868\u91cc\u7684\u52a8\u4f5c\u3002
    \u6a21\u578b\u5e38\u5728\u601d\u8003\u6536\u5c3e\u5904\u5199\u51fa 'the action is go to drawer 1' \u4e4b\u7c7b\uff0c\u8d8a\u9760\u540e\u8d8a\u63a5\u8fd1\u6700\u7ec8\u610f\u56fe\uff0c
    \u6240\u4ee5\u4ece\u540e\u5f80\u524d\u5339\u914d\u6700\u7a33\u3002\u8fd4\u56de (action_str, matched_bool)\u3002"""
    low = raw.lower()
    best = None  # (\u4f4d\u7f6e, \u52a8\u4f5c)
    for cmd in adm:
        if cmd == "help":
            continue
        pos = low.rfind(cmd.lower())   # \u8be5\u52a8\u4f5c\u5728\u6587\u672c\u4e2d\u6700\u540e\u4e00\u6b21\u51fa\u73b0\u7684\u4f4d\u7f6e
        if pos != -1 and (best is None or pos > best[0]):
            best = (pos, cmd)
    if best:
        return best[1], True
    return "", False


def alfworld_projection(actions: List[str], action_pools: List[List[str]],
                        return_details: bool = False):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.

    \u6bcf\u4e2a\u52a8\u4f5c\u5728\u8fd4\u56de\u524d\u90fd\u4f1a\u4e0e\u5bf9\u5e94\u7684 ``action_pools[i]``\uff08admissible_commands\uff09\u505a\u7cbe\u786e\u5339\u914d\uff1a
    \u62bd\u53d6\u5931\u8d25\u6216\u62bd\u53d6\u5230\u7684\u5185\u5bb9\u4e0d\u5728\u6c60\u5b50\u91cc\uff0c\u5148\u5c1d\u8bd5 :func:`salvage_action_from_back` \u4ece\u6a21\u578b\u539f\u59cb
    \u6587\u672c\u91cc\u635e\u4e00\u4e2a\u786e\u5b9e\u5728\u6c60\u5b50\u91cc\u7684\u52a8\u4f5c\uff1b\u4ecd\u7136\u627e\u4e0d\u5230\u5c31\u9000\u5316\u4e3a\u4e00\u4e2a\u786e\u5b9a\u5b89\u5168\u7684\u9ed8\u8ba4\u52a8\u4f5c\uff08\u540c
    rollout \u5faa\u73af\u91cc"\u975e\u6d3b\u8dc3 slot \u5360\u4f4d\u52a8\u4f5c"\u7684\u515c\u5e95\u903b\u8f91\u4e00\u81f4\uff09\u3002\u8fd9\u6837\u4fdd\u8bc1\u6700\u7ec8\u9001\u8fdb env.step()
    \u7684\u5b57\u7b26\u4e32\u4e00\u5b9a\u662f admissible_commands \u91cc\u7684\u5408\u6cd5\u52a8\u4f5c\uff0c\u7edd\u4e0d\u4f1a\u662f\u6a21\u578b\u7684\u534a\u6210\u54c1/\u601d\u8003\u6b8b\u7559\u6587\u672c\u3002
    ``valids[i]`` \u4ecd\u7136\u53ea\u53cd\u6620"\u683c\u5f0f\u662f\u5426\u89c4\u6574"\uff08<action>/<think> \u6807\u7b7e\u662f\u5426\u95ed\u5408\u3001\u6709\u65e0\u4e2d\u6587\uff09\uff0c
    \u4e0d\u56e0\u4e3a\u8d70\u4e86 salvage/\u9ed8\u8ba4\u515c\u5e95\u800c\u88ab\u62c9\u9ad8\u3002
    """

    valids = [0] * len(actions)
    details = []

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        pool = action_pools[i] if i < len(action_pools) else []
        lowered_pool = {cmd.strip().lower(): cmd for cmd in pool}
        lowered = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag)
        extracted_action = None
        try:
            if start_idx != -1 and end_idx != -1:
                # Extract just the content between the tags
                extracted_action = lowered[start_idx + len(start_tag):end_idx].strip().lower()
        except Exception:
            extracted_action = None

        format_valid = extracted_action is not None

        # Exact match against admissible_commands. Only a hit here is trusted
        # as-is; anything else goes through salvage / a safe default below.
        matched_directly = extracted_action is not None and extracted_action in lowered_pool
        execution_source = "direct"
        if matched_directly:
            actions[i] = lowered_pool[extracted_action]
        else:
            salvaged, ok = salvage_action_from_back(original_str, pool)
            if ok:
                actions[i] = salvaged
                execution_source = "salvaged"
            else:
                actions[i] = "look" if "look" in pool else (pool[0] if pool else "look")
                execution_source = "fallback"

        # valid means the model directly named an admissible action with no
        # rescue needed — salvage/default-fallback never counts as valid, even
        # though we still hand the environment a legal action either way.
        valids[i] = 1 if (format_valid and matched_directly) else 0

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        has_think_block = think_start_idx != -1 and think_end_idx != -1
        if not has_think_block:
            valids[i] = 0

        # check if contains any Chinese characters
        contains_cjk = bool(re.search(r'[\u4e00-\u9fff]', original_str))
        if contains_cjk:
            valids[i] = 0

        details.append({
            # ``valid_action`` intentionally keeps the historical ``valids``
            # semantics: the model emitted a tagged, directly admissible action
            # and a closed think block, without relying on rescue logic.
            "valid_action": bool(valids[i]),
            "execution_source": execution_source,
            "has_action_block": bool(format_valid),
            "direct_admissible_action": bool(matched_directly),
            "has_think_block": has_think_block,
            "contains_cjk": contains_cjk,
        })

    if return_details:
        return actions, valids, details
    return actions, valids
