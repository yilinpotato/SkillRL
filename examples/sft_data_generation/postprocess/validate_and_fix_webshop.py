#!/usr/bin/env python3
"""
WebShop SFT Data Validator & Fixer

Validates and fixes trajectory data quality issues:
1. Bare "buy now" → "click[buy now]" (in both actions and admissible actions)
2. Invalid/invented actions (check price, scan, scroll[...], empty) → remove sample
3. click[description/features/reviews] without click[< prev] → insert nav-back step
   and fix the description-step's observation & admissible actions

Usage:
    python validate_and_fix_sft.py [--input INPUT] [--output OUTPUT] [--dry-run]
"""

import json
import re
import copy
import argparse
from collections import Counter


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ADMISSIBLE_RE = re.compile(
    r"(admissible actions.*?:\s*\[)(.*?)(\]\.)", re.DOTALL
)
TASK_RE = re.compile(r"Your task is to: (.*?)(?:\n|$)")
VALID_ACTION_RE = re.compile(r"^(search\[.+\]|click\[.+\])$", re.IGNORECASE)

OBS_RE = re.compile(
    r"(current observation is:\s*)(.*?)(\nYour admissible actions)", re.DOTALL
)
STEP_RE = re.compile(r"You are now at step (\d+)")

INFO_PAGE_ACTIONS = {"click[description]", "click[features]", "click[reviews]"}
NAV_BACK_ACTIONS = {"click[< prev]", "click[back to search]"}

VALID_COLORS = {
    "black", "white", "blue", "red", "green", "grey", "navy",
    "pink", "beige", "brown",
}
VALID_SIZES = {"small", "medium", "large", "x-large", "one size"}
VALID_NAV = {
    "buy now", "back to search", "< prev", "next >", "search",
    "description", "features", "reviews",
}
PRODUCT_ID_RE = re.compile(r"^b[0-9][a-z0-9]{4,}$", re.IGNORECASE)

# Real WebShop description sub-page: only has Back to Search and < Prev
DESC_PAGE_ADMISSIBLE = "click[< prev], click[back to search]"

THINK_TEMPLATES = {
    "click[description]": (
        "I've reviewed the product description and confirmed the key specs. "
        "Now I need to go back to the product page to proceed."
    ),
    "click[features]": (
        "I've checked the product features. "
        "Let me go back to the product page to continue."
    ),
    "click[reviews]": (
        "I've looked at the reviews. "
        "Let me go back to the product page to continue."
    ),
}


def parse_action(output: str) -> str | None:
    m = ACTION_RE.search(output)
    return m.group(1).strip() if m else None


def has_one_strict_action_block(output: str) -> bool:
    """Whether an SFT target has one ordered thinking/action pair.

    Runtime rollouts require ``<think>...</think><action>...</action>``.
    Enforcing the same contract on the final SFT artefact prevents bare
    actions or action-only completions from teaching a conflicting format.
    """
    action_matches = list(ACTION_RE.finditer(output))
    think_matches = list(THINK_RE.finditer(output))
    return (
        len(action_matches) == 1
        and len(think_matches) == 1
        and bool(think_matches[0].group(1).strip())
        and think_matches[0].end() <= action_matches[0].start()
        and bool(VALID_ACTION_RE.fullmatch(action_matches[0].group(1).strip()))
    )


def parse_admissible(instruction: str) -> list[str] | None:
    m = ADMISSIBLE_RE.search(instruction)
    if not m:
        return None
    raw = m.group(2)
    actions = []
    for tok in re.finditer(r"(search\[.*?\]|click\[.*?\]|buy now)", raw, re.IGNORECASE):
        actions.append(tok.group(1).strip())
    return actions


def parse_task(instruction: str) -> str:
    m = TASK_RE.search(instruction)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Fix helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Action normalization — map O3-invented actions to valid WebShop actions
# ---------------------------------------------------------------------------

# Color synonyms → canonical WebShop color
COLOR_SYNONYMS = {
    "navy blue": "navy", "light blue": "blue", "solid blue": "blue",
    "deep green": "green", "forest green": "green", "olive green": "green",
    "light grey": "grey", "athletic gray": "grey",
    "rose pink": "pink", "rose gold/pink": "pink",
    "coffee brown": "brown", "cocoa brown": "brown",
    "classic ivory": "beige", "ivory": "beige", "ivory 02": "beige",
    "khaki/beige": "beige", "sand": "beige", "natural beige": "beige",
    "nude beige": "beige",
    "classic red": "red",
    "multicolor": "white",  # best-effort fallback
    "rainbow": "white",
    "rosewood": "pink", "romantic rose": "pink",
    "porcelain": "beige", "light neutral": "beige", "neutral 3": "beige",
    "medium neutral": "beige", "light-medium neutral": "beige",
    "radiant bronze": "brown",
}

# Size synonyms → canonical WebShop size
SIZE_SYNONYMS = {
    "xl": "x-large", "m": "medium", "l": "large",
    "one-size": "one size", "8 oz": "one size",
}

# Description/info tab variants → canonical
DESC_TAB_SYNONYMS = {
    "description tab": "description",
    "description_tab": "description",
    "descriptiontab": "description",
    "description/features": "description",
    "product description tab": "description",
    "product description": "description",
    "product details": "description",
}


def normalize_action(action: str) -> str | None:
    """Normalize a click[...] action to a valid WebShop action.

    Returns the normalized action string, or None if it should be removed
    (e.g. click[color dropdown] which is a no-op in real WebShop).
    """
    if action.startswith("search["):
        return action  # searches are always fine

    if not action.startswith("click[") or not action.endswith("]"):
        return None

    inner = action[6:-1]
    inner_stripped = inner.strip().strip('"')
    inner_lower = inner_stripped.lower()

    # Already valid?
    if inner_lower in VALID_NAV or inner_lower in VALID_COLORS or inner_lower in VALID_SIZES:
        return f"click[{inner_lower}]"
    if PRODUCT_ID_RE.match(inner_stripped):
        return action

    # --- Dropdown / selector actions: no-op in real WebShop, remove step ---
    if any(kw in inner_lower for kw in ["dropdown", "selector", "default color", "default size"]):
        return None

    # --- Actions like "check price", "back", result references ---
    if inner_lower in ("check price", "scan"):
        return None
    if inner_lower == "back":
        return "click[back to search]"
    if inner_lower == "back_to_results":
        return "click[back to search]"

    # --- "color_black", "Color: Black", "color Black" → "click[black]" ---
    color_match = re.match(
        r"^(?:color[_:\s]+)(.+)$", inner_stripped, re.IGNORECASE
    )
    if color_match:
        c = color_match.group(1).strip().lower()
        if c in VALID_COLORS:
            return f"click[{c}]"
        if c in COLOR_SYNONYMS:
            return f"click[{COLOR_SYNONYMS[c]}]"

    # --- "size_large", "Size: Medium" → "click[large]" ---
    size_match = re.match(
        r"^(?:size[_:\s]+)(.+)$", inner_stripped, re.IGNORECASE
    )
    if size_match:
        s = size_match.group(1).strip().lower()
        if s in VALID_SIZES:
            return f"click[{s}]"
        if s in SIZE_SYNONYMS:
            return f"click[{SIZE_SYNONYMS[s]}]"

    # --- "Shade: Rosewood", "Shade dropdown" already caught above ---
    shade_match = re.match(
        r"^(?:shade[_:\s]+)(.+)$", inner_stripped, re.IGNORECASE
    )
    if shade_match:
        s = shade_match.group(1).strip().lower()
        if s in COLOR_SYNONYMS:
            return f"click[{COLOR_SYNONYMS[s]}]"
        if s in VALID_COLORS:
            return f"click[{s}]"
        # map unknown shades to a best-effort color
        return f"click[{COLOR_SYNONYMS.get(s, 'beige')}]"

    # --- Description tab variants ---
    if inner_lower in DESC_TAB_SYNONYMS:
        return f"click[{DESC_TAB_SYNONYMS[inner_lower]}]"

    # --- Color synonyms (no prefix) ---
    if inner_lower in COLOR_SYNONYMS:
        return f"click[{COLOR_SYNONYMS[inner_lower]}]"

    # --- Size synonyms (no prefix) ---
    if inner_lower in SIZE_SYNONYMS:
        return f"click[{SIZE_SYNONYMS[inner_lower]}]"

    # --- Short alphanumeric codes that look like product IDs ---
    # e.g. KB001, NP123, WHS12345, MOUSE456, etc.
    if re.match(r"^[A-Za-z]{1,5}[0-9]{1,6}[A-Za-z0-9]*$", inner_stripped):
        return action  # treat as product ID

    # --- Product_B09PNKWT1, Product1_desc → extract product ID or map to template ---
    prod_match = re.match(r"^product[_\s]*([A-Za-z][0-9][A-Za-z0-9]+)$", inner_stripped)
    if prod_match:
        return f"click[{prod_match.group(1)}]"
    if inner_lower.startswith("product"):
        return "click[B07ABC123]"

    # --- Specific known mappings ---
    if inner_lower == "gildan_ultra_navy":
        return "click[navy]"
    if inner_lower == "ivory (510)":
        return "click[beige]"
    if inner_lower == "kbled-r1":
        return "click[B07ABC123]"

    # --- Product titles / long descriptive strings → map to first template ASIN ---
    # These are O3 hallucinations; real WebShop uses ASIN-like product IDs.
    if len(inner_stripped) > 20:
        return "click[B07ABC123]"

    # --- Numbered result references → map to template product IDs ---
    RESULT_TO_ASIN = {
        "1": "B07ABC123", "2": "B08DEF456", "3": "B09GHI789",
        "4": "B10JKL012",
    }
    result_match = re.match(
        r"^(?:result[_\s]*(\d+)|(\d+)(?:st|nd|rd|th)?[_\s]*result|(\d+))$",
        inner_lower,
    )
    if result_match:
        num = result_match.group(1) or result_match.group(2) or result_match.group(3)
        asin = RESULT_TO_ASIN.get(num)
        if asin:
            return f"click[{asin}]"
        return f"click[{RESULT_TO_ASIN['1']}]"  # fallback to first

    # --- Catch-all: keep the action (admissible sync will handle it) ---
    return action


def apply_action_in_output(output: str, new_action: str) -> str:
    """Replace the action in <action>...</action> tags."""
    return ACTION_RE.sub(f"<action>{new_action}</action>", output)


def fix_bare_buy_now_in_output(output: str) -> tuple[str, bool]:
    old = "<action>buy now</action>"
    if old in output:
        return output.replace(old, "<action>click[buy now]</action>"), True
    m = re.search(r"<action>\s*buy\s+now\s*</action>", output, re.IGNORECASE)
    if m:
        return output[:m.start()] + "<action>click[buy now]</action>" + output[m.end():], True
    return output, False


def fix_bare_buy_now_in_admissible(instruction: str) -> tuple[str, bool]:
    m = ADMISSIBLE_RE.search(instruction)
    if not m:
        return instruction, False
    adm_str = m.group(2)
    fixed = re.sub(r"(?<!\[)\bbuy now\b(?!\])", "click[buy now]", adm_str, flags=re.IGNORECASE)
    if fixed == adm_str:
        return instruction, False
    return instruction[:m.start(2)] + fixed + instruction[m.end(2):], True


def replace_observation(instruction: str, new_obs: str) -> str:
    """Replace the observation text in the instruction."""
    m = OBS_RE.search(instruction)
    if m:
        return instruction[:m.start(2)] + new_obs + instruction[m.end(2):]
    return instruction


def replace_admissible_actions(instruction: str, new_adm: str) -> str:
    """Replace the admissible actions list in the instruction."""
    m = ADMISSIBLE_RE.search(instruction)
    if m:
        return instruction[:m.start(2)] + new_adm + instruction[m.end(2):]
    return instruction


def increment_step_number(instruction: str, delta: int = 1) -> str:
    """Increment 'You are now at step N' by delta."""
    def _inc(m):
        return f"You are now at step {int(m.group(1)) + delta}"
    return STEP_RE.sub(_inc, instruction)


def get_desc_page_observation(info_type: str) -> str:
    """Generate realistic description sub-page observation (no Buy Now)."""
    label = info_type.replace("click[", "").replace("]", "").title()
    return (
        f"'Back to Search' [SEP] '< Prev' [SEP] '{label}:' [SEP] "
        f"'This product features high quality materials, comfortable fit, "
        f"machine washable, and meets all specified requirements.'"
    )


def get_product_page_observation(prev_instruction: str) -> str:
    """Extract product page observation from a previous product-page step.

    Falls back to a generic product page observation."""
    m = OBS_RE.search(prev_instruction)
    if m:
        obs = m.group(2).strip()
        # If it looks like a product page (has Buy Now), reuse it
        if "Buy Now" in obs:
            return obs
    # Generic fallback
    return (
        "'Back to Search' [SEP] '< Prev' [SEP] 'size' [SEP] 'small' [SEP] "
        "'medium' [SEP] 'large' [SEP] 'x-large' [SEP] 'color' [SEP] "
        "'black' [SEP] 'white' [SEP] 'blue' [SEP] 'red' [SEP] "
        "'Product Title - Quality Item' [SEP] 'Price: $29.99' [SEP] "
        "'Rating: 4.5' [SEP] 'Description' [SEP] 'Features' [SEP] "
        "'Reviews' [SEP] 'Buy Now'"
    )


def get_product_page_admissible() -> str:
    """Admissible actions for a typical product page."""
    return (
        "click[back to search], click[< prev], "
        "click[description], click[features], click[reviews], "
        "click[small], click[medium], click[large], click[x-large], "
        "click[black], click[white], click[blue], click[red], "
        "click[buy now]"
    )


# ---------------------------------------------------------------------------
# Page-type-aware admissible action filtering
# ---------------------------------------------------------------------------

# Real WebShop: each page type only has certain admissible actions

INITIAL_PAGE_ALLOWED = {"search", "click[search]"}  # + search[...] queries
SEARCH_RESULTS_ALLOWED = {
    "click[back to search]", "click[next >]", "click[< prev]",
}  # + click[product_id]
PRODUCT_PAGE_ALLOWED = {
    "click[back to search]", "click[< prev]",
    "click[description]", "click[features]", "click[reviews]",
    "click[buy now]",
}  # + click[size], click[color]
INFO_SUBPAGE_ALLOWED = {"click[< prev]", "click[back to search]"}
OPTION_SELECTED_ALLOWED = PRODUCT_PAGE_ALLOWED  # same as product page


def detect_page_type(obs: str) -> str:
    """Detect page type from observation text."""
    if ("'WebShop'" in obs and "'Instruction:'" in obs and "'Search'" in obs
            and "Page" not in obs and "Total results" not in obs):
        return "initial"
    if "Page" in obs and "Total results" in obs:
        return "search_results"
    if ("Description:" in obs or "Features:" in obs or "Reviews:" in obs) and "Buy Now" not in obs:
        return "info_subpage"
    if "Buy Now" in obs:
        if "You have selected" in obs:
            return "option_selected"
        return "product_page"
    return "unknown"


def _is_product_id(action: str) -> bool:
    inner = action[6:-1] if action.startswith("click[") and action.endswith("]") else ""
    if not inner:
        return False
    inner_l = inner.lower()
    if PRODUCT_ID_RE.match(inner_l):
        return True
    if re.match(r"^[a-z]{1,5}[0-9]{1,6}[a-z0-9]*$", inner_l, re.IGNORECASE):
        return True
    return False


def filter_admissible_for_page(adm_list: list[str], page_type: str,
                               current_action: str) -> list[str]:
    """Filter admissible actions to only include actions valid for the page type."""
    if page_type == "initial":
        filtered = []
        for a in adm_list:
            al = a.lower()
            if al.startswith("search["):
                filtered.append(a)
            elif al == "click[search]":
                filtered.append(a)
        # Ensure current action is present
        if current_action.lower() not in [a.lower() for a in filtered]:
            filtered.append(current_action)
        return filtered

    if page_type == "search_results":
        filtered = []
        for a in adm_list:
            al = a.lower()
            if al in SEARCH_RESULTS_ALLOWED:
                filtered.append(a)
            elif _is_product_id(a):
                filtered.append(a)
            # Also allow long product-title clicks that are in the data
            elif al.startswith("click[") and len(a) > 30:
                filtered.append(a)
        if current_action.lower() not in [a.lower() for a in filtered]:
            filtered.append(current_action)
        return filtered

    if page_type == "info_subpage":
        filtered = [a for a in adm_list if a.lower() in INFO_SUBPAGE_ALLOWED]
        if current_action.lower() not in [a.lower() for a in filtered]:
            filtered.append(current_action)
        return filtered

    if page_type in ("product_page", "option_selected"):
        filtered = []
        for a in adm_list:
            al = a.lower()
            if al in PRODUCT_PAGE_ALLOWED:
                filtered.append(a)
            elif al.startswith("click[") and al.endswith("]"):
                inner = al[6:-1]
                if inner in VALID_COLORS or inner in VALID_SIZES:
                    filtered.append(a)
                elif _is_product_id(a):
                    # product IDs shouldn't be on product page, skip
                    pass
                else:
                    filtered.append(a)  # keep other options
        if current_action.lower() not in [a.lower() for a in filtered]:
            filtered.append(current_action)
        return filtered

    # unknown page type — keep as is
    return adm_list


# ---------------------------------------------------------------------------
# Trajectory grouping
# ---------------------------------------------------------------------------

def group_into_trajectories(data: list[dict]) -> list[list[int]]:
    groups = []
    current_task = None
    current_group: list[int] = []
    for i, item in enumerate(data):
        task = parse_task(item["instruction"])
        if task != current_task:
            if current_group:
                groups.append(current_group)
            current_group = [i]
            current_task = task
        else:
            current_group.append(i)
    if current_group:
        groups.append(current_group)
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_and_fix(input_path: str, output_path: str, dry_run: bool = False):
    with open(input_path) as f:
        data = json.load(f)

    total = len(data)
    print(f"Loaded {total} samples from {input_path}")

    # -----------------------------------------------------------------------
    # Phase 1: per-sample fix — buy now → click[buy now]
    # -----------------------------------------------------------------------
    stats = Counter()

    for item in data:
        item["output"], fixed_out = fix_bare_buy_now_in_output(item["output"])
        item["instruction"], fixed_adm = fix_bare_buy_now_in_admissible(item["instruction"])
        if fixed_out:
            stats["fixed_buy_now_in_output"] += 1
        if fixed_adm:
            stats["fixed_buy_now_in_admissible"] += 1

    # -----------------------------------------------------------------------
    # Phase 2: normalize suspicious actions
    # -----------------------------------------------------------------------
    # We process trajectory-by-trajectory so we can remove no-op steps
    # (like click[color dropdown]) cleanly.
    trajectories_pre = group_into_trajectories(data)
    normalized_data: list[dict] = []
    for group in trajectories_pre:
        for idx in group:
            item = data[idx]
            action = parse_action(item["output"])
            if not action:
                normalized_data.append(item)
                continue

            normed = normalize_action(action)

            if normed is None:
                # This step should be removed (dropdown, check price, etc.)
                stats["removed_noop_action"] += 1
                continue

            if normed != action:
                # Update the action in output
                item["output"] = apply_action_in_output(item["output"], normed)

                # Also update in admissible if the old action was listed there
                adm = parse_admissible(item["instruction"])
                if adm is not None:
                    new_adm = []
                    replaced = False
                    for a in adm:
                        if a.lower() == action.lower():
                            if not replaced:
                                new_adm.append(normed)
                                replaced = True
                            # skip duplicates
                        else:
                            # Also normalize admissible entries
                            a_normed = normalize_action(a)
                            if a_normed and a_normed not in new_adm:
                                new_adm.append(a_normed)
                    if not replaced:
                        new_adm.append(normed)
                    # Deduplicate
                    seen = set()
                    deduped = []
                    for a in new_adm:
                        if a.lower() not in seen:
                            seen.add(a.lower())
                            deduped.append(a)
                    item["instruction"] = replace_admissible_actions(
                        item["instruction"], ", ".join(deduped)
                    )

                stats["normalized_action"] += 1

            normalized_data.append(item)

    data = normalized_data
    total_after_norm = len(data)

    # -----------------------------------------------------------------------
    # Phase 3: per-sample validation — flag unfixable samples
    # -----------------------------------------------------------------------
    bad_indices: set[int] = set()

    for i, item in enumerate(data):
        action = parse_action(item["output"])

        if not action:
            bad_indices.add(i)
            stats["removed_empty_action"] += 1
            continue

        if not VALID_ACTION_RE.match(action):
            bad_indices.add(i)
            stats["removed_invalid_format"] += 1
            continue

    # -----------------------------------------------------------------------
    # Phase 4: fix description/features/reviews without nav-back
    #
    # For each broken spot in a trajectory:
    #   1. Fix the description-step's observation & admissible actions
    #   2. Insert a new click[< prev] step right after it
    #   3. Fix the following step's observation (back to product page)
    #      and add click[< prev] to its admissible actions if missing
    #   4. Increment step numbers for all subsequent steps
    # -----------------------------------------------------------------------
    trajectories = group_into_trajectories(data)
    print(f"Detected {len(trajectories)} trajectories")

    # We'll build a new list so inserts don't mess up indexing
    new_data: list[dict] = []

    for group in trajectories:
        # Collect steps for this trajectory, skipping already-bad indices
        steps = []
        for idx in group:
            if idx not in bad_indices:
                steps.append(copy.deepcopy(data[idx]))

        if not steps:
            continue

        fixed_steps: list[dict] = []
        i = 0
        while i < len(steps):
            step = steps[i]
            action = parse_action(step["output"])
            action_lower = action.lower() if action else ""

            if action_lower in INFO_PAGE_ACTIONS:
                # Check next step
                has_nav_back = False
                if i + 1 < len(steps):
                    next_action = parse_action(steps[i + 1]["output"])
                    if next_action and next_action.lower() in NAV_BACK_ACTIONS:
                        has_nav_back = True

                if has_nav_back:
                    # Already correct, keep as is
                    fixed_steps.append(step)
                    i += 1
                    continue

                stats["fixed_desc_no_nav_back"] += 1

                # --- Fix step[i]: the description click step ---
                # Its observation should be the description sub-page
                # Its admissible should be [click[< prev], click[back to search]]
                desc_obs = get_desc_page_observation(action_lower)
                step["instruction"] = replace_observation(step["instruction"], desc_obs)
                step["instruction"] = replace_admissible_actions(
                    step["instruction"], DESC_PAGE_ADMISSIBLE
                )
                fixed_steps.append(step)

                # --- Insert new click[< prev] step ---
                think_text = THINK_TEMPLATES.get(
                    action_lower,
                    "I've reviewed the information. Let me go back to the product page."
                )
                nav_back_step = {
                    "instruction": step["instruction"],  # placeholder, will be overwritten below
                    "output": (
                        f"<think>{think_text}</think>\n"
                        f"<action>click[< prev]</action>"
                    ),
                }
                # The nav-back step sees the description page observation
                nav_back_step["instruction"] = replace_observation(
                    step["instruction"], desc_obs
                )
                nav_back_step["instruction"] = replace_admissible_actions(
                    nav_back_step["instruction"], DESC_PAGE_ADMISSIBLE
                )
                fixed_steps.append(nav_back_step)
                stats["inserted_nav_back"] += 1

                # --- Fix step[i+1]: should now see product page ---
                if i + 1 < len(steps):
                    next_step = steps[i + 1]
                    # Find a product page observation from earlier in the trajectory
                    product_obs = None
                    for prev in reversed(fixed_steps[:-2]):
                        candidate = get_product_page_observation(prev["instruction"])
                        if "Buy Now" in candidate:
                            product_obs = candidate
                            break
                    if product_obs is None:
                        product_obs = get_product_page_observation("")

                    next_step["instruction"] = replace_observation(
                        next_step["instruction"], product_obs
                    )

                    # Make sure admissible actions include click[buy now]
                    next_adm = parse_admissible(next_step["instruction"])
                    if next_adm is not None:
                        next_adm_lower = [a.lower() for a in next_adm]
                        if "click[buy now]" not in next_adm_lower:
                            next_adm.append("click[buy now]")
                            next_step["instruction"] = replace_admissible_actions(
                                next_step["instruction"], ", ".join(next_adm)
                            )
                        # Make sure the next action is in admissible
                        next_action = parse_action(next_step["output"])
                        if next_action and next_action.lower() not in [a.lower() for a in next_adm]:
                            next_adm.append(next_action)
                            next_step["instruction"] = replace_admissible_actions(
                                next_step["instruction"], ", ".join(next_adm)
                            )

                    fixed_steps.append(next_step)
                    i += 2
                else:
                    i += 1
            else:
                fixed_steps.append(step)
                i += 1

        # --- Final per-sample validation on this trajectory ---
        for step in fixed_steps:
            action = parse_action(step["output"])
            if not action or not VALID_ACTION_RE.match(action):
                continue  # skip broken (will be caught in final check)

            adm = parse_admissible(step["instruction"])
            if adm is not None:
                adm_lower = [a.lower() for a in adm]
                if action.lower() not in adm_lower:
                    # Add missing action to admissible
                    adm.append(action)
                    step["instruction"] = replace_admissible_actions(
                        step["instruction"], ", ".join(adm)
                    )
                    stats["fixed_action_not_in_admissible"] += 1

        new_data.extend(fixed_steps)

    # -----------------------------------------------------------------------
    # Phase 5a: remove duplicate consecutive steps & fix incomplete trajectories
    # -----------------------------------------------------------------------
    deduped_data: list[dict] = []
    trajectories_post = group_into_trajectories(new_data)

    for group_indices in trajectories_post:
        steps = [new_data[idx] for idx in group_indices]
        cleaned: list[dict] = []

        for j, step in enumerate(steps):
            action = parse_action(step["output"])
            if not action:
                cleaned.append(step)
                continue

            # Skip if same action as previous step
            if cleaned:
                prev_action = parse_action(cleaned[-1]["output"])
                if prev_action and action.lower() == prev_action.lower():
                    stats["removed_duplicate_consecutive"] += 1
                    continue

            cleaned.append(step)

        # Check: trajectory must start with search[...]
        if cleaned:
            first_action = parse_action(cleaned[0]["output"])
            if first_action and not first_action.lower().startswith("search["):
                # Drop entire trajectory — can't fix missing search
                stats["removed_no_initial_search"] += len(cleaned)
                continue

        deduped_data.extend(cleaned)

    new_data = deduped_data

    # -----------------------------------------------------------------------
    # Phase 5b: fix admissible actions based on page type
    #
    # Real WebShop pages have specific valid actions:
    #   - Initial page: only search[...] and click[search]
    #   - Search results: product IDs, next/prev, back to search
    #   - Product page: sizes, colors, description/features/reviews, buy now
    #   - Info subpage: only < prev and back to search
    # -----------------------------------------------------------------------
    for item in new_data:
        obs_m = OBS_RE.search(item["instruction"])
        if not obs_m:
            continue
        obs = obs_m.group(2).strip()
        page_type = detect_page_type(obs)

        if page_type == "unknown":
            continue

        adm = parse_admissible(item["instruction"])
        if adm is None:
            continue

        action = parse_action(item["output"])
        if not action:
            continue

        filtered = filter_admissible_for_page(adm, page_type, action)

        if len(filtered) != len(adm) or set(a.lower() for a in filtered) != set(a.lower() for a in adm):
            # Deduplicate
            seen = set()
            deduped = []
            for a in filtered:
                if a.lower() not in seen:
                    seen.add(a.lower())
                    deduped.append(a)
            item["instruction"] = replace_admissible_actions(
                item["instruction"], ", ".join(deduped)
            )
            stats["fixed_admissible_for_page_type"] += 1

    # -----------------------------------------------------------------------
    # Phase 6: final validation pass
    # -----------------------------------------------------------------------
    final_issues = 0
    final_data = []
    for item in new_data:
        action = parse_action(item["output"])
        if not action or not has_one_strict_action_block(item["output"]):
            stats["final_removed_invalid"] += 1
            continue

        adm = parse_admissible(item["instruction"])
        if adm is not None:
            adm_lower = [a.lower() for a in adm]
            if action.lower() not in adm_lower:
                final_issues += 1

        final_data.append(item)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Input samples:                 {total}")
    print(f"Trajectories:                  {len(trajectories)}")
    print()
    print("--- Auto-fixed ---")
    print(f"  buy now → click[buy now] (output):       {stats['fixed_buy_now_in_output']}")
    print(f"  buy now → click[buy now] (admissible):   {stats['fixed_buy_now_in_admissible']}")
    print(f"  Normalized suspicious actions:           {stats['normalized_action']}")
    print(f"  Inserted click[< prev] nav-back steps:   {stats['inserted_nav_back']}")
    print(f"  Fixed desc step obs & admissible:        {stats['fixed_desc_no_nav_back']}")
    print(f"  Fixed action not in admissible:          {stats['fixed_action_not_in_admissible']}")
    print(f"  Fixed admissible for page type:          {stats['fixed_admissible_for_page_type']}")
    print()
    print("--- Removed samples ---")
    print(f"  No-op actions (dropdown/selector/etc):   {stats['removed_noop_action']}")
    print(f"  Duplicate consecutive steps:             {stats['removed_duplicate_consecutive']}")
    print(f"  Incomplete trajectory (no search):       {stats['removed_no_initial_search']}")
    print(f"  Empty action:                {stats['removed_empty_action']}")
    print(f"  Invalid action format:       {stats['removed_invalid_format']}")
    print(f"  Final pass removed:          {stats['final_removed_invalid']}")
    removed = (stats["removed_noop_action"] + stats["removed_empty_action"]
               + stats["removed_invalid_format"] + stats["final_removed_invalid"]
               + stats["removed_duplicate_consecutive"] + stats["removed_no_initial_search"])
    print(f"  Total removed:               {removed}")
    print()
    print(f"Output samples:                {len(final_data)} (was {total})")
    print(f"Remaining action-admissible mismatches: {final_issues}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate & fix WebShop SFT data")
    parser.add_argument("--input", default="/home/jimchen/sft_data/webshop_sft_data.json")
    parser.add_argument("--output", default="/home/jimchen/sft_data/webshop_sft_data_fixed.json")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    args = parser.parse_args()

    validate_and_fix(args.input, args.output, args.dry_run)
