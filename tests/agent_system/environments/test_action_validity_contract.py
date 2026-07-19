"""Regression tests for CoSkill's strict/non-strict action accounting."""

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.env_package.webshop.projection import webshop_projection
from verl.trainer.ppo.metric_utils import compute_action_validity_metrics


def test_alfworld_penalty_validity_matches_original_skillrl_protocol():
    """A formatted but non-admissible action gets no extra CoSkill penalty."""
    raw = "reasoning</think><action>invent an action</action>"
    projected, valids, details = alfworld_projection(
        [raw], [["look", "go to drawer 1"]], return_details=True
    )

    assert projected == ["look"]
    assert valids == [1]
    assert details[0]["valid_action"] is True
    assert details[0]["strict_valid_action"] is False
    assert details[0]["direct_admissible_action"] is False


def test_alfworld_qwen_prompt_prefilled_think_is_non_strict_but_not_strict():
    raw = "reasoning</think><action>go to drawer 1</action>"
    _projected, valids, details = alfworld_projection(
        [raw], [["look", "go to drawer 1"]], return_details=True
    )

    assert valids == [1]
    assert details[0]["valid_action"] is True
    assert details[0]["strict_valid_action"] is False


def test_alfworld_plain_admissible_action_is_salvaged_without_format_penalty():
    projected, valids, details = alfworld_projection(
        ["I will now go to drawer 1"], [["look", "go to drawer 1"]],
        return_details=True,
    )

    assert projected == ["go to drawer 1"]
    assert valids == [1]
    assert details[0]["execution_source"] == "salvaged"
    assert details[0]["strict_valid_action"] is False


def test_alfworld_strict_diagnostic_requires_full_protocol_and_direct_action():
    raw = "<think>reasoning</think><action>go to drawer 1</action>"
    projected, valids, details = alfworld_projection(
        [raw], [["look", "go to drawer 1"]], return_details=True
    )

    assert projected == ["go to drawer 1"]
    assert valids == [1]
    assert details[0]["strict_valid_action"] is True


def test_webshop_uses_non_strict_penalty_and_keeps_strict_diagnostic_rate():
    _projected, valids, details = webshop_projection(
        ["<action>search[speaker]</action>"], return_details=True
    )

    assert valids == [1]
    assert details[0]["valid_action"] is True
    assert details[0]["strict_valid_action"] is False


def test_webshop_salvages_plain_or_unclosed_action_and_uses_safe_fallback():
    projected, valids, details = webshop_projection([
        "Action: CLICK[Back to Search]",
        "reasoning</think><action>search[waterproof speaker]",
        "unfinished reasoning only",
    ], return_details=True)

    assert projected == ["click[back to search]", "search[waterproof speaker]", "search[]"]
    assert valids == [1, 1, 0]
    assert [row["execution_source"] for row in details] == [
        "salvaged", "salvaged", "fallback"]


def test_validity_metrics_keep_penalty_and_both_diagnostics_distinct():
    metrics = compute_action_validity_metrics(
        [1, 0, 1, 0],
        strict_action_valid=[1, 0, 0, 0],
        non_strict_action_valid=[1, 1, 1, 0],
    )

    assert metrics == {
        "episode/valid_action_ratio": 0.5,
        "episode/strict_valid_action_ratio": 0.25,
        "episode/non_strict_valid_action_ratio": 0.75,
        "episode/relaxed_valid_action_ratio": 0.75,
    }
