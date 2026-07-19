"""Regression tests for CoSkill WebShop prompt/protocol compatibility."""

from agent_system.environments.env_package.webshop.projection import webshop_projection
from agent_system.environments.prompts.webshop import (
    fit_webshop_history_to_char_limit,
)
from mini_test_pen_shelf.webshop_utils import WebShopObsBuilder
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def test_qwen_prompt_prefilled_think_is_validated_as_complete_transcript():
    """The prompt-provided opener must not make a pure continuation invalid."""
    raw = "reason about the query</think><action>search[waterproof speaker]</action>"
    rendered, restored = TrajectoryCollector._render_protocol_response(raw, True)

    actions, valids, details = webshop_projection([rendered], return_details=True)
    assert restored is True
    assert rendered.startswith("<think>")
    assert actions == ["search[waterproof speaker]"]
    assert valids == [1]
    assert details[0]["strict_valid_action"] is True


def test_bare_completion_is_not_granted_prompt_prefill_compatibility():
    raw = "reason about the query</think><action>search[waterproof speaker]</action>"
    rendered, restored = TrajectoryCollector._render_protocol_response(raw, False)

    _actions, valids, details = webshop_projection([rendered], return_details=True)
    assert restored is False
    assert valids == [1]
    assert details[0]["valid_action"] is True
    assert details[0]["strict_valid_action"] is False


def test_history_compaction_keeps_the_newest_complete_pairs():
    records = [
        {"text_obs": "old-observation-" + "a" * 80, "action": "search[old]"},
        {"text_obs": "new-observation-" + "b" * 80, "action": "click[new]"},
    ]

    def render(history, history_length):
        return f"TASK\ncount={history_length}\n{history}\nCURRENT"

    prompt, kept, dropped, static_over_limit = fit_webshop_history_to_char_limit(
        records,
        first_step=4,
        render_prompt=render,
        char_limit=180,
    )
    assert kept == 1
    assert dropped == 1
    assert static_over_limit is False
    assert "Action 5: 'click[new]'" in prompt
    assert "search[old]" not in prompt


def test_static_prompt_is_never_replaced_by_a_no_skill_fallback():
    records = [{"text_obs": "ignored", "action": "ignored"}]

    prompt, kept, dropped, static_over_limit = fit_webshop_history_to_char_limit(
        records,
        first_step=1,
        render_prompt=lambda _history, _count: "RETRIEVED TREE AND CURRENT STATE",
        char_limit=5,
    )
    assert prompt == "RETRIEVED TREE AND CURRENT STATE"
    assert kept == 0
    assert dropped == 1
    assert static_over_limit is True


def test_no_rl_initial_prompt_does_not_duplicate_the_skill_tree():
    class _Memory:
        enable_playbook = True

        @staticmethod
        def retrieve(task_description, top_k):
            del task_description, top_k
            return {
                "playbook": "# Learned Tree",
                "general_skills": [],
                "task_specific_skills": [],
                "mistakes_to_avoid": [],
            }

        @staticmethod
        def format_for_prompt(_retrieved):
            # The real formatter puts the playbook first, too.
            return "# Learned Tree\n\n### General Principles\n- test"

    builder = WebShopObsBuilder(mem_lib=_Memory(), history_length=8)
    builder.reset("find a test product")
    prompt = builder.build(
        "Results: [1] Test product $1.",
        {"has_search_bar": True, "clickables": ["1"]},
        init=True,
    )
    assert prompt.count("# Learned Tree") == 1
