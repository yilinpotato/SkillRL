from types import SimpleNamespace

from mini_test_pen_shelf.agent_vllm import VLLMAgent


class _FakeLLM:
    def __init__(self, texts, finish_reasons):
        self.texts = texts
        self.finish_reasons = finish_reasons
        self.calls = []

    def generate(self, prompts, sampling, use_tqdm=False):
        self.calls.append((list(prompts), sampling, use_tqdm))
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    SimpleNamespace(
                        text=text,
                        token_ids=[4, 5],
                        finish_reason=finish_reason,
                    )
                ],
            )
            for text, finish_reason in zip(self.texts, self.finish_reasons)
        ]


def _bare_agent(texts, finish_reasons):
    agent = VLLMAgent.__new__(VLLMAgent)
    agent.llm = _FakeLLM(texts, finish_reasons)
    agent.sampling = object()
    agent.total_prompt_tokens = 0
    agent.total_response_tokens = 0
    agent._build_prompt = lambda value: f"prompt:{value}<think>"
    agent._single_sampling = lambda **kwargs: ("single", kwargs)
    return agent


def test_batch_with_meta_uses_one_request_and_counts_it_once():
    agent = _bare_agent(
        [
            "reason one</think><action>go north</action>",
            "reason two</think><action>look</action>",
        ],
        ["stop", "length"],
    )

    results = agent.act_batch_with_meta(["first", "second"], temperature=0.4, sampling_seed=17)

    assert len(agent.llm.calls) == 1
    assert agent.llm.calls[0][0] == ["prompt:first<think>", "prompt:second<think>"]
    assert results == [
        ("<think>\nreason one</think><action>go north</action>", False),
        ("<think>\nreason two</think><action>look</action>", True),
    ]
    assert agent.get_token_usage() == {"prompt": 6, "response": 4, "total": 10}


def test_single_with_meta_does_not_fabricate_missing_protocol_tags():
    agent = _bare_agent(["unfinished reasoning"], ["length"])

    text, forced = agent.act_with_meta("observation")

    assert len(agent.llm.calls) == 1
    assert text == "<think>\nunfinished reasoning"
    assert forced is True
    assert "<action>" not in text


def test_context_guard_usage_defaults_and_reports_cumulative_trims():
    agent = VLLMAgent.__new__(VLLMAgent)
    assert agent.get_context_guard_usage() == {
        "prompt_trims": 0,
        "trimmed_tokens": 0,
    }
    agent.context_guard_prompt_trims = 3
    agent.context_guard_trimmed_tokens = 127
    assert agent.get_context_guard_usage() == {
        "prompt_trims": 3,
        "trimmed_tokens": 127,
    }


def test_close_shuts_down_engine_core_once():
    calls = []
    agent = VLLMAgent.__new__(VLLMAgent)
    agent.llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: calls.append("shutdown"))))

    agent.close()
    agent.close()

    assert calls == ["shutdown"]
