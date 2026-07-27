from types import SimpleNamespace

import numpy as np
import torch

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto


class _Tokenizer:
    def batch_decode(self, responses, skip_special_tokens=False):
        return ["<think>x</think><action>go</action>" for _ in range(len(responses))]


class _Collector(TrajectoryCollector):
    """Avoid tokenizer/chat-template dependencies in this rollout unit test."""

    def preprocess_batch(self, gen_batch, obs):
        n = len(gen_batch)
        return DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((n, 2), dtype=torch.long),
                "attention_mask": torch.ones((n, 2), dtype=torch.long),
                "position_ids": torch.arange(2).repeat(n, 1),
            },
            non_tensors={"raw_prompt_ids": [[] for _ in range(n)]},
        )


class _Actor:
    world_size = 2

    def __init__(self):
        self.request_sizes = []

    def generate_sequences(self, batch):
        self.request_sizes.append(len(batch))
        n = len(batch)
        # A real vLLM worker returns its generated responses plus prompt data.
        return DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((n, 2), dtype=torch.long),
                "responses": torch.ones((n, 2), dtype=torch.long),
            }
        )


class _Envs:
    def __init__(self):
        self.step_count = 0
        self.actions = []

    def reset(self, kwargs=None):
        return {"text": ["a", "b", "c"], "anchor": ["a", "b", "c"]}, [{}, {}, {}]

    def step(self, actions):
        self.actions.append(list(actions))
        self.step_count += 1
        dones_by_step = (
            np.array([True, False, False]),
            np.array([True, True, False]),
            np.array([True, True, True]),
        )
        dones = dones_by_step[self.step_count - 1]
        infos = [
            {
                "is_action_valid": True,
                "non_strict_action_valid": True,
                "strict_action_valid": True,
                "tool_calling": 0,
                "won": bool(done),
            }
            for done in dones
        ]
        return (
            {"text": ["a", "b", "c"], "anchor": ["a", "b", "c"]},
            np.ones(3, dtype=np.float32),
            dones,
            infos,
        )

    def success_evaluator(self, **kwargs):
        return {"success": kwargs["episode_rewards"] > 0}


def _config(compact):
    return SimpleNamespace(
        env=SimpleNamespace(
            rollout=SimpleNamespace(n=1),
            max_steps=3,
            env_name="fake",
            compact_finished_trajectories=compact,
        ),
        trainer=SimpleNamespace(default_local_dir="."),
    )


def _gen_batch():
    return DataProto.from_dict(
        tensors={"input_ids": torch.ones((3, 2), dtype=torch.long)},
        non_tensors={"env_kwargs": [None, None, None]},
    )


def test_finished_rows_are_not_regenerated_when_compaction_enabled():
    actor = _Actor()
    envs = _Envs()
    collector = _Collector(_config(compact=True), _Tokenizer())

    batch_list, rewards, lengths, *_ = collector.vanilla_multi_turn_loop(
        _gen_batch(), actor, envs
    )

    # Active rows 3 -> 2 -> 1 are padded only to the two DP ranks: 4, 2, 2.
    assert actor.request_sizes == [4, 2, 2]
    assert [len(rows) for rows in batch_list] == [1, 2, 3]
    assert rewards.tolist() == [1.0, 2.0, 3.0]
    assert lengths.tolist() == [1.0, 2.0, 3.0]
    assert all(len(actions) == 3 for actions in envs.actions)
    metrics = collector.get_last_rollout_compaction_metrics()
    assert {
        key: metrics[key]
        for key in (
            "rollout/active_compaction_enabled",
            "rollout/active_generation_rounds",
            "rollout/vllm_full_batch_rows_legacy",
            "rollout/vllm_active_rows",
            "rollout/vllm_request_rows",
            "rollout/vllm_dp_padding_rows",
            "rollout/vllm_rows_avoided",
            "rollout/active_rows_per_round_mean",
            "rollout/active_rows_per_round_min",
            "rollout/active_rows_per_round_max",
            "rollout/dp_underfilled_rounds",
        )
    } == {
        "rollout/active_compaction_enabled": 1,
        "rollout/active_generation_rounds": 3,
        "rollout/vllm_full_batch_rows_legacy": 9,
        "rollout/vllm_active_rows": 6,
        "rollout/vllm_request_rows": 8,
        "rollout/vllm_dp_padding_rows": 2,
        "rollout/vllm_rows_avoided": 1,
        "rollout/active_rows_per_round_mean": 2.0,
        "rollout/active_rows_per_round_min": 1,
        "rollout/active_rows_per_round_max": 3,
        "rollout/dp_underfilled_rounds": 1,
    }
    assert metrics["timing_s/rollout_collector"] >= 0
    assert metrics["timing_s/rollout_vllm_generate"] >= 0


def test_legacy_switch_keeps_full_generation_batches():
    actor = _Actor()
    envs = _Envs()
    collector = _Collector(_config(compact=False), _Tokenizer())

    batch_list, rewards, lengths, *_ = collector.vanilla_multi_turn_loop(
        _gen_batch(), actor, envs
    )

    assert actor.request_sizes == [4, 4, 4]
    # Legacy finished rows are retained internally but filtered before PPO.
    assert [sum(bool(row["active_masks"]) for row in rows) for rows in batch_list] == [1, 2, 3]
    assert rewards.tolist() == [1.0, 2.0, 3.0]
    assert lengths.tolist() == [1.0, 2.0, 3.0]
