import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto


class _BatchTokenizer:
    pad_token_id = 0

    @staticmethod
    def _render(chat):
        return f"prompt::{chat[0]['content']}::assistant"

    def apply_chat_template(
        self, chats, *, add_generation_prompt, tokenize, **kwargs
    ):
        assert add_generation_prompt is True
        assert tokenize is False
        if chats and isinstance(chats[0], dict):
            return self._render(chats)
        return [self._render(chat) for chat in chats]

    @staticmethod
    def _encode(text):
        return [ord(char) % 97 + 1 for char in text]

    def __call__(
        self,
        prompts,
        *,
        return_tensors=None,
        add_special_tokens,
        padding=False,
        truncation=False,
    ):
        del padding, truncation
        assert add_special_tokens is False
        if isinstance(prompts, str):
            rows = [self._encode(prompts)]
        else:
            rows = [self._encode(prompt) for prompt in prompts]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
            }
        return {"input_ids": rows}

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return self._encode(prompt)


def _config(*, batch_tokenize):
    return OmegaConf.create(
        {
            "env": {
                "dump_raw_trajectories": False,
                "compact_finished_trajectories": True,
                "batch_tokenize_observations": batch_tokenize,
            },
            "trainer": {"default_local_dir": "."},
            "data": {
            "max_prompt_length": 64,
            "truncation": "left",
            "return_raw_chat": True,
            "apply_chat_template_kwargs": {},
            },
        },
    )


def _gen_batch():
    return DataProto.from_dict(
        tensors={"input_ids": torch.ones((2, 2), dtype=torch.long)},
        non_tensors={
            "raw_prompt": np.array([[{"content": "old", "role": "user"}]] * 2),
            "data_source": np.array(["a", "b"], dtype=object),
        },
    )


def test_batch_text_tokenization_matches_the_historical_row_path():
    obs = {
        "text": ["room one", "room two"],
        "anchor": ["anchor one", "anchor two"],
    }
    tokenizer = _BatchTokenizer()
    batched = TrajectoryCollector(
        _config(batch_tokenize=True), tokenizer
    ).preprocess_batch(_gen_batch(), obs)
    rowwise = TrajectoryCollector(
        _config(batch_tokenize=False), tokenizer
    ).preprocess_batch(_gen_batch(), obs)

    assert set(batched.batch.keys()) == set(rowwise.batch.keys())
    for key, value in batched.batch.items():
        assert torch.equal(value, rowwise.batch[key])
    assert set(batched.non_tensor_batch) == set(rowwise.non_tensor_batch)
    for key in batched.non_tensor_batch:
        assert batched.non_tensor_batch[key].tolist() == rowwise.non_tensor_batch[
            key
        ].tolist()
