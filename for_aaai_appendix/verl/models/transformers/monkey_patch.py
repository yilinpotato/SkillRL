# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import sys
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, sequence_length, key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch,
        sequence_length,
        key_value_heads,
        n_rep,
        head_dim,
    )
    return hidden_states.reshape(
        batch,
        sequence_length,
        key_value_heads * n_rep,
        head_dim,
    )


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if sequence_parallel_size > 1:
        if position_ids is None:
            raise ValueError("position_ids is required for sequence parallelism")
        repeats = max(sequence_parallel_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)
        query_states = gather_seq_scatter_heads(
            query_states,
            seq_dim=1,
            head_dim=2,
        )
        key_states = gather_seq_scatter_heads(
            key_states,
            seq_dim=1,
            head_dim=2,
        )
        value_states = gather_seq_scatter_heads(
            value_states,
            seq_dim=1,
            head_dim=2,
        )
        gathered_positions = [
            torch.empty_like(position_ids)
            for _ in range(sequence_parallel_size)
        ]
        torch.distributed.all_gather(
            gathered_positions,
            position_ids,
            group=get_ulysses_sequence_parallel_group(),
        )
        position_ids = torch.concat(gathered_positions, dim=-1)
    query_length = query_states.size(1)
    attention_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        *args,
        position_ids=position_ids,
        **kwargs,
    )
    if sequence_parallel_size > 1:
        attention_output = gather_heads_scatter_seq(
            attention_output,
            seq_dim=1,
            head_dim=2,
        )
    return attention_output


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    if use_fused_kernels:
        raise ValueError("Fused model-specific kernels are not included")
    if not use_remove_padding and ulysses_sp_size == 1:
        return
    module = sys.modules[model.__module__]
    if hasattr(module, "_flash_attention_forward"):
        module._flash_attention_forward = _ulysses_flash_attention_forward
    else:
        from transformers.integrations import flash_attention

        flash_attention._flash_attention_forward = (
            _ulysses_flash_attention_forward
        )
