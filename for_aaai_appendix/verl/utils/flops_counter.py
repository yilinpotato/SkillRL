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

import torch
from transformers import PretrainedConfig

from verl.utils.device import get_torch_device


def get_device_flops(unit="T"):
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        pointer = 0
        while pointer < len(units) and units[pointer] != level:
            number /= 1000
            pointer += 1
        return number

    device = get_torch_device()
    device_name = "CPU" if device == torch.cpu else device.get_device_name()
    flops = float("inf")
    if "CPU" in device_name:
        flops = 448e9
    elif "GB200" in device_name:
        flops = 2.5e15
    elif "B200" in device_name:
        flops = 2.25e15
    elif "MI300X" in device_name:
        flops = 1336e12
    elif any(name in device_name for name in ("H100", "H800", "H200")):
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40S" in device_name:
        flops = 362.05e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "A40" in device_name:
        flops = 149.7e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name or "Ascend910" in device_name:
        flops = 354e12
    elif "RTX 3070 Ti" in device_name:
        flops = 21.75e12
    return unit_convert(flops, unit)


class FlopsCounter:
    def __init__(self, config: PretrainedConfig):
        self.config = getattr(config, "text_config", config)

    def _estimate_dense_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = getattr(
            self.config,
            "head_dim",
            hidden_size // num_attention_heads,
        )
        query_size = num_attention_heads * head_dim
        key_size = num_key_value_heads * head_dim
        value_size = num_key_value_heads * head_dim
        mlp_parameters = hidden_size * intermediate_size * 3
        attention_parameters = hidden_size * (
            query_size
            + key_size
            + value_size
            + num_attention_heads * head_dim
        )
        embedding_parameters = vocab_size * hidden_size * 2
        dense_parameters = (
            (mlp_parameters + attention_parameters) * num_hidden_layers
            + embedding_parameters
        )
        dense_flops = 6 * dense_parameters * tokens_sum
        sequence_square_sum = sum(length * length for length in batch_seqlens)
        attention_flops = (
            12
            * sequence_square_sum
            * head_dim
            * num_attention_heads
            * num_hidden_layers
        )
        return (dense_flops + attention_flops) / delta_time / 1e12

    def estimate_flops(self, batch_seqlens, delta_time):
        estimated = self._estimate_dense_flops(
            sum(batch_seqlens),
            batch_seqlens,
            delta_time,
        )
        return estimated, get_device_flops()
