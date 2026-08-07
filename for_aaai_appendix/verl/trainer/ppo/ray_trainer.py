# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
""

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
from agent_system.task_taxonomy import (
    classify_task,
    normalize_benchmark,
    task_types_for_benchmark,
)
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_action_validity_metrics,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


class Role(Enum):
    ""

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    ""

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


def get_effective_train_batch_size(config) -> int:
    ""
    base_batch = int(config.data.train_batch_size)
    actor_repeat = int(config.actor_rollout_ref.rollout.n)
    env_repeat = int(OmegaConf.select(config, "env.rollout.n", default=1))
    if actor_repeat > 1:


        repeat = actor_repeat
    else:
        repeat = env_repeat
    return base_batch * repeat


@dataclass
class ResourcePoolManager:
    ""

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():




            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        ""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        ""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        ""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}


        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")


        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    ""
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]



    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()


    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def _extract_action_from_response(text: str) -> str:
    ""
    if not isinstance(text, str) or not text:
        return ""
    import re as _re
    m = _re.search(r"<action>(.*?)</action>", text, _re.DOTALL | _re.IGNORECASE)
    if m:
        return m.group(1).strip()


    stripped = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    if not stripped:
        return ""
    last_line = stripped.splitlines()[-1].strip()
    return last_line[:200]


def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)


        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids




    return data, {}

def compute_response_mask(data: DataProto):
    ""
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    ""

    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)


    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:

        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:

            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    ""
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    ""



    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        ""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()


        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0



        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)




        self._token_traffic_loaded = False
        self._token_traffic_totals = {}
        self._token_traffic_large_raw = None
        self._token_traffic_large_by_tt_raw = None
        self._token_traffic_large_mixed_raw = None
        self._episode_totals_loaded = False
        self._episode_totals = {}

    def _validate_config(self):
        config = self.config

        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes


        real_train_batch_size = get_effective_train_batch_size(config)
        assert real_train_batch_size % n_gpus == 0, (
            f"effective train trajectory batch ({real_train_batch_size}; "
            f"base_goals={config.data.train_batch_size}, "
            f"actor_rollout_n={config.actor_rollout_ref.rollout.n}, "
            f"env_rollout_n={OmegaConf.select(config, 'env.rollout.n', default=1)}) "
            f"must be divisible by total n_gpus ({n_gpus})."
        )



        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:

            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:

                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )


            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:

            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")


        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")






        if not config.actor_rollout_ref.actor.use_dynamic_bsz:

            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")


        if self.use_critic and not config.critic.use_dynamic_bsz:

            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus


        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")


        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"


        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        ""

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        ""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        ""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np


        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])


        rng = np.random.RandomState(42)
        rng.shuffle(samples)


        samples = samples[:generations_to_log]


        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        action_validity_list = defaultdict(list)
        success_rate_dict = {}


        sample_inputs = []
        sample_outputs = []
        sample_scores = []



        sample_traj_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)


            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)


            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}


            input_ids = test_batch.batch["input_ids"]

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            if os.environ.get("COSKILL_QUIET_LOGS", "0").strip().lower() not in {
                "1", "true", "yes", "on"
            }:
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")









            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)




            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            sample_traj_uids.extend(
                test_output_gen_batch.non_tensor_batch['traj_uid'].tolist()
            )

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            for validity_key in (
                'is_action_valid',
                'strict_action_valid',
                'non_strict_action_valid',
            ):
                if validity_key in test_output_gen_batch.non_tensor_batch:
                    action_validity_list[validity_key].append(
                        np.asarray(test_output_gen_batch.non_tensor_batch[validity_key], dtype=np.float32)
                    )

            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])

                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        action_validity = {
            key: np.concatenate(values, axis=0)
            for key, values in action_validity_list.items()
            if values
        }
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}


        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())



        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)



        if 'is_action_valid' in action_validity:
            for data_source in np.unique(data_sources):
                source_mask = data_sources == data_source
                metric_dict.update(compute_action_validity_metrics(
                    action_validity['is_action_valid'][source_mask],
                    action_validity.get('strict_action_valid', action_validity['is_action_valid'])[source_mask],
                    action_validity.get('non_strict_action_valid', action_validity['is_action_valid'])[source_mask],
                    prefix=f'val/{data_source}',
                ))

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        return metric_dict





    def _coskill_output_dir(self) -> str:
        return self.config.trainer.get('default_local_dir', './outputs')

    def _coskill_ingest_batch_to_pool(self, batch):
        ""
        if not hasattr(self, 'traces_pool'):
            from agent_system.memory import TracesPool
            tp_cfg = self.config.env.get('traces_pool', {})
            self.traces_pool = TracesPool(
                capacity_watermark=tp_cfg.get('capacity_watermark', 50_000),
                perf_watermark=tp_cfg.get('perf_watermark', 0.6),
                min_samples=tp_cfg.get('min_samples', 8),
                loop_threshold=tp_cfg.get('loop_threshold', 3),
                output_dir=self._coskill_output_dir(),
                enable_loop_filter=tp_cfg.get('enable_loop_filter', True),
                enable_obs_delta=tp_cfg.get('enable_obs_delta', True),
                enable_prefix_tree=tp_cfg.get('enable_prefix_tree', True),
                enable_consensus_prefix=tp_cfg.get('enable_consensus_prefix', True),
                cloud_evidence_mode=tp_cfg.get(
                    'cloud_evidence_mode', 'tree_only'
                ),
            )

        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        if "token_level_scores" not in batch.batch:
            return
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

        traj_uids = batch.non_tensor_batch.get('traj_uid', [None] * len(inputs))
        skill_lib = getattr(getattr(self, 'envs', None), 'retrieval_memory', None)
        can_record = skill_lib is not None and hasattr(skill_lib, 'record_usage')






        from collections import OrderedDict
        episode_rows = OrderedDict()
        for idx, (inp, out, score, uid) in enumerate(zip(inputs, outputs, scores, traj_uids)):
            key = uid if uid is not None else f'__row_{idx}'
            episode_rows.setdefault(key, []).append((inp, out, float(score)))


        ep_reward_arr = batch.non_tensor_batch.get('episode_rewards', None)
        if ep_reward_arr is None and 'episode_rewards' in batch.batch:
            ep_reward_arr = batch.batch['episode_rewards'].cpu().tolist()
        uid_to_epreward = {}
        if ep_reward_arr is not None:
            for uid, er in zip(traj_uids, ep_reward_arr):
                if uid is not None and uid not in uid_to_epreward:
                    uid_to_epreward[uid] = float(er)

        for key, rows in episode_rows.items():
            first_inp = rows[0][0]
            task_type = self._detect_task_type_from_input(first_inp)
            task_desc = self._extract_task_description(first_inp)


            indexed = []
            for inp, out, score in rows:
                indexed.append((
                    self._extract_step_index(inp),
                    self._extract_current_observation(inp),
                    _extract_action_from_response(out)[:2000],
                    score,
                ))


            indexed.sort(key=lambda t: t[0])
            steps = [
                {'step': i, 'observation': obs, 'action': action, 'reward': rew}
                for i, (_, obs, action, rew) in enumerate(indexed)
            ]

            uid = '' if str(key).startswith('__row_') else key
            if uid and uid in uid_to_epreward:
                ep_reward = uid_to_epreward[uid]
                success = ep_reward > 0
            else:
                ep_reward = max((r[3] for r in indexed), default=0.0)
                success = any(r[3] > 0 for r in indexed)

            raw_trace = {
                'traj_uid': uid,
                'task': task_desc,
                'task_type': task_type,
                'outcome': 'success' if success else 'failure',
                'episode_reward': float(ep_reward),
                'steps': steps,
                'meta': {'model_version': f'step_{self.global_steps}'},
            }
            self.traces_pool.add_trace(raw_trace)


            if can_record:
                try:
                    retrieved = skill_lib.retrieve(
                        task_description=task_desc,
                        top_k=self.config.env.skills_only_memory.get('top_k', 6),
                    )
                    injected = retrieved.get('injected_skill_ids', [])
                    skill_lib.record_usage(injected, success=success, task_type=task_type)
                    if (hasattr(skill_lib, 'record_playbook_usage')
                            and not self.config.env.get('skills_only_memory', {}).get(
                                'enable_tree_rl_internalize', False)):
                        skill_lib.record_playbook_usage(task_type, success=success)
                except Exception as e:
                    print(f"[CoSkill] record_usage failed: {e}")

    def _update_skills_coskill(self, batch):
        ""

        self._coskill_ingest_batch_to_pool(batch)





        if not (hasattr(self, 'envs') and getattr(self.envs, 'retrieval_memory', None)):
            print("[CoSkill] No retrieval_memory in training envs, skipping update")
            return
        self._get_coskill_loop().maybe_update(
            self.traces_pool, self.envs.retrieval_memory, self.global_steps)

    def _get_coskill_loop(self):
        ""
        if getattr(self, '_coskill_loop', None) is None:
            from agent_system.memory import CoSkillCloudLoop
            cfg = self.config.env.skills_only_memory
            self._coskill_loop = CoSkillCloudLoop(
                output_dir=self._coskill_output_dir(),
                enable_coskill=cfg.get('enable_coskill', False),
                enable_playbook_evolve=cfg.get('enable_playbook_evolve', False),
                enable_failure_analysis=cfg.get('enable_failure_analysis', True),
                max_new_skills=cfg.get('max_new_skills', 3),
                playbook_evolve_min_samples=cfg.get('playbook_evolve_min_samples', 6),
                coskill_debug=cfg.get('coskill_debug', False),
                environment_name=str(self.config.env.env_name),
            )
        return self._coskill_loop

    def _record_tree_rl_batch_outcomes(self, batch) -> None:
        ""
        skill_lib = getattr(getattr(self, "envs", None), "retrieval_memory", None)
        if skill_lib is None or not hasattr(skill_lib, "record_playbook_usage"):
            return
        if "token_level_scores" not in batch.batch:
            return
        try:
            from collections import OrderedDict

            inputs = self.tokenizer.batch_decode(
                batch.batch["prompts"], skip_special_tokens=True)
            traj_uids = batch.non_tensor_batch.get("traj_uid", [None] * len(inputs))
            episode_rewards = batch.non_tensor_batch.get("episode_rewards", None)
            if episode_rewards is None and "episode_rewards" in batch.batch:
                episode_rewards = batch.batch["episode_rewards"].detach().cpu().tolist()

            by_uid = OrderedDict()
            for index, (prompt, uid) in enumerate(zip(inputs, traj_uids)):
                key = uid if uid is not None else f"__row_{index}"
                by_uid.setdefault(key, (prompt, index))

            reward_by_uid = {}
            if episode_rewards is not None:
                for uid, reward in zip(traj_uids, episode_rewards):
                    if uid is not None and uid not in reward_by_uid:
                        reward_by_uid[uid] = float(reward)
            fallback_scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()

            for uid, (prompt, row_index) in by_uid.items():
                reward = reward_by_uid.get(uid, float(fallback_scores[row_index]))
                task_type = self._detect_task_type_from_input(prompt)
                skill_lib.record_playbook_usage(task_type, success=(reward > 0.0))
        except Exception as exc:


            print(f"[CoSkillTreeRL] outcome attribution failed: {exc}")

    def _advance_tree_rl_curriculum(self) -> None:
        ""
        som = self.config.env.get("skills_only_memory", {})
        if not som.get("enable_tree_rl_internalize", False):
            return
        skill_lib = getattr(getattr(self, "envs", None), "retrieval_memory", None)
        if skill_lib is None or not hasattr(skill_lib, "advance_tree_rl_curriculum"):
            print("[CoSkillTreeRL] hierarchical skill-tree memory is unavailable; curriculum skipped")
            return
        try:
            events = skill_lib.advance_tree_rl_curriculum(
                global_step=int(self.global_steps),
                order=str(som.get("tree_rl_order", "root")),
                min_rl_updates=int(som.get("tree_rl_min_updates", 5)),
                min_train_episodes=int(som.get("tree_rl_min_train_episodes", 24)),
                train_success_threshold=float(som.get("tree_rl_train_success_threshold", 0.7)),
                min_probe_episodes=int(som.get("tree_rl_min_probe_episodes", 24)),
                probe_success_threshold=float(som.get("tree_rl_probe_success_threshold", 0.7)),
            )
        except Exception as exc:
            print(f"[CoSkillTreeRL] curriculum transition failed: {exc}")
            return

        self._tree_rl_stats = {
            "last_step": int(self.global_steps),
            "last_events": len(events),
        }
        if events:
            output_dir = self._coskill_output_dir()
            try:
                event_path = os.path.join(output_dir, "skill_tree_rl_events.jsonl")
                with open(event_path, "a", encoding="utf-8") as handle:
                    for event in events:
                        handle.write(json.dumps({"step": int(self.global_steps), **event}, ensure_ascii=False) + "\n")
                print(f"[CoSkillTreeRL] step={self.global_steps} events={events}")
            except Exception as exc:
                print(f"[CoSkillTreeRL] event log write failed: {exc}")




        save_freq = max(1, int(som.get("tree_rl_state_save_freq", 1)))
        if events or int(self.global_steps) % save_freq == 0:
            try:
                save_dir = os.path.join(self._coskill_output_dir(), "skill_lib")
                os.makedirs(save_dir, exist_ok=True)
                skill_lib.save_skills(os.path.join(save_dir, "skills_tree_rl_latest.json"))
            except Exception as exc:
                print(f"[CoSkillTreeRL] state save failed: {exc}")





    def _group_metrics_jsonl_path(self) -> str:
        ""
        explicit = os.environ.get("JSONL_METRICS_PATH")
        if explicit:
            return os.path.expanduser(explicit)
        base = os.environ.get("JSONL_METRICS_DIR")
        if not base:
            base = self.config.trainer.get("default_local_dir", None)
        if not base:
            base = os.path.join(
                "outputs",
                str(self.config.trainer.get("project_name", "verl")),
                str(self.config.trainer.get("experiment_name", "experiment")),
            )
        return os.path.join(os.path.expanduser(str(base)), "group_metrics.jsonl")

    def _benchmark_name(self) -> str:
        return normalize_benchmark(str(self.config.env.get("env_name", "")))

    def _metric_task_types(self) -> tuple[str, ...]:
        return task_types_for_benchmark(self._benchmark_name())

    def _load_episode_totals(self) -> None:
        if self._episode_totals_loaded:
            return
        task_types = self._metric_task_types()
        totals = {
            "count": 0,
            "wins": 0,
            "by_task_type": {
                task_type: {"count": 0, "wins": 0}
                for task_type in task_types
            },
        }
        path = self._group_metrics_jsonl_path()
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        row = row.get("metrics", row) if isinstance(row, dict) else {}
                        record_type = row.get("record/type")
                        if record_type not in {None, "train_update"}:
                            continue
                        if "episode/count" not in row:
                            continue
                        totals["count"] += int(
                            self._metric_scalar(row.get("episode/count", 0), 0)
                        )
                        totals["wins"] += int(
                            self._metric_scalar(row.get("episode/wins", 0), 0)
                        )
                        for task_type in task_types:
                            prefix = f"episode/by_task_type/{task_type}"
                            totals["by_task_type"][task_type]["count"] += int(
                                self._metric_scalar(row.get(f"{prefix}/count", 0), 0)
                            )
                            totals["by_task_type"][task_type]["wins"] += int(
                                self._metric_scalar(row.get(f"{prefix}/wins", 0), 0)
                            )
            except OSError as exc:
                print(f"[group-metrics] cannot read '{path}': {exc}")
        self._episode_totals = totals
        self._episode_totals_loaded = True

    def _add_episode_cumulative_metrics(self, metrics: Dict) -> None:
        self._load_episode_totals()
        totals = self._episode_totals
        totals["count"] += int(self._metric_scalar(metrics.get("episode/count", 0), 0))
        totals["wins"] += int(self._metric_scalar(metrics.get("episode/wins", 0), 0))
        metrics["episode/wins_cumulative"] = totals["wins"]
        metrics["episode/success_rate_cumulative"] = (
            totals["wins"] / max(totals["count"], 1)
        )
        for task_type in self._metric_task_types():
            prefix = f"episode/by_task_type/{task_type}"
            task_totals = totals["by_task_type"][task_type]
            task_totals["count"] += int(
                self._metric_scalar(metrics.get(f"{prefix}/count", 0), 0)
            )
            task_totals["wins"] += int(
                self._metric_scalar(metrics.get(f"{prefix}/wins", 0), 0)
            )
            metrics[f"{prefix}/count_cumulative"] = task_totals["count"]
            metrics[f"{prefix}/wins_cumulative"] = task_totals["wins"]
            metrics[f"{prefix}/success_rate_cumulative"] = (
                task_totals["wins"] / max(task_totals["count"], 1)
            )

    @staticmethod
    def _metric_scalar(value, default=0.0):
        ""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item()) if value.numel() == 1 else default
        if isinstance(value, np.generic):
            return value.item()
        return value if isinstance(value, (int, float, bool)) else default

    def _load_token_traffic_totals(self) -> None:
        ""
        if self._token_traffic_loaded:
            return
        task_types = self._metric_task_types()
        totals = {
            "small_prompt": 0,
            "small_response": 0,
            "small_total": 0,
            "large_prompt": 0,
            "large_completion": 0,
            "large_total": 0,
            "small_by_tt": {tt: 0 for tt in task_types},
            "large_by_tt": {tt: 0 for tt in task_types},
            "large_mixed": 0,
        }
        path = self._group_metrics_jsonl_path()
        last_cumulative = None
        legacy_small = {"small_prompt": 0, "small_response": 0, "small_total": 0}
        legacy_large = None
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue


                        row = row.get("metrics", row)
                        if not isinstance(row, dict):
                            continue
                        if "tokens/small_model/total_cumulative" in row:
                            last_cumulative = {
                                "small_prompt": int(self._metric_scalar(
                                    row.get("tokens/small_model/prompt_cumulative", 0), 0)),
                                "small_response": int(self._metric_scalar(
                                    row.get("tokens/small_model/response_cumulative", 0), 0)),
                                "small_total": int(self._metric_scalar(
                                    row.get("tokens/small_model/total_cumulative", 0), 0)),
                                "large_prompt": int(self._metric_scalar(
                                    row.get("tokens/large_model/prompt_cumulative", 0), 0)),
                                "large_completion": int(self._metric_scalar(
                                    row.get("tokens/large_model/completion_cumulative", 0), 0)),
                                "large_total": int(self._metric_scalar(
                                    row.get("tokens/large_model/total_cumulative", 0), 0)),
                                "small_by_tt": {
                                    tt: int(self._metric_scalar(
                                        row.get(f"tokens/small_model/by_task_type/{tt}/total_cumulative", 0), 0))
                                    for tt in task_types
                                },
                                "large_by_tt": {
                                    tt: int(self._metric_scalar(
                                        row.get(f"tokens/large_model/by_task_type/{tt}/total_cumulative", 0), 0))
                                    for tt in task_types
                                },
                                "large_mixed": int(self._metric_scalar(
                                    row.get("tokens/large_model/mixed/total_cumulative", 0), 0)),
                            }
                        else:
                            legacy_small["small_prompt"] += max(0, int(self._metric_scalar(
                                row.get("tokens/small_model/prompt", 0), 0)))
                            legacy_small["small_response"] += max(0, int(self._metric_scalar(
                                row.get("tokens/small_model/response", 0), 0)))
                            legacy_small["small_total"] += max(0, int(self._metric_scalar(
                                row.get("tokens/small_model/total", 0), 0)))
                        if "coskill/cloud/large_model_total_tokens" in row:
                            legacy_large = {
                                "large_prompt": int(self._metric_scalar(
                                    row.get("coskill/cloud/large_model_prompt_tokens", 0), 0)),
                                "large_completion": int(self._metric_scalar(
                                    row.get("coskill/cloud/large_model_completion_tokens", 0), 0)),
                                "large_total": int(self._metric_scalar(
                                    row.get("coskill/cloud/large_model_total_tokens", 0), 0)),
                            }
            except OSError as exc:
                print(f"[token-traffic] cannot read '{path}': {exc}")

        if last_cumulative is not None:
            totals.update(last_cumulative)
        else:
            totals.update(legacy_small)
            if legacy_large is not None:
                totals.update(legacy_large)
        self._token_traffic_totals = totals
        self._token_traffic_loaded = True

    def _large_token_usage_snapshot(self):
        ""
        summary = None
        loop = getattr(self, "_coskill_loop", None)
        analyzer = getattr(loop, "cloud_analyzer", None) if loop is not None else None
        if analyzer is not None:
            summary = analyzer.get_update_summary() or {}
        if summary is None:
            summary = {}
        by_tt_prompt = summary.get("large_model_prompt_tokens_by_task_type", {}) or {}
        by_tt_completion = summary.get("large_model_completion_tokens_by_task_type", {}) or {}
        task_types = self._metric_task_types()
        by_tt_total = {
            tt: int(self._metric_scalar(by_tt_prompt.get(tt, 0), 0))
            + int(self._metric_scalar(by_tt_completion.get(tt, 0), 0))
            for tt in task_types
        }
        mixed_total = int(self._metric_scalar(summary.get("large_model_prompt_tokens_mixed", 0), 0)) + \
            int(self._metric_scalar(summary.get("large_model_completion_tokens_mixed", 0), 0))
        return (
            int(self._metric_scalar(summary.get("large_model_prompt_tokens", 0), 0)),
            int(self._metric_scalar(summary.get("large_model_completion_tokens", 0), 0)),
            int(self._metric_scalar(summary.get("large_model_total_tokens", 0), 0)),
            by_tt_total,
            mixed_total,
        )

    def _add_token_traffic_metrics(self, metrics: Dict) -> None:
        ""
        self._load_token_traffic_totals()
        totals = self._token_traffic_totals
        small_prompt = max(0, int(self._metric_scalar(
            metrics.get("tokens/small_model/prompt", 0), 0)))
        small_response = max(0, int(self._metric_scalar(
            metrics.get("tokens/small_model/response", 0), 0)))
        small_total = max(0, int(self._metric_scalar(
            metrics.get("tokens/small_model/total", 0), 0)))
        totals["small_prompt"] += small_prompt
        totals["small_response"] += small_response
        totals["small_total"] += small_total





        task_types = self._metric_task_types()
        small_by_tt = totals.setdefault("small_by_tt", {tt: 0 for tt in task_types})
        for tt in task_types:
            step_val = max(0, int(self._metric_scalar(
                metrics.get(f"tokens/small_model/by_task_type/{tt}/total", 0), 0)))
            small_by_tt[tt] = small_by_tt.get(tt, 0) + step_val

        raw_prompt, raw_completion, raw_total, raw_by_tt, raw_mixed = self._large_token_usage_snapshot()
        raw_scalar = (raw_prompt, raw_completion, raw_total)
        if self._token_traffic_large_raw is None:


            large_delta = raw_scalar
        elif all(now >= old for now, old in zip(raw_scalar, self._token_traffic_large_raw)):
            large_delta = tuple(now - old for now, old in zip(raw_scalar, self._token_traffic_large_raw))
        else:

            large_delta = raw_scalar
        self._token_traffic_large_raw = raw_scalar
        totals["large_prompt"] += max(0, large_delta[0])
        totals["large_completion"] += max(0, large_delta[1])
        totals["large_total"] += max(0, large_delta[2])


        old_by_tt = self._token_traffic_large_by_tt_raw
        large_by_tt = totals.setdefault("large_by_tt", {tt: 0 for tt in task_types})
        by_tt_delta = {}
        for tt in task_types:
            now = raw_by_tt.get(tt, 0)
            old = 0 if old_by_tt is None else old_by_tt.get(tt, 0)
            delta = now - old if now >= old else now
            by_tt_delta[tt] = max(0, delta)
            large_by_tt[tt] = large_by_tt.get(tt, 0) + by_tt_delta[tt]
        self._token_traffic_large_by_tt_raw = dict(raw_by_tt)




        old_mixed = self._token_traffic_large_mixed_raw
        if old_mixed is None:
            mixed_delta = raw_mixed
        elif raw_mixed >= old_mixed:
            mixed_delta = raw_mixed - old_mixed
        else:
            mixed_delta = raw_mixed
        mixed_delta = max(0, mixed_delta)
        self._token_traffic_large_mixed_raw = raw_mixed
        totals["large_mixed"] = totals.get("large_mixed", 0) + mixed_delta

        metrics.update({
            "tokens/small_model/accounting": "actor_rollout_request_tokens",
            "tokens/small_model/prompt_cumulative": totals["small_prompt"],
            "tokens/small_model/response_cumulative": totals["small_response"],
            "tokens/small_model/total_cumulative": totals["small_total"],
            "tokens/large_model/prompt": max(0, large_delta[0]),
            "tokens/large_model/completion": max(0, large_delta[1]),
            "tokens/large_model/total": max(0, large_delta[2]),
            "tokens/large_model/accounting": "provider_api_usage",
            "tokens/large_model/prompt_cumulative": totals["large_prompt"],
            "tokens/large_model/completion_cumulative": totals["large_completion"],
            "tokens/large_model/total_cumulative": totals["large_total"],
            **{
                f"tokens/small_model/by_task_type/{tt}/total_cumulative": small_by_tt[tt]
                for tt in task_types
            },
            **{
                f"tokens/large_model/by_task_type/{tt}/total": by_tt_delta[tt]
                for tt in task_types
            },
            **{
                f"tokens/large_model/by_task_type/{tt}/total_cumulative": large_by_tt[tt]
                for tt in task_types
            },
            "tokens/large_model/mixed/total": mixed_delta,
            "tokens/large_model/mixed/total_cumulative": totals["large_mixed"],
            "tokens/large_model/mixed/accounting": "provider_api_usage_unattributed_mixed_task_type",
        })
        small_reconciled = sum(
            max(0, int(self._metric_scalar(
                metrics.get(f"tokens/small_model/by_task_type/{tt}/total", 0), 0
            )))
            for tt in task_types
        )
        metrics["tokens/small_model/by_task_type/total_reconciled"] = small_reconciled
        metrics["tokens/small_model/by_task_type/reconciliation_error"] = (
            small_total - small_reconciled
        )
        large_reconciled = sum(by_tt_delta.values()) + mixed_delta
        metrics["tokens/large_model/attribution/total_reconciled"] = large_reconciled
        metrics["tokens/large_model/attribution/reconciliation_error"] = (
            max(0, large_delta[2]) - large_reconciled
        )

    def _coskill_metrics(self) -> dict:
        ""
        m = {}

        loop = getattr(self, '_coskill_loop', None)
        skill_lib = getattr(getattr(self, 'envs', None), 'retrieval_memory', None)
        if loop is not None:
            m.update(loop.metrics(getattr(self, 'traces_pool', None), skill_lib))

        istats = getattr(self, '_internalize_stats', None)
        if istats:
            m['coskill/internalize/last_step'] = istats.get('last_step', 0)
            m['coskill/internalize/n_skills'] = istats.get('n_skills', 0)
            m['coskill/internalize/n_samples'] = istats.get('n_samples', 0)
            m['coskill/internalize/seconds'] = istats.get('seconds', 0.0)
        if skill_lib is not None and hasattr(skill_lib, "tree_rl_metrics"):
            m.update(skill_lib.tree_rl_metrics())
        tree_stats = getattr(self, "_tree_rl_stats", None)
        if tree_stats:
            m["coskill/tree_rl/last_step"] = tree_stats.get("last_step", 0)
            m["coskill/tree_rl/last_events"] = tree_stats.get("last_events", 0)
        return m




    def _internalize_raw_dir(self) -> str:
        ""
        som = self.config.env.get('skills_only_memory', {})
        explicit = som.get('raw_traj_dir', None) if hasattr(som, 'get') else None
        if explicit:
            return explicit
        return os.path.join(self._coskill_output_dir(), 'raw_episodes')

    def _load_internalize_episodes(self, cold_skill_ids, max_episodes):
        ""
        import glob as _glob
        raw_dir = self._internalize_raw_dir()
        if not os.path.isdir(raw_dir):
            print(f"[Skill2param] raw dir not found: {raw_dir}; nothing to internalize from")
            return []
        cold_set = set(cold_skill_ids or [])

        subdirs = sorted(_glob.glob(os.path.join(raw_dir, 'rollout_*')), reverse=True)
        episodes = []
        for sd in subdirs:
            for fp in sorted(_glob.glob(os.path.join(sd, 'ep_*.json'))):
                try:
                    with open(fp, encoding='utf-8') as f:
                        ep = json.load(f)
                except Exception:
                    continue
                if not ep.get('won', False):
                    continue
                steps = ep.get('steps', []) or []
                if not steps:
                    continue


                if cold_set:
                    used = set()
                    for s in steps:
                        for sid in (s.get('injected_skill_ids') or []):
                            used.add(sid)
                    if used and not (used & cold_set):
                        continue
                episodes.append({
                    'task_goal': ep.get('task_goal'),
                    'steps': steps,
                })
                if len(episodes) >= max_episodes:
                    return episodes
        return episodes

    def _build_internalize_batch(self, episodes, max_prompt_len, max_resp_len):
        ""
        import verl.utils.torch_functional as verl_F
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.model import compute_position_id_with_mask

        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        samples = []
        for ep in episodes:
            for s in ep.get('steps', []):
                obs = s.get('observation')
                resp = s.get('raw_response')
                if not obs or not resp:
                    continue
                try:
                    p_ids, p_mask = verl_F.tokenize_and_postprocess_data(
                        prompt=obs, tokenizer=tokenizer, max_length=max_prompt_len,
                        pad_token_id=pad_id, left_pad=True, truncation='left')
                    r_ids, r_mask = verl_F.tokenize_and_postprocess_data(
                        prompt=resp, tokenizer=tokenizer, max_length=max_resp_len,
                        pad_token_id=pad_id, left_pad=False, truncation='right')
                except Exception:
                    continue
                input_ids = torch.cat([p_ids[0], r_ids[0]], dim=-1)
                attention_mask = torch.cat([p_mask[0], r_mask[0]], dim=-1)
                position_ids = compute_position_id_with_mask(attention_mask)
                samples.append({
                    'prompts': p_ids[0],
                    'responses': r_ids[0],
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                })
        if not samples:
            return None
        return DataProto.from_single_dict(data=collate_fn(samples))

    def _internalize_cold_skills(self):
        ""
        import time as _time
        t0 = _time.time()
        skill_lib = getattr(getattr(self, 'envs', None), 'retrieval_memory', None)
        if skill_lib is None or not hasattr(skill_lib, 'get_cold_skills'):
            return
        cold = skill_lib.get_cold_skills()
        if not cold:
            return
        cold_ids = [s.get('skill_id') for s in cold]
        som = self.config.env.get('skills_only_memory', {})
        max_eps = int(som.get('internalize_max_episodes', 8)) if hasattr(som, 'get') else 8
        episodes = self._load_internalize_episodes(cold_ids, max_eps)
        if not episodes:
            print(f"[Skill2param] no won episodes available; skip (cold={len(cold_ids)})")
            return

        max_prompt = int(self.config.data.max_prompt_length)
        max_resp = int(self.config.data.max_response_length)
        batch = self._build_internalize_batch(episodes, max_prompt, max_resp)
        if batch is None or len(batch.batch) == 0:
            print("[Skill2param] empty internalize batch; skip")
            return



        try:
            batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.get('temperature', 1.0)
            old_lp = self.actor_rollout_wg.compute_log_prob(batch)
            old_lp.batch.pop('entropys', None)
            batch = batch.union(old_lp)
            if self.use_reference_policy:
                if not self.ref_in_actor:
                    ref_lp = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_lp = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_lp)
            resp_len = batch.batch['responses'].size(1)
            response_mask = batch.batch['attention_mask'][:, -resp_len:]
            adv_const = float(som.get('internalize_adv', 1.0)) if hasattr(som, 'get') else 1.0
            advantages = adv_const * response_mask.float()
            batch.batch['advantages'] = advantages
            batch.batch['returns'] = advantages.clone()
            batch.batch['response_mask'] = response_mask
            batch.meta_info['multi_turn'] = False
            self.actor_rollout_wg.update_actor(batch)
            n_samples = len(batch.batch)
        except Exception as e:
            print(f"[Skill2param] update_actor failed (non-fatal): {e}")
            return

        skill_lib.mark_internalized(cold_ids)
        save_dir = os.path.join(self._coskill_output_dir(), 'skill_lib')
        try:
            os.makedirs(save_dir, exist_ok=True)
            skill_lib.save_skills(os.path.join(save_dir, f'skills_step{self.global_steps}_internalized.json'))
        except Exception as e:
            print(f"[Skill2param] skill save failed: {e}")
        self._internalize_stats = {
            'last_step': self.global_steps,
            'n_skills': len(cold_ids),
            'n_samples': n_samples,
            'seconds': _time.time() - t0,
        }
        print(f"[Skill2param] internalized {len(cold_ids)} skills from "
              f"{n_samples} samples in {self._internalize_stats['seconds']:.1f}s; "
              f"marked: {cold_ids}")

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
    ) -> list:
        ""
        failed = []
        for inp, out, score in zip(inputs, outputs, scores):
            if score <= 0:
                task_type = self._detect_task_type_from_input(inp)
                task_desc = self._extract_task_description(inp)
                trajectory = self._parse_conversation_to_steps(inp, out)
                failed.append({
                    'task': task_desc,
                    'trajectory': trajectory,
                    'task_type': task_type,
                })
        return failed[:10]

    def _extract_task_description(self, inp: str) -> str:
        ""
        import re

        patterns = [
            r'(?:Your task is to|Task:|task is to|you need to)[:\s]+(.*?)(?:\n|$)',
            r'(?:goal|objective)[:\s]+(.*?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, inp, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:1000]

        for marker in ('<|im_start|>user\n', '\nHuman: ', '\nUser: '):
            idx = inp.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                return inp[start:start + 1000]
        return inp[:1000]

    def _parse_conversation_to_steps(self, inp: str, out: str) -> list:
        ""
        import re
        steps = []


        user_turns = re.findall(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        asst_turns = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': _extract_action_from_response(act)[:1500],
                    'observation': obs.strip()[:800],
                })

            steps.append({'action': _extract_action_from_response(out)[:2000], 'observation': ''})
            return steps


        user_turns = re.findall(
            r'(?:Human|User):\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        asst_turns = re.findall(
            r'Assistant:\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': _extract_action_from_response(act)[:1500],
                    'observation': obs.strip()[:800],
                })
            steps.append({'action': _extract_action_from_response(out)[:2000], 'observation': ''})
            return steps


        steps.append({'action': '', 'observation': inp[:3000]})
        steps.append({'action': _extract_action_from_response(out)[:2000], 'observation': ''})
        return steps

    def _extract_current_observation(self, inp: str) -> str:
        ""
        import re
        m = re.search(
            r'current observation is:\s*(.*?)\s*Your admissible actions',
            inp, re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()[:1500]
        return inp.strip()[:1500]

    def _extract_step_index(self, inp: str) -> int:
        ""
        import re
        m = re.search(r'at step\s+(\d+)', inp, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    def _detect_task_type_from_input(self, inp: str) -> str:
        ""
        return classify_task(self._benchmark_name(), inp)

    def _compute_row_task_types(self, batch) -> np.ndarray:
        ""
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        traj_uids = batch.non_tensor_batch.get('traj_uid', [None] * len(inputs))
        first_seen: Dict[Any, str] = {}
        task_types = []
        for inp, uid in zip(inputs, traj_uids):
            key = uid if uid is not None else inp
            if key not in first_seen:
                first_seen[key] = self._detect_task_type_from_input(inp)
            task_types.append(first_seen[key])
        return np.asarray(task_types)

    def init_workers(self):
        ""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}


        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError


        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls


        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls


        if self.use_rm:

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls






        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()


        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()


        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):

        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)


        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)


        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0


        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)


        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")

        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)



        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        ""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        ""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0


        self._load_checkpoint()



        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            val_metrics["record/type"] = "validation"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return


        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")


        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)


                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps






                rollout_only = self.config.trainer.get("rollout_only", False)

                with _timer("step", timing_raw):

                    with _timer("gen", timing_raw):








                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output





                    del batch
                    batch = gen_batch_output





                    try:
                        metrics.update(
                            self.traj_collector.get_last_rollout_compaction_metrics()
                        )
                    except Exception as exc:
                        print(f"[rollout-compaction] metrics unavailable: {exc}")

                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)



                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)


                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):

                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)


                    if not rollout_only:
                      with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():

                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy and not rollout_only:

                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)


                    if self.use_critic and not rollout_only:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):

                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if os.environ.get("COSKILL_QUIET_LOGS", "0").strip().lower() not in {
                            "1", "true", "yes", "on"
                        }:
                            print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})


                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)


                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]



                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )




                    som_cfg = self.config.env.get('skills_only_memory', {})
                    if som_cfg.get('enable_tree_rl_internalize', False):
                        self._record_tree_rl_batch_outcomes(batch)


                    if (self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False)
                            and self.config.env.get('skills_only_memory', {}).get('update_skills_from_train', False)):
                        skill_update_freq = self.config.env.get('skills_only_memory', {}).get('skill_update_freq', self.config.trainer.get('test_freq', 5))
                        if self.global_steps > 0 and self.global_steps % skill_update_freq == 0:
                            som_cfg_upd = self.config.env.get('skills_only_memory', {})
                            if (som_cfg_upd.get('enable_coskill', False)
                                    or som_cfg_upd.get('enable_playbook_evolve', False)):
                                self._update_skills_coskill(batch)
                            else:
                                raise RuntimeError(
                                    "The appendix artifact supports only the CoSkill "
                                    "closed-loop skill update."
                                )




                    som_cfg = self.config.env.get('skills_only_memory', {})
                    if som_cfg.get('enable_internalize', False) and not rollout_only:
                        internalize_freq = som_cfg.get('internalize_freq', 10)
                        if (self.global_steps > 0
                                and self.global_steps % internalize_freq == 0):
                            skill_lib = getattr(getattr(self, 'envs', None), 'retrieval_memory', None)
                            if (skill_lib is not None
                                    and hasattr(skill_lib, 'has_cold_skills')
                                    and skill_lib.has_cold_skills()):
                                with _timer("internalize", timing_raw):
                                    self._internalize_cold_skills()


                    if self.use_critic and not rollout_only:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)


                    if not rollout_only and self.config.trainer.critic_warmup <= self.global_steps:

                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if som_cfg.get('enable_tree_rl_internalize', False):
                            with _timer("tree_rl_curriculum", timing_raw):
                                self._advance_tree_rl_curriculum()


                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            if os.environ.get("COSKILL_QUIET_LOGS", "0").strip().lower() not in {
                                "1", "true", "yes", "on"
                            }:
                                print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )


                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if (not rollout_only and self.config.trainer.save_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()


                metrics.update(
                    {
                        "record/type": "train_update",
                        "training/epoch": epoch,
                    }
                )

                metrics.update(compute_data_metrics(
                    batch=batch,
                    use_critic=self.use_critic,
                    task_types=self._compute_row_task_types(batch),
                    known_task_types=self._metric_task_types(),
                ))
                self._add_episode_cumulative_metrics(metrics)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))


                self._add_token_traffic_metrics(metrics)




                benchmark = self._benchmark_name()
                _som_for_comparison = self.config.env.get('skills_only_memory', {})
                metrics.update({
                    'comparison/schema_version': 3,



                    'comparison/method': 'coskill',
                    'comparison/benchmark': benchmark,
                    'comparison/rollout_accounting': 'active_env_decisions',
                    'comparison/timing_cloud_update_measured': int(bool(
                        _som_for_comparison.get('enable_coskill', False)
                        or _som_for_comparison.get('enable_playbook_evolve', False))),
                    'experiment/rl_enabled': 1,
                    'experiment/tree_rl_internalize_enabled': int(bool(
                        _som_for_comparison.get('enable_tree_rl_internalize', False))),
                })



                _som = self.config.env.get('skills_only_memory', {})
                if (_som.get('enable_coskill', False)
                        or _som.get('enable_playbook_evolve', False)
                        or _som.get('enable_tree_rl_internalize', False)):
                    try:
                        metrics.update(self._coskill_metrics())
                    except Exception as e:
                        print(f"[CoSkill] metrics collection failed: {e}")


                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
