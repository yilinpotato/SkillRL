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

import torch
import numpy as np
import os
import re
import json
import time
import uuid as _uuid_mod
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _cfg_get(cfg, key, default=None):
    ""
    if cfg is None:
        return default
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)

        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                return cfg.get(key, default)
            except Exception:
                pass
        return getattr(cfg, key, default)
    except Exception:
        return default


def _split_think_action(raw_response: str):
    ""
    if not isinstance(raw_response, str):
        return "", ""
    think = ""
    action = ""
    m = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL | re.IGNORECASE)
    if m:
        think = m.group(1).strip()
    m = re.search(r"<action>(.*?)</action>", raw_response, re.DOTALL | re.IGNORECASE)
    if m:
        action = m.group(1).strip()
    return think, action


def _select_observation_rows(obs: Dict, indices: np.ndarray) -> Dict:
    ""
    selected = {}
    index_list = [int(i) for i in indices]
    for key, value in obs.items():
        if value is None:
            selected[key] = None
        elif isinstance(value, torch.Tensor):
            selected[key] = value[torch.as_tensor(index_list, device=value.device)]
        elif isinstance(value, np.ndarray):
            selected[key] = value[indices]
        elif isinstance(value, list):
            selected[key] = [value[i] for i in index_list]
        elif isinstance(value, tuple):
            selected[key] = tuple(value[i] for i in index_list)
        else:


            selected[key] = value
    return selected


def _rollout_compaction_metrics(
    *,
    enabled: bool,
    rounds: int,
    full_batch_rows: int,
    active_rows: int,
    vllm_request_rows: int,
    active_rows_by_round: List[int] | None = None,
    world_size: int = 1,
    batch_tokenize_enabled: bool = False,
    collector_seconds: float = 0.0,
    preprocess_seconds: float = 0.0,
    generate_seconds: float = 0.0,
    env_step_seconds: float = 0.0,
) -> Dict[str, int | float]:
    ""
    per_round = active_rows_by_round or []
    return {
        "rollout/active_compaction_enabled": int(bool(enabled)),
        "rollout/batch_tokenize_observations_enabled": int(
            bool(batch_tokenize_enabled)
        ),
        "rollout/active_generation_rounds": int(rounds),


        "rollout/vllm_full_batch_rows_legacy": int(full_batch_rows),

        "rollout/vllm_active_rows": int(active_rows),


        "rollout/vllm_request_rows": int(vllm_request_rows),
        "rollout/vllm_dp_padding_rows": max(
            0, int(vllm_request_rows) - int(active_rows)
        ),
        "rollout/vllm_rows_avoided": max(0, int(full_batch_rows) - int(vllm_request_rows)),
        "rollout/active_rows_per_round_mean": (
            float(sum(per_round) / len(per_round)) if per_round else 0.0
        ),
        "rollout/active_rows_per_round_min": min(per_round) if per_round else 0,
        "rollout/active_rows_per_round_max": max(per_round) if per_round else 0,
        "rollout/dp_underfilled_rounds": sum(
            int(rows < max(1, int(world_size))) for rows in per_round
        ),
        "timing_s/rollout_collector": float(collector_seconds),
        "timing_s/rollout_preprocess": float(preprocess_seconds),
        "timing_s/rollout_vllm_generate": float(generate_seconds),
        "timing_s/rollout_env_step": float(env_step_seconds),
        "timing_s/rollout_other": max(
            0.0,
            float(collector_seconds)
            - float(preprocess_seconds)
            - float(generate_seconds)
            - float(env_step_seconds),
        ),
    }


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        ""
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor






        env_cfg = getattr(self.config, "env", None)
        self._dump_raw = bool(_cfg_get(env_cfg, "dump_raw_trajectories", False))

        trainer_cfg = getattr(self.config, "trainer", None)
        default_dir = _cfg_get(trainer_cfg, "default_local_dir", None) or "."
        self._raw_dump_dir = _cfg_get(env_cfg, "raw_traj_dir", None) or os.path.join(
            str(default_dir), "raw_episodes"
        )

        self._rollout_call_idx = 0



        self._compact_finished_trajectories = bool(
            _cfg_get(env_cfg, "compact_finished_trajectories", False)
        )
        self._batch_tokenize_observations = bool(
            _cfg_get(env_cfg, "batch_tokenize_observations", False)
        )
        self._batch_tokenize_fallback_warned = False
        self._last_rollout_compaction_metrics: Dict[str, int | float] = {}

    def get_last_rollout_compaction_metrics(self) -> Dict[str, int | float]:
        ""
        return dict(self._last_rollout_compaction_metrics)

    def _prompts_open_think(self, input_ids, attention_mask):
        ""
        flags = []
        for token_ids, mask in zip(input_ids, attention_mask):
            prompt_ids = token_ids[mask.to(dtype=torch.bool)][-32:]
            tail = self.tokenizer.decode(
                prompt_ids.detach().cpu().tolist(), skip_special_tokens=False
            )
            flags.append(tail.rstrip().endswith("<think>"))
        return flags

    @staticmethod
    def _render_protocol_response(response: str, prompt_opens_think: bool):
        ""
        response = "" if response is None else str(response)
        has_open = bool(re.search(r"<think>", response, flags=re.IGNORECASE))
        has_close = bool(re.search(r"</think>", response, flags=re.IGNORECASE))
        if prompt_opens_think and not has_open and has_close:
            return "<think>\n" + response, True
        return response, False

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        ""

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})


        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor







        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")


        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])


        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )


        row_dict = {}


        if is_multi_modal:

            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)



        if is_multi_modal:
            raise ValueError("The appendix artifact supports text input only")
        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")


        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
    ) -> DataProto:
        ""
        if self._batch_tokenize_observations and self._is_text_only_batch(obs):
            try:
                return self._preprocess_text_batch(gen_batch=gen_batch, obs=obs)
            except Exception as exc:
                if not self._batch_tokenize_fallback_warned:
                    print(
                        "[rollout] batch text tokenization unavailable; "
                        f"falling back to the equivalent row path: {exc}"
                    )
                    self._batch_tokenize_fallback_warned = True

        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []


        for item in range(batch_size):

            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)


        batch = collate_fn(processed_samples)


        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    @staticmethod
    def _is_text_only_batch(obs: Dict) -> bool:
        images = obs.get("image", None)
        if images is None:
            return True
        try:
            return all(image is None for image in images)
        except TypeError:
            return False

    def _preprocess_text_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        ""
        obs_texts = obs.get("text", None)
        obs_anchors = obs.get("anchor", None)
        batch_size = len(gen_batch.batch["input_ids"])
        if obs_texts is None or len(obs_texts) != batch_size:
            raise ValueError("text observations do not match the active batch")

        chats = []
        for obs_text in obs_texts:
            if obs_text is None:
                raise ValueError("text-only batch contains an empty observation")
            chats.append([{"content": str(obs_text), "role": "user"}])

        apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        rendered_prompts = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        if isinstance(rendered_prompts, str):
            rendered_prompts = [rendered_prompts]
        if len(rendered_prompts) != batch_size:
            raise ValueError(
                "batched chat rendering returned "
                f"{len(rendered_prompts)} rows for {batch_size} observations"
            )

        encoded = self.tokenizer(
            list(rendered_prompts),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        encoded_rows = encoded["input_ids"]
        if isinstance(encoded_rows, torch.Tensor):
            encoded_rows = encoded_rows.detach().cpu().tolist()
        data_sources = gen_batch.non_tensor_batch["data_source"]
        processed_samples = []

        for item, (chat, token_ids) in enumerate(zip(chats, encoded_rows)):
            token_ids = list(token_ids)
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            raw_prompt_ids = token_ids
            max_length = int(self.config.data.max_prompt_length)
            if len(raw_prompt_ids) > max_length:
                truncation = self.config.data.truncation
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_length]
                elif truncation == "middle":
                    left_half = max_length // 2
                    right_half = max_length - left_half
                    raw_prompt_ids = (
                        raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                    )
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than "
                        f"{max_length}."
                    )

            anchor = obs_anchors[item] if obs_anchors is not None else None
            anchor = (
                torch_to_numpy(anchor, is_object=True)
                if isinstance(anchor, torch.Tensor)
                else anchor
            )
            row = {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "anchor_obs": anchor,
                "index": item,
                "data_source": data_sources[item],
            }
            if self.config.data.get("return_raw_chat", False):
                row["raw_prompt"] = chat
            processed_samples.append(row)

        return DataProto.from_single_dict(
            data=collate_fn(processed_samples),
            meta_info=gen_batch.meta_info,
        )


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        ""
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        for bs in range(batch_size):

            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:

                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_won'] = float(success['success_rate'][bs])

                    data['episode_lengths'] = episode_lengths[bs]

                    data['tool_callings'] = tool_callings[bs]

                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)


        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        ""

        collector_started = time.perf_counter()
        batch_size = len(gen_batch.batch)


        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]

        raw_steps_per_ep = [[] for _ in range(batch_size)] if self._dump_raw else None
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)



        rollout_rounds = 0
        legacy_full_batch_rows = 0
        active_generation_rows = 0
        vllm_request_rows = 0
        active_rows_by_round = []
        preprocess_seconds = 0.0
        generate_seconds = 0.0
        env_step_seconds = 0.0




        last_protocol_actions = [""] * batch_size

        for _step in range(self.config.env.max_steps):
            live_indices = np.flatnonzero(np.logical_not(is_done))
            if len(live_indices) == 0:
                break
            rollout_rounds += 1
            legacy_full_batch_rows += batch_size
            active_generation_rows += len(live_indices)
            active_rows_by_round.append(len(live_indices))

            if self._compact_finished_trajectories:
                generation_indices = live_indices
                active_gen_batch = gen_batch.select_idxs(generation_indices)
                active_obs = _select_observation_rows(obs, generation_indices)
            else:


                generation_indices = np.arange(batch_size)
                active_gen_batch = gen_batch
                active_obs = obs

            phase_started = time.perf_counter()
            batch = self.preprocess_batch(gen_batch=active_gen_batch, obs=active_obs)
            preprocess_seconds += time.perf_counter() - phase_started
            is_webshop = "webshop" in str(self.config.env.env_name).lower()
            prompt_opens_think = (
                self._prompts_open_think(
                    batch.batch["input_ids"], batch.batch["attention_mask"]
                ) if is_webshop else [False] * len(generation_indices)
            )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = active_gen_batch.meta_info


            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            vllm_request_rows += len(batch_input_padded)
            phase_started = time.perf_counter()
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            generate_seconds += time.perf_counter() - phase_started

            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch[generation_indices]
            batch.non_tensor_batch['traj_uid'] = traj_uid[generation_indices]

            batch = batch.union(batch_output)



            raw_text_actions = self.tokenizer.batch_decode(
                batch.batch['responses'], skip_special_tokens=False
            )
            text_actions, restored_prompt_think = zip(*[
                self._render_protocol_response(response, opens_think)
                for response, opens_think in zip(raw_text_actions, prompt_opens_think)
            ])
            text_actions = list(text_actions)
            restored_prompt_think = list(restored_prompt_think)




            raw_outputs_snapshot = list(raw_text_actions) if self._dump_raw else None
            protocol_outputs_snapshot = list(text_actions) if self._dump_raw else None


            obs_anchor_snapshot = None
            if self._dump_raw:
                obs_anchor_snapshot = active_obs.get('anchor', None)
                if obs_anchor_snapshot is None:
                    obs_anchor_snapshot = active_obs.get('text', None)







            full_text_actions = list(last_protocol_actions)
            for local_index, global_index in enumerate(generation_indices):
                full_text_actions[int(global_index)] = text_actions[local_index]
                last_protocol_actions[int(global_index)] = text_actions[local_index]
            phase_started = time.perf_counter()
            next_obs, rewards, dones, infos = envs.step(full_text_actions)
            env_step_seconds += time.perf_counter() - phase_started



            env_action_snapshot = (
                [full_text_actions[int(i)] for i in generation_indices]
                if self._dump_raw else None
            )

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:

                dones = dones.squeeze(1)

            generation_infos = [infos[int(i)] for i in generation_indices]
            if generation_infos and 'is_action_valid' in generation_infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array(
                    [info['is_action_valid'] for info in generation_infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(len(generation_indices), dtype=bool)





            batch.non_tensor_batch['non_strict_action_valid'] = np.array(
                [info.get('non_strict_action_valid', info.get('is_action_valid', True)) for info in generation_infos],
                dtype=bool,
            )
            batch.non_tensor_batch['strict_action_valid'] = np.array(
                [info.get('strict_action_valid', info.get('is_action_valid', True)) for info in generation_infos],
                dtype=bool,
            )

            live_infos = [infos[int(i)] for i in live_indices]
            if live_infos and 'tool_calling' in live_infos[0]:
                tool_callings[live_indices] += np.array(
                    [info['tool_calling'] for info in live_infos], dtype=np.float32)

            rewards_np = torch_to_numpy(rewards)
            episode_rewards[live_indices] += rewards_np[live_indices]
            episode_lengths[live_indices] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = rewards_np[generation_indices].astype(object)



            batch.non_tensor_batch['active_masks'] = np.isin(
                generation_indices, live_indices)


            batch_list: list[dict] = to_list_of_dict(batch)

            for local_index, global_index in enumerate(generation_indices):
                total_batch_list[int(global_index)].append(batch_list[local_index])
                total_infos[int(global_index)].append(infos[int(global_index)])


            if self._dump_raw and raw_steps_per_ep is not None:
                try:


                    resp_ids = None
                    prompt_ids = None
                    try:
                        resp_ids = batch.batch['responses']
                        if 'prompts' in batch.batch:
                            prompt_ids = batch.batch['prompts']
                    except Exception:
                        resp_ids = None
                    for local_index, global_index in enumerate(generation_indices):
                        i = int(global_index)
                        if i not in live_indices:
                            continue
                        raw_resp = raw_outputs_snapshot[local_index] if raw_outputs_snapshot else ""
                        protocol_resp = (
                            protocol_outputs_snapshot[local_index]
                            if protocol_outputs_snapshot else raw_resp
                        )
                        think, action = _split_think_action(protocol_resp)
                        obs_text = None
                        if obs_anchor_snapshot is not None:
                            try:
                                obs_text = obs_anchor_snapshot[local_index]
                            except Exception:
                                obs_text = None

                        env_action = None
                        if env_action_snapshot is not None:
                            try:
                                env_action = env_action_snapshot[local_index]
                            except Exception:
                                env_action = None

                        next_obs_text = None
                        try:
                            _na = next_obs.get('anchor', None) if isinstance(next_obs, dict) else None
                            if _na is None and isinstance(next_obs, dict):
                                _na = next_obs.get('text', None)
                            if _na is not None:
                                next_obs_text = _na[i]
                        except Exception:
                            next_obs_text = None
                        rec = {
                            "step": int(episode_lengths[i]),
                            "active": True,
                            "observation": obs_text,
                            "raw_response": raw_resp,
                            "protocol_response": protocol_resp,
                            "restored_prompt_think": bool(restored_prompt_think[local_index]),
                            "think": think,
                            "action": action,
                            "action_text": action or None,
                            "model_output": raw_resp,
                            "env_action": env_action,
                            "is_action_valid": int(batch.non_tensor_batch['is_action_valid'][local_index]),
                            "non_strict_action_valid": int(
                                batch.non_tensor_batch['non_strict_action_valid'][local_index]
                            ),
                            "strict_action_valid": int(
                                batch.non_tensor_batch['strict_action_valid'][local_index]
                            ),
                            "reward": float(rewards[i]),
                            "done": bool(dones[i]),
                            "won": bool(infos[i].get('won', False)) if isinstance(infos[i], dict) else False,
                            "next_observation": next_obs_text,
                        }

                        try:
                            if resp_ids is not None:
                                rec["response_token_ids"] = resp_ids[local_index].tolist()
                            if prompt_ids is not None:
                                rec["prompt_token_ids"] = prompt_ids[local_index].tolist()
                        except Exception:
                            pass
                        raw_steps_per_ep[i].append(rec)
                except Exception as e:
                    print(f"[RawDump] per-step capture failed (non-fatal): {e}")


            is_done = np.logical_or(is_done, dones)


            obs = next_obs


            if is_done.all():
                break

        self._last_rollout_compaction_metrics = _rollout_compaction_metrics(
            enabled=self._compact_finished_trajectories,
            rounds=rollout_rounds,
            full_batch_rows=legacy_full_batch_rows,
            active_rows=active_generation_rows,
            vllm_request_rows=vllm_request_rows,
            active_rows_by_round=active_rows_by_round,
            world_size=actor_rollout_wg.world_size,
            batch_tokenize_enabled=self._batch_tokenize_observations,
            collector_seconds=time.perf_counter() - collector_started,
            preprocess_seconds=preprocess_seconds,
            generate_seconds=generate_seconds,
            env_step_seconds=env_step_seconds,
        )

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )


        if self._dump_raw and raw_steps_per_ep is not None:
            self._write_raw_episode_files(
                raw_steps_per_ep=raw_steps_per_ep,
                total_infos=total_infos,
                episode_rewards=episode_rewards,
                envs=envs,
                uid_batch=uid_batch,
                traj_uid=traj_uid,
            )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def _write_raw_episode_files(self, raw_steps_per_ep, total_infos,
                                 episode_rewards, envs, uid_batch, traj_uid):
        ""
        try:
            self._rollout_call_idx += 1
            call_idx = self._rollout_call_idx
            out_dir = os.path.join(self._raw_dump_dir, f"rollout_{call_idx:04d}")
            os.makedirs(out_dir, exist_ok=True)

            tasks = getattr(envs, 'tasks', None)
            for i, steps in enumerate(raw_steps_per_ep):
                if not steps:
                    continue
                last_info = total_infos[i][-1] if total_infos[i] else {}
                won = bool(last_info.get('won', False)) if isinstance(last_info, dict) else False
                task_goal = None
                if tasks is not None:
                    try:
                        task_goal = tasks[i]
                    except Exception:
                        task_goal = None
                episode = {
                    "rollout_index": call_idx,
                    "episode_index": i,
                    "trajectory_uid": str(traj_uid[i]) if i < len(traj_uid) else None,
                    "group_uid": str(uid_batch[i]) if i < len(uid_batch) else None,
                    "task_goal": task_goal,
                    "won": won,
                    "episode_reward": float(episode_rewards[i]),
                    "num_steps": len(steps),
                    "steps": steps,
                }
                tuid = (str(traj_uid[i])[:8] if i < len(traj_uid) else f"{i:03d}")
                fpath = os.path.join(out_dir, f"ep_{i:03d}_{tuid}.json")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(episode, f, ensure_ascii=False, indent=2)
            print(f"[RawDump] wrote {sum(1 for s in raw_steps_per_ep if s)} "
                  f"episode files → {out_dir}")
        except Exception as e:
            print(f"[RawDump] episode file write failed (non-fatal): {e}")

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        ""
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards,
                                                                                                episode_lengths=episode_lengths,
                                                                                                success=success,
                                                                                                traj_uid=traj_uid,
                                                                                                tool_callings=tool_callings,
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        ""
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)


        if self.config.algorithm.filter_groups.enable and is_train:

            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:

            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)



        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )

        return gen_batch_output
