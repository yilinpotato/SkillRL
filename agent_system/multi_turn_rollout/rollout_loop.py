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
    """Read a key from an OmegaConf/dataclass/dict config, tolerating absence."""
    if cfg is None:
        return default
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        # OmegaConf DictConfig supports .get; plain objects use getattr.
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
    """Extract (think, action) from a raw model output.

    Matches the project's parsing convention: thinking lives inside
    <think>...</think> and the chosen action inside <action>...</action>
    (see alfworld/projection.py). Returns ("", "") components that are missing.
    """
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


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        # --- Raw trajectory dumping (per-episode files) ---------------------
        # Optional: when enabled, every rollout's raw step-by-step trajectory is
        # written to one JSON file per episode (observation / raw model output /
        # think / action / reward / done / won). Controlled via config so it can
        # be turned off to avoid disk overhead. Disabled unless explicitly set.
        env_cfg = getattr(self.config, "env", None)
        self._dump_raw = bool(_cfg_get(env_cfg, "dump_raw_trajectories", False))
        # Where to write. Falls back to <trainer.default_local_dir>/raw_episodes.
        trainer_cfg = getattr(self.config, "trainer", None)
        default_dir = _cfg_get(trainer_cfg, "default_local_dir", None) or "."
        self._raw_dump_dir = _cfg_get(env_cfg, "raw_traj_dir", None) or os.path.join(
            str(default_dir), "raw_episodes"
        )
        # Monotonic counter labelling which rollout (collection call) we are on.
        self._rollout_call_idx = 0

    def _prompts_open_think(self, input_ids, attention_mask):
        """Return whether each rendered Qwen prompt already opened ``<think>``.

        Qwen Thinking's legacy chat template can append ``<think>`` to the
        prompt itself.  In that valid wire format the sampled response is only
        the continuation (reasoning text followed by ``</think>`` and the
        action), so strict WebShop accounting must validate the rendered
        prompt+completion transcript rather than the completion in isolation.
        """
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
        """Restore only a prompt-supplied opener for WebShop protocol checks.

        This leaves ``batch.batch['responses']`` untouched: PPO log-probs and
        gradients continue to use exactly the sampled token sequence.  The
        returned text is only the complete transcript passed to the environment
        and used for strict valid-action accounting/debug traces.
        """
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
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
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

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
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

        # Build final output dict
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
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
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
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        # Per-episode raw step records (only populated when dumping is enabled).
        raw_steps_per_ep = [[] for _ in range(batch_size)] if self._dump_raw else None
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            is_webshop = "webshop" in str(self.config.env.env_name).lower()
            prompt_opens_think = (
                self._prompts_open_think(
                    batch.batch["input_ids"], batch.batch["attention_mask"]
                ) if is_webshop else [False] * batch_size
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

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            # Preserve Qwen's <think>/<action> protocol tokens until WebShop
            # projection validates and extracts the executable action.
            raw_text_actions = self.tokenizer.batch_decode(
                batch.batch['responses'], skip_special_tokens=False
            )
            text_actions, restored_prompt_think = zip(*[
                self._render_protocol_response(response, opens_think)
                for response, opens_think in zip(raw_text_actions, prompt_opens_think)
            ])
            text_actions = list(text_actions)
            restored_prompt_think = list(restored_prompt_think)

            # Snapshot raw model outputs BEFORE envs.step(): the projection step
            # mutates text_actions in place (lowercases / truncates), so we must
            # copy here to preserve the true model output for the raw dump.
            raw_outputs_snapshot = list(raw_text_actions) if self._dump_raw else None
            protocol_outputs_snapshot = list(text_actions) if self._dump_raw else None
            # Raw env observation shown to the model this step (anchor = env text
            # before skills are concatenated); 'text' is the full built prompt.
            obs_anchor_snapshot = None
            if self._dump_raw:
                obs_anchor_snapshot = obs.get('anchor', None)
                if obs_anchor_snapshot is None:
                    obs_anchor_snapshot = obs.get('text', None)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            # After step(), text_actions has been projected in place to the
            # admissible env action (lowercased / extracted). Snapshot it as the
            # actual action sent to the env, for the raw dump.
            env_action_snapshot = list(text_actions) if self._dump_raw else None
            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # ``is_action_valid`` remains the predicate used for the reward
            # penalty.  Keep the strict/non-strict diagnostic predicates beside
            # it so their rates can be reported without changing any reward.
            # Environments without an explicit pair retain their legacy value.
            batch.non_tensor_batch['non_strict_action_valid'] = np.array(
                [info.get('non_strict_action_valid', info.get('is_action_valid', True)) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['strict_action_valid'] = np.array(
                [info.get('strict_action_valid', info.get('is_action_valid', True)) for info in infos],
                dtype=bool,
            )

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # --- Raw trajectory capture (per active episode) -----------------
            if self._dump_raw and raw_steps_per_ep is not None:
                try:
                    # Per-step token ids let the Skill2param internalization stage
                    # rebuild SFT batches without re-tokenizing. Decoded lazily.
                    resp_ids = None
                    prompt_ids = None
                    try:
                        resp_ids = batch.batch['responses']
                        if 'prompts' in batch.batch:
                            prompt_ids = batch.batch['prompts']
                    except Exception:
                        resp_ids = None
                    for i in range(batch_size):
                        if is_done[i]:
                            continue  # episode already finished; skip padding steps
                        raw_resp = raw_outputs_snapshot[i] if raw_outputs_snapshot else ""
                        protocol_resp = (
                            protocol_outputs_snapshot[i]
                            if protocol_outputs_snapshot else raw_resp
                        )
                        think, action = _split_think_action(protocol_resp)
                        obs_text = None
                        if obs_anchor_snapshot is not None:
                            try:
                                obs_text = obs_anchor_snapshot[i]
                            except Exception:
                                obs_text = None
                        # Projected/admissible action actually sent to the env.
                        env_action = None
                        if env_action_snapshot is not None:
                            try:
                                env_action = env_action_snapshot[i]
                            except Exception:
                                env_action = None
                        # Resulting observation after this action (next anchor).
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
                            "restored_prompt_think": bool(restored_prompt_think[i]),
                            "think": think,
                            "action": action,
                            "action_text": action or None,
                            "model_output": raw_resp,
                            "env_action": env_action,
                            "is_action_valid": int(batch.non_tensor_batch['is_action_valid'][i]),
                            "non_strict_action_valid": int(
                                batch.non_tensor_batch['non_strict_action_valid'][i]
                            ),
                            "strict_action_valid": int(
                                batch.non_tensor_batch['strict_action_valid'][i]
                            ),
                            "reward": float(rewards[i]),
                            "done": bool(dones[i]),
                            "won": bool(infos[i].get('won', False)) if isinstance(infos[i], dict) else False,
                            "next_observation": next_obs_text,
                        }
                        # Token ids for SFT reuse (optional; best-effort).
                        try:
                            if resp_ids is not None:
                                rec["response_token_ids"] = resp_ids[i].tolist()
                            if prompt_ids is not None:
                                rec["prompt_token_ids"] = prompt_ids[i].tolist()
                        except Exception:
                            pass
                        raw_steps_per_ep[i].append(rec)
                except Exception as e:
                    print(f"[RawDump] per-step capture failed (non-fatal): {e}")

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        # Dump each episode's raw trajectory to its own file (best-effort).
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
        """Write one JSON file per episode with its full raw trajectory.

        Layout: <raw_traj_dir>/rollout_<NNNN>/ep_<idx>_<traj_uid8>.json
        Each file: rollout index, episode index, task goal, won/reward, steps[].
        Wrapped so a dump failure never disrupts training.
        """
        try:
            self._rollout_call_idx += 1
            call_idx = self._rollout_call_idx
            out_dir = os.path.join(self._raw_dump_dir, f"rollout_{call_idx:04d}")
            os.makedirs(out_dir, exist_ok=True)
            # Task goals live on the env manager (set at reset()).
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
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
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
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
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
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
