"""WebShop adapters for the standalone frozen-model CoSkill driver.

The production GRPO path wraps WebShop in Ray actors.  The no-RL experiment
already uses persistent GPU worker processes, so nesting a second Ray runtime in
each worker is both expensive and unnecessary.  This module provides the same
observable contract in-process while sharing one product server across the
eight rollouts of each GRPO goal.  Sessions remain independent through unique
prefixes, and all eight replicas receive the same sampled instruction.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent_system.environments.prompts.webshop import (
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    WEBSHOP_TEMPLATE_WITH_MEMORY,
)
from agent_system.memory import SimpleMemory
from agent_system.memory.skills_only_memory import SkillsOnlyMemory


_WEBSHOP_PACKAGE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "agent_system",
    "environments", "env_package", "webshop", "webshop",
))


def _load_webshop_env_class(data_dir: Optional[str] = None):
    if _WEBSHOP_PACKAGE not in sys.path:
        sys.path.insert(0, _WEBSHOP_PACKAGE)
    from web_agent_site.envs import WebAgentTextEnv
    if data_dir:
        # WebShop's engine keeps HUMAN_ATTR_PATH and the Lucene index root as
        # imported module globals, so passing file_path/attr_path alone is not
        # enough when large assets live in another checkout. Redirect both to
        # the installation that owns WEBSHOP_DATA_DIR.
        import web_agent_site.engine.engine as engine
        import web_agent_site.utils as utils
        data_dir = os.path.realpath(data_dir)
        webshop_root = os.path.dirname(data_dir)
        human_attr = os.path.join(data_dir, "items_human_ins.json")
        search_index = os.path.join(webshop_root, "search_engine", "indexes")
        if not os.path.isfile(human_attr):
            raise FileNotFoundError(
                f"WebShop also requires items_human_ins.json beside the small data: {human_attr}")
        if not os.path.isdir(search_index):
            raise FileNotFoundError(
                f"WebShop Lucene index not found: {search_index}")
        engine.HUMAN_ATTR_PATH = human_attr
        utils.HUMAN_ATTR_PATH = human_attr
        # engine.init_search_engine resolves ../search_engine from BASE_DIR.
        engine.BASE_DIR = os.path.join(webshop_root, "web_agent_site")
    return WebAgentTextEnv


def extract_webshop_task(observation: str) -> str:
    """Extract the instruction using the production manager's SEP contract."""
    parts = observation.split(" [SEP] ")
    for i, part in enumerate(parts[:-1]):
        if part.strip().strip("'").lower() == "instruction:":
            return parts[i + 1].strip().strip("'")
    raise ValueError(f"WebShop observation has no Instruction field: {observation[:240]!r}")


def format_webshop_observation(observation: str, task: str) -> str:
    """Remove the repeated WebShop/instruction prefix like env_manager.py."""
    parts = observation.split(" [SEP] ")
    try:
        index = parts.index(task)
        return " [SEP] ".join(f"'{part}'" for part in parts[index + 1:])
    except ValueError:
        return observation


def format_webshop_actions(available: Dict[str, Any]) -> List[str]:
    unknown = set(available) - {"has_search_bar", "clickables"}
    if unknown:
        raise ValueError(f"Unknown WebShop available-action fields: {sorted(unknown)}")
    actions = []
    if available.get("has_search_bar"):
        actions.append("search[<your query>]")
    actions.extend(f"click[{text}]" for text in available.get("clickables", []))
    return actions


def webshop_trace_observation(observation: str) -> str:
    """Turn SEP pages into lines so TracesPool can produce useful line diffs."""
    return "\n".join(part.strip().strip("'") for part in observation.split(" [SEP] ")
                     if part.strip().strip("'"))


class LocalBatchWebShopEnv:
    """In-process WebShop batch matching ``train_data_size × group_size``.

    ``base_task_count`` is the number of distinct instructions owned by this
    data-parallel worker. Each instruction is repeated ``group_size`` times,
    exactly like ``env.rollout.n``; the CoSkill comparison defaults are 12×6.
    """

    def __init__(self, *, seed: int, base_task_count: int, group_size: int,
                 base_task_offset: int, total_base_tasks: int,
                 file_path: str, attr_path: str):
        WebAgentTextEnv = _load_webshop_env_class(os.path.dirname(file_path))
        self.seed = int(seed)
        self.base_task_count = int(base_task_count)
        self.group_size = int(group_size)
        self.base_task_offset = int(base_task_offset)
        self.total_base_tasks = int(total_base_tasks)
        self.batch_size = self.base_task_count * self.group_size
        self.envs = []
        self._env_groups = []
        self._last_tasks = []

        for local_base in range(self.base_task_count):
            global_base = self.base_task_offset + local_base
            shared_server = None
            replicas = []
            for replica in range(self.group_size):
                env = WebAgentTextEnv(
                    observation_mode="text",
                    num_products=None,
                    human_goals=False,
                    file_path=file_path,
                    attr_path=attr_path,
                    seed=self.seed + global_base,
                    server=shared_server,
                    session_prefix=f"dp{global_base:02d}r{replica:02d}-",
                )
                if shared_server is None:
                    shared_server = env.server
                replicas.append(env)
                self.envs.append(env)
            self._env_groups.append(replicas)

        if not self.envs:
            raise ValueError("LocalBatchWebShopEnv requires at least one environment")
        self.num_goals = len(self.envs[0].server.goals)
        if self.num_goals <= 500:
            raise ValueError(f"WebShop train split requires >500 goals, found {self.num_goals}")

    def _goal_indices(self, group_id: int) -> List[int]:
        # Reconstruct the production RandomState deterministically. Both GPU
        # workers therefore see disjoint shards of the same sampled goal set
        # (12 goals under the CoSkill comparison standard).
        rng = np.random.RandomState(self.seed)
        selected = None
        for _ in range(int(group_id)):
            selected = rng.choice(
                np.arange(500, self.num_goals),
                size=self.total_base_tasks,
                replace=False,
            ).tolist()
        return selected or []

    def reset(self, group_id: int = 1):
        goal_indices = self._goal_indices(group_id)
        owned = goal_indices[
            self.base_task_offset:self.base_task_offset + self.base_task_count
        ]
        observations, infos, tasks = [], [], []
        for goal_idx, replicas in zip(owned, self._env_groups):
            for env in replicas:
                observation, _ = env.reset(session=int(goal_idx))
                task = extract_webshop_task(observation)
                observations.append(observation)
                tasks.append(task)
                infos.append({
                    "available_actions": env.get_available_actions(),
                    "won": False,
                    "task_score": 0.0,
                    "goal_index": int(goal_idx),
                })
        self._last_tasks = tasks
        return observations, infos

    def step(self, actions: List[str]):
        if len(actions) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} actions, got {len(actions)}")
        observations, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            observation, raw_score, done, _ = env.step(action)
            raw_score = float(raw_score or 0.0)
            won = bool(done and abs(raw_score - 1.0) < 1e-8)
            observations.append(observation)
            rewards.append(10.0 if won else 0.0)
            dones.append(bool(done))
            infos.append({
                "available_actions": env.get_available_actions(),
                "won": won,
                "task_score": raw_score,
            })
        return observations, rewards, dones, infos

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


class WebShopObsBuilder:
    """Prompt builder mirroring ``WebshopEnvironmentManager.build_text_obs``."""

    def __init__(self, *, mem_lib=None, skills_json_path: Optional[str] = None,
                 history_length: int = 8, with_skills: bool = True,
                 top_k: int = 6, enable_skill_tree: bool = True,
                 prompt_char_limit: int = 13000):
        self.mem_lib = mem_lib or SkillsOnlyMemory(
            skills_json_path=skills_json_path,
            retrieval_mode="template",
            enable_playbook=enable_skill_tree,
        )
        self.history_length = int(history_length)
        self.with_skills = bool(with_skills)
        self.top_k = int(top_k)
        self.enable_skill_tree = bool(getattr(
            self.mem_lib, "enable_playbook", enable_skill_tree))
        self.prompt_char_limit = int(prompt_char_limit)
        self.history = SimpleMemory()
        self.retrieved = None
        self.task = ""

    def _retrieval_active(self):
        return self.enable_skill_tree or self.with_skills

    def reset(self, task: str):
        self.task = task
        self.history.reset(1)
        self.retrieved = self.mem_lib.retrieve(task_description=task, top_k=self.top_k)
        if not self.with_skills:
            self.retrieved = dict(self.retrieved)
            self.retrieved["general_skills"] = []
            self.retrieved["task_specific_skills"] = []
            self.retrieved["mistakes_to_avoid"] = []
            self.retrieved["injected_skill_ids"] = []

    @staticmethod
    def _actions_text(available: Dict[str, Any]) -> str:
        return "\n".join(f"'{action}'," for action in format_webshop_actions(available))

    def _initial_prompt(self, observation: str, available: Dict[str, Any]) -> str:
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=self.task,
            current_observation=observation,
            available_actions=self._actions_text(available),
        )
        # CoSkill uses the same first-step policy as the ALFWorld no-RL driver:
        # the compact skill tree is available immediately, while flat bullets
        # enter through the normal WITH_MEMORY branch from step 2 onward.
        tree = (self.retrieved or {}).get("playbook")
        if self.enable_skill_tree and isinstance(tree, str) and tree.strip():
            prompt = tree.strip() + "\n\n" + prompt
        return prompt

    def build(self, observation: str, available: Dict[str, Any], init: bool) -> str:
        if init or self.history_length <= 0:
            return self._initial_prompt(observation, available)

        memory_contexts, valid_lens = self.history.fetch(
            self.history_length, obs_key="text_obs", action_key="action")
        step_count = len(self.history[0])
        fields = dict(
            task_description=self.task,
            step_count=step_count,
            history_length=valid_lens[0],
            action_history=memory_contexts[0],
            current_step=step_count + 1,
            current_observation=observation,
            available_actions=self._actions_text(available),
        )
        if self._retrieval_active():
            fields["retrieved_memories"] = self.mem_lib.format_for_prompt(self.retrieved)
            prompt = WEBSHOP_TEMPLATE_WITH_MEMORY.format(**fields)
        else:
            prompt = WEBSHOP_TEMPLATE.format(**fields)

        # Match the production WebShop manager's character guard.  Keep the
        # current tree on fallback because it is CoSkill's learned policy state.
        if len(prompt) > self.prompt_char_limit:
            prompt = self._initial_prompt(observation, available)
        return prompt

    def record(self, observation: str, action: str):
        self.history.store({"text_obs": [observation], "action": [action]})
