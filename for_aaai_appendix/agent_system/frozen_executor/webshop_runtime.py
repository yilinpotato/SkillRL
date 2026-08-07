""

import os
import sys
from typing import List, Optional

import numpy as np

_WEBSHOP_PACKAGE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "environments",
    "env_package", "webshop", "webshop",
))


def _load_webshop_env_class(data_dir: Optional[str] = None):
    if _WEBSHOP_PACKAGE not in sys.path:
        sys.path.insert(0, _WEBSHOP_PACKAGE)
    from web_agent_site.envs import WebAgentTextEnv
    if data_dir:




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

        engine.BASE_DIR = os.path.join(webshop_root, "web_agent_site")
    return WebAgentTextEnv


def extract_webshop_task(observation: str) -> str:
    ""
    parts = observation.split(" [SEP] ")
    for i, part in enumerate(parts[:-1]):
        if part.strip().strip("'").lower() == "instruction:":
            return parts[i + 1].strip().strip("'")
    raise ValueError(f"WebShop observation has no Instruction field: {observation[:240]!r}")


def format_webshop_observation(observation: str, task: str) -> str:
    ""
    parts = observation.split(" [SEP] ")
    try:
        index = parts.index(task)
        return " [SEP] ".join(f"'{part}'" for part in parts[index + 1:])
    except ValueError:
        return observation


def webshop_trace_observation(observation: str) -> str:
    ""
    return "\n".join(part.strip().strip("'") for part in observation.split(" [SEP] ")
                     if part.strip().strip("'"))


class LocalBatchWebShopEnv:
    ""

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



        rng = np.random.RandomState(self.seed)
        selected = None
        for _ in range(int(group_id)):
            selected = rng.choice(
                np.arange(500, self.num_goals),
                size=self.total_base_tasks,
                replace=False,
            ).tolist()
        return selected or []

    def reset(self, group_id: int = 1, goal_indices: Optional[List[int]] = None):
        ""
        if goal_indices is None:
            all_goal_indices = self._goal_indices(group_id)
            owned = all_goal_indices[
                self.base_task_offset:self.base_task_offset + self.base_task_count
            ]
        else:
            owned = [int(index) for index in goal_indices]
            if len(owned) != self.base_task_count:
                raise ValueError(
                    f"Expected {self.base_task_count} explicit WebShop goals, "
                    f"got {len(owned)}")
            if any(index < 0 or index >= self.num_goals for index in owned):
                raise ValueError(
                    f"Explicit WebShop goal index out of range [0, {self.num_goals}): "
                    f"{owned}")
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
