import copy
import glob
import json
import os
import random

import yaml

from alfworld.agents.environment import get_environment


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TW_CONFIG = os.path.normpath(
    os.path.join(
        _THIS_DIR,
        "..",
        "environments",
        "env_package",
        "alfworld",
        "configs",
        "config_tw.yaml",
    )
)
_TASK_TYPE_TO_ID = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}


def extract_task(obs_text):
    marker = "Your task is to: "
    start = obs_text.find(marker)
    if start == -1:
        return ""
    return obs_text[start + len(marker):].strip()


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _split_root(alfworld_data, split):
    alfworld_data = alfworld_data or os.environ.get("ALFWORLD_DATA")
    if not alfworld_data:
        raise ValueError("ALFWORLD_DATA is required")
    split_dir = {
        "train": "json_2.1.1/train",
        "valid_seen": "json_2.1.1/valid_seen",
        "valid_unseen": "json_2.1.1/valid_unseen",
    }[split]
    root = os.path.join(alfworld_data, split_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    return root


def find_games_by_type(
    task_type,
    alfworld_data=None,
    split="train",
    limit=None,
    verbose=True,
    match=None,
    sample_n=None,
    sample_seed=0,
    diverse_by_object=True,
):
    root = _split_root(alfworld_data, split)
    results = []
    trajectories = sorted(
        glob.glob(os.path.join(root, "**", "traj_data.json"), recursive=True)
    )
    hard_limit = limit if limit and not sample_n else None
    for trajectory_path in trajectories:
        folder = os.path.dirname(trajectory_path)
        if "movable" in folder or "Sliced" in folder:
            continue
        game_file = os.path.join(folder, "game.tw-pddl")
        if not os.path.exists(game_file):
            continue
        trajectory = _read_json(trajectory_path)
        if not trajectory or trajectory.get("task_type") != task_type:
            continue
        if match and not match(trajectory):
            continue
        game = _read_json(game_file)
        if not game or not game.get("solvable", False):
            continue
        results.append((game_file, trajectory))
        if hard_limit and len(results) >= hard_limit:
            break
    if sample_n and len(results) > sample_n:
        results = _sample_diverse(
            results,
            sample_n,
            sample_seed,
            diverse_by_object,
        )
    if verbose:
        print(f"[games] task_type={task_type} count={len(results)} root={root}")
    return results


def _sample_diverse(results, count, seed, diverse_by_object):
    rng = random.Random(seed)
    if not diverse_by_object:
        indices = sorted(rng.sample(range(len(results)), count))
        return [results[index] for index in indices]
    groups = {}
    for item in results:
        params = item[1].get("pddl_params", {}) or {}
        object_type = str(params.get("object_target", "?")).lower()
        groups.setdefault(object_type, []).append(item)
    for values in groups.values():
        rng.shuffle(values)
    order = sorted(groups)
    rng.shuffle(order)
    selected = []
    while len(selected) < count and any(groups[key] for key in order):
        for key in order:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) >= count:
                    break
    return selected


def load_tw_config_types(task_type_ids, config_path=None, num_games=-1):
    config_path = config_path or DEFAULT_TW_CONFIG
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["env"]["type"] = "AlfredTWEnv"
    config["env"]["task_types"] = list(task_type_ids)
    config["env"]["domain_randomization"] = False
    config["env"]["expert_type"] = "handcoded"
    config["dataset"]["num_train_games"] = num_games
    config["dataset"]["num_eval_games"] = num_games
    return config


def make_single_env(game_files, config, seed=0):
    return make_batch_env(game_files, config, batch_size=1, seed=seed)


def make_batch_env(game_files, config, batch_size=1, seed=0):
    resolved = copy.deepcopy(config)
    environment_class = get_environment(resolved["env"]["type"])
    base = environment_class(resolved, train_eval="train")
    base.game_files = list(game_files)
    base.num_games = len(base.game_files)
    environment = base.init_env(batch_size=batch_size)
    environment.seed(seed)
    return environment
