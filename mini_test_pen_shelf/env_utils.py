"""
env_utils.py — 迷你测试的环境工具（纯 CPU，不依赖任何模型）

职责：
  1. load_tw_config():  读取项目里的 config_tw.yaml 并改成「只跑 pick_and_place_simple、单环境」的轻量配置
  2. find_pen_shelf_games(): 在 $ALFWORLD_DATA 里筛选出 "把 pen/pencil 放到 shelf" 的游戏
  3. extract_pen_ground_truth(): 不跑环境，直接解析 game.tw-pddl / traj_data.json 拿到 pen 的真值初始位置
  4. make_single_env():  用筛选好的 game 列表构造一个 batch_size=1 的 AlfredTWEnv

全部是文本/PDDL 解析，零 GPU。
"""
import os
import re
import glob
import json
import copy

# 项目内 alfworld 包路径
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
    get_environment,
)

# 与 envs.py 里 load_config_file 等价
import yaml

# 项目自带的 tw 配置
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TW_CONFIG = os.path.normpath(
    os.path.join(
        _THIS_DIR,
        "..",
        "agent_system",
        "environments",
        "env_package",
        "alfworld",
        "configs",
        "config_tw.yaml",
    )
)

# pen→shelf 属于 task_type 1: pick_and_place_simple
PEN_OBJECTS = {"pen", "pencil"}
SHELF_RECEPS = {"shelf"}


def load_tw_config(config_path=None, num_games=-1):
    """读取 config_tw.yaml，改成只跑 pick&place 的轻量纯文本配置。"""
    config_path = config_path or DEFAULT_TW_CONFIG
    assert os.path.exists(config_path), f"找不到配置文件: {config_path}"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 强制纯文本环境（不渲染、不吃 GPU）
    config["env"]["type"] = "AlfredTWEnv"
    # 只保留 task_type 1 (pick_and_place_simple)，pen→shelf 就在其中
    config["env"]["task_types"] = [1]
    # 关闭领域随机化，保证可复现
    config["env"]["domain_randomization"] = False
    # 不需要专家计划，省去 handcoded expert 开销
    config["env"]["expert_type"] = "handcoded"
    # 限制游戏数量（-1 表示全部）
    config["dataset"]["num_train_games"] = num_games
    config["dataset"]["num_eval_games"] = num_games
    return config


def _read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _traj_is_pen_shelf(traj_data):
    """根据 traj_data.json 判断是否 pen/pencil → shelf。"""
    if not traj_data:
        return False
    if traj_data.get("task_type") != "pick_and_place_simple":
        return False
    pp = traj_data.get("pddl_params", {}) or {}
    obj = str(pp.get("object_target", "")).lower()
    parent = str(pp.get("mrecep_target", "") or pp.get("parent_target", "")).lower()
    # parent_target 字段在不同版本叫法不同，这里都兼容
    if not parent:
        parent = str(pp.get("parent_target", "")).lower()
    obj_ok = any(p in obj for p in PEN_OBJECTS)
    parent_ok = any(s in parent for s in SHELF_RECEPS)
    return obj_ok and parent_ok


def find_pen_shelf_games(alfworld_data=None, split="train", limit=None, verbose=True):
    """
    遍历数据集目录，返回 pen→shelf 的 (game_file_path, traj_data) 列表。
    split: train / valid_seen / valid_unseen
    """
    alfworld_data = alfworld_data or os.environ.get("ALFWORLD_DATA")
    assert alfworld_data, "请先 export ALFWORLD_DATA=/path/to/alfworld"

    split_dir = {
        "train": "json_2.1.1/train",
        "valid_seen": "json_2.1.1/valid_seen",
        "valid_unseen": "json_2.1.1/valid_unseen",
    }[split]
    root = os.path.join(alfworld_data, split_dir)
    assert os.path.isdir(root), f"找不到数据目录: {root}"

    results = []
    traj_files = glob.glob(os.path.join(root, "**", "traj_data.json"), recursive=True)
    if verbose:
        print(f"[find] 扫描 {len(traj_files)} 个 traj_data.json @ {root}")

    for tj in traj_files:
        folder = os.path.dirname(tj)
        if "movable" in folder or "Sliced" in folder:
            continue
        game_file = os.path.join(folder, "game.tw-pddl")
        if not os.path.exists(game_file):
            continue
        traj = _read_json(tj)
        if not _traj_is_pen_shelf(traj):
            continue
        # 确认 game 标记 solvable
        gd = _read_json(game_file)
        if not gd or not gd.get("solvable", False):
            continue
        results.append((game_file, traj))
        if limit and len(results) >= limit:
            break

    if verbose:
        print(f"[find] 命中 pen→shelf 游戏: {len(results)} 个")
    return results


# ----------------------------------------------------------------------------
# 通用筛选：按 task_type 取游戏（pick_and_place_simple / pick_two_obj_and_place ...）
# ----------------------------------------------------------------------------
_TASK_TYPE_TO_ID = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}


def _split_root(alfworld_data, split):
    alfworld_data = alfworld_data or os.environ.get("ALFWORLD_DATA")
    assert alfworld_data, "请先 export ALFWORLD_DATA=/path/to/alfworld"
    split_dir = {
        "train": "json_2.1.1/train",
        "valid_seen": "json_2.1.1/valid_seen",
        "valid_unseen": "json_2.1.1/valid_unseen",
    }[split]
    root = os.path.join(alfworld_data, split_dir)
    assert os.path.isdir(root), f"找不到数据目录: {root}"
    return root


def find_games_by_type(task_type, alfworld_data=None, split="train",
                       limit=None, verbose=True, match=None,
                       sample_n=None, sample_seed=0, diverse_by_object=True):
    """
    返回指定 task_type 的 (game_file, traj_data) 列表。
    task_type: 'pick_and_place_simple' / 'pick_two_obj_and_place' / ...
    match: 可选回调 (traj_data)->bool，用于进一步精筛（如某个具体 object/parent）。
    limit: 朴素「取前 N」（glob 排序后），会让前 N 个全是同一物体 → 过拟合风险。
    sample_n: 若给定，则【扫全部命中后】抽 N 个，优先用 diverse_by_object 跨物体均匀抽样，
              避免样本集中在 newspaper/book 等少数物体上。可复现（sample_seed）。
    """
    root = _split_root(alfworld_data, split)
    results = []
    # glob 不保证顺序，排序以保证可复现
    traj_files = sorted(glob.glob(os.path.join(root, "**", "traj_data.json"), recursive=True))
    if verbose:
        print(f"[find] 扫描 {len(traj_files)} 个 traj_data.json @ {root}")
    # 朴素 limit 模式（仅当未要求 sample_n 时启用 early-break 省时）
    hard_limit = limit if (limit and not sample_n) else None
    for tj in traj_files:
        folder = os.path.dirname(tj)
        if "movable" in folder or "Sliced" in folder:
            continue
        game_file = os.path.join(folder, "game.tw-pddl")
        if not os.path.exists(game_file):
            continue
        traj = _read_json(tj)
        if not traj or traj.get("task_type") != task_type:
            continue
        if match and not match(traj):
            continue
        gd = _read_json(game_file)
        if not gd or not gd.get("solvable", False):
            continue
        results.append((game_file, traj))
        if hard_limit and len(results) >= hard_limit:
            break

    # 均匀抽样：跨物体分散，避免过拟合到少数物体
    if sample_n and len(results) > sample_n:
        results = _sample_diverse(results, sample_n, sample_seed, diverse_by_object)

    if verbose:
        objs = {}
        for _, tj in results:
            o = str((tj.get("pddl_params", {}) or {}).get("object_target", "?")).lower()
            objs[o] = objs.get(o, 0) + 1
        print(f"[find] 命中 {task_type} 游戏: {len(results)} 个  物体分布: {objs}")
    return results


def _sample_diverse(results, n, seed, diverse_by_object):
    """从 results 抽 n 个。diverse_by_object=True 时按 object_target 分组轮转抽取，
    使样本尽量覆盖多种物体；否则均匀间隔抽。可复现。"""
    import random as _random
    rng = _random.Random(seed)
    if not diverse_by_object:
        idx = sorted(rng.sample(range(len(results)), n))
        return [results[i] for i in idx]
    # 按物体分组
    groups = {}
    for item in results:
        o = str((item[1].get("pddl_params", {}) or {}).get("object_target", "?")).lower()
        groups.setdefault(o, []).append(item)
    for o in groups:
        rng.shuffle(groups[o])
    # 轮转：每种物体轮流取一个，直到取够 n
    order = sorted(groups)  # 可复现
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(groups[o] for o in order):
        for o in order:
            if groups[o]:
                picked.append(groups[o].pop())
                if len(picked) >= n:
                    break
    return picked


def extract_task_target(traj_data):
    """从 traj_data 的 pddl_params 取 (object, parent_recep, mrecep, count)。
    object: 要搬的物体类型（小写，如 'pen'/'newspaper'）。
    parent: 最终放置容器（小写，如 'shelf'/'sofa'）。
    mrecep: 中间可移动容器（如 bowl/box），多数任务为空。
    count:  需要搬几个（pick_two_obj_and_place=2，其余=1）。"""
    pp = (traj_data or {}).get("pddl_params", {}) or {}
    obj = str(pp.get("object_target", "")).strip().lower() or None
    parent = str(pp.get("parent_target", "")).strip().lower() or None
    mrecep = str(pp.get("mrecep_target", "")).strip().lower() or None
    tt = (traj_data or {}).get("task_type", "")
    count = 2 if tt == "pick_two_obj_and_place" else 1
    return {"object": obj, "parent": parent, "mrecep": mrecep, "count": count,
            "task_type": tt}


def load_tw_config_types(task_type_ids, config_path=None, num_games=-1):
    """与 load_tw_config 相同，但允许指定多个 task_type id。"""
    cfg = load_tw_config(config_path=config_path, num_games=num_games)
    cfg["env"]["task_types"] = list(task_type_ids)
    return cfg


# ----------------------------------------------------------------------------
# 真值提取：不跑环境，直接从 game.tw-pddl 解析 pen 的初始所在 receptacle
# ----------------------------------------------------------------------------
# game.tw-pddl 的 init facts 里，物体所在容器通常写成谓词:
#   (inreceptacle pen_1 drawer_2)  或  (in pen_1 drawer_2)  或  (on pen_1 desk_1)
# 物体/容器名形如 Pen_xxx、Drawer_xxx、Shelf_xxx、Desk_xxx ...
_RECEP_PRED = re.compile(
    r"\(\s*(?:inreceptacle|in|on|onreceptacle)\s+"
    r"([A-Za-z]+)[\w\-]*\s+"          # 物体类型，如 Pen
    r"([A-Za-z]+)[\w\-]*\s*\)",        # 容器类型，如 Drawer
    re.IGNORECASE,
)


def _normalize_type(name):
    """把 'Pen_bar1_xyz' / 'pen_1' 归一成 'pen'。"""
    # 去掉下划线后缀和数字
    base = re.split(r"[_\d]", name, maxsplit=1)[0]
    return base.lower()


def extract_pen_ground_truth(game_file, traj_data=None):
    """
    返回 dict:
      {
        'object_target': 'pen' / 'pencil',
        'parent_target': 'shelf',
        'pen_locations': ['drawer', 'desk', ...],   # pen/pencil 真值初始所在的容器类型（去重）
        'task_desc': '...'                           # 若 traj_data 里有
      }
    解析失败时 pen_locations 为空列表，不抛异常。
    """
    info = {
        "object_target": None,
        "parent_target": None,
        "pen_locations": [],
        "task_desc": None,
    }

    if traj_data:
        pp = traj_data.get("pddl_params", {}) or {}
        info["object_target"] = str(pp.get("object_target", "")).lower() or None
        info["parent_target"] = (
            str(pp.get("parent_target", "") or pp.get("mrecep_target", "")).lower()
            or None
        )
        # 任务自然语言描述（若有标注）
        anns = traj_data.get("turk_annotations", {}).get("anns", [])
        if anns:
            info["task_desc"] = anns[0].get("task_desc")

    gd = _read_json(game_file)
    if not gd:
        return info

    # 把整个 game 文件转成文本再正则（兼容任意嵌套结构）
    blob = json.dumps(gd)
    locations = []
    for obj_type, recep_type in _RECEP_PRED.findall(blob):
        if _normalize_type(obj_type) in PEN_OBJECTS:
            locations.append(_normalize_type(recep_type))
    # 去重并保持顺序
    seen = set()
    info["pen_locations"] = [x for x in locations if not (x in seen or seen.add(x))]
    return info


def make_single_env(game_files, config, seed=0):
    """
    用给定的 game_files 列表构造一个 batch_size=1 的 AlfredTWEnv（底层 gym env）。
    直接复用项目里的 AlfredTWEnv，但绕开 envs.py 的 Ray 封装 —— 我们只要一个进程内单环境。
    """
    cfg = copy.deepcopy(config)
    EnvClass = get_environment(cfg["env"]["type"])  # AlfredTWEnv
    base = EnvClass(cfg, train_eval="train")
    # 覆盖它扫描到的 game 列表，强制只用我们筛选好的 pen→shelf 游戏
    base.game_files = list(game_files)
    base.num_games = len(base.game_files)
    env = base.init_env(batch_size=1)
    env.seed(seed)
    return env


def make_batch_env(game_files, config, batch_size=1, seed=0):
    """
    用给定的 game_files 列表构造一个 batch_size=N 的 AlfredTWEnv（底层 gym env）。

    这是 no-RL CoSkill driver 的批量 rollout 入口：一次 reset 同时启动 N 个
    TextWorld 子环境，随后每一步把 N 条 prompt 批量送入 vLLM。与 make_single_env
    一样，必须在加载 vLLM/CUDA 之前调用，避免 CUDA-after-fork 卡死。
    """
    cfg = copy.deepcopy(config)
    EnvClass = get_environment(cfg["env"]["type"])  # AlfredTWEnv
    base = EnvClass(cfg, train_eval="train")
    base.game_files = list(game_files)
    base.num_games = len(base.game_files)
    env = base.init_env(batch_size=batch_size)
    env.seed(seed)
    return env
