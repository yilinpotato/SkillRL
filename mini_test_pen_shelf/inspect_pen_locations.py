"""
inspect_pen_locations.py — 零成本探查 pen 的真值位置分布（不加载任何模型，秒级）

直接解析数据集里所有 pen→shelf 游戏的 game.tw-pddl，统计 pen / pencil
在游戏开始时通常被放在哪些 receptacle（drawer / desk / shelf / sidetable ...）。

用法:
  export ALFWORLD_DATA=/path/to/alfworld
  python -m mini_test_pen_shelf.inspect_pen_locations --split train --limit 200
"""
import argparse
from collections import Counter

from mini_test_pen_shelf.env_utils import (
    find_pen_shelf_games,
    extract_pen_ground_truth,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "valid_seen", "valid_unseen"])
    ap.add_argument("--limit", type=int, default=None,
                    help="最多分析多少个游戏（默认全部）")
    args = ap.parse_args()

    games = find_pen_shelf_games(split=args.split, limit=args.limit)
    if not games:
        print("没有找到 pen→shelf 游戏，检查 ALFWORLD_DATA 和 split。")
        return

    loc_counter = Counter()       # pen 出现的容器类型 -> 次数
    obj_counter = Counter()       # object_target: pen vs pencil
    games_with_loc = 0

    print("\n" + "=" * 70)
    print(f"  逐游戏 pen 真值初始位置  (split={args.split}, 共 {len(games)} 个)")
    print("=" * 70)

    for idx, (game_file, traj) in enumerate(games):
        gt = extract_pen_ground_truth(game_file, traj)
        if gt["object_target"]:
            obj_counter[gt["object_target"]] += 1
        locs = gt["pen_locations"]
        if locs:
            games_with_loc += 1
            for l in locs:
                loc_counter[l] += 1
        desc = gt["task_desc"] or "(无标注描述)"
        print(f"\n[{idx+1:>3}] {gt['object_target'] or '?'} -> {gt['parent_target'] or '?'}")
        print(f"      任务: {desc}")
        print(f"      pen 初始位置: {', '.join(locs) if locs else '(未解析到)'}")

    # 汇总
    print("\n" + "=" * 70)
    print("  汇总统计")
    print("=" * 70)
    print(f"\n目标物体分布:")
    for obj, n in obj_counter.most_common():
        print(f"    {obj:<10} : {n}")

    print(f"\npen / pencil 通常出现的位置（按出现游戏数排序）:")
    total = sum(loc_counter.values())
    for loc, n in loc_counter.most_common():
        bar = "#" * int(40 * n / max(loc_counter.values()))
        pct = 100.0 * n / total if total else 0
        print(f"    {loc:<14} {n:>4}  ({pct:4.1f}%)  {bar}")

    print(f"\n成功解析出位置的游戏: {games_with_loc}/{len(games)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
