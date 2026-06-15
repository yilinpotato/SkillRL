"""
report.py — 详细易读的终端输出（带分隔线、缩进、emoji 标记）
"""

W = 76  # 行宽


def _hr(ch="="):
    return ch * W


def _wrap(text, indent=6, width=None):
    """简单按宽度折行，保留缩进。"""
    width = width or (W - indent)
    text = " ".join(str(text).split())
    out, line = [], ""
    for word in text.split(" "):
        if len(line) + len(word) + 1 > width:
            out.append(" " * indent + line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(" " * indent + line)
    return "\n".join(out)


def print_game_header(idx, task, gt, game_file):
    print("\n" + _hr("="))
    print(f"  游戏 #{idx}  ::  pen → shelf")
    print(_hr("="))
    print(f"  任务: {task}")
    obj = gt.get("object_target") or "?"
    parent = gt.get("parent_target") or "?"
    print(f"  目标: 把 [{obj}] 放到 [{parent}]")
    locs = gt.get("pen_locations") or []
    if locs:
        print(f"  🎯 pen 真值初始位置: {', '.join(locs)}")
    else:
        print(f"  🎯 pen 真值初始位置: (未从 game 文件解析到)")
    print(_hr("-"))


def print_step(step, think, action, valid, obs, added, removed, sighting, won):
    flag = "✅合法" if valid else "❌非法"
    print(f"\n  ── Step {step}  [{flag}] " + "-" * (W - 20))
    if think:
        print(f"  🧠 think:")
        print(_wrap(think, indent=8))
    print(f"  ▶️  action: {action}")
    print(f"  👁️  obs:")
    print(_wrap(obs, indent=8))
    if sighting:
        print(f"  🖊️  发现 pen! 位于: {sighting}")
    if added:
        shown = sorted(added)[:8]
        more = "" if len(added) <= 8 else f" (+{len(added)-8} 更多)"
        print(f"  ➕ 新增可选动作: {', '.join(shown)}{more}")
    if removed:
        shown = sorted(removed)[:8]
        more = "" if len(removed) <= 8 else f" (+{len(removed)-8} 更多)"
        print(f"  ➖ 消失可选动作: {', '.join(shown)}{more}")
    if won:
        print(f"  🏆 任务完成！")


def print_game_footer(won, discovered, steps):
    print("\n" + _hr("-"))
    status = "🏆 成功" if won else "🚧 未完成"
    print(f"  结果: {status}  |  用了 {steps} 步")
    if discovered:
        print(f"  本局发现 pen 的位置: {', '.join(sorted(discovered))}")
    else:
        print(f"  本局未在 observation 中明确定位到 pen")
    print(_hr("-"))


def print_final_summary(n_games, wins, total_loc):
    print("\n\n" + _hr("="))
    print("  最终汇总")
    print(_hr("="))
    print(f"  游戏数: {n_games}   成功: {wins}   成功率: {100.0*wins/max(n_games,1):.1f}%")
    print(f"\n  跨所有游戏，pen 被发现的位置统计:")
    if total_loc:
        mx = max(total_loc.values())
        for loc, c in total_loc.most_common():
            bar = "#" * int(30 * c / mx)
            print(f"    {loc:<14} {c:>3}  {bar}")
    else:
        print("    (未在 observation 中归因到具体位置；可看上面各局的真值位置)")
    print(_hr("="))
