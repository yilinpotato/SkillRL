"""compare_ab.py — 读取 strategy ON / OFF 两次 run 的 summary.json，打印并归档对比报告。

用法:
  python -m mini_test_pen_shelf.compare_ab \
      --with_strategy mini_test_pen_shelf/output_strategy/summary.json \
      --no_strategy   mini_test_pen_shelf/output_baseline/summary.json \
      --archive       mini_test_pen_shelf/output_ab/ab_report
产出:
  <archive>.txt  人类可读对比表
  <archive>.json 合并后的对比数据
"""
import os
import json
import argparse


def _load(path):
    with open(path) as f:
        return json.load(f)


def _fmt_pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def _fmt_num(x):
    return "—" if x is None else f"{x}"


def build_report(a, b):
    """a=带策略, b=无策略。返回 (文本, 合并dict)。"""
    lines = []
    bar = "=" * 72
    lines.append(bar)
    lines.append("  A/B 对比：有 strategy template  vs  无 template (baseline)")
    lines.append(bar)
    lines.append(f"  数据集 split : {a.get('split')}   "
                 f"每臂游戏数 : A={a['num_games']}  B={b['num_games']}   "
                 f"步数上限 : {a.get('max_steps')}")
    lines.append("")
    header = f"  {'指标':<22}{'A: 有template':>16}{'B: 无template':>16}{'Δ (A-B)':>14}"
    lines.append(header)
    lines.append("  " + "-" * 66)

    def row(label, ka, fmt, delta=True, pct=False):
        va, vb = a.get(ka), b.get(ka)
        sa = _fmt_pct(va) if pct else _fmt_num(va)
        sb = _fmt_pct(vb) if pct else _fmt_num(vb)
        d = ""
        if delta and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            dv = va - vb
            d = (f"{dv*100:+.1f}%" if pct else f"{dv:+.2f}")
        lines.append(f"  {label:<22}{sa:>16}{sb:>16}{d:>14}")

    row("成功率", "success_rate", None, pct=True)
    row("成功局平均步数", "avg_steps_win", None)
    row("全部局平均步数", "avg_steps_all", None)
    row("平均合法动作率", "avg_valid_rate", None, pct=True)
    row("成功局数", "wins", None)

    # 逐局对照
    lines.append("")
    lines.append("  逐局结果 (步数；✓=成功 ✗=失败):")
    lines.append(f"  {'#':<4}{'任务':<34}{'A 有':>10}{'B 无':>10}")
    lines.append("  " + "-" * 56)
    a_games = {g["game_idx"]: g for g in a.get("per_game", [])}
    b_games = {g["game_idx"]: g for g in b.get("per_game", [])}
    for idx in sorted(set(a_games) | set(b_games)):
        ga, gb = a_games.get(idx), b_games.get(idx)
        task = (ga or gb).get("task", "")[:32]

        def cell(g):
            if not g:
                return "—"
            mark = "✓" if g["won"] else "✗"
            return f"{mark}{g['used_steps']}"
        lines.append(f"  {idx:<4}{task:<34}{cell(ga):>10}{cell(gb):>10}")

    lines.append("")
    lines.append(bar)
    # 结论
    da = a["success_rate"] - b["success_rate"]
    verdict = ("template 显著提升" if da > 0.05 else
               "template 略有提升" if da > 0 else
               "template 无明显帮助" if da == 0 else "template 反而更差")
    lines.append(f"  结论: {verdict}  (成功率 {_fmt_pct(a['success_rate'])} vs "
                 f"{_fmt_pct(b['success_rate'])}, Δ={da*100:+.1f}%)")
    lines.append(bar)

    merged = {"with_strategy": a, "no_strategy": b,
              "delta_success_rate": round(da, 4), "verdict": verdict}
    return "\n".join(lines), merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_strategy", required=True, help="带策略 run 的 summary.json")
    ap.add_argument("--no_strategy", required=True, help="无策略 run 的 summary.json")
    ap.add_argument("--archive", default=None, help="归档路径前缀（不含扩展名）")
    args = ap.parse_args()

    a = _load(args.with_strategy)
    b = _load(args.no_strategy)
    text, merged = build_report(a, b)
    print("\n" + text + "\n")

    if args.archive:
        os.makedirs(os.path.dirname(args.archive) or ".", exist_ok=True)
        with open(args.archive + ".txt", "w") as f:
            f.write(text + "\n")
        with open(args.archive + ".json", "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"📦 对比报告已归档: {args.archive}.txt / .json")


if __name__ == "__main__":
    main()
