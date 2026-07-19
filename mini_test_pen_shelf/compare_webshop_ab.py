"""Compare fixed-task WebShop baseline/template summaries and render A/B reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

NATIVE_CATEGORIES = ("fashion", "garden", "beauty", "electronics", "grocery")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def compare(template, baseline):
    t_ids = [row["goal_index"] for row in template["per_task"]]
    b_ids = [row["goal_index"] for row in baseline["per_task"]]
    if t_ids != b_ids:
        raise ValueError(f"A/B task mismatch: template={t_ids}, baseline={b_ids}")
    merged = {
        "benchmark": "WebShop",
        "goal_indices": t_ids,
        "template": template,
        "baseline": baseline,
        "delta": {
            "success_rate": round(template["success_rate"] - baseline["success_rate"], 4),
            "mean_task_score": round(template["mean_task_score"] - baseline["mean_task_score"], 4),
            "purchase_rate": round(template["purchase_rate"] - baseline["purchase_rate"], 4),
            "mean_steps": round(template["mean_steps"] - baseline["mean_steps"], 2),
        },
        "by_category": {},
        "per_task": [],
    }
    for category in NATIVE_CATEGORIES:
        t = template["by_category"][category]
        b = baseline["by_category"][category]
        merged["by_category"][category] = {
            "template": t,
            "baseline": b,
            "delta_success_rate": round(t["success_rate"] - b["success_rate"], 4),
            "delta_task_score": round(t["mean_task_score"] - b["mean_task_score"], 4),
        }
    for t, b in zip(template["per_task"], baseline["per_task"]):
        merged["per_task"].append({
            "task_idx": t["task_idx"],
            "goal_index": t["goal_index"],
            "category": t["category"],
            "query": t["query"],
            "instruction": t["instruction"],
            "template_score": t["task_score"],
            "baseline_score": b["task_score"],
            "delta_score": round(t["task_score"] - b["task_score"], 4),
            "template_won": t["won"],
            "baseline_won": b["won"],
            "template_steps": t["used_steps"],
            "baseline_steps": b["used_steps"],
            "template_trajectory": t["files"]["trajectory"],
            "baseline_trajectory": b["files"]["trajectory"],
        })
    return merged


def render_text(report):
    t, b, d = report["template"], report["baseline"], report["delta"]
    lines = [
        "=" * 100,
        "WebShop 固定任务 A/B：Max-Score Template vs Baseline",
        "=" * 100,
        f"{'指标':<24}{'Template':>16}{'Baseline':>16}{'Δ':>16}",
        "-" * 100,
        f"{'成功率':<24}{t['success_rate']*100:>15.1f}%{b['success_rate']*100:>15.1f}%{d['success_rate']*100:>+15.1f}%",
        f"{'平均 task score':<24}{t['mean_task_score']:>16.4f}{b['mean_task_score']:>16.4f}{d['mean_task_score']:>+16.4f}",
        f"{'购买率':<24}{t['purchase_rate']*100:>15.1f}%{b['purchase_rate']*100:>15.1f}%{d['purchase_rate']*100:>+15.1f}%",
        f"{'平均步数':<24}{t['mean_steps']:>16.2f}{b['mean_steps']:>16.2f}{d['mean_steps']:>+16.2f}",
        "",
        f"{'类别':<16}{'T成功':>10}{'B成功':>10}{'T得分':>12}{'B得分':>12}{'Δ得分':>12}  可视化",
        "-" * 100,
    ]
    for category in NATIVE_CATEGORIES:
        row = report["by_category"][category]
        tc, bc = row["template"], row["baseline"]
        delta = row["delta_task_score"]
        t_success = f"{tc['wins']}/{tc['count']}"
        b_success = f"{bc['wins']}/{bc['count']}"
        bar = ("+" * int(round(max(delta, 0) * 20))
               if delta >= 0 else "-" * int(round(abs(delta) * 20)))
        lines.append(
            f"{category:<16}{t_success:>10}{b_success:>10}"
            f"{tc['mean_task_score']:>12.4f}{bc['mean_task_score']:>12.4f}{delta:>+12.4f}  {bar}"
        )
    lines.extend(["", "逐任务得分：", "-" * 100])
    for row in report["per_task"]:
        mark = "↑" if row["delta_score"] > 0 else ("↓" if row["delta_score"] < 0 else "=")
        lines.append(
            f"{mark} #{row['task_idx']:02d} {row['category']:<12} goal={row['goal_index']:<4} "
            f"template={row['template_score']:.4f} baseline={row['baseline_score']:.4f} "
            f"Δ={row['delta_score']:+.4f}  {row['query']}"
        )
    return "\n".join(lines) + "\n"


def render_html(report, template_summary: Path, baseline_summary: Path):
    t, b, d = report["template"], report["baseline"], report["delta"]
    cat_rows = []
    for category in NATIVE_CATEGORIES:
        row = report["by_category"][category]
        cat_rows.append(f"""
        <tr><td>{html.escape(category)}</td>
        <td>{row['template']['wins']}/{row['template']['count']}</td>
        <td>{row['baseline']['wins']}/{row['baseline']['count']}</td>
        <td>{row['template']['mean_task_score']:.4f}</td>
        <td>{row['baseline']['mean_task_score']:.4f}</td>
        <td class="{'positive' if row['delta_task_score'] >= 0 else 'negative'}">{row['delta_task_score']:+.4f}</td></tr>""")
    task_rows = []
    for row in report["per_task"]:
        cls = "positive" if row["delta_score"] > 0 else ("negative" if row["delta_score"] < 0 else "")
        template_href = template_summary.parent / row["template_trajectory"]
        baseline_href = baseline_summary.parent / row["baseline_trajectory"]
        task_rows.append(f"""
        <tr><td>#{row['task_idx']:02d}</td><td>{html.escape(row['category'])}</td>
        <td>{html.escape(row['query'])}</td><td>{row['template_score']:.4f}</td>
        <td>{row['baseline_score']:.4f}</td><td class="{cls}">{row['delta_score']:+.4f}</td>
        <td><a href="{html.escape(str(template_href))}">T轨迹</a> · <a href="{html.escape(str(baseline_href))}">B轨迹</a></td></tr>""")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>WebShop A/B</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0}}main{{max-width:1150px;margin:auto;padding:30px}}
.hero,table{{background:#fff;border:1px solid #e2e7ef;border-radius:14px;box-shadow:0 5px 20px #15213a0d}}.hero{{display:flex;gap:38px;padding:22px;margin:20px 0;flex-wrap:wrap}}
.m .n{{font-size:29px;font-weight:760}}.positive{{color:#12855b;font-weight:700}}.negative{{color:#cc4050;font-weight:700}}
table{{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;margin:18px 0}}th,td{{padding:11px;border-bottom:1px solid #edf0f5;text-align:left}}th{{background:#eef2f7}}a{{color:#315fd6;text-decoration:none}}</style></head>
<body><main><h1>WebShop 固定任务 A/B</h1><p>Max-Score Template vs Baseline · 同一组 5 类 × 2 任务</p>
<section class="hero"><div class="m"><div class="n">{t['success_rate']*100:.1f}% / {b['success_rate']*100:.1f}%</div><div>success: template / baseline</div></div>
<div class="m"><div class="n">{t['mean_task_score']:.4f} / {b['mean_task_score']:.4f}</div><div>mean score: template / baseline</div></div>
<div class="m"><div class="n {'positive' if d['mean_task_score'] >= 0 else 'negative'}">{d['mean_task_score']:+.4f}</div><div>score delta</div></div></section>
<h2>分类对比</h2><table><thead><tr><th>类别</th><th>T成功</th><th>B成功</th><th>T得分</th><th>B得分</th><th>Δ得分</th></tr></thead><tbody>{''.join(cat_rows)}</tbody></table>
<h2>逐任务</h2><table><thead><tr><th>#</th><th>类别</th><th>查询</th><th>T得分</th><th>B得分</th><th>Δ</th><th>轨迹</th></tr></thead><tbody>{''.join(task_rows)}</tbody></table>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()
    template_path, baseline_path = Path(args.template).resolve(), Path(args.baseline).resolve()
    report = compare(_load(template_path), _load(baseline_path))
    prefix = Path(args.out_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    text = render_text(report)
    prefix.with_suffix(".txt").write_text(text, encoding="utf-8")
    prefix.with_suffix(".html").write_text(
        render_html(report, template_path, baseline_path), encoding="utf-8")
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(text)
    print(f"A/B outputs: {prefix.with_suffix('.txt')} / .html / .json")


if __name__ == "__main__":
    main()
