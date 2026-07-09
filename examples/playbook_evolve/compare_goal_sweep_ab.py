"""Compare multi-arm goal-sweep CoSkill summaries.

Reads every ``<root>/<arm>/summary.json`` and writes a compact Markdown report
covering final success, post-evolution delta vs baseline, and final per-goal
success rates.
"""
import argparse
import json
import os
from pathlib import Path


def _load(path):
    with open(path) as f:
        return json.load(f)


def _pct(value):
    return f"{100 * float(value or 0):.1f}%"


def _phase(summary, key):
    return (summary.get("phase_stats") or {}).get(str(key), {})


def _last_phase(summary):
    phases = summary.get("phase_stats") or {}
    if not phases:
        return {}
    last = max((int(k) for k in phases.keys()), default=0)
    return _phase(summary, last)


def _arm_dirs(root):
    root = Path(root)
    order = {"none": 0, "tree_only": 1, "patch_only": 2, "tree_plus_patch": 3}
    arms = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "summary.json").is_file()
    ]
    return sorted(arms, key=lambda p: (order.get(p.name, 99), p.name))


def _metric(summary, key, default=0):
    return (summary.get("final_coskill_metrics") or {}).get(key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing arm subdirs")
    ap.add_argument("--out", required=True, help="output markdown report")
    ap.add_argument("--baseline", default="none", help="arm name used for deltas")
    args = ap.parse_args()

    arms = [(p.name, _load(p / "summary.json")) for p in _arm_dirs(args.root)]
    if not arms:
        raise SystemExit(f"no arm summary.json files found under {args.root}")

    baseline = next((s for name, s in arms if name == args.baseline), arms[0][1])
    base_final = _last_phase(baseline).get("success_rate", baseline.get("success_rate", 0))

    all_tts = sorted({
        tt
        for _, summary in arms
        for phase in (summary.get("phase_stats") or {}).values()
        for tt in (phase.get("by_task_type") or {}).keys()
    })

    lines = [
        "# Goal-sweep CoSkill ablation",
        "",
        f"Root: `{args.root}`",
        "",
        "Arms compare agent skill tree and flat skill-patch bullets under the same sampled goal pool.",
        "",
        "| Arm | Tree | Patches | Episodes | Final success | Δ vs baseline | Round 0 | Final round | Cloud patches | Tree updates | Tree nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, summary in arms:
        final_phase = _last_phase(summary)
        round0 = _phase(summary, 0)
        final_success = final_phase.get("success_rate", summary.get("success_rate", 0))
        final_round = max((int(k) for k in (summary.get("phase_stats") or {}).keys()), default=0)
        lines.append(
            f"| {name} | {int(bool(summary.get('skill_tree_enabled')))} | "
            f"{int(bool(summary.get('skill_bullets_enabled')))} | "
            f"{summary.get('total_episodes', final_phase.get('episodes', 0))} | "
            f"{_pct(final_success)} | "
            f"{100 * (float(final_success or 0) - float(base_final or 0)):+.1f} pp | "
            f"{_pct(round0.get('success_rate'))} | {final_round} | "
            f"{_metric(summary, 'coskill/cloud/total_patches')} | "
            f"{_metric(summary, 'coskill/skill_tree/updates', _metric(summary, 'coskill/playbook/updates'))} | "
            f"{summary.get('skill_tree_nodes', 0)} |"
        )

    if all_tts:
        lines.extend([
            "",
            "## Final success by goal type",
            "",
            "| Arm | " + " | ".join(all_tts) + " |",
            "|---" + "|---:" * len(all_tts) + "|",
        ])
        for name, summary in arms:
            by_tt = _last_phase(summary).get("by_task_type") or {}
            cells = [_pct((by_tt.get(tt) or {}).get("success_rate")) for tt in all_tts]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Notes:",
        "",
        "- `none`: no skill tree and no flat skill-patch bullets.",
        "- `tree_only`: per-task skill trees only; no flat skill-patch bullets.",
        "- `tree_plus_patch`: skill tree plus contrastively distilled flat skill-patch bullets.",
        "- `patch_only`: flat skill-patch bullets only; no skill tree.",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"goal-sweep comparison report -> {args.out}")


if __name__ == "__main__":
    main()
