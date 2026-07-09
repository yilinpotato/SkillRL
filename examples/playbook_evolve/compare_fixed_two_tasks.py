"""Compare the bullets-OFF/ON fixed-task evolution summaries."""
import argparse
import json
import os


def _load(path):
    with open(path) as f:
        return json.load(f)


def _pct(value):
    return f"{100 * float(value or 0):.1f}%"


def _phase(summary, round_id):
    return (summary.get("phase_stats") or {}).get(str(round_id), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="bullets_off/summary.json")
    ap.add_argument("--on", required=True, help="bullets_on/summary.json")
    ap.add_argument("--out", required=True, help="output markdown report")
    args = ap.parse_args()
    arms = [("bullets OFF", _load(args.off)), ("bullets ON", _load(args.on))]

    lines = [
        "# Fixed two-task CoSkill A/B",
        "",
        "Both arms use the same pick-one and pick-two game instances. Round 0 is before the first cloud update; round 1 is after one skill-tree/skill-bullet evolution update.",
        "",
        "| Arm | Round | Episodes | Wins | Success | pick_one | pick_two | Cloud patches | Skill-tree updates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in arms:
        final = summary.get("final_coskill_metrics") or {}
        for round_id in (0, 1):
            phase = _phase(summary, round_id)
            by_tt = phase.get("by_task_type") or {}
            pick_one = by_tt.get("pick_and_place", by_tt.get("pick_and_place_simple", {}))
            pick_two = by_tt.get("pick_two_obj_and_place", {})
            lines.append(
                f"| {name} | {round_id} | {phase.get('episodes', 0)} | "
                f"{phase.get('wins', 0)} | {_pct(phase.get('success_rate'))} | "
                f"{_pct(pick_one.get('success_rate'))} | {_pct(pick_two.get('success_rate'))} | "
                f"{final.get('coskill/cloud/total_patches', 0)} | "
                f"{final.get('coskill/skill_tree/updates', final.get('coskill/playbook/updates', 0))} |"
            )

    off_post = _phase(arms[0][1], 1).get("success_rate", 0)
    on_post = _phase(arms[1][1], 1).get("success_rate", 0)
    lines.extend([
        "",
        f"Post-update bullets ON − OFF: **{100 * (on_post - off_post):+.1f} percentage points**.",
        "",
        "Inspect each arm's `cloud_io/`, `trajectories/`, and `metrics.jsonl` before attributing a small-sample difference to the bullets.",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"comparison report -> {args.out}")


if __name__ == "__main__":
    main()
