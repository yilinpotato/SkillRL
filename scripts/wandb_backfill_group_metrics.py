#!/usr/bin/env python3
"""Backfill repaired CoSkill group metrics to a dedicated W&B mirror run.

The live benchmark run must not be opened by a second W&B writer because both
writers would compete for the monotonically increasing internal step.  This
sidecar creates a separate run in the same project, replays the canonical
group_metrics.jsonl from step 1, then tails it until the requested final group.
"""
import argparse
import json
import os
import signal
import time


def atomic_text(path, value):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(value)
    os.replace(tmp, path)


def read_rows(path):
    latest = {}
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            group = int(row.get("step", 0) or 0)
            if group > 0:
                latest[group] = row
    return latest


def numeric_metrics(row):
    payload = {
        key: value for key, value in (row.get("metrics") or {}).items()
        if isinstance(value, (bool, int, float))
    }
    payload["backfill/group"] = int(row["step"])
    payload["backfill/global_episode_end"] = int(row.get("global_episode_end", 0) or 0)
    payload["backfill/repaired_source"] = 1
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "coskill-alfworld"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--name", default="alfworld_playbook_evolve_norl_repaired_metrics")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--final-group", type=int, default=100)
    args = parser.parse_args()

    import wandb

    os.makedirs(args.outdir, exist_ok=True)
    metrics_path = os.path.join(args.outdir, "group_metrics.jsonl")
    run_id_path = os.path.join(args.outdir, "wandb_repaired_metrics_run_id.txt")
    state_path = os.path.join(args.outdir, "wandb_repaired_metrics_last_group.txt")
    if os.path.isfile(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        atomic_text(run_id_path, f"{run_id}\n")
    last_group = 0
    if os.path.isfile(state_path):
        with open(state_path) as f:
            last_group = int(f.read().strip() or 0)

    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        id=run_id,
        resume="allow",
        name=args.name,
        job_type="metrics-backfill",
        tags=["metrics-repaired", "group-metrics-backfill"],
        dir=args.outdir,
        config={
            "source": os.path.abspath(metrics_path),
            "benchmark": "alfworld",
            "method": "coskill_playbook_evolve_norl",
            "final_group": args.final_group,
            "accounting": "canonical repaired group_metrics.jsonl",
        },
    )
    print(f"[wandb-backfill] run={run.get_url()} resume_after_group={last_group}", flush=True)

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        while not stopping:
            rows = read_rows(metrics_path)
            for group in sorted(g for g in rows if last_group < g <= args.final_group):
                run.log(numeric_metrics(rows[group]), step=group)
                last_group = group
                atomic_text(state_path, f"{last_group}\n")
                print(f"[wandb-backfill] synced group={group}", flush=True)
            run.summary.update({
                "backfill_status": "complete" if last_group >= args.final_group else "tailing",
                "backfilled_through_group": last_group,
                "backfilled_global_episode_end": last_group * 72,
            })
            if last_group >= args.final_group:
                break
            time.sleep(max(5, args.poll_seconds))
    finally:
        run.finish(exit_code=0)


if __name__ == "__main__":
    main()
