#!/usr/bin/env python3
"""Repair historical ALFWorld no-RL group token metrics from per-episode data.

Older runs used concise rollout labels (``clean``/``cool``/...) while their
group-level metric schema used ALFWorld manifest labels.  Episode records keep
the original token counts, so this utility reconstructs canonical per-task
token fields, removes interrupted-run duplicate groups, and repairs the
matching checkpoint summary.  It always writes timestamped backups first.
"""
import argparse
import copy
import glob
import json
import os
import shutil
import time


ALIASES = {
    "pick_and_place": "pick_and_place_simple",
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
}


def canonical_task_type(name):
    return ALIASES.get(name, name)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def atomic_json(path, value):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_jsonl(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def token_bucket(records, task_types):
    result = {task: {"prompt": 0, "response": 0, "total": 0} for task in task_types}
    for record in records:
        task = canonical_task_type(record.get("detected_type", "unknown"))
        bucket = result.setdefault(task, {"prompt": 0, "response": 0, "total": 0})
        for key, source in (("prompt", "tokens_prompt"), ("response", "tokens_response"),
                            ("total", "tokens_total")):
            bucket[key] += int(record.get(source, 0) or 0)
    return result


def cloud_task_deltas(rows, task_types):
    """Recover per-task cloud usage from CoSkill's cumulative task counters.

    A resumed driver restarts those counters from zero; detect that boundary
    and start a new delta segment.  Cross-task cloud calls remain in ``mixed``.
    """
    concise_tasks = ("clean", "cool", "heat", "look_at_obj_in_light",
                     "pick_and_place", "pick_two_obj_and_place")
    previous = {task: {"prompt": 0, "completion": 0, "total": 0} for task in concise_tasks}
    result = []
    for row in rows:
        metrics = row["metrics"]
        current = {
            task: {
                key: int(metrics.get(
                    f"coskill/cloud/by_task_type/{task}/large_model_{source}_tokens", 0) or 0)
                for key, source in (("prompt", "prompt"), ("completion", "completion"),
                                    ("total", "total"))
            }
            for task in concise_tasks
        }
        reset = any(current[task]["total"] < previous[task]["total"] for task in concise_tasks)
        baseline = ({task: {"prompt": 0, "completion": 0, "total": 0} for task in concise_tasks}
                    if reset else previous)
        bucket = {task: {"prompt": 0, "completion": 0, "total": 0} for task in task_types}
        for task in concise_tasks:
            canonical = canonical_task_type(task)
            target = bucket.setdefault(canonical, {"prompt": 0, "completion": 0, "total": 0})
            for key in ("prompt", "completion", "total"):
                target[key] += max(0, current[task][key] - baseline[task][key])
        # Very early pre-recovery logs did not preserve every per-task cloud
        # counter.  Keep their missing portion explicit as ``unknown`` rather
        # than inventing a task allocation; this preserves exact accounting.
        expected = {
            key: int(metrics.get(f"tokens/large_model/{source}", 0) or 0) - int(
                metrics.get(f"tokens/large_model/mixed/{source}", 0) or 0)
            for key, source in (("prompt", "prompt"), ("completion", "completion"),
                                ("total", "total"))
        }
        observed_total = sum(value["total"] for value in bucket.values())
        unknown = bucket.setdefault("unknown", {"prompt": 0, "completion": 0, "total": 0})
        if observed_total > expected["total"]:
            # Counter reset boundary is ambiguous: preserve total accounting,
            # but do not manufacture a per-task allocation for this group.
            for task in bucket:
                bucket[task] = {"prompt": 0, "completion": 0, "total": 0}
            unknown = bucket["unknown"]
            unknown.update(expected)
        else:
            for key in ("prompt", "completion", "total"):
                observed = sum(value[key] for value in bucket.values())
                unknown[key] += max(0, expected[key] - observed)
        result.append(bucket)
        previous = current
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="defaults to latest formal step*.json checkpoint")
    parser.add_argument("--apply", action="store_true",
                        help="write repairs; without this flag only validate/report")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = sorted(glob.glob(os.path.join(args.outdir, "checkpoints", "step*.json")))
        if not candidates:
            raise FileNotFoundError("no formal checkpoint found")
        checkpoint = candidates[-1]
    summary = load_json(checkpoint)
    # Use the earliest pre-repair checkpoint as the source of any cloud-task
    # counters that were directly recorded.  Later repaired copies may contain
    # non-comparable counters accumulated across restarted driver processes.
    backup_candidates = sorted(glob.glob(f"{checkpoint}.pre_task_token_repair_*"))
    cloud_summary_source = load_json(backup_candidates[0]) if backup_candidates else summary
    per_game = summary.get("per_game") or []
    batch_size = int(summary.get("batch_rollout_size", 0) or 0)
    completed_groups = int(summary.get("completed_rollout_groups", 0) or 0)
    if batch_size <= 0 or len(per_game) != completed_groups * batch_size:
        raise ValueError("checkpoint episode/group boundary is not complete; refusing repair")

    task_types = list((summary.get("token_usage", {}).get("small_model", {})
                       .get("by_task_type", {}) or {}).keys())
    if not task_types:
        raise ValueError("checkpoint has no small-model task token schema")
    rebuilt_total = token_bucket(per_game, task_types)
    total_tokens = sum(row["total"] for row in rebuilt_total.values())
    expected_total = int(summary["token_usage"]["small_model"].get("total", 0) or 0)
    if total_tokens != expected_total:
        raise ValueError(f"per-episode tokens {total_tokens} != checkpoint total {expected_total}")

    metrics_path = os.path.join(args.outdir, "group_metrics.jsonl")
    with open(metrics_path) as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]
    # Last occurrence is the completed/recovered execution; earlier duplicate
    # rows were interrupted attempts with the same logical group boundary.
    latest_by_key = {}
    for row in raw_rows:
        key = (int(row.get("step", 0)), int(row.get("global_episode_end", 0)))
        latest_by_key[key] = row

    repaired_rows = []
    cumulative = {task: {"prompt": 0, "response": 0, "total": 0} for task in task_types}
    selected_rows = []
    for group_id in range(1, completed_groups + 1):
        selected_rows.append(latest_by_key[(group_id, group_id * batch_size)])
    large_group_tokens = cloud_task_deltas(selected_rows, task_types)
    large_cumulative = {task: {"prompt": 0, "completion": 0, "total": 0} for task in task_types}

    for group_id in range(1, completed_groups + 1):
        episode_end = group_id * batch_size
        original = latest_by_key.get((group_id, episode_end))
        if original is None:
            raise ValueError(f"missing group metric for group {group_id} / episode {episode_end}")
        row = copy.deepcopy(original)
        metrics = row["metrics"]
        records = per_game[(group_id - 1) * batch_size:episode_end]
        group_tokens = token_bucket(records, task_types)
        for task in task_types:
            for key in ("prompt", "response", "total"):
                cumulative[task][key] += group_tokens[task][key]
                metrics[f"tokens/small_model/by_task_type/{task}/{key}"] = group_tokens[task][key]
            metrics[f"tokens/small_model/by_task_type/{task}/total_cumulative"] = cumulative[task]["total"]

        metrics["tokens/small_model/prompt"] = sum(item["prompt"] for item in group_tokens.values())
        metrics["tokens/small_model/response"] = sum(item["response"] for item in group_tokens.values())
        metrics["tokens/small_model/total"] = sum(item["total"] for item in group_tokens.values())
        metrics["tokens/small_model/prompt_cumulative"] = sum(item["prompt"] for item in cumulative.values())
        metrics["tokens/small_model/response_cumulative"] = sum(item["response"] for item in cumulative.values())
        metrics["tokens/small_model/total_cumulative"] = sum(item["total"] for item in cumulative.values())
        for task in task_types:
            for key in ("prompt", "completion", "total"):
                large_cumulative[task][key] += large_group_tokens[group_id - 1][task][key]
                metrics[f"tokens/large_model/by_task_type/{task}/{key}"] = \
                    large_group_tokens[group_id - 1][task][key]
            metrics[f"tokens/large_model/by_task_type/{task}/total_cumulative"] = \
                large_cumulative[task]["total"]
        metrics["repair/task_token_schema"] = "alfworld_manifest_v1"
        metrics["repair/rebuilt_from_per_game"] = True
        metrics["repair/source_checkpoint"] = os.path.basename(checkpoint)
        repaired_rows.append(row)

    repaired_summary = copy.deepcopy(summary)
    repaired_summary["token_usage"]["small_model"]["by_task_type"] = rebuilt_total
    original_large_by_tt = copy.deepcopy(
        cloud_summary_source["token_usage"]["large_model"].get("by_task_type", {}))
    for task in task_types:
        original_large_by_tt.setdefault(task, {"prompt": 0, "completion": 0, "total": 0})
    large_total = repaired_summary["token_usage"]["large_model"]
    for key in ("prompt", "completion", "total"):
        known = sum(int(value.get(key, 0) or 0) for task, value in original_large_by_tt.items()
                    if task != "unknown")
        mixed = int(large_total.get("mixed", {}).get(key, 0) or 0)
        original_large_by_tt["unknown"][key] = max(0, int(large_total.get(key, 0) or 0) - mixed - known)
    repaired_summary["token_usage"]["large_model"]["by_task_type"] = original_large_by_tt
    repaired_summary["token_usage_repair"] = {
        "task_token_schema": "alfworld_manifest_v1",
        "source": "per_game",
        "repaired_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"checkpoint={checkpoint}")
    print(f"groups={completed_groups} input_rows={len(raw_rows)} repaired_rows={len(repaired_rows)}")
    print(f"small_model_tokens={total_tokens} task_sum={sum(row['total'] for row in rebuilt_total.values())}")
    if not args.apply:
        print("dry-run only; pass --apply to write backups and repairs")
        return

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_metrics = f"{metrics_path}.pre_task_token_repair_{stamp}"
    shutil.copy2(metrics_path, backup_metrics)
    atomic_jsonl(metrics_path, repaired_rows)

    summary_paths = [checkpoint, os.path.join(args.outdir, "summary_partial.json")]
    for path in summary_paths:
        if not os.path.isfile(path):
            continue
        old = load_json(path)
        if int(old.get("total_episodes", -1)) != len(per_game):
            continue
        backup = f"{path}.pre_task_token_repair_{stamp}"
        shutil.copy2(path, backup)
        fixed = copy.deepcopy(old)
        fixed["token_usage"]["small_model"]["by_task_type"] = rebuilt_total
        fixed["token_usage"]["large_model"]["by_task_type"] = original_large_by_tt
        fixed["token_usage_repair"] = repaired_summary["token_usage_repair"]
        atomic_json(path, fixed)
    print(f"wrote {metrics_path}; backup={backup_metrics}")


if __name__ == "__main__":
    main()
