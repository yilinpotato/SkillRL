"""Reproducible ALFWorld fixed-trajectory ablations for CoSkill.

The runner deliberately separates four phases so expensive rollouts never need
to be repeated: ``manifests`` -> ``bootstrap`` -> ``artifacts`` -> ``evaluate``.
All generated assets are immutable once written and are linked by SHA-256 in
``artifact_manifest.json``.  The normal no-RL driver remains the sole rollout
implementation; this file only orchestrates fixed inputs and reports results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from agent_system.memory import CoSkillCloudLoop, HierarchicalSkillLib, TracesPool
from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.memory.skill_updater import SkillUpdater
from mini_test_pen_shelf.env_utils import find_games_by_type


TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
# The dataset/manifest names differ from the names emitted by ``extract_task``
# and used by SkillsOnlyMemory retrieval. Artifact banks must use the latter
# or a perfectly generated skill would never enter the prompt.
TASK_TYPE_TO_RUNTIME = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "examine",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
}
RUNTIME_TASK_TYPES = tuple(TASK_TYPE_TO_RUNTIME[tt] for tt in TASK_TYPES)
RUNTIME_TO_TASK_TYPE = {runtime: canonical for canonical, runtime in TASK_TYPE_TO_RUNTIME.items()}
TREE_ARMS = tuple(f"tree_depth_{i}" for i in range(1, 6))
REPRESENTATION_ARMS = ("none", "flat_claude", *TREE_ARMS)
COMPRESSION_ARMS = ("compression_all_on", "compression_all_off")
MAX_TREE_GENERATION_ATTEMPTS = 20


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return os.environ.get("COSKILL_SOURCE_COMMIT", "unknown")


def _relative_game(game_file: str, data_root: Path) -> str:
    return os.path.relpath(os.path.realpath(game_file), os.path.realpath(data_root))


def create_manifests(root: Path, data_root: Path, split: str, sample_seed: int) -> Tuple[Path, Path]:
    """Select two distinct solvable games per type and write portable manifests."""
    manifest_dir = root / "manifests"
    boot_path, eval_path = manifest_dir / "bootstrap_games.json", manifest_dir / "eval_games.json"
    if boot_path.exists() and eval_path.exists():
        validate_manifest_pair(boot_path, eval_path, data_root)
        return boot_path, eval_path

    bootstrap, evaluation = [], []
    for offset, task_type in enumerate(TASK_TYPES):
        games = find_games_by_type(
            task_type, alfworld_data=str(data_root), split=split, sample_n=2,
            sample_seed=sample_seed + offset, verbose=False,
        )
        if len(games) < 2:
            raise RuntimeError(f"need two solvable games for {task_type}, found {len(games)}")
        for target, (game_file, _traj), role in (
            (bootstrap, games[0], "bootstrap"), (evaluation, games[1], "eval"),
        ):
            target.append({
                "label": f"{role}_{task_type}", "task_type": task_type,
                "game_file": _relative_game(game_file, data_root),
            })
    _write_json(boot_path, {"split": split, "role": "bootstrap", "games": bootstrap})
    _write_json(eval_path, {"split": split, "role": "eval", "games": evaluation})
    validate_manifest_pair(boot_path, eval_path, data_root)
    return boot_path, eval_path


def validate_manifest_pair(bootstrap_path: Path, eval_path: Path, data_root: Path) -> Dict[str, Any]:
    boot, ev = _read_json(bootstrap_path), _read_json(eval_path)
    b_games, e_games = boot.get("games", []), ev.get("games", [])
    if {x.get("task_type") for x in b_games} != set(TASK_TYPES):
        raise ValueError("bootstrap manifest must contain all six task types exactly once")
    if {x.get("task_type") for x in e_games} != set(TASK_TYPES):
        raise ValueError("eval manifest must contain all six task types exactly once")
    b_paths = {os.path.realpath(data_root / x["game_file"]) for x in b_games}
    e_paths = {os.path.realpath(data_root / x["game_file"]) for x in e_games}
    if len(b_paths) != 6 or len(e_paths) != 6 or b_paths & e_paths:
        raise ValueError("bootstrap/eval manifests must each have six unique, non-overlapping games")
    missing = [p for p in b_paths | e_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"manifest games missing, first: {missing[0]}")
    return {"bootstrap_sha256": _sha256_path(bootstrap_path), "eval_sha256": _sha256_path(eval_path)}


def _driver_cmd(args, outdir: Path, manifest: Path, episodes: int, batch: int,
                enable_coskill: int, enable_tree: int, enable_tree_evolve: int,
                enable_cloud_updates: int, extra: Iterable[str] = ()) -> List[str]:
    cmd = [sys.executable, "-u", "-m", "examples.playbook_evolve.run_playbook_evolve",
           "--outdir", str(outdir), "--fixed_games_manifest", str(manifest),
           "--epochs", "6", "--group_size", "6", "--max_episodes", str(episodes),
           "--batch_rollout_size", str(batch), "--max_steps", str(args.max_steps),
           "--seed", str(args.seed), "--retrieval_mode", args.retrieval_mode,
           "--enable_coskill", str(enable_coskill), "--enable_skill_tree", str(enable_tree),
           "--enable_skill_tree_evolve", str(enable_tree_evolve),
           "--enable_cloud_updates", str(enable_cloud_updates),
           "--log_trajectories", str(args.log_trajectories)]
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if args.data_parallel_workers:
        cmd.extend(["--data_parallel_workers", str(args.data_parallel_workers)])
    if args.rollout_worker_gpus:
        cmd.extend(["--rollout_worker_gpus", args.rollout_worker_gpus])
    cmd.extend(args.driver_arg)
    cmd.extend(extra)
    return cmd


def _run(cmd: List[str], root: Path) -> None:
    print("[ablation]", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap(args, root: Path, bootstrap_manifest: Path) -> Path:
    """Generate the common corpus; retry only task types with no success."""
    frozen = root / "frozen" / "raw_traces.jsonl"
    empty_skills = _ensure_empty_bootstrap_skills(root)
    coverage_path = root / "frozen" / "bootstrap_coverage.json"
    if frozen.exists():
        # A frozen corpus made before the explicit empty-bank protocol cannot
        # be distinguished from one collected with the driver's default seed
        # library.  Never relabel it as clean: require a new root instead.
        if not coverage_path.exists():
            raise RuntimeError(
                "existing frozen raw traces have no bootstrap protocol record; "
                "they may contain historical seed skills. Use a new --root.")
        coverage = _read_json(coverage_path)
        bank = coverage.get("bootstrap_skill_library", {})
        if not (bank.get("is_empty") is True and
                bank.get("sha256") == _sha256_path(empty_skills) and
                bank.get("enable_coskill") is True and
                bank.get("enable_skill_tree") is True):
            raise RuntimeError(
                "existing frozen raw traces were not collected with the empty "
                "bootstrap skill library. Use a new --root for this protocol.")
        return frozen
    run_dir = root / "bootstrap" / "initial"
    _run(_driver_cmd(args, run_dir, bootstrap_manifest, 36, 36, 1, 1, 0, 0,
                     extra=("--skills_json", str(empty_skills))), args.project_root)
    traces = _read_jsonl(run_dir / "traces_pool" / "raw_traces.jsonl")
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_type[trace.get("task_type", "unknown")].append(trace)

    missing_success = [tt for tt in TASK_TYPES if not any(
        (x.get("outcome") == "success" or x.get("episode_reward", 0) > 0)
        for x in by_type[TASK_TYPE_TO_RUNTIME[tt]]
    )]
    supplement_dirs = []
    if missing_success:
        boot = _read_json(bootstrap_manifest)
        selected = [x for x in boot["games"] if x["task_type"] in missing_success]
        supplement_manifest = root / "manifests" / "bootstrap_supplement_games.json"
        _write_json(supplement_manifest, {"split": boot["split"], "role": "bootstrap_supplement", "games": selected})
        supplement = root / "bootstrap" / "supplement"
        _run(_driver_cmd(args, supplement, supplement_manifest, 6 * len(selected), 6 * len(selected),
                         1, 1, 0, 0,
                         extra=("--seed", str(args.seed + 1),
                                "--skills_json", str(empty_skills))), args.project_root)
        supplement_dirs.append(supplement)

    all_traces = list(traces)
    for directory in supplement_dirs:
        all_traces.extend(_read_jsonl(directory / "traces_pool" / "raw_traces.jsonl"))
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in all_traces))
    coverage = {tt: {
        "runtime_task_type": TASK_TYPE_TO_RUNTIME[tt],
        "traces": sum(1 for x in all_traces if x.get("task_type") == TASK_TYPE_TO_RUNTIME[tt]),
        "successes": sum(1 for x in all_traces if x.get("task_type") == TASK_TYPE_TO_RUNTIME[tt] and
                         (x.get("outcome") == "success" or x.get("episode_reward", 0) > 0)),
    } for tt in TASK_TYPES}
    _write_json(coverage_path, {
        "raw_traces": str(frozen), "raw_traces_sha256": _sha256_path(frozen),
        "bootstrap_skill_library": {
            "path": str(empty_skills), "sha256": _sha256_path(empty_skills),
            "is_empty": True, "enable_coskill": True, "enable_skill_tree": True,
            "enable_cloud_updates": False, "enable_skill_tree_evolve": False,
        },
        "coverage": coverage, "supplemented_task_types": missing_success,
    })
    return frozen


def _compress_raw(raw_traces: List[Dict[str, Any]], outdir: Path, **flags: bool) -> Dict[str, Any]:
    pool = TracesPool(output_dir=str(outdir), min_samples=999999, **flags)
    for trace in raw_traces:
        pool.add_trace(trace)
    batch = pool.export_batch(trigger_reason="fixed_trajectory_ablation")
    _write_json(outdir / "compressed_batch.json", batch)
    return batch


def _payload_accounting(batch: Dict[str, Any]) -> Dict[str, Any]:
    text = json.dumps(batch, ensure_ascii=False, sort_keys=True)
    return {"cloud_payload_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "cloud_payload_chars": len(text), "cloud_payload_tokens_chars_div_4": _approx_tokens(text)}


def _empty_skill_bank() -> Dict[str, Any]:
    return {
        "general_skills": [],
        "task_specific_skills": {tt: [] for tt in RUNTIME_TASK_TYPES},
        "common_mistakes": [],
        "metadata": {"source": "fixed_trajectory_ablation", "handwritten_seed_skills": False},
    }


def _ensure_empty_bootstrap_skills(root: Path) -> Path:
    """Persist the content-free bootstrap library used by every ablation arm.

    Bootstrap intentionally keeps the normal retrieval/tree switches enabled,
    so the small model sees the same prompt framework as the CoSkill driver.
    It must not, however, silently inherit the driver's historical seed skills.
    """
    path = root / "bootstrap" / "empty_skills.json"
    expected = _empty_skill_bank()
    if path.exists() and _read_json(path) != expected:
        raise RuntimeError(
            f"bootstrap empty skill library is not empty: {path}; use a new --root")
    _write_json(path, expected)
    return path


def _raw_to_skillrl_failure(trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task": trace.get("task", ""), "task_type": trace.get("task_type", "unknown"),
        "trajectory": [{"action": step.get("action", ""), "observation": step.get("observation", "")}
                       for step in (trace.get("steps") or [])],
        "traj_uid": trace.get("traj_uid"),
    }


def _tree_stats(markdown: str) -> Dict[str, Any]:
    nodes, stack = [], []
    for line in (markdown or "").splitlines():
        if not (line.startswith("#") and line.lstrip("#").startswith(" ")):
            continue
        depth = len(line) - len(line.lstrip("#"))
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
        node = {"depth": depth, "label": line[depth:].strip(),
                "parent": stack[-1]["label"] if stack else None}
        nodes.append(node)
        stack.append(node)
    parent_labels = {n["parent"] for n in nodes if n["parent"]}
    return {
        "nodes": nodes, "node_count": len(nodes), "root_count": sum(n["parent"] is None for n in nodes),
        "internal_count": sum(n["label"] in parent_labels for n in nodes),
        "leaf_count": sum(n["label"] not in parent_labels for n in nodes),
        "edge_count": max(0, len(nodes) - sum(n["parent"] is None for n in nodes)),
        "max_depth": max((n["depth"] for n in nodes), default=0),
        "text_chars": len(markdown or ""), "text_tokens_chars_div_4": _approx_tokens(markdown or ""),
    }


def _skill_stats(skills: Dict[str, Any]) -> Dict[str, Any]:
    flat = list(skills.get("general_skills", []))
    for _, entries in (skills.get("task_specific_skills", {}) or {}).items():
        flat.extend(entries or [])
    return {
        "total": len(flat), "general": len(skills.get("general_skills", [])),
        "task_specific": len(flat) - len(skills.get("general_skills", [])),
        "skills": [{k: s.get(k) for k in ("skill_id", "title", "principle", "when_to_apply", "task_type", "scope")}
                   for s in flat],
        "text_tokens_chars_div_4": _approx_tokens(json.dumps(flat, ensure_ascii=False)),
    }


def _save_artifact_manifest(directory: Path, arm: str, raw_path: Path, skills: Dict[str, Any],
                            cloud_calls: List[Dict[str, Any]], extra: Dict[str, Any]) -> Path:
    skills_path = directory / "skills.json"
    _write_json(skills_path, skills)
    trees = {
        tt: _tree_stats((record or {}).get("content", ""))
        for tt, record in (skills.get("skill_trees", {}) or {}).items()
    }
    metadata = dict(extra)
    metadata.setdefault("status", "ready")
    metadata.setdefault("evaluation_eligible", metadata["status"] == "ready")
    manifest = {
        "arm": arm, "raw_traces_sha256": _sha256_path(raw_path),
        "skills_sha256": _sha256_path(skills_path), "cloud_calls": cloud_calls,
        "flat_skills": _skill_stats(skills), "skill_trees": trees, **metadata,
    }
    path = directory / "artifact_manifest.json"
    _write_json(path, manifest)
    return path


def build_flat_artifact(raw_path: Path, directory: Path) -> Path:
    raw = _read_jsonl(raw_path)
    skills, audit = _empty_skill_bank(), []
    source_trace_ids: Dict[str, List[str]] = {}
    failures = defaultdict(list)
    for trace in raw:
        if trace.get("outcome") != "success" and trace.get("episode_reward", 0) <= 0:
            failures[trace.get("task_type", "unknown")].append(_raw_to_skillrl_failure(trace))
    updater = SkillUpdater(max_new_skills_per_update=3)
    for tt in RUNTIME_TASK_TYPES:
        generated = updater.analyze_failures(failures.get(tt, []), skills) if failures.get(tt) else []
        for skill in generated:
            # Keep the persisted flat skill exactly in the SkillRL Claude
            # schema.  Placement/provenance belongs in the audit manifest,
            # not as extra prompt-visible fields.
            source_trace_ids[skill["skill_id"]] = [x.get("traj_uid") for x in failures[tt]]
        skills["task_specific_skills"][tt].extend(generated)
        call_dir = directory / "cloud_io"
        call_dir.mkdir(parents=True, exist_ok=True)
        if updater.last_prompt:
            (call_dir / f"flat_{tt}_prompt.txt").write_text(updater.last_prompt)
        if updater.last_response:
            (call_dir / f"flat_{tt}_response.txt").write_text(updater.last_response)
        audit.append({"purpose": "skillrl_flat_failure_analysis", "task_type": tt,
                      "failure_count": len(failures.get(tt, [])), "generated_ids": [x["skill_id"] for x in generated],
                      **updater.last_usage,
                      "prompt_sha256": _sha256_json(updater.last_prompt or ""),
                      "response_sha256": _sha256_json(updater.last_response or "")})
    return _save_artifact_manifest(directory, "flat_claude", raw_path, skills, audit,
                                   {"generation_protocol": "SkillRL SkillUpdater-compatible failure-only Claude JSON",
                                    "flat_skill_source_trace_ids": source_trace_ids})


def build_tree_artifact(raw_path: Path, directory: Path, depth: int) -> Path:
    raw = _read_jsonl(raw_path)
    batch = _compress_raw(raw, directory / "traces_pool", enable_loop_filter=True,
                          enable_obs_delta=True, enable_prefix_tree=True, enable_consensus_prefix=True)
    empty_path = directory / "empty_skills.json"
    _write_json(empty_path, _empty_skill_bank())
    lib = HierarchicalSkillLib(str(empty_path), retrieval_mode="template", enable_playbook=True)
    analyzer = CloudAnalyzer(output_dir=str(directory), environment_name="ALFWorld")
    diagnoses = analyzer.diagnose_failures(batch)
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"success": [], "failure": []})
    for trace in (batch.get("success_samples", []) + batch.get("failure_samples", [])):
        grouped[trace.get("task_type", "unknown")][trace.get("outcome", "failure")].append(trace)
    status = {}
    for tt in RUNTIME_TASK_TYPES:
        bucket = grouped[tt]
        result, attempts, repair_candidate = None, 0, None
        while attempts < MAX_TREE_GENERATION_ATTEMPTS:
            result = analyzer.evolve_playbook(
                tt, None, bucket["success"], bucket["failure"], diagnoses.get(tt, []),
                target_depth=depth, repair_candidate=repair_candidate,
            )
            attempts += 1
            if result and result.get("depth_valid", False):
                break
            repair_candidate = (result or {}).get("skill_tree") or repair_candidate
        if not result or not result.get("depth_valid", False):
            status[tt] = {
                "status": "generation_failed", "attempts": attempts,
                "actual_depth": (result or {}).get("actual_depth"),
                "heading_levels_present": (result or {}).get("heading_levels_present", []),
                "depth_validation_errors": (result or {}).get("depth_validation_errors", ["cloud_no_result"]),
            }
            continue
        rec = lib.update_playbook(tt, result["skill_tree"], result.get("level", "outline"),
                                  {"target_depth": depth, "actual_depth": result.get("actual_depth"),
                                   "depth_generation_attempts": attempts, "critique": result.get("critique", ""),
                                   "changelog": result.get("changelog", "")})
        status[tt] = {"status": "ok", "version": rec["version"], "attempts": attempts,
                      "actual_depth": result.get("actual_depth")}
    failed = {tt: info for tt, info in status.items() if info["status"] != "ok"}
    _write_json(directory / "generation_status.json", status)
    return _save_artifact_manifest(directory, f"tree_depth_{depth}", raw_path, lib.skills,
                                   analyzer.call_audit,
                                   {"generation_protocol": "CoSkill diagnose_failures + evolve_playbook",
                                    "target_depth": depth, "tree_generation_status": status,
                                    "tree_generation_max_attempts": MAX_TREE_GENERATION_ATTEMPTS,
                                    "status": "N.A." if failed else "ready",
                                    "evaluation_eligible": not bool(failed),
                                    "unavailable_reason": ("depth_validation_failed" if failed else None),
                                    "failed_task_types": sorted(failed),
                                    "compression": batch.get("compression", {}),
                                    **_payload_accounting(batch)})


def build_compression_artifact(raw_path: Path, directory: Path, all_on: bool) -> Path:
    raw = _read_jsonl(raw_path)
    flags = ({"enable_loop_filter": True, "enable_obs_delta": True,
              "enable_prefix_tree": True, "enable_consensus_prefix": True}
             if all_on else
             {"enable_loop_filter": False, "enable_obs_delta": False,
              "enable_prefix_tree": False, "enable_consensus_prefix": False})
    pool = TracesPool(output_dir=str(directory / "traces_pool"), min_samples=999999, **flags)
    for trace in raw:
        pool.add_trace(trace)
    empty_path = directory / "empty_skills.json"
    _write_json(empty_path, _empty_skill_bank())
    lib = HierarchicalSkillLib(str(empty_path), retrieval_mode="template", enable_playbook=True)
    loop = CoSkillCloudLoop(str(directory), enable_coskill=True, enable_playbook_evolve=True,
                            enable_failure_analysis=True, max_new_skills=3,
                            playbook_evolve_min_samples=6, environment_name="ALFWorld")
    if not loop.maybe_update(pool, lib, global_step=36, force_reason="fixed_trajectory_ablation"):
        raise RuntimeError(f"cloud generation failed for {'all_on' if all_on else 'all_off'}")
    batch_files = sorted((directory / "traces_pool").glob("batch_*.json"))
    batch = _read_json(batch_files[-1]) if batch_files else {}
    return _save_artifact_manifest(directory,
                                   "compression_all_on" if all_on else "compression_all_off",
                                   raw_path, lib.skills, (loop.cloud_analyzer.call_audit if loop.cloud_analyzer else []),
                                   {"generation_protocol": "full CoSkill cloud loop", "compression": batch.get("compression", {}),
                                    **_payload_accounting(batch)})


def build_artifacts(root: Path, raw_path: Path) -> Dict[str, Path]:
    artifacts = root / "artifacts"
    result = {}
    none_dir = artifacts / "none"
    none_manifest = none_dir / "artifact_manifest.json"
    result["none"] = (none_manifest if none_manifest.exists() else
                      _save_artifact_manifest(none_dir, "none", raw_path, _empty_skill_bank(), [],
                                              {"generation_protocol": "no skill"}))
    flat_manifest = artifacts / "flat_claude" / "artifact_manifest.json"
    result["flat_claude"] = (flat_manifest if flat_manifest.exists() else
                              build_flat_artifact(raw_path, artifacts / "flat_claude"))
    for depth in range(1, 6):
        path = artifacts / f"tree_depth_{depth}" / "artifact_manifest.json"
        result[f"tree_depth_{depth}"] = (path if path.exists() else
                                          build_tree_artifact(raw_path, artifacts / f"tree_depth_{depth}", depth))
    on = artifacts / "compression_all_on" / "artifact_manifest.json"
    off = artifacts / "compression_all_off" / "artifact_manifest.json"
    result["compression_all_on"] = (on if on.exists() else
                                    build_compression_artifact(raw_path, artifacts / "compression_all_on", True))
    result["compression_all_off"] = (off if off.exists() else
                                     build_compression_artifact(raw_path, artifacts / "compression_all_off", False))
    return result


def evaluate_arm(args, root: Path, eval_manifest: Path, arm: str) -> Path:
    output = root / "arms" / arm
    summary = output / "summary.json"
    if summary.exists():
        return summary
    artifact_manifest = root / "artifacts" / arm / "artifact_manifest.json"
    if not artifact_manifest.exists():
        raise FileNotFoundError(f"missing artifact manifest for {arm}: {artifact_manifest}")
    artifact_meta = _read_json(artifact_manifest)
    if not artifact_meta.get("evaluation_eligible", artifact_meta.get("status", "ready") == "ready"):
        _write_json(summary, {
            "status": "N.A.", "arm": arm, "evaluation_skipped": True,
            "reason": artifact_meta.get("unavailable_reason", "artifact_not_eligible"),
            "failed_task_types": artifact_meta.get("failed_task_types", []),
            "total_episodes": 0, "wins": 0, "success_rate": None,
        })
        return summary
    artifact = root / "artifacts" / arm / "skills.json"
    if not artifact.exists():
        raise FileNotFoundError(f"missing artifact for {arm}: {artifact}")
    tree = int(arm.startswith("tree_depth_") or arm.startswith("compression_"))
    bullets = int(arm == "flat_claude" or arm.startswith("compression_"))
    _run(_driver_cmd(args, output, eval_manifest, 36, 36, bullets, tree, 0, 0,
                     ["--skills_json", str(artifact)]), args.project_root)
    return summary


def _latest_group_metrics(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    return (rows[-1].get("metrics", {}) if rows else {})


def write_summary(root: Path, manifests: Dict[str, str]) -> None:
    rows, detailed = [], {}
    for arm in (*REPRESENTATION_ARMS, *COMPRESSION_ARMS):
        summary_path = root / "arms" / arm / "summary.json"
        artifact_path = root / "artifacts" / arm / "artifact_manifest.json"
        if not summary_path.exists() or not artifact_path.exists():
            continue
        summary, artifact = _read_json(summary_path), _read_json(artifact_path)
        if not artifact.get("evaluation_eligible", artifact.get("status", "ready") == "ready"):
            reason = artifact.get("unavailable_reason", "artifact_not_eligible")
            detailed[arm] = {
                "summary": summary, "artifact": artifact, "status": "N.A.",
                "evaluation_skipped": True, "reason": reason,
            }
            rows.append({
                "arm": arm, "status": "N.A.", "evaluation_skipped": True,
                "reason": reason,
                "failed_task_types": ",".join(artifact.get("failed_task_types", [])),
                "target_tree_depth": artifact.get("target_depth"),
                "raw_traces_sha256": artifact.get("raw_traces_sha256"),
            })
            continue
        metrics = _latest_group_metrics(root / "arms" / arm / "group_metrics.jsonl")
        per_type = defaultdict(lambda: {"episodes": 0, "wins": 0})
        injected = []
        for episode in summary.get("per_game", []):
            tt = episode.get("detected_type", "unknown")
            per_type[tt]["episodes"] += 1
            per_type[tt]["wins"] += int(bool(episode.get("won")))
            injected.append({"step": episode.get("step"), "task_type": tt,
                             "skill_ids": episode.get("skill_ids_used", [])})
        episodes = summary.get("per_game", [])
        action_count = sum(int(x.get("used_steps", 0) or 0) for x in episodes)
        normal_valid = sum(int(x.get("valid_actions", 0) or 0) for x in episodes)
        strict_valid = sum(int(x.get("strict_valid_actions", 0) or 0) for x in episodes)
        token_usage = summary.get("token_usage", {}) or {}
        small_usage = token_usage.get("small_model", {}) or {}
        large_usage = token_usage.get("large_model", {}) or {}
        flat_tokens = {
            item.get("skill_id"): _approx_tokens(json.dumps(item, ensure_ascii=False))
            for item in (artifact.get("flat_skills", {}) or {}).get("skills", [])
            if item.get("skill_id")
        }
        flat_injection_count = sum(len(item["skill_ids"]) for item in injected)
        flat_injection_tokens = sum(flat_tokens.get(sid, 0) for item in injected for sid in item["skill_ids"])
        tree_tokens = {tt: int(info.get("text_tokens_chars_div_4", 0) or 0)
                       for tt, info in (artifact.get("skill_trees", {}) or {}).items()}
        tree_injection_count = sum(1 for episode in episodes if episode.get("detected_type") in tree_tokens)
        tree_injection_tokens = sum(tree_tokens.get(episode.get("detected_type"), 0) for episode in episodes)
        detailed[arm] = {"summary": summary, "artifact": artifact, "last_group_metrics": metrics,
                         "per_task_type": {tt: {**v, "success_rate": v["wins"] / max(v["episodes"], 1)}
                                           for tt, v in per_type.items()},
                         "episode_injected_skill_ids": injected,
                         "injection_accounting": {
                             "flat_skill_injection_count": flat_injection_count,
                             "flat_skill_tokens_chars_div_4": flat_injection_tokens,
                             "tree_injection_count": tree_injection_count,
                             "tree_text_tokens_chars_div_4": tree_injection_tokens,
                         }}
        row = {"arm": arm, "status": "ready", "episodes": summary.get("total_episodes", 0),
               "success_rate": summary.get("success_rate", 0), "wins": summary.get("wins", 0),
               "strict_valid_action_rate": strict_valid / max(action_count, 1),
               "normal_valid_action_rate": normal_valid / max(action_count, 1),
               "small_model_prompt_tokens_cumulative": small_usage.get("prompt"),
               "small_model_response_tokens_cumulative": small_usage.get("response"),
               "small_model_total_tokens_cumulative": small_usage.get("total"),
               "small_model_token_accounting": metrics.get(
                   "tokens/small_model/accounting", small_usage.get("accounting")),
               "large_model_prompt_tokens_cumulative": large_usage.get("prompt"),
               "large_model_completion_tokens_cumulative": large_usage.get("completion"),
               "large_model_total_tokens_cumulative": large_usage.get("total"),
               "perf_total_num_tokens_last_group": metrics.get("perf/total_num_tokens"),
               "perf_total_num_tokens_cumulative_equivalent": small_usage.get("total"),
               "flat_skill_injection_count": flat_injection_count,
               "flat_skill_tokens_chars_div_4": flat_injection_tokens,
               "tree_injection_count": tree_injection_count,
               "tree_text_tokens_chars_div_4": tree_injection_tokens,
               "artifact_sha256": artifact.get("skills_sha256"),
               "raw_traces_sha256": artifact.get("raw_traces_sha256")}
        rows.append(row)
    _write_json(root / "ablation_summary.json", {"manifests": manifests, "arms": detailed,
                                                   "token_accounting": {"trajectory_estimate": "chars_div_4",
                                                                        "small_model": "reported_per_arm_in_small_model_token_accounting",
                                                                        "large_model": "provider_api_usage",
                                                                        "perf/total_num_tokens": "driver compatibility metric"}})
    with (root / "ablation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ablation output root; phases are resumable")
    ap.add_argument("--phase", choices=("manifests", "bootstrap", "artifacts", "evaluate", "all"), default="all")
    ap.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--retrieval_mode", default="template", choices=("template", "embedding"))
    ap.add_argument("--data_parallel_workers", type=int, default=0)
    ap.add_argument("--rollout_worker_gpus", default=None)
    ap.add_argument("--log_trajectories", type=int, default=0)
    ap.add_argument("--driver_arg", action="append", default=[], help="extra one-token argument forwarded to the main driver")
    args = ap.parse_args()
    if not args.alfworld_data:
        ap.error("--alfworld_data or ALFWORLD_DATA is required")
    args.project_root = Path(__file__).resolve().parents[2]
    root, data_root = Path(args.root).resolve(), Path(args.alfworld_data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bootstrap_empty_skills = _ensure_empty_bootstrap_skills(root)
    boot_manifest, eval_manifest = create_manifests(root, data_root, args.split, args.sample_seed)
    manifest_hashes = validate_manifest_pair(boot_manifest, eval_manifest, data_root)
    _write_json(root / "run_config.json", {"task_types": TASK_TYPES,
                                             "runtime_task_type_map": TASK_TYPE_TO_RUNTIME,
                                             "bootstrap_rollouts_per_type": 6,
                                             "eval_rollouts_per_type": 6, "seed": args.seed,
                                             "sample_seed": args.sample_seed, "git_commit": _git_commit(args.project_root),
                                             "sampling_seed_accounting": "sha256(base_seed|game_id|replica_index|env_step)",
                                             "bootstrap_supplement_seed_offset": 1,
                                             "bootstrap_skill_library": {
                                                 "path": str(bootstrap_empty_skills),
                                                 "sha256": _sha256_path(bootstrap_empty_skills),
                                                 "is_empty": True,
                                                 "enable_coskill": True,
                                                 "enable_skill_tree": True,
                                             },
                                             "tree_depth_generation_max_attempts": MAX_TREE_GENERATION_ATTEMPTS,
                                             **manifest_hashes})
    if args.phase == "manifests":
        return
    raw = bootstrap(args, root, boot_manifest)
    if args.phase == "bootstrap":
        return
    build_artifacts(root, raw)
    if args.phase == "artifacts":
        return
    for arm in (*REPRESENTATION_ARMS, *COMPRESSION_ARMS):
        evaluate_arm(args, root, eval_manifest, arm)
    write_summary(root, manifest_hashes)


if __name__ == "__main__":
    main()
