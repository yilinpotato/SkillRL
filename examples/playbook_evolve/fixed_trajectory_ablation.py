"""Reproducible ALFWorld fixed-trajectory L0-L5 skill-representation experiment.

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

from agent_system.memory import HierarchicalSkillLib, TracesPool
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
    # SkillsOnlyMemory._detect_task_type deliberately folds both the
    # "look at X under the Y" and "examine the X with the Y" templates into
    # one label, "look_at_obj_in_light" (see skills_only_memory.py), so raw
    # traces are tagged with that name, never "examine". Mapping to "examine"
    # here made every by-runtime-type lookup for this task miss (an always-
    # empty bucket), starving evolve_playbook of samples and failing depth
    # validation for every tree_depth_N arm.
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
}
RUNTIME_TASK_TYPES = tuple(TASK_TYPE_TO_RUNTIME[tt] for tt in TASK_TYPES)
SKILL_LEVEL_ARMS = tuple(f"skill_level_l{i}" for i in range(6))
TREE_ARMS = SKILL_LEVEL_ARMS[1:]
MAX_TREE_GENERATION_ATTEMPTS = 20
# V2 receives a large fixed corpus.  Keep every trace in the compression
# statistics/prefix tree, then choose the *prompt examples* deterministically
# rather than letting JSONL input order decide what the cloud sees.
TREE_EVIDENCE_SAMPLING = {
    "algorithm": "hash_stratified_outcome_length_loop_task_diversity_v1",
    "seed": 20260723,
    # Diagnosis is per task type in V2, so this remains a bounded prompt.
    "diagnose": {"success": 12, "failure": 18},
    # The actual tree-authoring prompt gets more evidence than diagnosis.
    "evolve": {"success": 16, "failure": 24},
    # L0 preserves SkillRL's original five-example prompt budget.  Only its
    # choice of those five failures is made order-independent.
    "flat_skillrl": {"failure": 5},
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Atomically replace a derived JSONL report.

    The controller is resumable, so appending here would duplicate completed
    arms every time ``--phase evaluate`` is re-entered.  Per-rollout driver
    logs remain append-only; these root-level comparison files are rebuilt
    deterministically from those canonical outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _stable_trace_rank(trace: Dict[str, Any], *, salt: str) -> str:
    """Stable pseudo-random order without depending on JSONL line order."""
    value = f"{TREE_EVIDENCE_SAMPLING['seed']}:{salt}:{trace.get('traj_uid', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trace_task_key(trace: Dict[str, Any]) -> str:
    """Prefer one example per natural-language goal before repeating a goal."""
    task = " ".join(str(trace.get("task", "")).split())
    return task or str(trace.get("traj_uid", ""))


def _stratified_trace_sample(traces: List[Dict[str, Any]], limit: int, *, salt: str
                             ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pick bounded, reproducible evidence across length/loop/task strata.

    Callers already split by task type and outcome.  Within that bucket this
    avoids a corpus sorted by seed, game, or outcome variant silently turning
    into a first-N prompt.  A first pass prefers different natural-language
    task descriptions; a second pass fills any remaining budget.
    """
    if limit <= 0 or not traces:
        return [], {"available": len(traces), "requested": limit, "selected": 0,
                    "selected_traj_uids": [], "selected_strata": {}}
    ranked = sorted(traces, key=lambda trace: (
        len(trace.get("steps") or []), _stable_trace_rank(trace, salt=salt)))
    length_bucket: Dict[str, str] = {}
    total = len(ranked)
    for index, trace in enumerate(ranked):
        # Tertiles are local to this task_type/outcome bucket, which makes the
        # same rule usable for both short successes and long failures.
        length_bucket[str(trace.get("traj_uid", ""))] = ("short" if index * 3 < total
                                                            else "medium" if index * 3 < total * 2
                                                            else "long")
    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        loop = "loop" if int(trace.get("dropped_loops", 0) or 0) > 0 else "no_loop"
        bucket = length_bucket[str(trace.get("traj_uid", ""))]
        strata[f"{loop}:{bucket}"].append(trace)
    for values in strata.values():
        values.sort(key=lambda trace: _stable_trace_rank(trace, salt=salt))

    ordered_strata = sorted(strata)
    positions = {name: 0 for name in ordered_strata}
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    selected_tasks = set()

    def take_one(*, require_new_task: bool) -> bool:
        progressed = False
        for name in ordered_strata:
            values = strata[name]
            while positions[name] < len(values):
                trace = values[positions[name]]
                positions[name] += 1
                uid = str(trace.get("traj_uid", ""))
                if uid in selected_ids:
                    continue
                if require_new_task and _trace_task_key(trace) in selected_tasks:
                    continue
                selected.append(trace)
                selected_ids.add(uid)
                selected_tasks.add(_trace_task_key(trace))
                progressed = True
                break
            if len(selected) >= limit:
                break
        return progressed

    while len(selected) < limit and take_one(require_new_task=True):
        pass
    # Reset cursors so examples skipped solely because their task duplicated a
    # first-pass example remain eligible to fill the requested quota.
    positions = {name: 0 for name in ordered_strata}
    while len(selected) < limit and take_one(require_new_task=False):
        pass

    selected_strata: Dict[str, int] = defaultdict(int)
    for trace in selected:
        loop = "loop" if int(trace.get("dropped_loops", 0) or 0) > 0 else "no_loop"
        selected_strata[f"{loop}:{length_bucket[str(trace.get('traj_uid', ''))]}"] += 1
    audit = {
        "available": len(traces), "requested": limit, "selected": len(selected),
        "selected_traj_uids": [str(trace.get("traj_uid", "")) for trace in selected],
        "selected_strata": dict(sorted(selected_strata.items())),
    }
    return selected, audit


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return os.environ.get("COSKILL_SOURCE_COMMIT", "unknown")


def _relative_game(game_file: str, data_root: Path) -> str:
    return os.path.relpath(os.path.realpath(game_file), os.path.realpath(data_root))


def create_manifests(root: Path, data_root: Path, split: str, sample_seed: int,
                     eval_games_per_type: int = 1) -> Tuple[Path, Path]:
    """Select one bootstrap and N distinct evaluation games per task type."""
    if eval_games_per_type < 1:
        raise ValueError("eval_games_per_type must be at least 1")
    manifest_dir = root / "manifests"
    boot_path, eval_path = manifest_dir / "bootstrap_games.json", manifest_dir / "eval_games.json"
    if boot_path.exists() and eval_path.exists():
        validate_manifest_pair(boot_path, eval_path, data_root)
        return boot_path, eval_path

    bootstrap, evaluation = [], []
    for offset, task_type in enumerate(TASK_TYPES):
        games = find_games_by_type(
            task_type, alfworld_data=str(data_root), split=split,
            sample_n=1 + eval_games_per_type,
            sample_seed=sample_seed + offset, verbose=False,
        )
        if len(games) < 1 + eval_games_per_type:
            raise RuntimeError(
                f"need {1 + eval_games_per_type} solvable games for {task_type}, found {len(games)}")
        game_file, _traj = games[0]
        bootstrap.append({
                "label": f"bootstrap_{task_type}", "task_type": task_type,
                "game_file": _relative_game(game_file, data_root),
            })
        for index, (game_file, _traj) in enumerate(games[1:], start=1):
            evaluation.append({
                "label": f"eval_{task_type}_{index}", "task_type": task_type,
                "game_file": _relative_game(game_file, data_root),
            })
    _write_json(boot_path, {"split": split, "role": "bootstrap", "games": bootstrap,
                            "games_per_task_type": 1})
    _write_json(eval_path, {"split": split, "role": "eval", "games": evaluation,
                            "games_per_task_type": eval_games_per_type})
    validate_manifest_pair(boot_path, eval_path, data_root)
    return boot_path, eval_path


def validate_manifest_pair(bootstrap_path: Path, eval_path: Path, data_root: Path) -> Dict[str, Any]:
    boot, ev = _read_json(bootstrap_path), _read_json(eval_path)
    b_games, e_games = boot.get("games", []), ev.get("games", [])
    if {x.get("task_type") for x in b_games} != set(TASK_TYPES) or len(b_games) != len(TASK_TYPES):
        raise ValueError("bootstrap manifest must contain all six task types exactly once")
    if {x.get("task_type") for x in e_games} != set(TASK_TYPES):
        raise ValueError("eval manifest must contain every task type")
    expected_eval_per_type = ev.get("games_per_task_type")
    if expected_eval_per_type is not None:
        if any(sum(x.get("task_type") == tt for x in e_games) != expected_eval_per_type
               for tt in TASK_TYPES):
            raise ValueError("eval manifest does not contain the declared number of games per task type")
    b_paths = {os.path.realpath(data_root / x["game_file"]) for x in b_games}
    e_paths = {os.path.realpath(data_root / x["game_file"]) for x in e_games}
    if len(b_paths) != len(b_games) or len(e_paths) != len(e_games) or b_paths & e_paths:
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
           # One complete fixed-manifest epoch has exactly six games times the
           # requested replica count.  Keeping this explicit is what makes
           # 1/2/4/8-GPU DP plans use the identical seeded trajectory multiset.
           "--epochs", "1", "--group_size", str(args.rollouts_per_type),
           "--max_episodes", str(episodes),
           "--batch_rollout_size", str(batch), "--max_steps", str(args.max_steps),
           "--seed", str(args.seed), "--retrieval_mode", args.retrieval_mode,
           "--gpu_mem_util", str(args.gpu_mem_util),
           "--vllm_enforce_eager", str(args.vllm_enforce_eager),
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


def _canonical_runtime_task_type(task_type: Any) -> str:
    """Accept either dataset-manifest or runtime skill-retrieval task labels."""
    value = str(task_type or "")
    return TASK_TYPE_TO_RUNTIME.get(value, value if value in RUNTIME_TASK_TYPES else "")


def _normalize_external_trace(trace: Dict[str, Any], line_number: int) -> Dict[str, Any]:
    """Validate and normalize an externally collected ALFWorld trace row.

    The interface intentionally accepts the normal driver ``raw_traces.jsonl``
    schema without requiring a run directory.  Only immutable trajectory data
    is imported: no source skills, model state, or prior cloud outputs enter
    the V2 artifact generation.
    """
    if not isinstance(trace, dict):
        raise ValueError(f"external trace line {line_number} must be a JSON object")
    task_type = _canonical_runtime_task_type(trace.get("task_type"))
    if not task_type:
        raise ValueError(
            f"external trace line {line_number} has unsupported task_type={trace.get('task_type')!r}")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"external trace line {line_number} must contain a non-empty steps list")
    normalized_steps = []
    for step_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"external trace line {line_number} step {step_index} must be an object")
        action, observation = step.get("action"), step.get("observation")
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"external trace line {line_number} step {step_index} has no action")
        if not isinstance(observation, str):
            raise ValueError(f"external trace line {line_number} step {step_index} has no observation")
        normalized_steps.append(dict(step))
    success = trace.get("outcome") == "success" or float(trace.get("episode_reward", 0) or 0) > 0
    normalized = dict(trace)
    normalized["task_type"] = task_type
    normalized["outcome"] = "success" if success else "failure"
    normalized["episode_reward"] = float(trace.get("episode_reward", 1.0 if success else 0.0) or 0.0)
    normalized["steps"] = normalized_steps
    normalized.setdefault("traj_uid", f"external_line_{line_number}")
    normalized.setdefault("task", "")
    return normalized


def import_external_raw_traces(source: Path, root: Path) -> Path:
    """Import a supplied JSONL corpus once, with reproducible source lineage."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"external raw trace JSONL not found: {source}")
    destination = root / "frozen" / "raw_traces.jsonl"
    import_record = root / "frozen" / "external_trace_import.json"
    source_sha256 = _sha256_path(source)
    if destination.exists() or import_record.exists():
        if not (destination.exists() and import_record.exists()):
            raise RuntimeError("external trace import is incomplete; use a new --root")
        existing = _read_json(import_record)
        if existing.get("source_sha256") != source_sha256:
            raise RuntimeError(
                "external trace source differs from the corpus already imported into this root; "
                "use a new --root rather than mixing trajectory evidence")
        if existing.get("normalized_sha256") != _sha256_path(destination):
            raise RuntimeError("imported external trace corpus checksum mismatch; use a new --root")
        return destination

    counts, success_counts = defaultdict(int), defaultdict(int)
    normalized_lines: List[str] = []
    seen_uids = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"external trace line {line_number} is not valid JSON: {exc}") from exc
            trace = _normalize_external_trace(raw, line_number)
            uid = str(trace["traj_uid"])
            if uid in seen_uids:
                raise ValueError(f"external trace traj_uid is duplicated: {uid!r}")
            seen_uids.add(uid)
            counts[trace["task_type"]] += 1
            success_counts[trace["task_type"]] += int(trace["outcome"] == "success")
            normalized_lines.append(json.dumps(trace, ensure_ascii=False) + "\n")
    missing = [tt for tt in RUNTIME_TASK_TYPES if not counts[tt]]
    if missing:
        raise ValueError(f"external traces are missing task types: {', '.join(missing)}")
    if not normalized_lines:
        raise ValueError("external trace JSONL is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(normalized_lines), encoding="utf-8")
    _write_json(import_record, {
        "source_path": str(source), "source_sha256": source_sha256,
        "normalized_raw_traces": str(destination), "normalized_sha256": _sha256_path(destination),
        "trace_count": len(normalized_lines), "task_type_counts": dict(counts),
        "success_counts": dict(success_counts),
        "schema": {"required": ["task_type", "steps[].action", "steps[].observation"],
                   "accepted_task_type_labels": list(TASK_TYPES) + list(RUNTIME_TASK_TYPES)},
    })
    return destination


def bootstrap(args, root: Path, bootstrap_manifest: Path) -> Path:
    """Generate the common corpus; retry only task types with no success."""
    if args.external_raw_traces:
        return import_external_raw_traces(Path(args.external_raw_traces), root)
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
        if coverage.get("rollouts_per_type") != args.rollouts_per_type:
            raise RuntimeError(
                "existing frozen corpus uses a different rollout count per task type "
                f"({coverage.get('rollouts_per_type')!r} versus requested "
                f"{args.rollouts_per_type}); use a new --root rather than mixing "
                "the 36- and 72-rollout protocols.")
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
    total_rollouts = len(TASK_TYPES) * args.rollouts_per_type
    _run(_driver_cmd(args, run_dir, bootstrap_manifest, total_rollouts, total_rollouts, 1, 1, 0, 0,
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
        supplement_rollouts = args.rollouts_per_type * len(selected)
        _run(_driver_cmd(args, supplement, supplement_manifest, supplement_rollouts, supplement_rollouts,
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
        "rollouts_per_type": args.rollouts_per_type,
        "initial_total_rollouts": total_rollouts,
        "bootstrap_skill_library": {
            "path": str(empty_skills), "sha256": _sha256_path(empty_skills),
            "is_empty": True, "enable_coskill": True, "enable_skill_tree": True,
            "enable_cloud_updates": False, "enable_skill_tree_evolve": False,
        },
        "coverage": coverage, "supplemented_task_types": missing_success,
    })
    return frozen


def _compress_raw(raw_traces: List[Dict[str, Any]], outdir: Path, **flags: Any) -> Dict[str, Any]:
    flags.setdefault(
        "cloud_evidence_mode",
        "tree_only" if flags.get("enable_prefix_tree", True) else "flat",
    )
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


def _tree_evidence_selection(batch: Dict[str, Any], directory: Path) -> Dict[str, Dict[str, Any]]:
    """Select and persist the exact prompt evidence for canonical-tree V2.

    The compressed batch remains an all-corpus artifact.  This separate file
    makes the bounded cloud prompt reproducible and auditable without treating
    unshown traces as if they had been individually read by the provider.
    """
    by_type: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"success": [], "failure": []})
    for trace in batch.get("success_samples", []) or []:
        by_type[str(trace.get("task_type", "unknown"))]["success"].append(trace)
    for trace in batch.get("failure_samples", []) or []:
        by_type[str(trace.get("task_type", "unknown"))]["failure"].append(trace)

    selected: Dict[str, Dict[str, Any]] = {}
    for task_type in RUNTIME_TASK_TYPES:
        bucket = by_type[task_type]
        diag_success, diag_success_audit = _stratified_trace_sample(
            bucket["success"], TREE_EVIDENCE_SAMPLING["diagnose"]["success"],
            salt=f"diagnose:{task_type}:success")
        diag_failure, diag_failure_audit = _stratified_trace_sample(
            bucket["failure"], TREE_EVIDENCE_SAMPLING["diagnose"]["failure"],
            salt=f"diagnose:{task_type}:failure")
        evolve_success, evolve_success_audit = _stratified_trace_sample(
            bucket["success"], TREE_EVIDENCE_SAMPLING["evolve"]["success"],
            salt=f"evolve:{task_type}:success")
        evolve_failure, evolve_failure_audit = _stratified_trace_sample(
            bucket["failure"], TREE_EVIDENCE_SAMPLING["evolve"]["failure"],
            salt=f"evolve:{task_type}:failure")
        selected[task_type] = {
            "diagnose": {"success": diag_success, "failure": diag_failure,
                         "audit": {"success": diag_success_audit, "failure": diag_failure_audit}},
            "evolve": {"success": evolve_success, "failure": evolve_failure,
                       "audit": {"success": evolve_success_audit, "failure": evolve_failure_audit}},
        }
    # Keep only lineage/stratum metadata on disk: raw trajectory content is
    # already preserved exactly once in compressed_batch.json.
    audit = {
        "sampling": TREE_EVIDENCE_SAMPLING,
        "task_types": {
            task_type: {
                phase: values[phase]["audit"]
                for phase in ("diagnose", "evolve")
            }
            for task_type, values in selected.items()
        },
    }
    _write_json(directory / "tree_evidence_selection.json", audit)
    return selected


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
    """Convert one frozen failure to the original SkillRL updater schema."""
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


def _tree_node_token_accounting(markdown: str) -> Dict[str, Any]:
    """Attribute rendered tree text to exclusive nodes for V2 cost reporting.

    Provider API usage is available only per request, not per Markdown node.
    Therefore ``generated_tokens_chars_div_4`` is explicitly a reproducible
    rendered-text estimate, while the unmodified provider totals remain in
    ``cloud_calls``.  A node receives its heading and direct non-heading text,
    never text owned by a descendant, so node totals compose exactly.
    """
    nodes: List[Dict[str, Any]] = []
    stack: List[int] = []
    current_index = None
    for line in (markdown or "").splitlines():
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            depth = len(line) - len(line.lstrip("#"))
            while stack and nodes[stack[-1]]["depth"] >= depth:
                stack.pop()
            parent_index = stack[-1] if stack else None
            index = len(nodes)
            node = {
                "node_id": f"n{index + 1}", "parent_node_id": (
                    nodes[parent_index]["node_id"] if parent_index is not None else None),
                "depth": depth, "label": line[depth:].strip(), "direct_lines": [line],
            }
            nodes.append(node)
            stack.append(index)
            current_index = index
        elif current_index is not None:
            nodes[current_index]["direct_lines"].append(line)
    for node in nodes:
        rendered = "\n".join(node.pop("direct_lines")).strip()
        node["rendered_chars"] = len(rendered)
        node["generated_tokens_chars_div_4"] = _approx_tokens(rendered)
    return {
        "method": "exclusive_node_rendered_text_chars_div_4",
        "nodes": nodes,
        "generated_tokens_chars_div_4": sum(
            int(node["generated_tokens_chars_div_4"]) for node in nodes),
    }


def _truncate_tree_to_depth(markdown: str, target_depth: int) -> str:
    """Derive L1-L5 from one canonical tree without another cloud request."""
    if target_depth < 1:
        raise ValueError("target_depth must be at least 1")
    lines, include_current = [], False
    for line in (markdown or "").splitlines():
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            depth = len(line) - len(line.lstrip("#"))
            include_current = depth <= target_depth
            if include_current:
                lines.append(line)
        elif include_current:
            lines.append(line)
    result = "\n".join(lines).strip()
    validation = CloudAnalyzer._validate_tree_depth(result, target_depth)
    if not validation.get("depth_valid", False):
        raise ValueError(
            f"canonical tree cannot derive exact L{target_depth}: "
            f"{validation.get('depth_validation_errors', [])}")
    return result


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


def build_l0_artifact(
    raw_path: Path,
    directory: Path,
    *,
    balanced_evidence_per_outcome: int | None = None,
) -> Path:
    """Build L0 as a flat SkillRL JSON skill bank.

    The default preserves the original five-failure protocol.  Representation
    experiments may request the same balanced full evidence used by tree arms;
    only the output representation remains flat in that mode.
    """
    raw = _read_jsonl(raw_path)
    skills, audit = _empty_skill_bank(), []
    source_trace_ids: Dict[str, List[str]] = {}
    successes = defaultdict(list)
    failures = defaultdict(list)
    for trace in raw:
        converted = _raw_to_skillrl_failure(trace)
        if trace.get("outcome") == "success" or trace.get("episode_reward", 0) > 0:
            successes[trace.get("task_type", "unknown")].append(converted)
        else:
            failures[trace.get("task_type", "unknown")].append(converted)
    updater = SkillUpdater(max_new_skills_per_update=3)
    evidence_selection = {
        "sampling": TREE_EVIDENCE_SAMPLING,
        "mode": (
            "balanced_full_success_failure"
            if balanced_evidence_per_outcome is not None
            else "original_skillrl_failure_only"
        ),
        "task_types": {},
    }
    for tt in RUNTIME_TASK_TYPES:
        if balanced_evidence_per_outcome is None:
            selected_successes = []
            selected_failures, failure_audit = _stratified_trace_sample(
                failures.get(tt, []), TREE_EVIDENCE_SAMPLING["flat_skillrl"]["failure"],
                salt=f"flat_skillrl:{tt}:failure")
            success_audit = {
                "available": len(successes.get(tt, [])),
                "requested": 0,
                "selected": 0,
                "selected_traj_uids": [],
            }
            generated = updater.analyze_failures(selected_failures, skills) if selected_failures else []
        else:
            selected_successes, success_audit = _stratified_trace_sample(
                successes.get(tt, []),
                balanced_evidence_per_outcome,
                salt=f"flat_skillrl_balanced:{tt}:success",
            )
            selected_failures, failure_audit = _stratified_trace_sample(
                failures.get(tt, []),
                balanced_evidence_per_outcome,
                salt=f"flat_skillrl_balanced:{tt}:failure",
            )
            if (
                len(selected_successes) != balanced_evidence_per_outcome
                or len(selected_failures) != balanced_evidence_per_outcome
            ):
                raise ValueError(
                    f"balanced L0 needs {balanced_evidence_per_outcome} success and "
                    f"{balanced_evidence_per_outcome} failure traces for {tt}"
                )
            generated = updater.analyze_balanced_trajectories(
                selected_successes,
                selected_failures,
                skills,
            )
        selected = selected_successes + selected_failures
        evidence_selection["task_types"][tt] = {
            "success": success_audit,
            "failure": failure_audit,
        }
        for skill in generated:
            source_trace_ids[skill["skill_id"]] = [x.get("traj_uid") for x in selected]
        skills["task_specific_skills"][tt].extend(generated)
        call_dir = directory / "cloud_io"
        call_dir.mkdir(parents=True, exist_ok=True)
        if updater.last_prompt:
            (call_dir / f"l0_{tt}_prompt.txt").write_text(updater.last_prompt)
        if updater.last_response:
            (call_dir / f"l0_{tt}_response.txt").write_text(updater.last_response)
        usage = dict(updater.last_usage)
        audit.append({
            "purpose": (
                "skillrl_flat_balanced_full_evidence"
                if balanced_evidence_per_outcome is not None
                else "skillrl_flat_failure_analysis"
            ),
            "task_type": tt,
            "success_count_available": len(successes.get(tt, [])),
            "success_count_prompt_selected": len(selected_successes),
            "failure_count_available": len(failures.get(tt, [])),
            "failure_count_prompt_selected": len(selected_failures),
            "trajectory_count_prompt_selected": len(selected),
            "source_traj_uids": [x.get("traj_uid") for x in selected],
            "generated_ids": [x["skill_id"] for x in generated],
            "prompt_sha256": _sha256_text(updater.last_prompt or ""),
            "response_sha256": _sha256_text(updater.last_response or ""),
            # Canonical fields match CloudAnalyzer call audits.  Keep the
            # legacy aliases so older summaries and external readers remain
            # compatible.
            "prompt_tokens": int(usage.get("prompt", 0) or 0),
            "completion_tokens": int(usage.get("completion", 0) or 0),
            "total_tokens": int(usage.get("total", 0) or 0),
            **usage,
        })
    _write_json(directory / "flat_evidence_selection.json", evidence_selection)
    return _save_artifact_manifest(
        directory, "skill_level_l0", raw_path, skills, audit,
        {"skill_level": "L0", "target_depth": 0,
         "generation_protocol": (
             "flat SkillRL JSON from balanced complete success/failure evidence"
             if balanced_evidence_per_outcome is not None
             else "original SkillRL flat failure-analysis JSON"
         ),
         "flat_skill_source_trace_ids": source_trace_ids,
         "evidence_sampling": TREE_EVIDENCE_SAMPLING},
    )


def build_tree_artifact(raw_path: Path, directory: Path, depth: int,
                        max_attempts: int = MAX_TREE_GENERATION_ATTEMPTS,
                        max_tree_nodes: int | None = None,
                        max_tree_chars: int | None = None) -> Path:
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
        while attempts < max_attempts:
            result = analyzer.evolve_playbook(
                tt, None, bucket["success"], bucket["failure"], diagnoses.get(tt, []),
                target_depth=depth, repair_candidate=repair_candidate,
                repair_feedback=({
                    "actual_depth": result.get("actual_depth"),
                    "depth_validation_errors": result.get("depth_validation_errors", []),
                } if result else None), max_tree_nodes=max_tree_nodes,
                max_tree_chars=max_tree_chars,
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
    return _save_artifact_manifest(directory, f"skill_level_l{depth}", raw_path, lib.skills,
                                   analyzer.call_audit,
                                   {"generation_protocol": "CoSkill diagnose_failures + evolve_playbook",
                                    "skill_level": f"L{depth}",
                                    "target_depth": depth, "tree_generation_status": status,
                                    "tree_generation_max_attempts": max_attempts,
                                    "tree_max_nodes": max_tree_nodes,
                                    "tree_max_chars": max_tree_chars,
                                    "status": "N.A." if failed else "ready",
                                    "evaluation_eligible": not bool(failed),
                                    "unavailable_reason": ("depth_validation_failed" if failed else None),
                                    "failed_task_types": sorted(failed),
                                    "compression": batch.get("compression", {}),
                                    **_payload_accounting(batch)})


def build_canonical_tree_artifact(raw_path: Path, directory: Path,
                                  max_attempts: int = MAX_TREE_GENERATION_ATTEMPTS) -> Path:
    """Generate one exact L5 tree per task, the only cloud tree-generation step in V2."""
    raw = _read_jsonl(raw_path)
    batch = _compress_raw(raw, directory / "traces_pool", enable_loop_filter=True,
                          enable_obs_delta=True, enable_prefix_tree=True, enable_consensus_prefix=True)
    empty_path = directory / "empty_skills.json"
    _write_json(empty_path, _empty_skill_bank())
    lib = HierarchicalSkillLib(str(empty_path), retrieval_mode="template", enable_playbook=True)
    analyzer = CloudAnalyzer(output_dir=str(directory), environment_name="ALFWorld")
    evidence = _tree_evidence_selection(batch, directory)
    diagnoses: Dict[str, List[Dict[str, Any]]] = {}
    # One type per call permits substantially more evidence without a six-task
    # mega-prompt and makes cloud usage honestly attributable to that type.
    for tt in RUNTIME_TASK_TYPES:
        diag_evidence = evidence[tt]["diagnose"]
        diag_batch = dict(batch)
        diag_batch.update({"task_type": tt, "success_samples": diag_evidence["success"],
                           "failure_samples": diag_evidence["failure"]})
        diagnoses[tt] = analyzer.diagnose_failures(
            diag_batch, task_type=tt,
            max_success_examples=TREE_EVIDENCE_SAMPLING["diagnose"]["success"],
            max_failure_examples=TREE_EVIDENCE_SAMPLING["diagnose"]["failure"],
        ).get(tt, [])
    status, node_accounting = {}, {}
    for tt in RUNTIME_TASK_TYPES:
        bucket = evidence[tt]["evolve"]
        result, attempts, repair_candidate = None, 0, None
        while attempts < max_attempts:
            result = analyzer.evolve_playbook(
                tt, None, bucket["success"], bucket["failure"], diagnoses.get(tt, []),
                target_depth=5, repair_candidate=repair_candidate,
                max_success_examples=TREE_EVIDENCE_SAMPLING["evolve"]["success"],
                max_failure_examples=TREE_EVIDENCE_SAMPLING["evolve"]["failure"],
                repair_feedback=({
                    "actual_depth": result.get("actual_depth"),
                    "depth_validation_errors": result.get("depth_validation_errors", []),
                } if result else None),
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
        rec = lib.update_playbook(tt, result["skill_tree"], result.get("level", "outline"), {
            "target_depth": 5, "actual_depth": result.get("actual_depth"),
            "depth_generation_attempts": attempts, "critique": result.get("critique", ""),
            "changelog": result.get("changelog", ""),
            "generation_protocol": "v2_canonical_l5_then_truncate",
        })
        node_accounting[tt] = _tree_node_token_accounting(rec["content"])
        status[tt] = {"status": "ok", "version": rec["version"], "attempts": attempts,
                      "actual_depth": result.get("actual_depth")}
    failed = {tt: info for tt, info in status.items() if info["status"] != "ok"}
    _write_json(directory / "generation_status.json", status)
    _write_json(directory / "tree_node_token_accounting.json", node_accounting)
    return _save_artifact_manifest(
        directory, "canonical_tree_l5", raw_path, lib.skills, analyzer.call_audit,
        {"generation_protocol": "one_canonical_l5_per_task_then_deterministic_truncation",
         "canonical_tree_depth": 5, "tree_generation_status": status,
         "tree_generation_max_attempts": max_attempts,
         "evidence_sampling": TREE_EVIDENCE_SAMPLING,
         "evidence_selection_sha256": _sha256_path(directory / "tree_evidence_selection.json"),
         "node_token_accounting": node_accounting,
         "node_generated_tokens_chars_div_4_total": sum(
             int(value.get("generated_tokens_chars_div_4", 0) or 0)
             for value in node_accounting.values()),
         "status": "N.A." if failed else "ready", "evaluation_eligible": False,
         "failed_task_types": sorted(failed), "compression": batch.get("compression", {}),
         **_payload_accounting(batch)})


def build_derived_tree_artifact(raw_path: Path, directory: Path, depth: int,
                                canonical_directory: Path) -> Path:
    """Materialize one evaluation arm by truncating the shared canonical L5 trees."""
    canonical_manifest = _read_json(canonical_directory / "artifact_manifest.json")
    canonical_skills = _read_json(canonical_directory / "skills.json")
    skills = _empty_skill_bank()
    status, node_accounting = {}, {}
    for tt in RUNTIME_TASK_TYPES:
        canonical_status = (canonical_manifest.get("tree_generation_status", {}) or {}).get(tt, {})
        source = (canonical_skills.get("skill_trees", {}) or {}).get(tt)
        if canonical_status.get("status") != "ok" or not isinstance(source, dict):
            status[tt] = dict(canonical_status or {"status": "generation_failed"})
            continue
        content = _truncate_tree_to_depth(str(source.get("content", "")), depth)
        record = dict(source)
        record["content"] = content
        record["level"] = f"derived_l{depth}"
        metadata = dict(record.get("metadata", {}) or {})
        metadata.update({"target_depth": depth, "derived_from": "canonical_tree_l5",
                         "canonical_content_sha256": _sha256_text(str(source.get("content", "")))})
        record["metadata"] = metadata
        skills.setdefault("skill_trees", {})[tt] = record
        node_accounting[tt] = _tree_node_token_accounting(content)
        status[tt] = {"status": "ok", "derivation": "truncate_canonical_l5",
                      "actual_depth": depth}
    failed = {tt: info for tt, info in status.items() if info["status"] != "ok"}
    _write_json(directory / "generation_status.json", status)
    _write_json(directory / "tree_node_token_accounting.json", node_accounting)
    return _save_artifact_manifest(
        directory, f"skill_level_l{depth}", raw_path, skills, [],
        {"generation_protocol": "deterministic_truncation_of_shared_canonical_l5",
         "generation_cost_scope": "shared_canonical_tree_l5",
         "canonical_artifact_sha256": canonical_manifest.get("skills_sha256"),
         "canonical_cloud_call_count": len(canonical_manifest.get("cloud_calls", []) or []),
         "canonical_node_generated_tokens_chars_div_4_total": canonical_manifest.get(
             "node_generated_tokens_chars_div_4_total", 0),
         "skill_level": f"L{depth}", "target_depth": depth,
         "tree_generation_status": status, "node_token_accounting": node_accounting,
         "node_generated_tokens_chars_div_4_total": sum(
             int(value.get("generated_tokens_chars_div_4", 0) or 0)
             for value in node_accounting.values()),
         "status": "N.A." if failed else "ready", "evaluation_eligible": not bool(failed),
         "unavailable_reason": "canonical_tree_generation_failed" if failed else None,
         "failed_task_types": sorted(failed)})


def build_artifacts(args, root: Path, raw_path: Path) -> Dict[str, Path]:
    artifacts = root / "artifacts"
    result = {}
    l0 = artifacts / "skill_level_l0" / "artifact_manifest.json"
    result["skill_level_l0"] = (
        l0 if l0.exists() else build_l0_artifact(raw_path, artifacts / "skill_level_l0")
    )
    if getattr(args, "external_raw_traces", None):
        canonical_dir = artifacts / "canonical_tree_l5"
        canonical_manifest = canonical_dir / "artifact_manifest.json"
        if not canonical_manifest.exists():
            build_canonical_tree_artifact(raw_path, canonical_dir, args.tree_generation_attempts)
        for depth in range(1, 6):
            arm = f"skill_level_l{depth}"
            path = artifacts / arm / "artifact_manifest.json"
            result[arm] = (path if path.exists() else
                           build_derived_tree_artifact(raw_path, artifacts / arm, depth, canonical_dir))
    else:
        for depth in range(1, 6):
            arm = f"skill_level_l{depth}"
            path = artifacts / arm / "artifact_manifest.json"
            result[arm] = (path if path.exists() else
                           build_tree_artifact(raw_path, artifacts / arm, depth,
                                               args.tree_generation_attempts))
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
    level = int(arm.rsplit("l", 1)[1])
    use_flat_skills = int(level == 0)
    use_skill_tree = int(level > 0)
    eval_games = _read_json(eval_manifest).get("games", [])
    # Unit-level callers historically supplied a skeletal manifest because
    # they inspect only the generated driver command. Real manifests are
    # validated before this point; retain the original six-game fallback for
    # that narrow compatibility path.
    eval_game_count = len(eval_games) if eval_games else len(TASK_TYPES)
    rollouts_per_group = eval_game_count * args.rollouts_per_type
    total_rollouts = rollouts_per_group * args.eval_groups_per_level
    cmd = _driver_cmd(args, output, eval_manifest, total_rollouts, rollouts_per_group,
                      use_flat_skills, use_skill_tree, 0, 0,
                      ["--enable_hierarchy", str(int(level > 0)),
                       "--skills_json", str(artifact)])
    cmd[cmd.index("--epochs") + 1] = str(args.eval_groups_per_level)
    _run(cmd, args.project_root)
    return summary


def _latest_group_metrics(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    return (rows[-1].get("metrics", {}) if rows else {})


def _cloud_usage_status_counts(calls: List[Dict[str, Any]]) -> Dict[str, int]:
    """Separate reported zero usage from absent/legacy provider usage."""
    return {
        "missing": sum(1 for call in calls if call.get("usage_reported") is False),
        # Artifacts written before explicit usage_status cannot distinguish a
        # real provider-reported zero from a missing usage object.  Flag them
        # as unverifiable instead of retrospectively asserting a zero cost.
        "unverifiable": sum(1 for call in calls if "usage_reported" not in call),
    }


def _canonical_shared_usage_by_task(root: Path) -> Dict[str, Dict[str, int]]:
    """Return actual canonical-tree provider usage for each runtime task.

    L1--L5 are deterministic materializations of one canonical L5 artifact.
    Their own manifests intentionally have no cloud calls, but reporting those
    rows as literal zero obscures the shared build cost.  Keep the direct-arm
    cost at zero and expose the shared provider usage in separate fields.
    """
    usage = {
        tt: {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
        for tt in RUNTIME_TASK_TYPES
    }
    path = root / "artifacts" / "canonical_tree_l5" / "artifact_manifest.json"
    if not path.exists():
        return usage
    for call in _read_json(path).get("cloud_calls", []) or []:
        tt = call.get("task_type")
        if tt not in usage:
            continue
        bucket = usage[tt]
        bucket["calls"] += 1
        bucket["prompt"] += int(call.get("prompt_tokens", 0) or 0)
        bucket["completion"] += int(call.get("completion_tokens", 0) or 0)
        bucket["total"] += int(call.get("total_tokens", 0) or 0)
    return usage


def _runtime_task_type(task_type: str) -> str:
    """Use the skill-retrieval vocabulary for every derived report."""
    return TASK_TYPE_TO_RUNTIME.get(task_type, task_type if task_type in RUNTIME_TASK_TYPES else "unknown")


def _small_tokens_from_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Rebuild task-scoped SLM usage from the authoritative per-episode ledger."""
    usage = {tt: {"prompt": 0, "response": 0, "total": 0} for tt in RUNTIME_TASK_TYPES}
    for episode in episodes:
        bucket = usage.get(_runtime_task_type(episode.get("detected_type", "unknown")))
        if bucket is None:
            continue
        for key, field in (("prompt", "tokens_prompt"),
                           ("response", "tokens_response"),
                           ("total", "tokens_total")):
            bucket[key] += int(episode.get(field, 0) or 0)
    return usage


def write_summary(root: Path, manifests: Dict[str, str]) -> None:
    rows, detailed = [], {}
    per_task_rows = []
    canonical_shared_usage = _canonical_shared_usage_by_task(root)
    for arm in SKILL_LEVEL_ARMS:
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
            for tt in RUNTIME_TASK_TYPES:
                per_task_rows.append({
                    "arm": arm, "status": "N.A.",
                    "target_tree_depth": artifact.get("target_depth"),
                    "task_type": tt, "episodes": 0, "wins": 0,
                    "success_rate": None,
                    "small_model_prompt_tokens": None,
                    "small_model_response_tokens": None,
                    "small_model_total_tokens": None,
                    "artifact_task_scoped_large_model_calls": None,
                    "artifact_task_scoped_large_model_prompt_tokens": None,
                    "artifact_task_scoped_large_model_completion_tokens": None,
                    "artifact_task_scoped_large_model_total_tokens": None,
                    "reason": reason,
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
        # ``per_game`` retains the actual runtime task name and token count.
        # Rebuild from it so historical reports are repaired even when their
        # serialized ``small_model.by_task_type`` used the old long labels.
        has_episode_token_ledger = any("tokens_total" in episode for episode in episodes)
        small_by_type = (_small_tokens_from_episodes(episodes) if has_episode_token_ledger
                         else (small_usage.get("by_task_type", {}) or {}))
        small_by_type_total = sum(v["total"] for v in small_by_type.values())
        small_by_type_reconciliation_error = int(small_usage.get("total", 0) or 0) - small_by_type_total
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
        # Emit a complete six-task matrix even if a malformed/interrupted arm
        # missed a task. A missing row would otherwise look like missing data
        # rather than the observable zero-episode condition.
        for tt in RUNTIME_TASK_TYPES:
            values = per_type[tt]
            usage = small_by_type.get(tt, {}) or {}
            artifact_calls = [
                call for call in (artifact.get("cloud_calls", []) or [])
                if call.get("task_type") == tt
            ]
            artifact_usage_status = _cloud_usage_status_counts(artifact_calls)
            shared_usage = (canonical_shared_usage.get(tt, {}) if artifact.get(
                "generation_cost_scope") == "shared_canonical_tree_l5" else {})
            per_task_rows.append({
                "arm": arm,
                "status": "ready",
                "target_tree_depth": artifact.get("target_depth"),
                "task_type": tt,
                "episodes": values["episodes"],
                "wins": values["wins"],
                "success_rate": (values["wins"] / values["episodes"]
                                 if values["episodes"] else None),
                "small_model_prompt_tokens": usage.get("prompt", 0),
                "small_model_response_tokens": usage.get("response", 0),
                "small_model_total_tokens": usage.get("total", 0),
                "artifact_task_scoped_large_model_calls": len(artifact_calls),
                "artifact_task_scoped_large_model_usage_missing_calls": artifact_usage_status["missing"],
                "artifact_task_scoped_large_model_usage_unverifiable_calls": artifact_usage_status["unverifiable"],
                "artifact_task_scoped_large_model_prompt_tokens": sum(
                    int(call.get("prompt_tokens", call.get("prompt", 0)) or 0)
                    for call in artifact_calls
                ),
                "artifact_task_scoped_large_model_completion_tokens": sum(
                    int(call.get("completion_tokens", call.get("completion", 0)) or 0)
                    for call in artifact_calls
                ),
                "artifact_task_scoped_large_model_total_tokens": sum(
                    int(call.get("total_tokens", call.get("total", 0)) or 0)
                    for call in artifact_calls
                ),
                # These fields are deliberately separate from direct artifact
                # calls.  The same canonical calls must not be counted once
                # per L1--L5 arm, but analysts still need to see their cost.
                "shared_canonical_large_model_calls": shared_usage.get("calls", 0),
                "shared_canonical_large_model_prompt_tokens": shared_usage.get("prompt", 0),
                "shared_canonical_large_model_completion_tokens": shared_usage.get("completion", 0),
                "shared_canonical_large_model_total_tokens": shared_usage.get("total", 0),
            })
        artifact_calls_all = artifact.get("cloud_calls", []) or []
        artifact_usage_status_all = _cloud_usage_status_counts(artifact_calls_all)
        artifact_calls_unscoped = [call for call in artifact_calls_all if not call.get("task_type")]
        row = {"arm": arm, "status": "ready", "episodes": summary.get("total_episodes", 0),
               "success_rate": summary.get("success_rate", 0), "wins": summary.get("wins", 0),
               "strict_valid_action_rate": strict_valid / max(action_count, 1),
               "normal_valid_action_rate": normal_valid / max(action_count, 1),
               "small_model_prompt_tokens_cumulative": small_usage.get("prompt"),
               "small_model_response_tokens_cumulative": small_usage.get("response"),
               "small_model_total_tokens_cumulative": small_usage.get("total"),
               "small_model_by_task_type_total_rebuilt": small_by_type_total,
               "small_model_by_task_type_reconciliation_error": small_by_type_reconciliation_error,
               "small_model_token_accounting": metrics.get(
                   "tokens/small_model/accounting", small_usage.get("accounting")),
               "large_model_prompt_tokens_cumulative": large_usage.get("prompt"),
               "large_model_completion_tokens_cumulative": large_usage.get("completion"),
               "large_model_total_tokens_cumulative": large_usage.get("total"),
               "artifact_generation_large_model_calls": len(artifact_calls_all),
               "artifact_generation_large_model_usage_missing_calls": artifact_usage_status_all["missing"],
               "artifact_generation_large_model_usage_unverifiable_calls": artifact_usage_status_all["unverifiable"],
               "artifact_generation_large_model_total_tokens": sum(
                   int(call.get("total_tokens", call.get("total", 0)) or 0)
                   for call in artifact_calls_all),
               "artifact_generation_unscoped_large_model_calls": len(artifact_calls_unscoped),
               "artifact_generation_unscoped_large_model_total_tokens": sum(
                   int(call.get("total_tokens", call.get("total", 0)) or 0)
                   for call in artifact_calls_unscoped),
               "perf_total_num_tokens_last_group": metrics.get("perf/total_num_tokens"),
               "perf_total_num_tokens_cumulative_equivalent": small_usage.get("total"),
               "flat_skill_injection_count": flat_injection_count,
               "flat_skill_tokens_chars_div_4": flat_injection_tokens,
               "tree_injection_count": tree_injection_count,
               "tree_text_tokens_chars_div_4": tree_injection_tokens,
               "artifact_sha256": artifact.get("skills_sha256"),
               "raw_traces_sha256": artifact.get("raw_traces_sha256")}
        rows.append(row)
    canonical_generation = None
    canonical_path = root / "artifacts" / "canonical_tree_l5" / "artifact_manifest.json"
    if canonical_path.exists():
        canonical = _read_json(canonical_path)
        canonical_calls = canonical.get("cloud_calls", []) or []
        canonical_generation = {
            "status": canonical.get("status"),
            "generation_cost_scope": "shared_once_for_all_l1_l5_arms",
            "provider_api_call_count": len(canonical_calls),
            "provider_usage_missing_calls": _cloud_usage_status_counts(canonical_calls)["missing"],
            "provider_usage_unverifiable_calls": _cloud_usage_status_counts(canonical_calls)["unverifiable"],
            "provider_total_tokens": sum(
                int(call.get("total_tokens", call.get("total", 0)) or 0)
                for call in canonical_calls),
            "node_generated_tokens_chars_div_4_total": canonical.get(
                "node_generated_tokens_chars_div_4_total", 0),
            "node_token_accounting": canonical.get("node_token_accounting", {}),
            "artifact_sha256": canonical.get("skills_sha256"),
        }
    _write_json(root / "ablation_summary.json", {"manifests": manifests, "arms": detailed,
                                                   "canonical_tree_generation": canonical_generation,
                                                   "token_accounting": {"trajectory_estimate": "chars_div_4",
                                                                        "small_model": "reported_per_arm_in_small_model_token_accounting",
                                                                        "large_model": "provider_api_usage",
                                                                        "perf/total_num_tokens": "driver compatibility metric"}})
    _write_jsonl(root / "metrics.jsonl", rows)
    _write_jsonl(root / "metrics_by_task.jsonl", per_task_rows)
    with (root / "ablation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)
    with (root / "skill_level_by_task.csv").open("w", newline="") as f:
        fields = ("arm", "status", "target_tree_depth", "task_type", "episodes", "wins", "success_rate",
                  "small_model_prompt_tokens", "small_model_response_tokens", "small_model_total_tokens",
                  "artifact_task_scoped_large_model_calls",
                  "artifact_task_scoped_large_model_usage_missing_calls",
                  "artifact_task_scoped_large_model_usage_unverifiable_calls",
                  "artifact_task_scoped_large_model_prompt_tokens",
                  "artifact_task_scoped_large_model_completion_tokens",
                  "artifact_task_scoped_large_model_total_tokens",
                  "shared_canonical_large_model_calls",
                  "shared_canonical_large_model_prompt_tokens",
                  "shared_canonical_large_model_completion_tokens",
                  "shared_canonical_large_model_total_tokens", "reason")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(per_task_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ablation output root; phases are resumable")
    ap.add_argument("--phase", choices=("manifests", "bootstrap", "artifacts", "evaluate", "summary", "all"), default="all")
    ap.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--rollouts_per_type", type=int, default=12,
                    help="each of the six fixed games is sampled this many times; "
                    "the formal default is 12 x 6 = 72 rollouts per bootstrap/evaluation arm")
    ap.add_argument("--eval_games_per_type", type=int, default=None,
                    help="distinct fixed evaluation games per task type; V2 external-trace runs default to 5, "
                    "legacy self-bootstrap runs retain 1")
    ap.add_argument("--external_raw_traces", default=None,
                    help="existing ALFWorld raw_traces.jsonl to import instead of generating bootstrap rollouts")
    ap.add_argument("--eval_groups_per_level", type=int, default=1,
                    help="number of 72-rollout evaluation groups for each L0-L5 arm; default 1")
    ap.add_argument("--tree_generation_attempts", type=int,
                    default=MAX_TREE_GENERATION_ATTEMPTS,
                    help="same-evidence cloud attempts per task/tree; formal default 20")
    ap.add_argument("--model_path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--retrieval_mode", default="template", choices=("template", "embedding"))
    ap.add_argument("--data_parallel_workers", type=int, default=0)
    ap.add_argument("--rollout_worker_gpus", default=None)
    ap.add_argument("--gpu_mem_util", type=float,
                    default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
                    help="vLLM per-replica GPU memory fraction; must match the launcher preflight")
    ap.add_argument("--vllm_enforce_eager", type=int, choices=(0, 1),
                    default=int(os.environ.get("VLLM_ENFORCE_EAGER", "0")),
                    help="0 enables vLLM CUDA Graph after warm-up; 1 is an explicit debug fallback")
    ap.add_argument("--log_trajectories", type=int, default=0)
    ap.add_argument("--driver_arg", action="append", default=[], help="extra one-token argument forwarded to the main driver")
    args = ap.parse_args()
    args.project_root = Path(__file__).resolve().parents[2]
    root = Path(args.root).resolve()
    if args.phase == "summary":
        config_path = root / "run_config.json"
        if not config_path.exists():
            ap.error(f"--phase summary requires an existing run_config.json under {root}")
        existing = _read_json(config_path)
        if existing.get("experiment_kind") != "alfworld_skill_level_l0_l5":
            ap.error("--phase summary only supports the independent L0-L5 skill-level experiment")
        write_summary(root, {
            "bootstrap_sha256": existing.get("bootstrap_sha256"),
            "eval_sha256": existing.get("eval_sha256"),
        })
        return
    if args.rollouts_per_type < 1:
        ap.error("--rollouts_per_type must be at least 1")
    if args.eval_games_per_type is None:
        args.eval_games_per_type = 5 if args.external_raw_traces else 1
    if args.eval_games_per_type < 1:
        ap.error("--eval_games_per_type must be at least 1")
    if args.eval_groups_per_level < 1:
        ap.error("--eval_groups_per_level must be at least 1")
    if args.tree_generation_attempts < 1:
        ap.error("--tree_generation_attempts must be at least 1")
    if not 0 < args.gpu_mem_util <= 1:
        ap.error("--gpu_mem_util must be in (0, 1]")
    if not args.alfworld_data:
        ap.error("--alfworld_data or ALFWORLD_DATA is required")
    root, data_root = root, Path(args.alfworld_data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "run_config.json"
    if config_path.exists():
        existing = _read_json(config_path)
        if existing.get("experiment_kind") != "alfworld_skill_level_l0_l5":
            raise RuntimeError(
                "existing root belongs to the legacy combined fixed-trajectory ablation. "
                "Use a new --root for the independent L0-L5 skill-level experiment.")
        existing_rollouts = existing.get("bootstrap_rollouts_per_type")
        if existing_rollouts is not None and existing_rollouts != args.rollouts_per_type:
            raise RuntimeError(
                "existing ablation root uses bootstrap_rollouts_per_type="
                f"{existing_rollouts}; requested {args.rollouts_per_type}. "
                "Use a new --root so different rollout multiplicities never mix.")
        existing_groups = existing.get("eval_groups_per_level", 1)
        if existing_groups != args.eval_groups_per_level:
            raise RuntimeError(
                "existing skill-tree experiment uses eval_groups_per_level="
                f"{existing_groups}; requested {args.eval_groups_per_level}. Use a new --root.")
        existing_attempts = existing.get(
            "tree_depth_generation_max_attempts", MAX_TREE_GENERATION_ATTEMPTS)
        if existing_attempts != args.tree_generation_attempts:
            raise RuntimeError(
                "existing skill-tree experiment uses tree_generation_attempts="
                f"{existing_attempts}; requested {args.tree_generation_attempts}. Use a new --root.")
        existing_eval_games = existing.get("eval_games_per_type", 1)
        if existing_eval_games != args.eval_games_per_type:
            raise RuntimeError(
                "existing skill-tree experiment uses eval_games_per_type="
                f"{existing_eval_games}; requested {args.eval_games_per_type}. Use a new --root.")
        existing_external = bool(existing.get("external_raw_traces"))
        if existing_external != bool(args.external_raw_traces):
            raise RuntimeError("existing root uses a different bootstrap/external-trace protocol; use a new --root.")
    bootstrap_empty_skills = _ensure_empty_bootstrap_skills(root)
    boot_manifest, eval_manifest = create_manifests(
        root, data_root, args.split, args.sample_seed, args.eval_games_per_type)
    manifest_hashes = validate_manifest_pair(boot_manifest, eval_manifest, data_root)
    external_source = None
    if args.external_raw_traces:
        external_path = Path(args.external_raw_traces).expanduser().resolve()
        if not external_path.is_file():
            ap.error(f"--external_raw_traces does not exist: {external_path}")
        external_source = {"path": str(external_path), "sha256": _sha256_path(external_path)}
    _write_json(config_path, {"experiment_kind": "alfworld_skill_level_l0_l5",
                                             "schema_version": 2 if external_source else 1,
                                             "task_types": TASK_TYPES,
                                             "skill_levels": list(SKILL_LEVEL_ARMS),
                                             "skill_level_definition": {
                                                 "L0": "original SkillRL flat skills; no hierarchy",
                                                 "L1-L5": "CoSkill skill trees with exact maximum Markdown heading depth",
                                             },
                                             "runtime_task_type_map": TASK_TYPE_TO_RUNTIME,
                                             "bootstrap_rollouts_per_type": args.rollouts_per_type,
                                             "eval_rollouts_per_type": args.rollouts_per_type,
                                             "eval_games_per_type": args.eval_games_per_type,
                                             "eval_groups_per_level": args.eval_groups_per_level,
                                             "rollouts_per_group": (len(TASK_TYPES) * args.eval_games_per_type *
                                                                    args.rollouts_per_type),
                                             "external_raw_traces": external_source,
                                             "tree_representation_protocol": (
                                                 "one_canonical_l5_per_task_then_deterministic_l1_l5_truncation"
                                                 if external_source else "independent_tree_generation_per_depth"),
                                             "gpu_mem_util": args.gpu_mem_util,
                                             "vllm_enforce_eager": bool(args.vllm_enforce_eager),
                                             "seed": args.seed,
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
                                             "tree_depth_generation_max_attempts": args.tree_generation_attempts,
                                             **manifest_hashes})
    if args.phase == "manifests":
        return
    raw = bootstrap(args, root, boot_manifest)
    if args.phase == "bootstrap":
        return
    build_artifacts(args, root, raw)
    if args.phase == "artifacts":
        return
    for arm in SKILL_LEVEL_ARMS:
        evaluate_arm(args, root, eval_manifest, arm)
    write_summary(root, manifest_hashes)


if __name__ == "__main__":
    main()
