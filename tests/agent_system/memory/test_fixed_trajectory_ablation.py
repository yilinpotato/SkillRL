import json
import sys
from types import SimpleNamespace
from pathlib import Path

# pytest inserts tests/ ahead of the project root, where tests/agent_system
# would otherwise shadow the real agent_system package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.memory.coskill_loop import CoSkillCloudLoop
from agent_system.memory.skill_updater import SkillUpdater
from agent_system.memory.traces_pool import TracesPool
from examples.playbook_evolve.fixed_trajectory_ablation import (
    RUNTIME_TASK_TYPES, TASK_TYPE_TO_RUNTIME, _empty_skill_bank, _ensure_empty_bootstrap_skills,
    _driver_cmd, _tree_stats, build_tree_artifact, validate_manifest_pair,
)
from examples.playbook_evolve.run_playbook_evolve import (
    _fixed_manifest_dp_plan, _fixed_request_seed, _stable_game_id,
)
from mini_test_pen_shelf.agent_vllm import VLLMAgent


def _trace(outcome="failure"):
    return {
        "traj_uid": "t1", "task_type": "pick_heat_then_place_in_recep", "outcome": outcome,
        "steps": [
            {"observation": "room\nobject", "action": "go to kitchen", "reward": 0},
            {"observation": "room\nobject", "action": "go to kitchen", "reward": 0},
        ],
    }


def test_all_compression_off_has_raw_observations_and_no_prefix_fields():
    pool = TracesPool(enable_loop_filter=False, enable_obs_delta=False,
                      enable_prefix_tree=False, enable_consensus_prefix=False)
    pool.add_trace(_trace())
    batch = pool.export_batch()
    assert "prefix_tree" not in batch
    assert "consensus_prefix" not in batch
    step = batch["failure_samples"][0]["steps"][0]
    assert step["observation"] == "room\nobject"
    assert "obs_delta" not in step
    assert batch["compression"]["enable_obs_delta"] is False


def test_driver_forwards_gpu_memory_fraction_to_every_rollout(tmp_path):
    args = SimpleNamespace(
        rollouts_per_type=12, max_steps=40, seed=0, retrieval_mode="template",
        gpu_mem_util=0.76, vllm_enforce_eager=0, log_trajectories=0,
        model_path=None, data_parallel_workers=1, rollout_worker_gpus="0", driver_arg=[],
    )
    cmd = _driver_cmd(args, tmp_path / "out", tmp_path / "games.json", 72, 72, 1, 1, 0, 0)
    assert cmd[cmd.index("--gpu_mem_util") + 1] == "0.76"


def test_default_compression_remains_delta_compatible():
    pool = TracesPool()
    pool.add_trace(_trace())
    batch = pool.export_batch()
    assert "prefix_tree" in batch
    assert "consensus_prefix" in batch
    assert "obs_delta" in batch["failure_samples"][0]["steps"][0]


def _trace_with_actions(traj_uid, actions, outcome="failure"):
    return {
        "traj_uid": traj_uid, "task_type": "clean", "outcome": outcome,
        "steps": [{"observation": "room", "action": a, "reward": 0} for a in actions],
    }


def test_prefix_tree_merges_on_normalized_action_not_instance_number():
    # Two episodes make the SAME semantic decision (check a cabinet next) but
    # sample different receptacle instance numbers.  Merging on the literal
    # action string would fork them immediately at step 0 ("go to cabinet 1"
    # vs "go to cabinet 7"), hiding the fact that both did the same thing.
    pool = TracesPool()
    pool.add_trace(_trace_with_actions("a", ["go to cabinet 1", "open cabinet 1"]))
    pool.add_trace(_trace_with_actions("b", ["go to cabinet 7", "open cabinet 7"], outcome="success"))
    batch = pool.export_batch()
    root = batch["prefix_tree"]
    assert list(root["children"].keys()) == ["go to cabinet #"]
    merged = root["children"]["go to cabinet #"]
    assert merged["count"] == 2
    assert merged["n_success"] == 1 and merged["n_failure"] == 1
    assert merged["n_variants"] == 2
    assert set(merged["example_actions"]) == {"go to cabinet 1", "go to cabinet 7"}


def test_prefix_tree_still_forks_on_genuinely_different_actions():
    pool = TracesPool()
    pool.add_trace(_trace_with_actions("a", ["go to cabinet 1"]))
    pool.add_trace(_trace_with_actions("b", ["go to drawer 2"]))
    batch = pool.export_batch()
    root = batch["prefix_tree"]
    assert set(root["children"].keys()) == {"go to cabinet #", "go to drawer #"}


def test_format_forks_shows_normalized_branch_with_variant_hint():
    pool = TracesPool()
    pool.add_trace(_trace_with_actions("a", ["go to cabinet 1"]))
    pool.add_trace(_trace_with_actions("b", ["go to cabinet 9"]))
    pool.add_trace(_trace_with_actions("c", ["go to drawer 2"]))
    batch = pool.export_batch()
    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    fork_txt = analyzer._format_forks(batch["prefix_tree"])
    assert "'go to cabinet #' [2 instance variants" in fork_txt
    assert "go to cabinet 1" in fork_txt and "go to cabinet 9" in fork_txt
    assert "'go to drawer #'" in fork_txt


def _trace_with_observations(traj_uid, obs_action_pairs, outcome="failure"):
    return {
        "traj_uid": traj_uid, "task_type": "clean", "outcome": outcome,
        "steps": [{"observation": o, "action": a, "reward": 0} for o, a in obs_action_pairs],
    }


def test_diff_compress_references_earlier_identical_observation():
    # Revisiting the same receptacle a second time (without having changed
    # its state) produces the exact same observation text again. This should
    # collapse to a short back-reference, not a repeat of the full text --
    # real ablation data shows ~37% of steps are exact repeats of an earlier
    # observation in the same trace, and the pre-fix code stored every one
    # of them in full (0% real compression).
    pool = TracesPool()
    pool.add_trace(_trace_with_observations("a", [
        ("You arrive at cabinet 1. The cabinet 1 is closed.", "go to cabinet 1"),
        ("You open the cabinet 1. In it, you see nothing.", "open cabinet 1"),
        ("You arrive at cabinet 2. The cabinet 2 is closed.", "go to cabinet 2"),
        ("You arrive at cabinet 1. The cabinet 1 is closed.", "go to cabinet 1"),
    ]))
    batch = pool.export_batch()
    steps = batch["failure_samples"][0]["steps"]
    assert steps[3]["obs_is_full"] is False
    # References the FIRST time this exact text appeared (step 0, right at
    # episode start), not the most recent occurrence -- only a step that
    # itself stored the full raw text can be a valid anchor, and only the
    # first occurrence is guaranteed to have done so.
    assert steps[3]["obs_delta"] == "(same as after '(episode start)')"


def test_diff_compress_consecutive_repeat_is_not_stored_in_full():
    # A repeated ineffective action (e.g. bumping into a closed receptacle
    # again) yields the identical observation as the immediately prior step.
    # The pre-fix code explicitly excluded "no change" deltas from the
    # worth-it check, so this was stored as full raw text every time.
    pool = TracesPool()
    pool.add_trace(_trace_with_observations("a", [
        ("You arrive at cabinet 1. The cabinet 1 is closed.", "go to cabinet 1"),
        ("Nothing happens.", "open cabinet 1"),
        ("Nothing happens.", "open cabinet 1"),
    ]))
    batch = pool.export_batch()
    steps = batch["failure_samples"][0]["steps"]
    # "Nothing happens." is short enough that a "(no change)" marker beats a
    # "(same as after '<action>')" back-reference on length -- either is a
    # correct compressed representation, so just check it got compressed at
    # all and never grew past the raw text.
    assert steps[2]["obs_is_full"] is False
    assert len(steps[2]["obs_delta"]) < len("Nothing happens.")


def test_diff_compress_strips_alfworld_welcome_boilerplate_and_task_suffix():
    pool = TracesPool()
    pool.add_trace(_trace_with_observations("a", [
        ("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. "
         "Looking quickly around you, you see a cabinet 1.\n\nYour task is to: "
         "find a mug and put it in the microwave.", "go to cabinet 1"),
    ]))
    batch = pool.export_batch()
    step = batch["failure_samples"][0]["steps"][0]
    assert "Welcome to TextWorld" not in step["obs_delta"]
    assert "Your task is to" not in step["obs_delta"]
    assert step["obs_delta"] == "a cabinet 1."


def test_diff_compress_never_makes_a_short_observation_longer():
    # A short exact repeat whose anchor action is itself long enough that
    # the back-reference text would be *longer* than the raw observation
    # must fall back to the raw text, not a needlessly bigger reference.
    pool = TracesPool()
    pool.add_trace(_trace_with_observations("a", [
        ("You arrive at cabinet 1.", "go to a very specific and unusually long named cabinet receptacle 1"),
        ("ok", "go to cabinet 2"),
        ("ok", "go to cabinet 3"),
    ]))
    batch = pool.export_batch()
    steps = batch["failure_samples"][0]["steps"]
    assert steps[2]["obs_delta"] == "ok"
    assert steps[2]["obs_is_full"] is True


def test_diff_compress_genuinely_new_observation_keeps_raw_text():
    pool = TracesPool()
    pool.add_trace(_trace_with_observations("a", [
        ("You arrive at cabinet 1. The cabinet 1 is closed.", "go to cabinet 1"),
        ("You open the cabinet 1. In it, you see a bowl 1.", "open cabinet 1"),
    ]))
    batch = pool.export_batch()
    steps = batch["failure_samples"][0]["steps"]
    assert steps[1]["obs_delta"] == "You open the cabinet 1. In it, you see a bowl 1."
    assert steps[1]["obs_is_full"] is True


def test_tree_depth_and_structure_accounting():
    stats = _tree_stats("# Root\n## Branch\n### Leaf\n## Second")
    assert stats["max_depth"] == 3
    assert stats["node_count"] == 4
    assert stats["edge_count"] == 3
    assert stats["leaf_count"] == 2


def test_skillrl_json_parser_requires_claude_fields():
    updater = SkillUpdater.__new__(SkillUpdater)
    parsed = updater._parse_skills_response(
        '[{"skill_id":"dyn_001","title":"Check State First",'
        '"principle":"Inspect state before acting.","when_to_apply":"Before a state-changing action."},'
        '{"skill_id":"bad","title":"Missing principle"}]'
    )
    assert len(parsed) == 1
    assert parsed[0]["when_to_apply"] == "Before a state-changing action."


def test_tree_depth_helper_counts_markdown_heading_levels_only():
    assert CloudAnalyzer._tree_depth("intro\n# A\n## B\n### C") == 3
    assert CloudAnalyzer._tree_depth("no headings") == 0


def test_fixed_depth_validation_requires_every_intermediate_level():
    assert CloudAnalyzer._validate_tree_depth("# A\n## B\n### C", 3)["depth_valid"] is True
    skipped = CloudAnalyzer._validate_tree_depth("# A\n### C", 3)
    assert skipped["depth_valid"] is False
    assert skipped["missing_heading_levels"] == [2]


def test_artifacts_use_runtime_retrieval_categories_not_manifest_categories():
    bank = _empty_skill_bank()
    assert TASK_TYPE_TO_RUNTIME["pick_heat_then_place_in_recep"] == "heat"
    assert "heat" in bank["task_specific_skills"]
    assert "pick_heat_then_place_in_recep" not in bank["task_specific_skills"]
    assert len(RUNTIME_TASK_TYPES) == 6
    # SkillsOnlyMemory._detect_task_type folds both phrasing templates into
    # one label, "look_at_obj_in_light" (never "examine") -- raw traces are
    # tagged with that name, so this mapping must be the identity or every
    # by-runtime-type lookup for this task silently sees zero samples.
    assert TASK_TYPE_TO_RUNTIME["look_at_obj_in_light"] == "look_at_obj_in_light"


def test_fixed_depth_cloud_prompt_is_explicit_without_dummy_local_rewrite():
    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.environment_name = "ALFWorld"
    prompt = analyzer._build_evolve_prompt(
        "pick_heat_then_place_in_recep", None, [], [_trace()], [], target_depth=3
    )
    assert "EXACTLY 3 semantic Markdown heading levels" in prompt
    assert "Do not use empty, dummy" in prompt
    assert "EXPERIMENTAL OVERRIDE" in prompt
    assert "FORCE a semantic deepening" not in prompt


def test_depth_repair_prompt_forces_semantic_deepening():
    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.environment_name = "ALFWorld"
    prompt = analyzer._build_evolve_prompt(
        "pick_heat_then_place_in_recep", None, [], [_trace()], [], target_depth=3,
        repair_candidate="# Existing root",
    )
    assert "FORCE a semantic deepening" in prompt
    assert "same-evidence grounding" in prompt


def test_tree_arm_becomes_na_after_twenty_invalid_cloud_attempts(tmp_path, monkeypatch):
    import examples.playbook_evolve.fixed_trajectory_ablation as ablation

    class FakeLib:
        def __init__(self, *_args, **_kwargs):
            self.skills = _empty_skill_bank()

    class FakeAnalyzer:
        calls = 0
        def __init__(self, *_args, **_kwargs): self.call_audit = []
        def diagnose_failures(self, _batch): return {}
        def evolve_playbook(self, *_args, **_kwargs):
            type(self).calls += 1
            return {
                "skill_tree": "# still shallow", "action": "rewrite", "actual_depth": 1,
                "heading_levels_present": [1], "depth_validation_errors": ["missing_heading_levels:2,3"],
                "depth_valid": False,
            }

    monkeypatch.setattr(ablation, "HierarchicalSkillLib", FakeLib)
    monkeypatch.setattr(ablation, "CloudAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(ablation, "_compress_raw", lambda *_args, **_kwargs: {
        "success_samples": [], "failure_samples": [], "compression": {},
    })
    raw = tmp_path / "raw.jsonl"
    raw.write_text("")
    manifest = json.loads(build_tree_artifact(raw, tmp_path / "tree", 3).read_text())
    assert FakeAnalyzer.calls == 20 * len(RUNTIME_TASK_TYPES)
    assert manifest["status"] == "N.A."
    assert manifest["evaluation_eligible"] is False
    assert manifest["tree_generation_max_attempts"] == 20
    assert set(manifest["failed_task_types"]) == set(RUNTIME_TASK_TYPES)


def test_empty_bootstrap_skill_library_is_content_free(tmp_path):
    path = _ensure_empty_bootstrap_skills(tmp_path)
    bank = json.loads(path.read_text())
    assert bank["general_skills"] == []
    assert all(not entries for entries in bank["task_specific_skills"].values())
    assert bank["common_mistakes"] == []
    assert bank["metadata"]["handwritten_seed_skills"] is False


def test_tree_depth_repair_reuses_same_evidence_once(tmp_path):
    class FakeAnalyzer:
        def __init__(self): self.calls = []
        def diagnose_failures(self, _): return {}
        def evolve_playbook(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"skill_tree": "# one", "depth_valid": False, "actual_depth": 1,
                        "action": "rewrite"}
            return {"skill_tree": "# one\n## two", "depth_valid": True, "actual_depth": 2,
                    "action": "rewrite", "level": 2, "critique": "", "changelog": ""}

    class FakeLib:
        def __init__(self): self.records = {}
        def get_playbook_record(self, tt): return self.records.get(tt)
        def update_playbook(self, task_type, content, level, meta):
            self.records[task_type] = {"version": 1, "content": content, "level": level, "nodes": {}, **meta}
            return self.records[task_type]

    loop = CoSkillCloudLoop(str(tmp_path), enable_playbook_evolve=True,
                            playbook_evolve_min_samples=1, required_tree_depth=2,
                            tree_depth_repair_attempts=1)
    fake, lib = FakeAnalyzer(), FakeLib()
    batch = {"success_samples": [], "failure_samples": [_trace()]}
    loop._evolve_playbooks(fake, lib, batch, global_step=1)
    assert len(fake.calls) == 2
    assert fake.calls[1]["repair_candidate"] == "# one"
    assert lib.records["pick_heat_then_place_in_recep"]["depth_repair_attempts"] == 1


def test_manifest_pair_rejects_overlap(tmp_path):
    root = tmp_path / "alfworld"
    game = root / "json_2.1.1" / "train" / "x" / "game.tw-pddl"
    game.parent.mkdir(parents=True)
    game.write_text("{}")
    tasks = [
        "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
    ]
    games = [{"task_type": task, "game_file": "json_2.1.1/train/x/game.tw-pddl"} for task in tasks]
    b, e = tmp_path / "bootstrap.json", tmp_path / "eval.json"
    b.write_text(json.dumps({"games": games})); e.write_text(json.dumps({"games": games}))
    try:
        validate_manifest_pair(b, e, root)
    except ValueError as exc:
        assert "non-overlapping" in str(exc)
    else:
        raise AssertionError("overlapping manifests must be rejected")


def test_fixed_manifest_dp_plan_preserves_formal_twelve_rollouts_per_game():
    games = [f"game-{i}" for i in range(6)]
    for workers in (1, 2, 4, 6, 8):
        plan = _fixed_manifest_dp_plan(games, replicas_per_game=12, workers=workers)
        assert len(plan) == workers
        assert sum(batch for _, batch in plan) == 72
        counts = {game: 0 for game in games}
        for assigned, batch in plan:
            assert batch % len(assigned) == 0
            per_game = batch // len(assigned)
            for game in assigned:
                counts[game] += per_game
        assert set(counts.values()) == {12}


def test_fixed_request_seed_is_path_and_worker_independent():
    local = "/home/user/.cache/alfworld/json_2.1.1/train/type/game/traj/game.tw-pddl"
    container = "/opt/data/alfworld/json_2.1.1/train/type/game/traj/game.tw-pddl"
    assert _stable_game_id(local) == _stable_game_id(container)
    assert _fixed_request_seed(7, _stable_game_id(local), 2, 4) == _fixed_request_seed(
        7, _stable_game_id(container), 2, 4)
    assert _fixed_request_seed(7, _stable_game_id(local), 3, 4) != _fixed_request_seed(
        7, _stable_game_id(local), 2, 4)


def test_vllm_batch_accepts_one_seed_per_request():
    class FakeLLM:
        def __init__(self): self.sampling = None
        def generate(self, prompts, sampling, use_tqdm=False):
            self.sampling = sampling
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text=f"answer-{i}", finish_reason="stop")]) for i, _ in enumerate(prompts)]

    agent = VLLMAgent.__new__(VLLMAgent)
    agent.llm = FakeLLM()
    agent._build_prompt = lambda text: f"prompt:{text}"
    agent._single_sampling = lambda *, temperature=None, seed=None: {"seed": seed}
    agent._record_token_usage = lambda outputs: None
    agent._restore_think = lambda prompt, text: text
    result = agent.act_batch_with_meta(["a", "b"], sampling_seeds=[101, 202])
    assert agent.llm.sampling == [{"seed": 101}, {"seed": 202}]
    assert result == [("answer-0", False), ("answer-1", False)]
