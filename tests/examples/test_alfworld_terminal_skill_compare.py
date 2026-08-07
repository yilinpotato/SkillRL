import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "terminal_compare"
    / "alfworld_skill_compare.py"
)
SPEC = importlib.util.spec_from_file_location("alfworld_skill_compare", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_arm(root: Path, arm: str, episodes):
    arm_dir = root / arm
    _write_json(
        arm_dir / "summary.json",
        {
            "per_game": [
                {
                    "detected_type": episode["task_type"],
                    "used_steps": episode["steps"],
                    "won": episode["success"],
                }
                for episode in episodes
            ]
        },
    )
    trace_path = arm_dir / "traces_pool" / "raw_traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(
            json.dumps({"task_type": episode["task_type"], "task": episode["task"]}) + "\n"
            for episode in episodes
        ),
        encoding="utf-8",
    )


def test_build_combined_rows_matches_tasks_independent_of_episode_order(tmp_path):
    tasks = [
        {"task_type": "clean", "task": "clean the mug.", "steps": 12, "success": True},
        {"task_type": "heat", "task": "heat the apple.", "steps": 40, "success": False},
    ]
    _write_arm(tmp_path, "coskill", tasks)
    _write_arm(
        tmp_path,
        "skillrl",
        [
            {"task_type": "heat", "task": "heat the apple.", "steps": 31, "success": True},
            {"task_type": "clean", "task": "clean the mug.", "steps": 40, "success": False},
        ],
    )

    rows = MODULE.build_combined_rows(tmp_path)

    assert rows == [
        {
            "task_type": "clean",
            "task": "clean the mug.",
            "coskill_steps": 12,
            "coskill_success": True,
            "skillrl_steps": 40,
            "skillrl_success": False,
        },
        {
            "task_type": "heat",
            "task": "heat the apple.",
            "coskill_steps": 40,
            "coskill_success": False,
            "skillrl_steps": 31,
            "skillrl_success": True,
        },
    ]


def test_validate_skill_artifacts_requires_tree_vs_flat(tmp_path):
    tree_keys = list(MODULE.RUNTIME_TASK_TYPES.values())
    coskill = tmp_path / "coskill.json"
    skillrl = tmp_path / "skillrl.json"
    _write_json(coskill, {"skill_trees": {key: "tree" for key in tree_keys}})
    _write_json(
        skillrl,
        {
            "general_skills": [{"skill_id": "general_1"}],
            "task_specific_skills": {"clean": [{"skill_id": "clean_1"}]},
        },
    )

    result = MODULE.validate_skill_artifacts(coskill, skillrl)

    assert result["coskill"]["training_rollouts"] == 3600
    assert result["skillrl"]["training_rollouts"] == 3600
    assert result["coskill"]["injection"] == "task_skill_tree_only"
    assert result["skillrl"]["injection"] == "flat_skills_only"


def test_driver_commands_change_only_the_skill_injection_arm(tmp_path):
    manifest = tmp_path / "fixed_tasks.json"
    _write_json(manifest, {"games": [{"game_file": f"game_{index}"} for index in range(24)]})
    args = SimpleNamespace(
        model_path=Path("/models/qwen3-4b"),
        batch_size=6,
        max_steps=40,
        enforce_eager=0,
        seed=0,
        gpu_memory_utilization=0.75,
        temperature=0.4,
    )
    coskill = MODULE._driver_command(args, "coskill", tmp_path / "coskill", manifest, Path("tree.json"))
    skillrl = MODULE._driver_command(args, "skillrl", tmp_path / "skillrl", manifest, Path("flat.json"))

    assert coskill[coskill.index("--enable_coskill") + 1] == "0"
    assert coskill[coskill.index("--enable_skill_tree") + 1] == "1"
    assert skillrl[skillrl.index("--enable_coskill") + 1] == "1"
    assert skillrl[skillrl.index("--enable_skill_tree") + 1] == "0"
    assert "lora" not in " ".join(coskill).lower()
    assert "lora" not in " ".join(skillrl).lower()
