import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "terminal_compare"
    / "replay_alfworld_single_task.py"
)
SPEC = importlib.util.spec_from_file_location("replay_alfworld_single_task", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_single_task_replay_outputs_every_action(tmp_path, capsys):
    run_dir = tmp_path / "coskill"
    _write_json(
        tmp_path / "comparison_manifest.json",
        {
            "base_model_shared": "/models/qwen3-4b",
            "max_environment_steps": 40,
            "total_tasks_per_arm": 1,
            "tasks_per_type": 1,
            "batch_size": 12,
        },
    )
    _write_json(run_dir / "terminal_run_summary.json", {"wall_time_seconds": 100.0})
    _write_json(
        run_dir / "summary.json",
        {
            "per_game": [
                {
                    "detected_type": "clean",
                    "used_steps": 2,
                    "won": True,
                    "tokens_prompt": 60,
                    "tokens_response": 40,
                    "tokens_total": 100,
                }
            ]
        },
    )
    trace_path = run_dir / "traces_pool" / "raw_traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "task_type": "clean",
                "task": "put a clean mug on the shelf",
                "steps": [
                    {
                        "step": 1,
                        "observation": "You are in the kitchen.",
                        "action": "go to sinkbasin 1",
                        "reward": 0.0,
                        "valid_action": True,
                        "strict_valid_action": True,
                        "execution_source": "direct",
                    },
                    {
                        "step": 2,
                        "observation": "You arrive at sinkbasin 1.",
                        "action": "clean mug 1 with sinkbasin 1",
                        "reward": 1.0,
                        "valid_action": True,
                        "strict_valid_action": False,
                        "execution_source": "salvaged",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    timeline = MODULE.build_single_task_timeline(run_dir, task_index=1, speedup=100)

    assert timeline["estimated_actual_task_seconds"] == 100.0
    assert timeline["replay_task_seconds"] == 1.0
    assert timeline["replay_step_delay_seconds"] == 0.5
    assert [step["action"] for step in timeline["steps"]] == [
        "go to sinkbasin 1",
        "clean mug 1 with sinkbasin 1",
    ]

    MODULE.replay_single_task(
        timeline,
        label="COSKILL",
        skill_source_label="CoSkill tree",
        skill_source="tree.json",
        gpu_index=0,
        gpu_name="NVIDIA GeForce RTX 3090",
        no_wait=True,
    )

    output = capsys.readouterr().out
    assert "[PLAN] model=/models/qwen3-4b" in output
    assert "[SELECT] task=01/01 | task_type=clean" in output
    assert "[COSKILL][STEP 01/40] action=go to sinkbasin 1" in output
    assert "[COSKILL][STEP 02/40] action=clean mug 1 with sinkbasin 1" in output
    assert "[COSKILL][TASK TOTAL] steps=2/40 | SUCCESS" in output
    assert "valid=" not in output
    assert "strict=" not in output
    assert "source=" not in output
    assert "reward=" not in output
