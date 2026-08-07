import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "terminal_compare"
    / "replay_alfworld_serial.py"
)
SPEC = importlib.util.spec_from_file_location("replay_alfworld_serial", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_timeline_preserves_total_time_and_token_weighted_relative_time(tmp_path):
    run_dir = tmp_path / "coskill"
    _write_json(tmp_path / "comparison_manifest.json", {"max_environment_steps": 40})
    _write_json(
        run_dir / "terminal_run_summary.json",
        {"wall_time_seconds": 400.0},
    )
    _write_json(
        run_dir / "summary.json",
        {
            "per_game": [
                {
                    "detected_type": "clean",
                    "used_steps": 12,
                    "won": True,
                    "tokens_prompt": 200,
                    "tokens_response": 100,
                    "tokens_total": 300,
                },
                {
                    "detected_type": "heat",
                    "used_steps": 20,
                    "won": False,
                    "tokens_prompt": 60,
                    "tokens_response": 40,
                    "tokens_total": 100,
                },
            ]
        },
    )
    trace_path = run_dir / "traces_pool" / "raw_traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps({"task_type": "clean", "task": "put a clean mug on the shelf"})
        + "\n"
        + json.dumps({"task_type": "heat", "task": "heat the apple"})
        + "\n",
        encoding="utf-8",
    )

    timeline = MODULE.build_timeline(run_dir, speedup=100.0)

    assert timeline["timing_basis"] == "small_model_total_tokens"
    assert timeline["max_environment_steps"] == 40
    configuration = timeline["run_configuration"]
    assert configuration["split"] == "valid_unseen"
    assert configuration["tasks"] == 2
    assert configuration["tasks_per_type"] == 1
    assert configuration["seed"] == 0
    assert configuration["history_length"] == 8
    assert configuration["max_response_tokens"] == 4096
    assert configuration["retrieval_mode"] == "template"
    assert configuration["top_k"] == 6
    assert configuration["rl_weights_loaded"] is False
    assert configuration["cloud_updates_enabled"] is False
    assert configuration["batch_size"] == 1
    assert configuration["cuda_graph"] is True
    assert timeline["measured_total_wall_seconds"] == 400.0
    assert timeline["replay_total_seconds"] == 4.0
    assert timeline["rows"][0]["estimated_actual_seconds"] == 300.0
    assert timeline["rows"][0]["replay_delay_seconds"] == 3.0
    assert timeline["rows"][1]["estimated_actual_seconds"] == 100.0
    assert timeline["rows"][1]["replay_delay_seconds"] == 1.0


def test_replay_prints_serial_task_lines_and_measured_total_time(tmp_path, capsys):
    run_dir = tmp_path / "coskill"
    _write_json(tmp_path / "comparison_manifest.json", {"max_environment_steps": 40})
    _write_json(run_dir / "terminal_run_summary.json", {"wall_time_seconds": 400.0})
    _write_json(
        run_dir / "summary.json",
        {
            "per_game": [
                {
                    "detected_type": "clean",
                    "used_steps": 12,
                    "won": True,
                    "tokens_total": 300,
                }
            ]
        },
    )
    trace_path = run_dir / "traces_pool" / "raw_traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps({"task_type": "clean", "task": "put a clean mug on the shelf"})
        + "\n",
        encoding="utf-8",
    )
    timeline = MODULE.build_timeline(run_dir, speedup=100.0)

    MODULE.replay(
        timeline,
        label="COSKILL",
        skill_source_label="CoSkill tree",
        skill_source="tree.json",
        gpu_index=0,
        gpu_name="NVIDIA GeForce RTX 3090",
        no_wait=True,
    )

    output = capsys.readouterr().out
    assert "[PLAN] split=valid_unseen tasks=1 (1 per type) seed=0 max_steps=40" in output
    assert "[PLAN] CoSkill tree: tree.json" in output
    assert "temperature=" not in output
    assert "RL/LoRA=" not in output
    assert (
        "[RUN] COSKILL | GPU 0 NVIDIA GeForce RTX 3090 | 1 fixed tasks | "
        "max steps 40 | batch=1 | CUDA Graph=on"
    ) in output
    assert "[COSKILL][01/01]" in output
    assert "steps=12/40 | SUCCESS" in output
    assert "[COSKILL][TOTAL] success=1/1 (100.0%)" in output
    assert "total_time=00:06:40" in output
    assert "task_time=" not in output
