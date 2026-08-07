#!/usr/bin/env python3

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> None:
    from agent_system.task_taxonomy import (
        ALFWORLD_TASK_TYPES,
        WEBSHOP_TASK_TYPES,
        classify_task,
    )

    alfworld = json.loads(
        (ROOT / "memory_data/alfworld/initial_skills.json").read_text()
    )
    webshop = json.loads(
        (ROOT / "memory_data/webshop/initial_skills.json").read_text()
    )
    assert tuple(alfworld["task_specific_skills"]) == ALFWORLD_TASK_TYPES
    assert tuple(webshop["task_specific_skills"]) == WEBSHOP_TASK_TYPES

    cases = (
        ("webshop", "Find a laptop with 16 GB RAM.", "electronics"),
        ("webshop", "Find a women's top under $20.", "apparel"),
        ("webshop", "Find a cotton tablecloth.", "home_decor"),
        ("webshop", "Find waterproof hiking shoes.", "footwear"),
        ("webshop", "Find a leather wallet.", "accessories"),
        ("webshop", "Find vitamin C serum.", "beauty_health"),
        ("alfworld", "put two apples in the fridge", "pick_two_obj_and_place"),
        ("alfworld", "look at the mug under the desklamp", "look_at_obj_in_light"),
        ("alfworld", "put a clean plate on the table", "clean"),
    )
    for benchmark, text, expected in cases:
        assert classify_task(benchmark, text) == expected

    contaminated_prompt = """
Your task is to: Find a laptop with 16 GB RAM.

## Retrieved Relevant Experience

Search for shirts, dresses, and tops.

## Current Progress
"""
    assert classify_task("webshop", contaminated_prompt) == "electronics"

    source_paths = (
        ROOT / "examples/playbook_evolve/run_playbook_evolve.py",
        ROOT / "examples/playbook_evolve/run_webshop_evolve.py",
        ROOT / "verl/trainer/ppo/metric_utils.py",
        ROOT / "verl/trainer/ppo/ray_trainer.py",
        ROOT / "verl/utils/tracking.py",
    )
    source = "\n".join(path.read_text() for path in source_paths)
    assert source.count("group_metrics.jsonl") >= 4
    assert "episode_metrics.jsonl" not in source
    assert "thinking_samples.jsonl" not in source
    assert '"metrics.jsonl"' not in source
    assert "validation_metrics.jsonl" not in source
    assert '"training/group"' not in source
    assert '"training/global_step"' not in source
    assert '"global_episode_end"' not in source
    assert '"episode/count_cumulative"' not in source
    assert "relaxed_valid_action_ratio" not in source
    assert source.count('"comparison/schema_version": 3') >= 2
    assert "'comparison/schema_version': 3" in source
    print("metrics schema check passed")


if __name__ == "__main__":
    main()
