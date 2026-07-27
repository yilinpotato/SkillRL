from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.playbook_evolve.webshop_trace_compression_one_step_ablation import (
    DISTINCT_GOALS,
    MAX_ENVIRONMENT_STEPS,
    ROLLOUTS_PER_GOAL,
    TOTAL_ROLLOUTS,
    WEBSHOP_TASK_TYPES,
    _driver_cmd,
    _validate_shared_raw,
)


def _shared_raw():
    rows = []
    for goal in range(DISTINCT_GOALS):
        category = WEBSHOP_TASK_TYPES[goal % len(WEBSHOP_TASK_TYPES)]
        score = 1.0 if goal % 2 == 0 else 0.5
        for replica in range(ROLLOUTS_PER_GOAL):
            rows.append(
                {
                    "traj_uid": f"goal-{goal}-replica-{replica}",
                    "task": f"buy product for goal {goal}",
                    "task_type": category,
                    "outcome": "success" if score == 1.0 else "failure",
                    "episode_reward": 10.0 if score == 1.0 else 0.0,
                    "steps": [
                        {
                            "step": 1,
                            "observation": "search page",
                            "action": f"search[product model {goal}]",
                            "reward": 0.0,
                        },
                        {
                            "step": 2,
                            "observation": "product page",
                            "action": f"click[size {10 + goal}]",
                            "reward": 10.0 if score == 1.0 else 0.0,
                        },
                    ],
                    "meta": {
                        "environment": "WebShop",
                        "goal_index": 500 + goal,
                        "task_score": score,
                    },
                }
            )
    return rows


def test_shared_capture_matches_webshop_goal_replica_contract():
    stats = _validate_shared_raw(_shared_raw())
    assert stats["rollouts"] == TOTAL_ROLLOUTS
    assert stats["distinct_goals"] == DISTINCT_GOALS
    assert stats["replicas_per_goal"] == ROLLOUTS_PER_GOAL
    assert stats["max_environment_steps"] == MAX_ENVIRONMENT_STEPS
    assert stats["observed_steps_per_trace"]["total"] == TOTAL_ROLLOUTS * 2
    assert stats["capture_diagnostics"]["wins"] == TOTAL_ROLLOUTS // 2
    assert stats["capture_diagnostics"]["success_rate"] == 0.5
    assert stats["capture_diagnostics"]["mean_task_score"] == 0.75
    assert stats["actions"] == {
        "click": TOTAL_ROLLOUTS,
        "search": TOTAL_ROLLOUTS,
    }


def test_shared_capture_rejects_goal_replica_instruction_drift():
    rows = _shared_raw()
    rows[1]["task"] = "a different instruction"
    with pytest.raises(RuntimeError, match="changed instruction/category"):
        _validate_shared_raw(rows)


def test_shared_capture_rejects_non_webshop_actions():
    rows = _shared_raw()
    rows[0]["steps"][0]["action"] = "go to cabinet 1"
    with pytest.raises(RuntimeError, match="search/click action contract"):
        _validate_shared_raw(rows)


def test_capture_driver_disables_cloud_mutation_but_keeps_normal_skill_prompt():
    args = SimpleNamespace(
        model_path="/models/qwen",
        webshop_file_path=Path("/data/items.json"),
        webshop_attr_path=Path("/data/attrs.json"),
        seed=0,
        data_parallel_workers=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        vllm_max_num_seqs=0,
        vllm_enforce_eager=0,
        prompt_char_limit=24000,
        max_model_len=12288,
        max_tokens=4096,
        think_budget=3840,
        action_budget=256,
        temperature=1.0,
        gpu_mem_util=0.8,
        retrieval_mode="template",
        rollout_worker_gpus="0,1",
        driver_arg=[],
    )
    command = _driver_cmd(
        args,
        Path("/output/capture"),
        Path("/output/shared/initial_skills.json"),
    )

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--train_data_size") == "12"
    assert value("--group_size") == "6"
    assert value("--max_episodes") == "72"
    assert value("--max_steps") == "15"
    assert value("--enable_coskill") == "1"
    assert value("--enable_skill_tree") == "1"
    assert value("--enable_skill_tree_evolve") == "0"
    assert value("--enable_cloud_updates") == "0"
    assert value("--data_parallel_workers") == "2"
    assert value("--rollout_worker_gpus") == "0,1"
