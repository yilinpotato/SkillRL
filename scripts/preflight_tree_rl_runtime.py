#!/usr/bin/env python3
# ruff: noqa: E402
"""Offline compatibility preflight for the CoSkill Tree-RL runtime.

This check intentionally performs no cloud or GPU work.  It protects the
runtime overlay from pairing a new trainer/call site with stale memory classes
whose constructors or methods do not accept the expected protocol fields.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

# Running a file under ``scripts/`` sets sys.path[0] to that directory.  An
# editable installation of a sibling SkillRL checkout could otherwise shadow
# this image's /workspace/CoSkill package and make the preflight validate the
# wrong implementation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.memory import (
    CloudAnalyzer,
    CoSkillCloudLoop,
    HierarchicalSkillLib,
    SkillsOnlyMemory,
    TracesPool,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _require_keywords(target, names: Iterable[str]) -> None:
    """Raise with a useful error when ``target`` rejects required keywords."""

    signature = inspect.signature(target)
    parameters = signature.parameters
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    missing = sorted(
        name for name in names if name not in parameters and not accepts_extra
    )
    if missing:
        qualified_name = getattr(target, "__qualname__", repr(target))
        raise TypeError(
            f"{qualified_name}{signature} is missing Tree-RL keywords: {missing}"
        )


def _check_signatures() -> None:
    traces_pool_source = Path(inspect.getsourcefile(TracesPool) or "").resolve()
    trainer_source = Path(inspect.getsourcefile(RayPPOTrainer) or "").resolve()
    expected_memory_root = (PROJECT_ROOT / "agent_system" / "memory").resolve()
    expected_trainer = (
        PROJECT_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).resolve()
    if expected_memory_root not in traces_pool_source.parents:
        raise ImportError(
            f"TracesPool was imported from {traces_pool_source}, "
            f"expected under {expected_memory_root}"
        )
    if trainer_source != expected_trainer:
        raise ImportError(
            f"RayPPOTrainer was imported from {trainer_source}, "
            f"expected {expected_trainer}"
        )

    _require_keywords(
        TracesPool,
        (
            "capacity_watermark",
            "perf_watermark",
            "min_samples",
            "loop_threshold",
            "output_dir",
            "enable_loop_filter",
            "enable_obs_delta",
            "enable_prefix_tree",
            "enable_consensus_prefix",
            "cloud_evidence_mode",
        ),
    )
    _require_keywords(
        CoSkillCloudLoop,
        (
            "output_dir",
            "enable_coskill",
            "enable_playbook_evolve",
            "enable_failure_analysis",
            "max_new_skills",
            "playbook_evolve_min_samples",
            "coskill_debug",
            "environment_name",
        ),
    )
    _require_keywords(
        CloudAnalyzer,
        ("max_new_skills_per_update", "output_dir", "environment_name"),
    )
    _require_keywords(
        CloudAnalyzer.evolve_playbook,
        (
            "task_type",
            "current_playbook",
            "success_traces",
            "failure_traces",
            "diagnoses",
            "history",
            "target_depth",
            "repair_candidate",
            "repair_feedback",
            "tree_evidence",
            "max_tree_nodes",
            "max_tree_chars",
        ),
    )
    _require_keywords(
        HierarchicalSkillLib,
        (
            "skills_json_path",
            "retrieval_mode",
            "embedding_model_path",
            "task_specific_top_k",
            "enable_hierarchy",
            "stable_cycles_l1",
            "stable_cycles_l2",
            "success_l1",
            "demote_threshold",
            "min_calls",
            "enable_playbook",
        ),
    )
    _require_keywords(
        SkillsOnlyMemory.update_playbook,
        ("task_type", "content", "level", "meta"),
    )
    _require_keywords(
        SkillsOnlyMemory.advance_tree_rl_curriculum,
        (
            "global_step",
            "order",
            "min_rl_updates",
            "min_train_episodes",
            "train_success_threshold",
            "min_probe_episodes",
            "probe_success_threshold",
        ),
    )


def _check_tree_codec() -> None:
    pool = TracesPool(
        min_samples=1,
        enable_loop_filter=True,
        enable_obs_delta=True,
        enable_prefix_tree=True,
        enable_consensus_prefix=True,
        cloud_evidence_mode="tree_only",
    )
    pool.add_trace(
        {
            "traj_uid": "runtime-preflight-1",
            "task": "put a mug in the cabinet",
            "task_type": "pick_and_place",
            "outcome": "success",
            "episode_reward": 1.0,
            "steps": [
                {
                    "step": 0,
                    "observation": "You are in a room.",
                    "action": "go to cabinet 1",
                    "reward": 1.0,
                }
            ],
        }
    )
    exported = pool.export_batch(trigger_reason="runtime_preflight")
    projected = pool.project_cloud_batch(exported)
    if projected.get("cloud_projection", {}).get("mode") != "tree_only":
        raise AssertionError("tree-only cloud projection was not selected")
    if not projected.get("tree_evidence", {}).get("records"):
        raise AssertionError("tree-only cloud projection contains no codec records")
    if any("steps" in trace for trace in projected.get("success_samples", [])):
        raise AssertionError("tree-only cloud projection leaked flat trajectory steps")


class _FakeTokenizer:
    def batch_decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return list(values)


def _check_trainer_ingest_contract() -> None:
    """Exercise the exact call path that previously failed after rollout."""

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": "/tmp/coskill-runtime-preflight"},
            "env": {
                "traces_pool": {
                    "capacity_watermark": 50_000,
                    "perf_watermark": 0.6,
                    "min_samples": 16,
                    "loop_threshold": 3,
                    "enable_loop_filter": True,
                    "enable_obs_delta": True,
                    "enable_prefix_tree": True,
                    "enable_consensus_prefix": True,
                    "cloud_evidence_mode": "tree_only",
                },
                "skills_only_memory": {
                    "top_k": 6,
                    "enable_tree_rl_internalize": True,
                },
            },
        }
    )
    trainer.tokenizer = _FakeTokenizer()
    trainer.global_steps = 1
    trainer.envs = SimpleNamespace(retrieval_memory=None)
    batch = SimpleNamespace(
        batch={
            "prompts": [
                "Your task is to: put a mug in the cabinet\n"
                "Step 0\nObservation: You are in a room."
            ],
            "responses": [
                "<think>navigate</think><action>go to cabinet 1</action>"
            ],
            "token_level_scores": torch.tensor([[0.0]]),
        },
        non_tensor_batch={
            "traj_uid": np.array(["runtime-preflight-1"], dtype=object),
            "episode_rewards": np.array([0.0]),
        },
    )
    trainer._coskill_ingest_batch_to_pool(batch)
    if trainer.traces_pool.cloud_evidence_mode != "tree_only":
        raise AssertionError("trainer did not configure tree-only evidence")
    if trainer.traces_pool.stats().get("pending_added") != 1:
        raise AssertionError("trainer did not ingest exactly one preflight trajectory")


def main() -> None:
    _check_signatures()
    _check_tree_codec()
    _check_trainer_ingest_contract()
    print("Tree-RL runtime compatibility preflight: OK")


if __name__ == "__main__":
    main()
