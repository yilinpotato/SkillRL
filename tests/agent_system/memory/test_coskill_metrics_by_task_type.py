"""Per-subtask (ALFWorld task_type) breakdown of the CoSkill skill-tree /
cloud-token output metrics, shared by both the RL (ray_trainer) and no-RL
(run_playbook_evolve) callers via CoSkillCloudLoop.metrics().

evolve_playbook is one call per task_type (cleanly attributable);
contrastive_distill/diagnose_failures mix every task_type into a single call
each and must land in an honest "mixed" bucket instead of a fabricated split.
"""

import sys
from pathlib import Path

# pytest inserts tests/ ahead of the project root, where tests/agent_system
# would otherwise shadow the real agent_system package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.memory.coskill_loop import CoSkillCloudLoop


def _analyzer_with_history():
    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.update_history = []
    # 2 evolve_playbook calls for tt_a (one kept, one refined) and 1 for tt_b.
    analyzer.playbook_history = [
        {"task_type": "heat", "action": "rewrite", "level": "outline", "had_current": False},
        {"task_type": "heat", "action": "keep", "level": "outline", "had_current": True},
        {"task_type": "cool", "action": "refine", "level": "detailed", "had_current": True},
    ]
    analyzer.n_diagnose_calls = 2
    analyzer.n_evolve_calls = 3
    analyzer.total_prompt_tokens = 130
    analyzer.total_completion_tokens = 40
    analyzer.total_prompt_tokens_by_task_type = {"heat": 80, "cool": 30}
    analyzer.total_completion_tokens_by_task_type = {"heat": 25, "cool": 10}
    analyzer.total_prompt_tokens_mixed = 20
    analyzer.total_completion_tokens_mixed = 5
    return analyzer


def test_get_update_summary_groups_evolve_calls_and_updates_by_task_type():
    summary = _analyzer_with_history().get_update_summary()

    assert summary["evolve_calls_by_task_type"] == {"heat": 2, "cool": 1}
    # Only the non-"keep" calls count as an actual update.
    assert summary["skill_tree_updates_by_task_type"] == {"heat": 1, "cool": 1}
    assert summary["evolve_calls"] == 3
    assert summary["skill_tree_updates"] == 2

    # Cloud-token invariant: mixed + sum(by_task_type) == raw total (no
    # double-counting between the attributable and mixed buckets).
    by_tt_prompt_total = sum(summary["large_model_prompt_tokens_by_task_type"].values())
    by_tt_completion_total = sum(summary["large_model_completion_tokens_by_task_type"].values())
    assert by_tt_prompt_total + summary["large_model_prompt_tokens_mixed"] == summary["large_model_prompt_tokens"]
    assert (by_tt_completion_total + summary["large_model_completion_tokens_mixed"]
            == summary["large_model_completion_tokens"])


def test_coskill_loop_metrics_exposes_by_task_type_and_mixed_buckets(tmp_path):
    loop = CoSkillCloudLoop(str(tmp_path))
    loop.cloud_analyzer = _analyzer_with_history()

    m = loop.metrics(traces_pool=None, skill_lib=None)

    assert m["coskill/cloud/by_task_type/heat/large_model_prompt_tokens"] == 80
    assert m["coskill/cloud/by_task_type/heat/large_model_completion_tokens"] == 25
    assert m["coskill/cloud/by_task_type/heat/large_model_total_tokens"] == 105
    assert m["coskill/cloud/by_task_type/cool/large_model_total_tokens"] == 40
    assert m["coskill/skill_tree/by_task_type/heat/evolve_calls"] == 2
    assert m["coskill/skill_tree/by_task_type/heat/updates"] == 1
    assert m["coskill/skill_tree/by_task_type/cool/evolve_calls"] == 1
    assert m["coskill/skill_tree/by_task_type/cool/updates"] == 1

    assert m["coskill/cloud/mixed/large_model_prompt_tokens"] == 20
    assert m["coskill/cloud/mixed/large_model_completion_tokens"] == 5
    assert m["coskill/cloud/mixed/large_model_total_tokens"] == 25

    # Hard invariant at the metrics-dict level too: by_task_type + mixed must
    # reconcile exactly with the existing flat cumulative totals (unchanged).
    by_tt_total = sum(
        v for k, v in m.items()
        if k.startswith("coskill/cloud/by_task_type/") and k.endswith("/large_model_total_tokens")
    )
    assert by_tt_total + m["coskill/cloud/mixed/large_model_total_tokens"] == m["coskill/cloud/large_model_total_tokens"]


def test_coskill_loop_metrics_without_cloud_analyzer_omits_breakdown(tmp_path):
    loop = CoSkillCloudLoop(str(tmp_path))
    m = loop.metrics(traces_pool=None, skill_lib=None)
    assert not any(k.startswith("coskill/cloud/by_task_type/") for k in m)
    assert not any(k.startswith("coskill/skill_tree/by_task_type/") for k in m)
