"""CPU-only checks for progressive skill-tree RL state transitions."""

import json
import tempfile
import unittest
from pathlib import Path

from agent_system.memory.skills_only_memory import SkillsOnlyMemory


TREE = """# Plan
root rule
## Verify
child rule
"""


def make_memory():
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "skills.json"
    path.write_text(json.dumps({
        "general_skills": [],
        "task_specific_skills": {"demo": []},
        "common_mistakes": [],
        "skill_trees": {"demo": {"content": TREE, "version": 1}},
    }), encoding="utf-8")
    memory = SkillsOnlyMemory(str(path), enable_playbook=True)
    # Legacy records are upgraded through the normal cloud install API so the
    # per-node lifecycle table mirrors a real run.
    memory.update_playbook("demo", TREE, level="outline")
    return directory, memory


class TreeRLCurriculumTest(unittest.TestCase):
    def test_root_layer_is_elided_but_child_remains_after_probe(self):
        directory, memory = make_memory()
        self.addCleanup(directory.cleanup)

        memory.advance_tree_rl_curriculum(
            global_step=0, order="root", min_rl_updates=1,
            min_train_episodes=2, min_probe_episodes=2,
            train_success_threshold=0.5, probe_success_threshold=0.5,
        )
        state = memory.skills["skill_tree_rl"]["demo"]
        self.assertEqual(state["target_level"], 1)
        self.assertIn("# Plan", memory.get_playbook("demo"))

        memory.record_playbook_usage("demo", True)
        memory.record_playbook_usage("demo", True)
        events = memory.advance_tree_rl_curriculum(
            global_step=1, order="root", min_rl_updates=1,
            min_train_episodes=2, min_probe_episodes=2,
            train_success_threshold=0.5, probe_success_threshold=0.5,
        )
        self.assertEqual(events[0]["event"], "probe_started")
        probe_text = memory.get_playbook("demo")
        self.assertNotIn("# Plan", probe_text)
        self.assertIn("# Verify", probe_text)

        memory.record_playbook_usage("demo", True)
        memory.record_playbook_usage("demo", True)
        events = memory.advance_tree_rl_curriculum(
            global_step=2, order="root", min_rl_updates=1,
            min_train_episodes=2, min_probe_episodes=2,
            train_success_threshold=0.5, probe_success_threshold=0.5,
        )
        self.assertEqual(events[0]["event"], "layer_internalized")
        state = memory.skills["skill_tree_rl"]["demo"]
        self.assertEqual(state["target_level"], 2)
        self.assertIn("# Verify", memory.get_playbook("demo"))

    def test_leaf_order_starts_from_deepest_layer(self):
        directory, memory = make_memory()
        self.addCleanup(directory.cleanup)
        memory.advance_tree_rl_curriculum(
            global_step=0, order="leaf", min_rl_updates=1,
            min_train_episodes=1, min_probe_episodes=1,
        )
        state = memory.skills["skill_tree_rl"]["demo"]
        self.assertEqual(state["target_level"], 2)


if __name__ == "__main__":
    unittest.main()
