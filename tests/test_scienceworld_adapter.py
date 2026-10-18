import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_system.scienceworld_adapter import ScienceSkillLib, action_examples, build_manifest, parse_action, run_episode
from examples.playbook_evolve.run_scienceworld import parse_args, run, summarize
from examples.playbook_evolve.run_searchqa_live import normalize_chat_message


class FakeEnvironment:
    def __init__(self, score=50):
        self.score = score
        self.actions = []
        self.loads = []

    def get_task_names(self):
        return ['boil', 'melt']

    def load(self, task, variation, simplifications, generateGoldPath):
        assert generateGoldPath is False
        self.loads.append((task, variation))

    def get_variations_train(self):
        return [0, 1]

    def get_variations_dev(self):
        return [2]

    def get_variations_test(self):
        return [3]

    def info(self, score):
        return {'score': score, 'look': 'Room', 'inv': 'Empty', 'valid': ['look around']}

    def reset(self):
        return 'Room', self.info(0)

    def get_task_description(self):
        return 'Complete the experiment'

    def step(self, action):
        self.actions.append(action)
        return 'Observation', self.score, True, self.info(self.score)


class Model:
    response = '<think>Observe first</think><action>look around</action>'

    def __init__(self, *args):
        pass

    def generate(self, messages):
        return self.response, {'prompt': 10, 'response': 5, 'usage_missing_calls': 0}


class ScienceWorldTests(unittest.TestCase):
    def test_native_qwen_prefilled_thinking_tag(self):
        raw = 'Inspect the room.</think>\n<action>look around</action>'
        self.assertEqual(parse_action(normalize_chat_message({'content': raw})), 'look around')
        self.assertEqual(normalize_chat_message({'content': Model.response}), Model.response)
        self.assertIsNone(parse_action(normalize_chat_message({'content': 'Unfinished reasoning'})))
        self.assertEqual(parse_action(normalize_chat_message({
            'reasoning_content': 'Inspect the room.', 'content': '<action>look around</action>',
        })), 'look around')

    def test_action_examples_cover_verbs_without_inventing_actions(self):
        actions = [f'connect a to wire {i}' for i in range(100)] + ['open door', 'look around', 'go to kitchen']
        selected = action_examples(actions, 4)
        self.assertEqual(selected, ['connect a to wire 0', 'open door', 'look around', 'go to kitchen'])
        self.assertTrue(set(selected) <= set(actions))

    def make_lib(self, path):
        path.write_text(json.dumps({'task_specific_skills': {'boil': [], 'melt': []}}))
        return ScienceSkillLib(str(path))

    def test_official_split_and_reproducibility(self):
        env = FakeEnvironment()
        a = build_manifest(env, ['boil', 'melt'], 1, 0)
        self.assertEqual(a, build_manifest(env, ['boil', 'melt'], 1, 0))
        self.assertEqual(len(a['train']), 2)
        self.assertTrue(all(row['variation'] == 3 for row in a['test']))

    def test_overlap_rejected(self):
        env = FakeEnvironment()
        env.get_variations_dev = lambda: [0]
        with self.assertRaises(ValueError):
            build_manifest(env, ['boil'], 1, 0)

    def test_action_contract(self):
        self.assertEqual(parse_action(Model.response), 'look around')
        self.assertIsNone(parse_action(Model.response + Model.response))
        self.assertIsNone(parse_action('<action>look around</action>'))

    def test_partial_score_is_not_success_and_type_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self.make_lib(Path(tmp) / 'skills.json')
            result = run_episode(FakeEnvironment(50), {'task': 'melt', 'variation': 0}, Model(), lib, 5, 2, '')
            self.assertEqual(result['outcome'], 'failure')
            self.assertEqual(result['task_score'], 0.5)
            self.assertEqual(lib.active_task, 'melt')
            self.assertEqual(result['tokens']['prompt'], 10)

    def test_success_and_terminal_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self.make_lib(Path(tmp) / 'skills.json')
            for score, expected in [(100, 'success'), (-100, 'failure')]:
                result = run_episode(FakeEnvironment(score), {'task': 'boil', 'variation': 0}, Model(), lib, 5, 2, '')
                self.assertEqual(result['outcome'], expected)
                self.assertEqual(len(result['steps']), 1)
                self.assertTrue(result['environment_done'])

    def test_action_quoted_in_reasoning_is_not_executed(self):
        from examples.playbook_evolve.run_searchqa_live import normalize_chat_message
        raw = 'Consider <action>wait</action>. </think>\n<action>look around</action>'
        self.assertEqual(parse_action(normalize_chat_message({'content': raw})), 'look around')
        self.assertIsNone(parse_action('<think>plan</think><action>wait</action><action>look around</action>'))

    def test_invalid_format_consumes_budget_without_environment_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self.make_lib(Path(tmp) / 'skills.json')
            model = Model()
            model.response = 'invalid'
            env = FakeEnvironment()
            result = run_episode(env, {'task': 'boil', 'variation': 0}, model, lib, 3, 2, '')
            self.assertEqual(env.actions, [])
            self.assertEqual(len(result['steps']), 3)
            self.assertTrue(result['decision_limit_reached'])
            self.assertEqual(summarize([result], 'train', 1, 0)['episode/format_valid_rate'], 0)

    def test_evaluation_does_not_learn(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--output-dir', str(Path(tmp) / 'out'), '--mode', 'eval', '--per-task', '1'])
            with patch('examples.playbook_evolve.run_scienceworld.ChatModel', Model), \
                    patch('examples.playbook_evolve.run_scienceworld.TracesPool.add_trace') as add, \
                    patch('examples.playbook_evolve.run_scienceworld.CoSkillCloudLoop.maybe_update') as update:
                run(args, FakeEnvironment(100), '1.2.3')
                add.assert_not_called()
                update.assert_not_called()
            metric = json.loads((Path(tmp) / 'out/group_metrics.jsonl').read_text())
            self.assertEqual(metric['episode/success_rate'], 1)
            self.assertEqual(metric['split'], 'test')

    def test_env_smoke_has_no_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--output-dir', str(Path(tmp) / 'out'), '--env-smoke', '--per-task', '1'])
            with patch('examples.playbook_evolve.run_scienceworld.ChatModel') as model:
                run(args, FakeEnvironment(), '1.2.3')
                model.assert_not_called()
            self.assertEqual(len((Path(tmp) / 'out/env_smoke.jsonl').read_text().splitlines()), 6)

    def test_only_training_enters_learning_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--output-dir', str(Path(tmp) / 'out'), '--per-task', '1', '--groups', '1'])
            with patch('examples.playbook_evolve.run_scienceworld.ChatModel', Model), \
                    patch('examples.playbook_evolve.run_scienceworld.CoSkillCloudLoop._get_analyzer', return_value=object()), \
                    patch('examples.playbook_evolve.run_scienceworld.TracesPool.add_trace') as add, \
                    patch('examples.playbook_evolve.run_scienceworld.CoSkillCloudLoop.maybe_update', return_value=False) as update:
                run(args, FakeEnvironment(), '1.2.3')
                self.assertEqual(add.call_count, 2)
                self.assertEqual(update.call_count, 1)
                self.assertTrue(all(call.args[0]['variation'] in (0, 1) for call in add.call_args_list))
            self.assertEqual(len((Path(tmp) / 'out/validation_metrics.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
