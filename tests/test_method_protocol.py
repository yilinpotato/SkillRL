import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_system.memory import HierarchicalSkillLib
from agent_system.method_protocol import (
    action_metrics,
    fit_history,
    history_messages,
    project_action,
    skill_fingerprint,
    skill_options,
)
from agent_system.scienceworld_adapter import ScienceSkillLib
from agent_system.scienceworld_adapter import run_episode as science_episode
from agent_system.searchqa_live import run_episode as search_episode
from examples.playbook_evolve.run_scienceworld import parse_args, run
from examples.playbook_evolve.run_searchqa_live import ChatModel, parser


class MethodTests(unittest.TestCase):
    def test_shared_defaults_and_lifecycle(self):
        science = parse_args(['--output-dir', 'unused'])
        search = parser().parse_args(['--output-dir', 'unused', '--data', 'unused'])
        for key in ('temperature', 'eval_temperature', 'eval_seed', 'history_length',
                    'max_tokens', 'max_prompt_tokens', 'min_calls', 'top_k', 'task_specific_top_k'):
            self.assertEqual(getattr(science, key), getattr(search, key))
        self.assertEqual(science.min_calls, 10)
        self.assertEqual((science.temperature, science.eval_temperature), (1.0, 0.4))
        self.assertEqual((science.max_steps, search.max_steps), (100, 15))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'skills.json'
            path.write_text(json.dumps({'task_specific_skills': {'searchqa': []}}))
            lib = HierarchicalSkillLib(str(path), **skill_options(search))
            self.assertEqual(lib.min_calls, 10)

    def test_action_recovery_never_executes_reasoning_or_ambiguity(self):
        result = project_action('<think>Consider <action>wrong</action></think>Explanation <action>look around</action>')
        self.assertEqual(result['parsed_action'], 'look around')
        self.assertEqual(result['execution_source'], 'recovered')
        self.assertFalse(result['format_valid'])
        self.assertEqual(project_action('<action>wait</action>')['parsed_action'], 'wait')
        for raw in ['<think>unfinished <action>wrong</action>',
                    '<think>x</think><action>a</action><action>b</action>',
                    '<think>x</think><action>partial',
                    '<think>x</think><action>a</action></action>']:
            self.assertIsNone(project_action(raw)['parsed_action'])

    def test_history_keeps_eight_pairs_and_current_state(self):
        rows = history_messages('rules', 'task', [(f'obs{i}', f'act{i}') for i in range(12)], 'current', 8)
        self.assertEqual(len(rows), 19)
        self.assertEqual(rows[2]['content'], 'obs4')
        self.assertEqual(rows[-1]['content'], 'current')
        self.assertNotIn('<think>', json.dumps(rows))
        missing = history_messages('rules', 'task', [('obs', None)], 'invalid feedback', 8)
        self.assertEqual(missing[3]['content'], '[No action executed]')

    def test_budget_drops_only_whole_oldest_pairs(self):
        rows = history_messages('rules', 'task', [('o1', 'a1'), ('o2', 'a2')], 'current', 8)
        fitted, dropped, count = fit_history(rows, len, 5)
        self.assertEqual((dropped, count), (1, 5))
        self.assertEqual(fitted[2]['content'], 'o2')
        self.assertEqual(fitted[-1]['content'], 'current')
        self.assertEqual(len(rows), 7)
        with self.assertRaises(ValueError):
            fit_history(rows, len, 2)

    def test_transport_sends_phase_sampling_and_actual_usage(self):
        class Tokenizer:
            def apply_chat_template(self, rows, **kwargs):
                return list(range(len(rows)))
        model = ChatModel('model', 'http://localhost/v1', 4096, 1000)
        model._tokenizer = Tokenizer()
        model.temperature = 0.4
        model.max_prompt_tokens = 3
        rows = history_messages('rules', 'task', [('old', 'act')], 'current', 8)
        reply = {'choices': [{'message': {'content': '<action>wait</action>'}}],
                 'usage': {'prompt_tokens': 20, 'completion_tokens': 7}}
        with patch('examples.playbook_evolve.run_searchqa_live.post_json', return_value=reply) as post:
            _, usage = model.generate(rows)
        request = post.call_args.args[1]
        self.assertEqual((request['temperature'], request['seed']), (0.4, 1000))
        self.assertEqual(len(request['messages']), 3)
        self.assertEqual(usage['prompt'], 20)
        self.assertEqual(usage['history_pairs_dropped'], 1)

    def test_both_adapters_remove_thoughts_and_keep_actual_actions(self):
        class Model:
            def __init__(self, commands):
                self.commands = iter(commands)
                self.messages = []
            def generate(self, messages):
                self.messages.append(messages)
                return '<think>SECRET_THOUGHT</think>Explanation <action>' + next(self.commands) + '</action>', {'prompt': 5, 'response': 3}
        class Search:
            def search(self, query):
                return [{'title': 'source', 'url': 'https://example.org', 'content': 'EVIDENCE'}]
        class Environment:
            def load(self, *args, **kwargs):
                pass
            def reset(self):
                return 'Room', {'score': 0, 'valid': ['look around']}
            def get_task_description(self):
                return 'TASK'
            def step(self, command):
                return 'NEW_STATE', 0, False, {'score': 0, 'valid': ['look around']}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'skills.json'
            path.write_text(json.dumps({'task_specific_skills': {'searchqa': []}}))
            lib = HierarchicalSkillLib(str(path))
            search_model = Model(['search[query]', 'answer[A]'])
            before = skill_fingerprint(lib)
            result = search_episode({'id': '1', 'question': 'Q', 'answers': ['A']}, search_model, Search(), lib)
            self.assertEqual(skill_fingerprint(lib), before)
            self.assertEqual(result['episode_reward'], 1)
            self.assertNotIn('SECRET_THOUGHT', json.dumps(search_model.messages[1]))
            self.assertIn('EVIDENCE', json.dumps(search_model.messages[1]))
            self.assertEqual(result['steps'][0]['executed_action'], 'search[query]')
            science_model = Model(['look around', 'look around'])
            result = science_episode(Environment(), {'task': 'boil', 'variation': 0}, science_model,
                                     ScienceSkillLib(str(path)), 2, 8, '')
            self.assertNotIn('SECRET_THOUGHT', json.dumps(science_model.messages[1]))
            self.assertIn('NEW_STATE', json.dumps(science_model.messages[1]))
            self.assertEqual(result['termination_reason'], 'decision_limit')
            metrics = action_metrics([result])
            self.assertEqual(metrics['episode/format_valid_rate'], 0)
            self.assertEqual(metrics['episode/executed_action_ratio'], 1)

    def test_science_progress_uses_episode_count_and_validation_is_frozen(self):
        from tests.test_scienceworld_adapter import FakeEnvironment, Model
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--output-dir', tmp, '--groups', '1'])
            with patch('examples.playbook_evolve.run_scienceworld.ChatModel', Model), \
                    patch('examples.playbook_evolve.run_scienceworld.CoSkillCloudLoop._get_analyzer', return_value=object()), \
                    patch('examples.playbook_evolve.run_scienceworld.CoSkillCloudLoop.maybe_update', return_value=False) as update:
                run(args, FakeEnvironment(), '1.2.3')
            self.assertEqual(update.call_args.args[2], 4)
            val = [json.loads(line) for line in (Path(tmp) / 'validation_metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['skills/fingerprint_before'] == r['skills/fingerprint_after'] for r in val))
            self.assertTrue(all(r['sampling/temperature'] == 0.4 for r in val))
            train = json.loads((Path(tmp) / 'group_metrics.jsonl').read_text())
            self.assertEqual(train['sampling/temperature'], 1.0)
            self.assertEqual(train['episode/count_cumulative'], 4)


if __name__ == '__main__':
    unittest.main()
