import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_system.memory import HierarchicalSkillLib, TracesPool
from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.searchqa_live import (
    SearchQAEnvironment,
    check_disjoint,
    load_examples,
    run_episode,
    score_answer,
)
from examples.playbook_evolve.run_searchqa_live import main


class Search:
    calls = 0
    credits = 0
    missing_usage = 0

    def __init__(self, *args):
        pass

    def search(self, query):
        self.calls += 1
        return [{'title': 'Evidence', 'url': 'https://example.org/evidence', 'content': 'Evidence text'}]

    def open(self, url):
        self.calls += 1
        return 'More evidence'


def action(value):
    return f'<think>Reasoning</think><action>{value}</action>'


class SearchQATests(unittest.TestCase):
    def test_scoring(self):
        self.assertEqual(score_answer('The Eiffel Tower.', ['Eiffel Tower'])['exact_match'], 1)
        self.assertAlmostEqual(score_answer('New York', ['New York City'])['f1'], 0.8)
        self.assertEqual(score_answer('', ['Paris'])['f1'], 0)

    def test_environment_budget_and_action_validation(self):
        env = SearchQAEnvironment('Question', Search(), max_steps=5, max_searches=1)
        self.assertFalse(env.step(action('open[1]'))[1])
        self.assertTrue(env.step(action('search[first]'))[1])
        self.assertFalse(env.step(action('search[second]'))[1])
        self.assertTrue(env.step(action('open[1]'))[1])
        self.assertTrue(env.step(action('answer[Paris]'))[1])
        self.assertTrue(env.done)
        self.assertEqual(env.answer, 'Paris')
        with self.assertRaises(RuntimeError):
            env.step(action('search[again]'))

    def test_multiple_actions_and_step_limit(self):
        env = SearchQAEnvironment('Question', Search(), max_steps=1)
        self.assertFalse(env.step(action('search[x]') + action('answer[y]'))[1])
        self.assertTrue(env.done)
        self.assertEqual(env.search.calls, 0)

    def test_no_reference_or_stored_context_leak_and_pool_compatibility(self):
        class Model:
            messages = []

            def generate(self, messages):
                self.messages.append(messages)
                return action('answer[Paris]'), {'prompt': 10, 'response': 5}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'skills.json'
            path.write_text(json.dumps({'task_specific_skills': {'searchqa': []}}))
            lib, model = HierarchicalSkillLib(str(path)), Model()
            row = {'id': '1', 'question': 'What city?', 'answers': ['SECRET_REFERENCE'],
                   'context': 'STORED_CONTEXT'}
            result = run_episode(row, model, Search(), lib)
            serialized = json.dumps(model.messages) + json.dumps(result)
            self.assertNotIn('SECRET_REFERENCE', serialized)
            self.assertNotIn('STORED_CONTEXT', serialized)
            pool = TracesPool()
            pool.add_trace(result)
            batch = pool.export_batch(trigger_reason='test')
            self.assertEqual(batch['stats']['n_failure'], 1)

    def test_split_overlap_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            check_disjoint([{'id': 'a', 'question': 'Same?'}], [{'id': 'b', 'question': 'Same?'}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            row = json.dumps({'id': 'a', 'question': 'Q?', 'answers': ['A']})
            path.write_text(row + '\n' + row)
            with self.assertRaises(ValueError):
                load_examples(path)

    def test_materializer_uses_only_available_source_splits(self):
        from tools.prepare_searchqa import main as prepare

        def dataset(_name, split, **kwargs):
            self.assertIn(split, ('train', 'validation'))
            return ({'key': f'{split}-{i}', 'question': f'{split} question {i}',
                     'answers': ['Answer'], 'context': 'DROP THIS CONTEXT'} for i in range(1000))

        class Api:
            def dataset_info(self, *args, **kwargs):
                return types.SimpleNamespace(sha='fixed-revision')

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict('sys.modules', {
                    'datasets': types.SimpleNamespace(load_dataset=dataset),
                    'huggingface_hub': types.SimpleNamespace(HfApi=Api),
                }), patch('sys.argv', ['prepare', '--output-dir', tmp, '--limit', '2']):
            prepare()
            for split in ('train', 'validation', 'test'):
                content = (Path(tmp) / f'{split}.jsonl').read_text()
                self.assertEqual(len(content.splitlines()), 2)
                self.assertNotIn('DROP THIS CONTEXT', content)
            check_disjoint(load_examples(Path(tmp) / 'train.jsonl'), load_examples(Path(tmp) / 'validation.jsonl'))

    def test_eval_never_updates_skill_or_pool(self):
        class Model:
            def __init__(self, *args):
                pass

            def generate(self, messages):
                return action('answer[Paris]'), {'prompt': 10, 'response': 5}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            path.write_text(json.dumps({'id': 'test1', 'question': 'Capital of France?', 'answers': ['Paris']}))
            output = Path(tmp) / 'output'
            with patch.dict('os.environ', {'TAVILY_API_KEY': 'test'}), \
                    patch('examples.playbook_evolve.run_searchqa_live.ChatModel', Model), \
                    patch('examples.playbook_evolve.run_searchqa_live.TavilySearch', Search), \
                    patch('examples.playbook_evolve.run_searchqa_live.CoSkillCloudLoop.maybe_update') as update, \
                    patch('examples.playbook_evolve.run_searchqa_live.TracesPool.add_trace') as add:
                main(['--data', str(path), '--mode', 'eval', '--output-dir', str(output)])
                update.assert_not_called()
                add.assert_not_called()
            metric = json.loads((output / 'group_metrics.jsonl').read_text())
            self.assertEqual(metric['qa/exact_match'], 1)
            self.assertEqual(metric['tokens/small_model/total'], 15)
            self.assertTrue((output / 'completed.json').exists())

    def test_training_reuses_cloud_tree_and_separates_validation(self):
        class Model:
            def __init__(self, *args):
                pass

            def generate(self, messages):
                return action('answer[Paris]'), {'prompt': 10, 'response': 5}

        class Analyzer:
            _tree_depth = staticmethod(CloudAnalyzer._tree_depth)

            def contrastive_distill(self, batch, current):
                return [{'skill_id': 'dyn_001', 'title': 'Verify evidence',
                         'principle': 'Cross-check named entities', 'scope': 'general'}]

            def diagnose_failures(self, batch):
                return {}

            def evolve_playbook(self, **kwargs):
                return {'action': 'rewrite', 'skill_tree': '# Verify evidence\nCheck entity identity.',
                        'depth_valid': True, 'actual_depth': 1}

        with tempfile.TemporaryDirectory() as tmp:
            train, val = Path(tmp) / 'train.jsonl', Path(tmp) / 'val.jsonl'
            train.write_text('\n'.join(json.dumps({
                'id': f'train{i}', 'question': f'Training question {i}?', 'answers': ['Other'],
            }) for i in range(6)))
            val.write_text(json.dumps({'id': 'val', 'question': 'Validation question?', 'answers': ['Paris']}))
            output = Path(tmp) / 'out'
            with patch.dict('os.environ', {'TAVILY_API_KEY': 'test'}), \
                    patch('examples.playbook_evolve.run_searchqa_live.ChatModel', Model), \
                    patch('examples.playbook_evolve.run_searchqa_live.TavilySearch', Search), \
                    patch('agent_system.memory.coskill_loop.CoSkillCloudLoop._get_analyzer', return_value=Analyzer()):
                main(['--data', str(train), '--validation-data', str(val), '--output-dir', str(output),
                      '--groups', '1', '--batch-size', '6', '--min-samples', '1'])
            skills = json.loads((output / 'skills_latest.json').read_text())
            self.assertEqual(skills['general_skills'][0]['skill_id'], 'dyn_001')
            self.assertIn('searchqa', skills['skill_trees'])
            traces = (output / 'traces_pool/raw_traces.jsonl').read_text()
            self.assertNotIn('Validation question', traces)
            self.assertEqual(len((output / 'validation_metrics.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
