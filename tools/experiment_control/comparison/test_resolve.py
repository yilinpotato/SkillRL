import unittest
from resolve import resolve


class ComparisonTests(unittest.TestCase):
    def test_independent_groups(self):
        a = resolve('norl_primary', 'skill0', 'alfworld', 16384)
        b = resolve('all_rl', 'skill0', 'alfworld', 16384)
        self.assertNotEqual(a['output_namespace'], b['output_namespace'])
        self.assertEqual(a['hydra_overrides'], b['hydra_overrides'])

    def test_rl_shared_parameters(self):
        for benchmark in ('alfworld', 'webshop', 'searchqa_live', 'scienceworld'):
            a = resolve('all_rl', 'skill0', benchmark, 16384)
            b = resolve('all_rl', 'skillrl', benchmark, 16384)
            self.assertEqual(a['rl'], b['rl'])
            self.assertEqual(a['common'], b['common'])
            self.assertEqual(a['environment'], b['environment'])

    def test_norl_has_no_optimizer(self):
        plan = resolve('norl_primary', 'coskill_norl', 'searchqa_live', 16384)
        self.assertIsNone(plan['rl'])
        self.assertNotIn('hydra_overrides', plan)
        self.assertFalse(plan['blockers'])

    def test_science_sft_not_silently_assumed(self):
        self.assertFalse(resolve('all_rl', 'coskill_tree_rl', 'searchqa_live', 16384)['blockers'])
        self.assertTrue(resolve('all_rl', 'coskill_tree_rl', 'scienceworld', 16384)['blockers'])

    def test_budget_and_environment_contracts(self):
        for bench, steps, history in [('alfworld', 40, 8), ('webshop', 15, 8), ('searchqa_live', 15, 0), ('scienceworld', 100, 0)]:
            p = resolve('all_rl', 'skill0', bench, 16384)
            c = p['common']
            self.assertEqual(c['training_episodes'], c['training_updates'] * c['tasks_per_update'] * c['rollouts_per_task'])
            self.assertEqual(p['environment']['max_steps'], steps)
            self.assertEqual(p['environment']['history_length'], history)
            self.assertFalse(p['training_launched'])

    def test_wrong_group_rejected(self):
        with self.assertRaises(ValueError):
            resolve('all_rl', 'coskill_norl', 'alfworld', 16384)


if __name__ == '__main__':
    unittest.main()
