"""Exercise the real generator method without loading CUDA/vLLM on CPU."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


def load_method():
    source = Path(__file__).resolve().parents[2] / 'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_generate_with_budget_forcing')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope[method.name]


class BudgetForcingPrefillTest(unittest.TestCase):
    def run_case(self, prompt, response, forced):
        calls = []
        def generate(**kw):
            calls.append(kw)
            text = response if len(calls) == 1 else 'search[test]</action>'
            ids = list(map(ord, text))
            return [NS(outputs=[NS(token_ids=ids, logprobs=[{i: NS(logprob=-0.5)} for i in ids])])]
        obj = NS(config=NS(response_length=4096), budget_forcing={'action_reserve_tokens': 256},
                 sampling_params=None, update_sampling_params=lambda **kw: nullcontext(),
                 tokenizer=NS(decode=lambda ids, **kw: ''.join(map(chr, ids)),
                              encode=lambda s, **kw: list(map(ord, s)), eos_token_id=0),
                 inference_engine=NS(generate=generate))
        tokens, probs = load_method()(obj, [{'prompt_token_ids': list(map(ord, prompt))}], {}, None)
        self.assertEqual(len(calls), 2 if forced else 1)
        if not forced:
            self.assertEqual(tokens[0], list(map(ord, response)))
            self.assertEqual(probs[0], [-0.5] * len(response))
        return calls

    def test_prefilled_complete_action_is_not_duplicated(self):
        self.run_case('assistant\n<think>\n', 'Reason.</think><action>search[test]</action>', False)

    def test_explicit_complete_transcript_unchanged(self):
        self.run_case('assistant\n', '<think>Reason.</think><action>answer[test]</action>', False)

    def test_unfinished_prefilled_thought_is_closed_once(self):
        calls = self.run_case('assistant\n<think>\n', 'Reason', True)
        suffix = ''.join(map(chr, calls[1]['prompts'][0]['prompt_token_ids']))
        self.assertTrue(suffix.endswith('Reason\n</think>\n\n<action>'))
        self.assertNotIn('Choose one admissible action.', suffix)

    def test_closed_prefilled_thought_only_adds_action_tag(self):
        calls = self.run_case('assistant\n<think>\n', 'Reason</think>', True)
        suffix = ''.join(map(chr, calls[1]['prompts'][0]['prompt_token_ids']))
        self.assertTrue(suffix.endswith('Reason</think>\n<action>'))

    def test_earlier_think_tag_is_not_an_assistant_prefill(self):
        self.run_case('user <think> example\nassistant\n', 'Reason</think><action>search[test]</action>', True)


if __name__ == '__main__':
    unittest.main()
