import json

import pytest
from omegaconf import OmegaConf
from agent_system.environments.benchmark_sessions import SearchSession, ScienceSession
from agent_system.environments.benchmark_protocol import project_action
from agent_system.environments.env_manager import make_envs


class Search:
    calls = 0
    def search(self, query):
        self.calls += 1
        return [{'url': 'https://example.org', 'title': 'Source', 'content': 'Evidence'}]


def test_search_answer_and_f1_are_separate():
    s = SearchSession({'question': 'Who?', 'answers': ['Hemingway']}, Search(), 15, 8)
    assert 'Hemingway' not in json.dumps(s.messages())
    reward, done, info = s.step('<action>answer[Ernest Hemingway]</action>')
    assert done and reward == 0 and info['f1'] == pytest.approx(2 / 3)
    assert s.step('<action>search[ignored]</action>')[0] == 0


def test_search_open_rejected_and_budgets_independent():
    provider = Search()
    s = SearchSession({'question': 'Who?', 'answers': ['X']}, provider, 4, 1)
    assert not s.step('<action>open[https://example.org]</action>')[2]['is_action_valid']
    s.step('<action>search[q]</action>')
    assert not s.step('<action>search[q2]</action>')[2]['is_action_valid']
    assert provider.calls == 1
    assert s.step('<action>answer[X]</action>')[0] == 1


@pytest.mark.parametrize('response', ['<think><action>look</action>', '<action>look', '<action>a</action><action>b</action>'])
def test_ambiguous_or_thought_action_never_executes(response):
    assert project_action(response)['parsed_action'] is None


def test_search_history_full_and_no_reasoning_replay():
    s = SearchSession({'question': 'Who?', 'answers': ['X']}, Search(), 15, 8)
    for _ in range(10):
        s.step('<think>SECRET_THOUGHT</think><action>open[bad]</action>')
    text = json.dumps(s.messages())
    assert text.count('Who?') == 1 and 'SECRET_THOUGHT' not in text
    assert len([m for m in s.messages() if m['role'] == 'assistant']) == 10


class Science:
    def __init__(self):
        self.calls = []; self.closed = False
    def load(self, *args, **kwargs):
        assert kwargs['generateGoldPath'] is False
    def get_variations_test(self):
        return [1]
    def reset(self):
        return 'ROOM', {'score': 0, 'valid': ['look'], 'look': 'DUPLICATE', 'inv': 'SECRET'}
    def get_task_description(self):
        return 'Experiment'
    def step(self, command):
        self.calls.append(command)
        return 'RESULT', 100, True, {'score': 100, 'valid': []}
    def close(self):
        self.closed = True


def test_science_observation_and_unlisted_action_are_not_filtered():
    env = Science()
    s = ScienceSession({'task': 'test', 'variation': 1, 'split': 'test'}, env, 100)
    text = json.dumps(s.messages())
    assert text.count('ROOM') == 1 and 'DUPLICATE' not in text and 'SECRET' not in text
    reward, done, info = s.step('<action>unlisted command</action>')
    assert env.calls == ['unlisted command'] and done and reward == 1
    assert info['scienceworld_score'] == 100 and not info['is_action_valid']
    assert s.step('<action>look</action>')[0] == 0
    s.close()
    assert env.closed


def test_science_invalid_official_split_closes_environment():
    env = Science()
    with pytest.raises(ValueError):
        ScienceSession({'task': 'test', 'variation': 2, 'split': 'test'}, env, 100)
    assert env.closed


def config():
    return OmegaConf.create({'env': {'env_name': 'searchqa_live', 'rollout': {'n': 2}, 'resources_per_worker': {'num_cpus': .1},
                                     'max_steps': 15, 'history_length': 0}, 'data': {'truncation': 'error'}})


def test_native_factory_batch_routing_done_and_no_answer_leak():
    train, evaluation = make_envs(config())
    train.search = Search()
    row = {'question': 'Who?', 'answers': ['HiddenAnswer'], 'split': 'train'}
    obs, _ = train.reset([row, row])
    assert len(obs['messages']) == 2 and 'HiddenAnswer' not in json.dumps(obs)
    _, rewards, dones, _ = train.step(['<action>answer[HiddenAnswer]</action>', '<action>search[question]</action>'])
    assert rewards.tolist() == [1., 0.] and dones.tolist() == [True, False]
    _, rewards, _, _ = train.step(['ignored', '<action>answer[wrong]</action>'])
    assert rewards.tolist() == [0., 0.]
    with pytest.raises(ValueError):
        evaluation.reset([row])
    train.close(); evaluation.close()


def test_reject_silent_context_trimming():
    cfg = config(); cfg.data.truncation = 'left'
    with pytest.raises(ValueError):
        make_envs(cfg)


def test_existing_skill_memory_is_retrieved_on_reset():
    train, evaluation = make_envs(config())
    class Memory:
        text = 'DOMAIN_SKILL'
        def retrieve(self, task, **kwargs):
            return {'text': self.text}
        def format_for_prompt(self, memory):
            return memory['text']
    train.retrieval_memory = Memory()
    train.config.env.retrieval_memory = {'top_k': 6}
    train.search = Search()
    row = {'question': 'Who?', 'answers': ['X'], 'split': 'train'}
    obs, _ = train.reset([row])
    assert 'DOMAIN_SKILL' in obs['messages'][0][0]['content']
    train.retrieval_memory.text = 'UPDATED_SKILL'
    obs, _ = train.reset([row])
    assert 'UPDATED_SKILL' in obs['messages'][0][0]['content']
    train.close(); evaluation.close()


def test_preparation_rejects_cross_split_questions(tmp_path):
    from examples.data_preprocess.prepare_new_benchmarks import prepare
    paths = {}
    for split in ('train', 'dev', 'test'):
        paths[split] = tmp_path / (split + '.jsonl')
        paths[split].write_text(json.dumps({'id': split, 'question': 'Same question', 'answers': ['X']}))
    with pytest.raises(ValueError, match='overlap'):
        prepare('searchqa_live', paths, tmp_path / 'output')


def test_native_rollout_preserves_chat_roles():
    import torch
    from types import SimpleNamespace
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    cfg = config()
    cfg.trainer = {}
    cfg.data.max_prompt_length = 256
    cfg.data.return_raw_chat = True
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, chat, **kwargs):
            self.chat = list(chat)
            return 'rendered'
        def encode(self, text, **kwargs):
            return [1, 2]
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.tensor([[1, 1]])}
    tokenizer = Tokenizer()
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.config, collector.tokenizer, collector.processor = cfg, tokenizer, None
    gen = SimpleNamespace(non_tensor_batch={'raw_prompt': [[{'role': 'user', 'content': 'ignored'}]], 'data_source': ['searchqa_live']})
    messages = [{'role': 'system', 'content': 'instruction'}, {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': '<action>search[q]</action>'}, {'role': 'user', 'content': 'evidence'}]
    obs = {'messages': [messages], 'text': ['legacy text'], 'image': None, 'anchor': ['evidence']}
    result = collector.preprocess_single_sample(0, gen, obs)
    assert result['raw_prompt'] == messages and tokenizer.chat == messages
    obs.pop('messages')
    result = collector.preprocess_single_sample(0, gen, obs)
    assert result['raw_prompt'] == [{'role': 'user', 'content': 'legacy text'}]
