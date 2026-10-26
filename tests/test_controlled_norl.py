import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import json

path = Path(__file__).resolve().parents[1]/'examples/playbook_evolve/run_controlled_norl.py'
spec = importlib.util.spec_from_file_location('controlled_norl', path)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class Memory:
    def __init__(self): self.updates = 0
    def _detect_task_type(self, task): return 'searchqa'
    def record_usage(self, *args): self.updates += 1
    def record_playbook_usage(self, *args): pass
    def save_skills(self, path): pass


class Env:
    def __init__(self): self.retrieval_memory = Memory(); self.resets = []
    def reset(self, rows):
        self.resets.append(list(rows)); self.tasks = list(rows)
        self.retrieved_memories = [{} for _ in rows]
        obs = {'messages': [[{'role':'user', 'content': str(row)}] for row in rows],
               'text': list(rows), 'anchor': list(rows)}
        self.obs = obs
        return obs, [{} for _ in rows]
    def step(self, actions):
        return self.obs, [1.]*len(actions), [True]*len(actions), [dict(won=True, executed_action='answer[A]') for _ in actions]


def test_exact_budget_eval_temperature_and_no_eval_learning(tmp_path):
    cfg = NS(data=NS(train_batch_size=12), env=NS(rollout=NS(n=6), max_steps=15),
             trainer=NS(total_training_steps=100, val_before_train=True, test_freq=5),
             actor_rollout_ref=NS(rollout=NS(temperature=1., val_kwargs=NS(temperature=.4))))
    train, dev = Env(), Env()
    traces, updates, temperatures = [], [], []
    pool = NS(add_trace=traces.append)
    cloud = NS(maybe_update=lambda pool, memory, n: updates.append(n))
    def generate(messages, temperature):
        temperatures.append(temperature)
        return '<action>answer[A]</action>', {'prompt_tokens': 10, 'completion_tokens': 5}
    completed = runner.execute(cfg, list(range(1200)), ['dev']*32, (train, dev), generate, pool, cloud, tmp_path)
    assert completed == len(traces) == train.retrieval_memory.updates == 7200
    assert len({t['traj_uid'] for t in traces}) == 7200
    assert len(dev.resets) == 21 and all(x == ['dev']*32 for x in dev.resets)
    assert temperatures.count(.4) == 21*32 and temperatures.count(1.) == 7200
    assert updates == list(range(72, 7201, 72))
    assert train.resets[0][:6] == [0]*6 and train.resets[-1][-6:] == [1199]*6
    assert json.loads((tmp_path/'progress.json').read_text())['train_episodes'] == 7200
