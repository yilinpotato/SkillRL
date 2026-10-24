"""Native rollout manager: shared environment protocol, repository-native skill memory."""
from pathlib import Path
from typing import Any
import numpy as np

from .env_manager import SearchEnvironmentManager
from .benchmark_protocol import TavilySearch
from .benchmark_sessions import SearchSession, ScienceSession


class BenchmarkEnvironmentManager(SearchEnvironmentManager):
    def __init__(self, config: Any, is_train: bool):
        super().__init__(None, None, config)
        self.is_train = is_train
        self.kind = config.env.env_name.lower()
        self.sessions = []
        self.search = None
        if config.env.history_length != 0:
            raise ValueError('These protocols require full history: env.history_length=0')
        if config.data.truncation != 'error':
            raise ValueError('Use data.truncation=error; do not silently trim benchmark prompts')

    def reset(self, kwargs: Any) -> tuple:
        self.close()
        if kwargs is None or len(kwargs) == 0:
            raise ValueError('Explicit benchmark env_kwargs are required')
        self.tasks, self.retrieved_memories = [], []
        try:
            for row in kwargs:
                if self.is_train and row['split'] != 'train':
                    raise ValueError('Training must use train split')
                if not self.is_train and row['split'] not in ('dev', 'test'):
                    raise ValueError('Evaluation must use dev/test split')
                if self.kind == 'searchqa_live':
                    cfg = self.config.env.get('searchqa_live', {})
                    if self.search is None:
                        audit = Path(cfg.get('audit_dir', 'outputs/searchqa-provider'))
                        audit.mkdir(parents=True, exist_ok=True)
                        self.search = TavilySearch(audit / ('train.jsonl' if self.is_train else 'eval.jsonl'), max_results=5)
                    session = SearchSession(row, self.search, self.config.env.max_steps, cfg.get('max_searches', 8))
                else:
                    from scienceworld import ScienceWorldEnv
                    cfg = self.config.env.get('scienceworld', {})
                    env = ScienceWorldEnv('', envStepLimit=self.config.env.max_steps)
                    session = ScienceSession(row, env, self.config.env.max_steps, cfg.get('simplifications', ''))
                self.sessions.append(session)
                self.tasks.append(session.goal)
                memory = {}
                if self.retrieval_memory is not None:
                    cfg = self.config.env.skills_only_memory if self.config.env.get('use_skills_only_memory', False) else self.config.env.retrieval_memory
                    memory = self.retrieval_memory.retrieve(session.goal, top_k=cfg.get('top_k', 6))
                self.retrieved_memories.append(memory)
        except Exception:
            self.close()
            raise
        return self.observations(), [s.info.copy() for s in self.sessions]

    def observations(self) -> dict:
        messages = []
        for session, memory in zip(self.sessions, self.retrieved_memories):
            skills = self.retrieval_memory.format_for_prompt(memory) if self.retrieval_memory is not None else ''
            messages.append(session.messages(skills))
        return {'messages': messages, 'text': ['\n'.join(m['content'] for m in ms) for ms in messages],
                'image': None, 'anchor': [s.current for s in self.sessions]}

    def step(self, text_actions: list[str]) -> tuple:
        if len(text_actions) != len(self.sessions):
            raise ValueError('Action batch size differs from reset batch')
        results = [session.step(action) for session, action in zip(self.sessions, text_actions)]
        rewards, dones, infos = zip(*results)
        return self.observations(), np.asarray(rewards), np.asarray(dones), list(infos)

    def _process_batch(self, batch_idx: int, total_batch_list: list, total_infos: list, success: Any) -> None:
        for step in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][step]['active_masks']:
                info = total_infos[batch_idx][step]
                success['success_rate'].append(float(info['won']))
                for key in ('task_score', 'exact_match', 'f1', 'scienceworld_score'):
                    if key in info:
                        success[key].append(float(info[key]))
                return
        raise ValueError('Episode has no active decision')

    def close(self) -> None:
        for session in self.sessions:
            session.close()
        self.sessions = []
