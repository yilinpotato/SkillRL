"""Method-independent sessions for the pinned CoSkill benchmark protocols."""
from typing import Any

from .benchmark_protocol import (
    SearchQAEnvironment, SYSTEM_PROMPT, PROMPT, initial_prompt, qa_messages,
    history_messages, project_action, score_answer,
)


class SearchSession:
    def __init__(self, row: dict, search: Any, max_steps: int, max_searches: int):
        self.row = row
        self.env = SearchQAEnvironment(row['question'], search, max_steps, max_searches)
        self.goal = row['question']
        self.current = initial_prompt(self.goal, max_steps, max_searches)
        self.initial = self.current
        self.history = []
        self.done = False
        self.info = {'won': False, 'exact_match': 0., 'f1': 0., 'task_score': 0.}

    def messages(self, skills: str = '') -> list[dict]:
        return qa_messages(SYSTEM_PROMPT + ('\n\n' + skills if skills else ''),
                           self.initial, self.history, self.current)

    def step(self, response: str) -> tuple:
        if self.done:
            return 0., True, dict(self.info, is_action_valid=False)
        previous = self.current
        self.current, valid = self.env.step(response)
        self.history.append((previous, self.env.executed_action))
        self.done = self.env.done
        scores = score_answer(self.env.answer, self.row['answers']) if self.done else {'exact_match': 0., 'f1': 0.}
        self.info = dict(scores, won=bool(scores['exact_match']), task_score=scores['f1'],
                         is_action_valid=valid, searches=self.env.searches,
                         executed_action=self.env.executed_action, **self.env.last_projection)
        return scores['exact_match'], self.done, self.info

    def close(self) -> None:
        pass


class ScienceSession:
    def __init__(self, row: dict, env: Any, max_steps: int, simplifications: str = ''):
        self.env, self.row, self.max_steps = env, row, max_steps
        self.history = []
        try:
            env.load(row['task'], int(row['variation']), simplifications, generateGoldPath=False)
            split = row.get('split')
            if split not in ('train', 'dev', 'test'):
                raise ValueError('ScienceWorld row requires official split: train/dev/test')
            if int(row['variation']) not in getattr(env, 'get_variations_' + split)():
                raise ValueError('Variation is outside its declared official split')
            self.current, self.raw_info = env.reset()
            self.goal = env.get_task_description()
        except Exception:
            env.close()
            raise
        self.done = False
        self.info = self.metrics(False)

    def metrics(self, valid: bool) -> dict:
        score = float(self.raw_info['score'])
        return dict(won=score >= 100, task_score=max(0., score) / 100.,
                    scienceworld_score=score, is_action_valid=valid)

    def messages(self, skills: str = '') -> list[dict]:
        return history_messages(PROMPT + ('\n' + skills if skills else ''),
                                f'Task: {self.goal}\nDecision budget: {self.max_steps}',
                                self.history, self.current, len(self.history))

    def step(self, response: str) -> tuple:
        if self.done:
            return 0., True, dict(self.info, is_action_valid=False)
        previous = self.current
        projection = project_action(response)
        action = projection['parsed_action']
        valid = bool(action and action in self.raw_info.get('valid', []))
        if action:
            self.current, _, terminated, self.raw_info = self.env.step(action)
        else:
            self.current = 'No action executed: use one complete action outside reasoning.\n' + self.current
            terminated = False
        self.history.append((previous, action))
        score = float(self.raw_info['score'])
        self.done = bool(terminated or score >= 100 or score < 0 or len(self.history) >= self.max_steps)
        self.info = dict(self.metrics(valid), **projection, executed_action=action)
        # Same terminal success objective as CoSkill; raw score remains a separate metric.
        return float(score >= 100), self.done, self.info

    def close(self) -> None:
        self.env.close()
