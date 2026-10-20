"""ScienceWorld 1.2.3 interaction without oracle paths or hidden world state."""

import random
import re
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from agent_system.memory import HierarchicalSkillLib
from agent_system.method_protocol import history_messages, project_action


class ScienceSkillLib(HierarchicalSkillLib):
    active_task = ''

    def _detect_task_type(self, task_description: str) -> str:
        return self.active_task


def build_manifest(env: Any, tasks: list[str], per_task: int, seed: int) -> dict:
    manifest = {'train': [], 'dev': [], 'test': []}
    known = env.get_task_names()
    if len(set(tasks)) != len(tasks) or any(task not in known for task in tasks):
        raise ValueError('Tasks must be unique names from env.get_task_names()')
    rng = random.Random(seed)
    for task in tasks:
        env.load(task, 0, '', generateGoldPath=False)
        splits = {split: list(getattr(env, f'get_variations_{split}')()) for split in manifest}
        if any(set(splits[a]) & set(splits[b]) for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]):
            raise ValueError(f'Overlapping official variations for {task}')
        for split, variations in splits.items():
            rng.shuffle(variations)
            manifest[split].extend({'task': task, 'variation': int(v)} for v in variations[:per_task or None])
    for rows in manifest.values():
        rng.shuffle(rows)
    return manifest


def parse_action(response: str) -> str | None:
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return None
    match = re.fullmatch(r'\s*<think>.*?</think>\s*<action>([^<>]+)</action>\s*', response, re.DOTALL)
    return match.group(1).strip() if match and match.group(1).strip() else None


def action_examples(actions: list[str], limit: int) -> list[str]:
    buckets: dict[str, deque] = defaultdict(deque)
    for action in actions:
        buckets[action.split(maxsplit=1)[0] if action.strip() else ''].append(action)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.popleft())
    return selected


def format_observation(observation: str, info: dict, action_limit: int = 80) -> str:
    actions = list(info.get('valid', []))
    return (
        f'{observation}\n\nVisible surroundings:\n{info.get("look", "")}\n'
        f'Inventory:\n{info.get("inv", "")}\n'
        f'Currently valid action examples ({min(len(actions), action_limit)} of {len(actions)}; not exhaustive):\n'
        + '\n'.join(action_examples(actions, action_limit))
    )


PROMPT = '''Complete the ScienceWorld task through text actions in the simulated environment.
Use only observed objects and evidence. Return exactly one <think>reasoning</think>
followed by one <action>environment action</action>. The action is a literal
ScienceWorld command, not a web search, shopping action or final-answer wrapper.
Use observations and inventory to plan actions. Action examples are a partial list.
Do not infer object locations from unrelated tasks. Follow the task's focus or
measurement requirements. Task completion is determined by the environment.'''


def run_episode(env: Any, instance: dict, model: Any, lib: ScienceSkillLib,
                max_steps: int, history_length: int, simplifications: str,
                action_limit: int = 80, on_step: Callable[[dict], None] | None = None, top_k: int = 6) -> dict:
    task, variation = instance['task'], instance['variation']
    env.load(task, variation, simplifications, generateGoldPath=False)
    observation, info = env.reset()
    goal = env.get_task_description()
    lib.active_task = task
    retrieved = lib.retrieve(goal, top_k=top_k)
    skill_text = lib.format_for_prompt(retrieved)
    current = format_observation(observation, info, action_limit)
    history, traces = [], []
    tokens = {'prompt': 0, 'response': 0, 'usage_missing_calls': 0}
    score = float(info['score'])
    done = score >= 100 or score < 0
    for decision in range(max_steps):
        if done:
            break
        messages = history_messages(PROMPT + '\n' + skill_text,
                                    f'Task: {goal}\nDecision budget: {max_steps}',
                                    history, current, history_length)
        response, usage = model.generate(messages)
        for key in tokens:
            tokens[key] += usage.get(key, 0)
        projection = project_action(response)
        command = projection['parsed_action']
        valid = command in info.get('valid', []) if command else False
        if command:
            observation, reward, done, info = env.step(command)
            score = float(info['score'])
            next_observation = format_observation(observation, info, action_limit)
        else:
            reward = 0
            next_observation = format_observation(observation, info, action_limit)
            next_observation = 'No action executed: use one complete action outside reasoning.\n' + next_observation
        traces.append({'observation': current, 'action': command or '',
                       'response': response, 'next_observation': next_observation,
                       'reward': reward, 'score': score, 'valid': valid,
                       'simulator_moves': info.get('moves'),
                       **projection, 'executed_action': command,
                       'environment_feedback': observation if command else 'No action executed'})
        if on_step is not None:
            on_step({'decision': decision + 1, **traces[-1], 'tokens': usage})
        history.append((current, command))
        current = next_observation
        done = bool(done or score >= 100 or score < 0)
    return {
        'traj_uid': f'{task}:{variation}', 'task': goal, 'task_type': task,
        'variation': variation, 'outcome': 'success' if score >= 100 else 'failure',
        'episode_reward': float(score >= 100), 'task_score': max(0, score) / 100,
        'scienceworld_score': score, 'environment_done': done,
        'simulator_moves': info.get('moves'),
        'termination_reason': ('success' if score >= 100 else 'environment_failure' if done else 'decision_limit'),
        'decision_limit_reached': not done and len(traces) >= max_steps,
        'steps': traces, 'tokens': tokens,
        'meta': {'skill_ids_used': retrieved.get('injected_skill_ids', []),
                 'injected_skill_text': skill_text},
    }
