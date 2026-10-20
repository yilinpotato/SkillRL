"""Shared no-RL method contract for ScienceWorld and live SearchQA only.

Legacy ALFWorld/WebShop imports and execution are intentionally unaffected.
"""

import argparse
import hashlib
import json
import os
import re
from typing import Any

PROTOCOL = 'coskill-norl-v2'


def add_method_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eval-temperature', type=float, default=0.4)
    parser.add_argument('--eval-seed', type=int, default=1000)
    parser.add_argument('--max-prompt-tokens', type=int, default=8192)
    parser.add_argument('--tokenizer-path', default=os.environ.get('EDGE_TOKENIZER_PATH'), help='Tokenizer used for the explicit input budget; defaults to model ID')
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--task-specific-top-k', type=int, default=6)
    parser.add_argument('--min-calls', type=int, default=10)
    parser.add_argument('--stable-cycles-l1', type=int, default=3)
    parser.add_argument('--stable-cycles-l2', type=int, default=5)
    parser.add_argument('--success-l1', type=float, default=0.7)
    parser.add_argument('--demote-threshold', type=float, default=0.3)
    parser.add_argument('--perf-watermark', type=float, default=0.6)
    parser.add_argument('--max-new-skills', type=int, default=3)
    parser.add_argument('--tree-min-samples', type=int, default=6)


def validate_method_args(args: argparse.Namespace) -> None:
    for field in ('max_prompt_tokens', 'top_k', 'task_specific_top_k', 'min_calls',
                  'stable_cycles_l1', 'stable_cycles_l2', 'max_new_skills', 'tree_min_samples'):
        if getattr(args, field) <= 0:
            raise ValueError(f'{field} must be positive')
    for field in ('temperature', 'eval_temperature'):
        if not 0 <= getattr(args, field) <= 2:
            raise ValueError(f'{field} must be in [0, 2]')
    for field in ('success_l1', 'demote_threshold', 'perf_watermark'):
        if not 0 <= getattr(args, field) <= 1:
            raise ValueError(f'{field} must be in [0, 1]')


def skill_options(args: argparse.Namespace) -> dict:
    return dict(retrieval_mode='template', task_specific_top_k=args.task_specific_top_k,
                enable_hierarchy=True, enable_playbook=True, min_calls=args.min_calls,
                stable_cycles_l1=args.stable_cycles_l1, stable_cycles_l2=args.stable_cycles_l2,
                success_l1=args.success_l1, demote_threshold=args.demote_threshold)


def method_manifest(args: argparse.Namespace) -> dict:
    return dict(protocol=PROTOCOL, history='observation_executed_action',
                action_recovery='unique_complete_tag_outside_reasoning',
                retrieval_mode='template', initial_skills='provided' if args.skills else 'empty',
                cloud_evidence_mode='tree_only', loop_threshold=3,
                update_check='after_existing_group', update_progress_unit='completed_train_episodes',
                tokenizer=args.tokenizer_path or args.model, repeat_stop='disabled',
                resume_supported=False)


def project_action(response: str) -> dict:
    """Never execute quoted thought actions, partial tags or ambiguous actions."""
    text = response.strip()
    closed = text.count('</think>')
    if closed == 1 and text.count('<think>') <= 1:
        final = text.split('</think>', 1)[1]
    elif closed == 0 and '<think>' not in text:
        final = text
    else:
        final = ''
    matches = re.findall(r'<action>([^<>]+)</action>', final, re.DOTALL)
    command = matches[0].strip() if (len(matches) == 1 and final.count('<action>') == 1
                                   and final.count('</action>') == 1) else None
    command = command or None
    strict = bool(command and text.count('<think>') == 1 and closed == 1 and re.fullmatch(
        r'<think>.*?</think>\s*<action>[^<>]+</action>', text, re.DOTALL))
    return dict(parsed_action=command, format_valid=strict,
                execution_source=('direct' if strict else 'recovered') if command else 'none')


def history_messages(system: str, task: str, history: list, current: str, length: int) -> list[dict]:
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': task}]
    for observation, executed_action in history[-length:]:
        messages.extend([{'role': 'user', 'content': observation},
                         {'role': 'assistant', 'content': (
                             f'<action>{executed_action}</action>' if executed_action is not None
                             else '[No action executed]')}])
    messages.append({'role': 'user', 'content': current})
    return messages


def fit_history(messages: list[dict], count: Any, limit: int) -> tuple[list[dict], int, int]:
    """Drop only whole oldest observation/action pairs; never cut task/current state."""
    fitted = list(messages)
    dropped = 0
    while count(fitted) > limit and len(fitted) > 3:
        del fitted[2:4]
        dropped += 1
    size = count(fitted)
    if size > limit:
        raise ValueError('Current task, skills and observation exceed max_prompt_tokens; no request sent')
    return fitted, dropped, size


def skill_fingerprint(lib: Any) -> str:
    state = {'skills': lib.skills, 'trees': getattr(lib, 'task_playbooks', {})}
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def action_metrics(results: list[dict]) -> dict:
    steps = [step for result in results for step in result['steps']]
    n = max(len(steps), 1)
    return {
        'episode/strict_valid_action_ratio': sum(s['format_valid'] and s['valid'] for s in steps) / n,
        'episode/non_strict_valid_action_ratio': sum(s['valid'] for s in steps) / n,
        'episode/format_valid_rate': sum(s['format_valid'] for s in steps) / n,
        'episode/action_extracted_ratio': sum(s['parsed_action'] is not None for s in steps) / n,
        'episode/executed_action_ratio': sum(s['executed_action'] is not None for s in steps) / n,
        'episode/recovered_action_ratio': sum(s['execution_source'] == 'recovered' for s in steps) / n,
        'episode/length/mean': len(steps) / max(len(results), 1),
        'episode/task_score/mean': sum(r['task_score'] for r in results) / max(len(results), 1),
    }


def cloud_usage(analyzer: Any) -> dict:
    prompt = int(getattr(analyzer, 'total_prompt_tokens', 0))
    response = int(getattr(analyzer, 'total_completion_tokens', 0))
    return dict(prompt=prompt, response=response, total=prompt + response,
                usage_missing_calls=int(getattr(analyzer, 'usage_missing_calls', 0)))
