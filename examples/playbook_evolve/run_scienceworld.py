"""Frozen CoSkill learning on official ScienceWorld variation splits."""

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from agent_system.memory import CoSkillCloudLoop, TracesPool
from agent_system.method_protocol import (
    PROTOCOL,
    action_metrics,
    add_method_args,
    cloud_usage,
    method_manifest,
    skill_fingerprint,
    skill_options,
    validate_method_args,
)
from agent_system.scienceworld_adapter import ScienceSkillLib, build_manifest, run_episode
from agent_system.searchqa_live import append_json
from examples.playbook_evolve.run_searchqa_live import ChatModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_method_args(p)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--split', choices=['dev', 'test'], default='test', help='Evaluation split')
    p.add_argument('--tasks', nargs='+', help='Official task names; default all tasks')
    p.add_argument('--per-task', type=int, default=0, help='Instances per task per split; 0 means all')
    p.add_argument('--groups', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=72)
    p.add_argument('--validation-size', type=int, default=32)
    p.add_argument('--validation-every', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--history-length', type=int, default=8)
    p.add_argument('--action-examples', type=int, default=80)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--simplifications', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skills', help='Previously learned domain skill checkpoint; no preset skills are supplied')
    p.add_argument('--no-cloud', action='store_true')
    p.add_argument('--min-samples', type=int, default=16)
    p.add_argument('--capacity-watermark', type=int, default=50000)
    p.add_argument('--model', default=os.environ.get('EDGE_MODEL', 'Qwen/Qwen3-4B-Thinking-2507'))
    p.add_argument('--base-url', default=os.environ.get('EDGE_API_BASE', 'http://127.0.0.1:8000/v1'))
    p.add_argument('--env-smoke', action='store_true', help='Real environment checks only; no model/API calls')
    return p.parse_args(argv)


def summarize(results: list[dict], split: str, group: int, elapsed: float) -> dict:
    steps = [step for result in results for step in result['steps']]
    metrics = {
        'benchmark': 'scienceworld', 'split': split, 'group': group,
        'episode/count': len(results),
        'episode/success_rate': sum(r['episode_reward'] for r in results) / len(results),
        'episode/score': sum(r['scienceworld_score'] for r in results) / len(results),
        'episode/avg_steps': len(steps) / len(results),
        'episode/valid_action_rate': sum(s['valid'] for s in steps) / max(1, len(steps)),
        'episode/format_valid_rate': sum(s['format_valid'] for s in steps) / max(1, len(steps)),
        'timing_s/rollout': elapsed,
        'tokens/small_model/accounting': 'provider_api_usage',
    }
    for key in ('prompt', 'response', 'usage_missing_calls'):
        metrics[f'tokens/small_model/{key}'] = sum(r['tokens'][key] for r in results)
    metrics['tokens/small_model/total'] = metrics['tokens/small_model/prompt'] + metrics['tokens/small_model/response']
    for task in sorted({r['task_type'] for r in results}):
        subset = [r for r in results if r['task_type'] == task]
        metrics[f'episode/by_task/{task}/success_rate'] = sum(r['episode_reward'] for r in subset) / len(subset)
        metrics[f'episode/by_task/{task}/score'] = sum(r['scienceworld_score'] for r in subset) / len(subset)
    return metrics


def run(args: argparse.Namespace, env: Any, installed_version: str) -> None:
    validate_method_args(args)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    tasks = args.tasks or env.get_task_names()
    manifest = build_manifest(env, tasks, args.per_task, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'benchmark': 'scienceworld', 'scienceworld_version': installed_version,
              'oracle_enabled': False, 'manifest': manifest, 'method': method_manifest(args)}
    if args.skills:
        config['skills_sha256'] = hashlib.sha256(Path(args.skills).read_bytes()).hexdigest()
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if args.env_smoke:
        for split, rows in manifest.items():
            for row in rows:
                env.load(row['task'], row['variation'], args.simplifications, generateGoldPath=False)
                observation, info = env.reset()
                observation, reward, done, info = env.step('look around')
                append_json(output / 'env_smoke.jsonl', {
                    **row, 'split': split, 'score': info['score'], 'done': done,
                    'observation_chars': len(observation), 'valid_actions': len(info['valid']),
                })
        (output / 'completed.json').write_text(json.dumps({'status': 'env_smoke_complete'}))
        print(json.dumps({'status': 'env_smoke_complete', 'instances': {k: len(v) for k, v in manifest.items()}}))
        return
    seed = output / 'initial_skills.json'
    seed.write_text(Path(args.skills).read_text(encoding='utf-8') if args.skills else json.dumps({
        'general_skills': [], 'task_specific_skills': {task: [] for task in tasks}, 'common_mistakes': [],
    }), encoding='utf-8')
    lib = ScienceSkillLib(str(seed), **skill_options(args))
    if set(lib.skills.get('task_specific_skills', {})) - set(env.get_task_names()):
        raise ValueError('Skill library contains non-ScienceWorld task types')
    learn = args.mode == 'train' and not args.no_cloud
    loop = CoSkillCloudLoop(str(output), enable_coskill=learn, enable_playbook_evolve=learn,
                           environment_name='scienceworld', max_new_skills=args.max_new_skills,
                           playbook_evolve_min_samples=args.tree_min_samples)
    if learn and loop._get_analyzer() is None:
        raise ValueError('Cloud configuration unavailable; configure credentials or use --no-cloud')
    pool = TracesPool(output_dir=str(output), min_samples=args.min_samples,
                      capacity_watermark=args.capacity_watermark, perf_watermark=args.perf_watermark)
    model = ChatModel(args.model, args.base_url, args.max_tokens, args.seed)
    model.max_prompt_tokens = args.max_prompt_tokens
    model.tokenizer_path = args.tokenizer_path or args.model
    cumulative: dict[str, dict] = {}
    episode_counts: dict[str, int] = {}

    def execute(rows: list[dict], split: str, group: int) -> dict:
        frozen = skill_fingerprint(lib)
        model.temperature = args.temperature
        model.seed = args.seed
        start, results = time.monotonic(), []
        for row in rows:
            def log_step(step: dict, row: dict = row) -> None:
                append_json(output / 'step_events.jsonl', {'group': group, 'split': split, **row, **step})
                print(f"[{split}] {row['task']} variation={row['variation']} "
                      f"step={step['decision']} score={step['score']} format_valid={step['format_valid']}", flush=True)

            result = run_episode(env, row, model, lib, args.max_steps, args.history_length,
                                 args.simplifications, args.action_examples, on_step=log_step, top_k=args.top_k)
            append_json(output / f'{split}_trajectories.jsonl', {'group': group, **result})
            results.append(result)
            if split == 'train' and learn:
                pool.add_trace(result)
                lib.record_usage(result['meta']['skill_ids_used'], bool(result['episode_reward']), result['task_type'])
                lib.record_playbook_usage(result['task_type'], bool(result['episode_reward']))
        metrics = summarize(results, split, group, time.monotonic() - start)
        metrics.update(action_metrics(results))
        totals = cumulative.setdefault(split, {})
        for key in ('prompt', 'response', 'total', 'usage_missing_calls'):
            field = f'tokens/small_model/{key}'
            totals[key] = totals.get(key, 0) + metrics[field]
            metrics[field + '_cumulative'] = totals[key]
        episode_counts[split] = episode_counts.get(split, 0) + len(results)
        metrics['episode/count_cumulative'] = episode_counts[split]
        metrics['rollout/global_episode_end'] = episode_counts.get('train', 0)
        metrics['method/protocol'] = PROTOCOL
        metrics['sampling/temperature'] = model.temperature
        metrics['sampling/seed'] = model.seed
        metrics['skills/fingerprint_before'] = frozen
        metrics['skills/fingerprint_after'] = skill_fingerprint(lib)
        if split != 'train' and frozen != skill_fingerprint(lib):
            raise RuntimeError('Evaluation mutated the skill library')
        for result in results:
            append_json(output / 'metrics.jsonl', {
                'split': split, 'group': group, 'instance_id': result['traj_uid'],
                'episode/won': bool(result['episode_reward']), 'episode/task_score': result['task_score'],
                'episode/length': len(result['steps']), 'termination_reason': result['termination_reason'],
                'tokens': result['tokens'], **action_metrics([result]),
            })
        return metrics

    rows = manifest['train' if args.mode == 'train' else args.split]
    validation = manifest['dev'][:args.validation_size] if args.mode == 'train' else []
    if not rows:
        raise ValueError('Selected split is empty')
    if validation:
        append_json(output / 'validation_metrics.jsonl', execute(validation, 'dev', 0))
    for group in range(1, args.groups + 1):
        offset = (group - 1) * args.batch_size
        batch = rows[offset:offset + args.batch_size]
        if not batch:
            break
        metrics = execute(batch, 'train' if args.mode == 'train' else args.split, group)
        large_before = cloud_usage(loop.cloud_analyzer)
        start = time.monotonic()
        metrics['coskill/updated'] = loop.maybe_update(pool, lib, episode_counts.get('train', 0)) if learn else False
        metrics['timing_s/cloud_update'] = time.monotonic() - start
        analyzer = loop.cloud_analyzer
        if analyzer:
            metrics.update({
                'tokens/large_model/prompt_cumulative': analyzer.total_prompt_tokens,
                'tokens/large_model/completion_cumulative': analyzer.total_completion_tokens,
                'tokens/large_model/total_cumulative': analyzer.total_prompt_tokens + analyzer.total_completion_tokens,
                'tokens/large_model/usage_missing_calls': analyzer.usage_missing_calls,
                'tokens/large_model/accounting': 'provider_api_usage',
            })
        large_after = cloud_usage(loop.cloud_analyzer)
        for key, value in large_after.items():
            metrics[f'tokens/large_model/{key}'] = value - large_before[key]
            metrics[f'tokens/large_model/{key}_cumulative'] = value
        metrics['tokens/large_model/accounting'] = 'provider_api_usage'
        append_json(output / 'group_metrics.jsonl', metrics)
        lib.save_skills(str(output / 'skills_latest.json'))
        (output / 'summary_partial.json').write_text(json.dumps({
            'protocol': PROTOCOL, 'completed_group': group, 'episode_counts': episode_counts,
            'skill_fingerprint': skill_fingerprint(lib), 'resume_supported': False,
            'skills_checkpoint': 'skills_latest.json',
        }, indent=2), encoding='utf-8')
        print(json.dumps(metrics), flush=True)
        if validation and (group % args.validation_every == 0 or offset + len(batch) >= len(rows) or group == args.groups):
            append_json(output / 'validation_metrics.jsonl', execute(validation, 'dev', group))
    (output / 'completed.json').write_text(json.dumps({'status': 'complete', 'mode': args.mode}))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for field in ('groups', 'batch_size', 'validation_size', 'validation_every', 'max_steps',
                  'history_length', 'action_examples', 'max_tokens', 'min_samples', 'capacity_watermark'):
        if getattr(args, field) <= 0:
            raise ValueError(f'{field} must be positive')
    if args.per_task < 0:
        raise ValueError('per_task must be nonnegative')
    installed = version('scienceworld')
    if installed != '1.2.3':
        raise ValueError(f'Expected scienceworld==1.2.3, found {installed}')
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv(envStepLimit=args.max_steps)
    try:
        run(args, env, installed)
    finally:
        env.close()


if __name__ == '__main__':
    main()
