"""Frozen-model SearchQA live-search learning and held-out evaluation."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

from agent_system.memory import CoSkillCloudLoop, HierarchicalSkillLib, TracesPool
from agent_system.method_protocol import (
    PROTOCOL,
    action_metrics,
    add_method_args,
    cloud_usage,
    fit_history,
    method_manifest,
    skill_fingerprint,
    skill_options,
    validate_method_args,
)
from agent_system.searchqa_live import (
    TavilySearch,
    append_json,
    check_disjoint,
    load_examples,
    post_json,
    run_episode,
)


def normalize_chat_message(message: dict) -> str:
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning')
    if reasoning and '<think>' not in content:
        return f'<think>{reasoning}</think>\n{content}'
    if (content.count('<think>') == 0 and content.count('</think>') == 1
            and content.split('</think>', 1)[1].lstrip().startswith('<action>')):
        return '<think>' + content
    return content


class ChatModel:
    def __init__(self, model, base_url, max_tokens, seed):
        self.model, self.base_url = model, base_url.rstrip('/')
        self.max_tokens, self.seed = max_tokens, seed
        self.temperature = 1.0
        self.max_prompt_tokens = 8192
        self.tokenizer_path = model
        self._tokenizer = None

    def generate(self, messages):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        def count(rows: list[dict]) -> int:
            return len(self._tokenizer.apply_chat_template(rows, tokenize=True, add_generation_prompt=True))
        messages, dropped, budget_tokens = fit_history(messages, count, self.max_prompt_tokens)
        response = post_json(self.base_url + '/chat/completions', {
            'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens,
            'temperature': self.temperature, 'seed': self.seed,
        }, os.environ.get('EDGE_API_KEY', 'local'), timeout=300)
        message = response['choices'][0]['message']
        content = normalize_chat_message(message)
        usage = response.get('usage') or {}
        return content, {
            'history_pairs_dropped': dropped, 'budget_prompt_tokens': budget_tokens,
            'prompt': usage.get('prompt_tokens', 0),
            'response': usage.get('completion_tokens', 0),
            'usage_missing_calls': int('prompt_tokens' not in usage or 'completion_tokens' not in usage),
        }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Training JSONL, or evaluation JSONL in eval mode')
    p.add_argument('--validation-data', help='Disjoint validation JSONL; never enters the trajectory pool')
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    add_method_args(p)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--skills', help='Previously learned SearchQA skill checkpoint; no preset skills are supplied')
    p.add_argument('--model', default=os.environ.get('EDGE_MODEL', 'Qwen/Qwen3-4B-Thinking-2507'))
    p.add_argument('--base-url', default=os.environ.get('EDGE_API_BASE', 'http://127.0.0.1:8000/v1'))
    p.add_argument('--groups', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=72)
    p.add_argument('--limit', type=int, default=0, help='Cap selected examples; 0 means all')
    p.add_argument('--validation-size', type=int, default=32)
    p.add_argument('--validation-every', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=15)
    p.add_argument('--max-searches', type=int, default=8)
    p.add_argument('--search-backend', choices=['tavily', 'search_r1'], default='tavily')
    p.add_argument('--retrieval-endpoint', default='')
    p.add_argument('--retrieval-corpus-id', default='')
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--history-length', type=int, default=8)
    p.add_argument('--observation-chars', type=int, default=12000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-cloud', action='store_true', help='Fixed-skill baseline; disable learning')
    p.add_argument('--min-samples', type=int, default=16)
    p.add_argument('--capacity-watermark', type=int, default=50000)
    p.add_argument('--check-only', action='store_true', help='Validate data and splits without API calls')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    validate_method_args(args)
    for name in ('groups', 'batch_size', 'validation_size', 'validation_every', 'max_steps',
                 'max_searches', 'max_tokens', 'history_length', 'observation_chars',
                 'min_samples', 'capacity_watermark'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.limit < 0:
        raise ValueError('limit must be nonnegative')
    examples = load_examples(args.data)
    validation = load_examples(args.validation_data) if args.validation_data else []
    check_disjoint(examples, validation)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    rng.shuffle(validation)
    if args.limit:
        examples = examples[:args.limit]
    validation = validation[:args.validation_size]
    if args.check_only:
        print(json.dumps({'examples': len(examples), 'validation': len(validation), 'mode': args.mode}))
        return
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; existing experiments are never overwritten')
    if args.search_backend == 'tavily' and not os.environ.get('TAVILY_API_KEY'):
        raise ValueError('Set TAVILY_API_KEY for live search')
    output.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    manifest.update(benchmark='searchqa_live', protocol='live-search-v2', method=method_manifest(args),
                    search_provider=args.search_backend,
                    search_depth='basic' if args.search_backend == 'tavily' else None, search_max_results=5,
                    excluded_domains=['j-archive.com', 'j-archive.org'],
                    train_ids=[x['id'] for x in examples] if args.mode == 'train' else [],
                    eval_ids=[x['id'] for x in examples] if args.mode == 'eval' else [],
                    validation_ids=[x['id'] for x in validation],
                    data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    if args.validation_data:
        manifest['validation_sha256'] = hashlib.sha256(Path(args.validation_data).read_bytes()).hexdigest()
    if args.skills:
        manifest['skills_sha256'] = hashlib.sha256(Path(args.skills).read_bytes()).hexdigest()
    (output / 'run_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    seed = output / 'initial_skills.json'
    seed.write_text(Path(args.skills).read_text(encoding='utf-8') if args.skills else json.dumps({
        'general_skills': [], 'task_specific_skills': {'searchqa': []}, 'common_mistakes': [],
    }), encoding='utf-8')
    lib = HierarchicalSkillLib(str(seed), **skill_options(args))
    if set(lib.skills.get('task_specific_skills', {})) - {'searchqa'}:
        raise ValueError('Use a SearchQA skill library, not ALFWorld/WebShop skills')
    from agent_system.local_search import create_search
    search = create_search({'backend': args.search_backend,
                            'retrieval_endpoint': args.retrieval_endpoint,
                            'retrieval_corpus_id': args.retrieval_corpus_id},
                           output / 'search_audit.jsonl', TavilySearch)
    model = ChatModel(args.model, args.base_url, args.max_tokens, args.seed)
    model.max_prompt_tokens = args.max_prompt_tokens
    model.tokenizer_path = args.tokenizer_path or args.model
    learn = args.mode == 'train' and not args.no_cloud
    loop = CoSkillCloudLoop(str(output), enable_coskill=learn, enable_playbook_evolve=learn,
                           environment_name='searchqa_live', max_new_skills=args.max_new_skills,
                           playbook_evolve_min_samples=args.tree_min_samples)
    if learn and loop._get_analyzer() is None:
        raise ValueError('Cloud analyzer unavailable; configure cloud credentials or use --no-cloud')
    pool = TracesPool(output_dir=str(output), min_samples=args.min_samples,
                      capacity_watermark=args.capacity_watermark, perf_watermark=args.perf_watermark)
    totals = {}
    search_totals = {}
    episode_counts: dict[str, int] = {}

    def execute(rows, split, group):
        frozen = skill_fingerprint(lib)
        model.temperature = args.temperature
        model.seed = args.seed
        start = time.monotonic()
        results = []
        before_calls, before_credits = search.calls, search.credits
        before_missing = search.missing_usage
        for example in rows:
            def log_step(step: dict, example: dict = example) -> None:
                append_json(output / 'step_events.jsonl', {'group': group, 'split': split,
                                                         'instance_id': example['id'], **step})

            result = run_episode(example, model, search, lib, args.max_steps, args.max_searches,
                                 args.history_length, args.observation_chars, top_k=args.top_k, on_step=log_step)
            append_json(output / f'{split}_trajectories.jsonl', {'group': group, **result})
            results.append(result)
            if split == 'train' and learn:
                pool.add_trace(result)
                lib.record_usage(result['meta']['skill_ids_used'], bool(result['episode_reward']), 'searchqa')
                lib.record_playbook_usage('searchqa', bool(result['episode_reward']))
        metrics = {'benchmark': 'searchqa_live', 'split': split, 'group': group,
                   'episode/count': len(results), 'timing_s/rollout': time.monotonic() - start,
                   'search/calls': search.calls - before_calls, 'search/credits': search.credits - before_credits,
                   'search/usage_missing_calls': search.missing_usage - before_missing}
        for metric in ('exact_match', 'f1'):
            metrics[f'qa/{metric}'] = sum(r['metrics'][metric] for r in results) / len(results)
        metrics['episode/success_rate'] = metrics['qa/exact_match']
        metrics['episode/avg_steps'] = sum(len(r['steps']) for r in results) / len(results)
        metrics['episode/valid_action_rate'] = sum(
            sum(s['valid'] for s in r['steps']) for r in results
        ) / sum(len(r['steps']) for r in results)
        metrics.update(action_metrics(results))
        search_phase = search_totals.setdefault(split, {})
        for key in ('calls', 'credits', 'usage_missing_calls'):
            field = f'search/{key}'
            search_phase[key] = search_phase.get(key, 0) + metrics[field]
            metrics[field + '_cumulative'] = search_phase[key]
        cumulative = totals.setdefault(split, {'prompt': 0, 'response': 0, 'usage_missing_calls': 0})
        for key in cumulative:
            value = sum(r['tokens'][key] for r in results)
            cumulative[key] += value
            metrics[f'tokens/small_model/{key}'] = value
            metrics[f'tokens/small_model/{key}_cumulative'] = cumulative[key]
        metrics['tokens/small_model/total'] = metrics['tokens/small_model/prompt'] + metrics['tokens/small_model/response']
        metrics['tokens/small_model/total_cumulative'] = cumulative['prompt'] + cumulative['response']
        metrics['tokens/small_model/accounting'] = 'provider_api_usage'
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

    if validation and args.mode == 'train':
        append_json(output / 'validation_metrics.jsonl', execute(validation, 'validation', 0))
    for group in range(1, args.groups + 1):
        offset = (group - 1) * args.batch_size
        rows = examples[offset:offset + args.batch_size]
        if not rows:
            break
        metrics = execute(rows, args.mode, group)
        large_before = cloud_usage(loop.cloud_analyzer)
        start = time.monotonic()
        metrics['coskill/updated'] = loop.maybe_update(pool, lib, episode_counts.get('train', 0)) if learn else False
        metrics['cloud/updated'] = metrics['coskill/updated']  # legacy alias
        metrics['timing_s/cloud_update'] = time.monotonic() - start
        if loop.cloud_analyzer is not None:
            analyzer = loop.cloud_analyzer
            metrics['tokens/large_model/prompt_cumulative'] = analyzer.total_prompt_tokens
            metrics['tokens/large_model/completion_cumulative'] = analyzer.total_completion_tokens
            metrics['tokens/large_model/total_cumulative'] = analyzer.total_prompt_tokens + analyzer.total_completion_tokens
            metrics['tokens/large_model/accounting'] = 'provider_api_usage'
            metrics['tokens/large_model/usage_missing_calls'] = analyzer.usage_missing_calls
        large_after = cloud_usage(loop.cloud_analyzer)
        for key, value in large_after.items():
            metrics[f'tokens/large_model/{key}'] = value - large_before[key]
            metrics[f'tokens/large_model/{key}_cumulative'] = value
        metrics['tokens/large_model/accounting'] = 'provider_api_usage'
        append_json(output / 'group_metrics.jsonl', metrics)
        print(json.dumps(metrics))
        lib.save_skills(str(output / 'skills_latest.json'))
        (output / 'summary_partial.json').write_text(json.dumps({
            'protocol': PROTOCOL, 'completed_group': group, 'episode_counts': episode_counts,
            'skill_fingerprint': skill_fingerprint(lib), 'resume_supported': False,
            'skills_checkpoint': 'skills_latest.json',
        }, indent=2), encoding='utf-8')
        if validation and args.mode == 'train' and (group % args.validation_every == 0 or offset + len(rows) >= len(examples) or group == args.groups):
            append_json(output / 'validation_metrics.jsonl', execute(validation, 'validation', group))
    (output / 'completed.json').write_text(json.dumps({'status': 'complete', 'mode': args.mode}), encoding='utf-8')


if __name__ == '__main__':
    main()
