"""Execute controlled experiments using repository-native trainers and environment managers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from resolve import resolve


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def method_settings(method: str, benchmark: str) -> dict:
    settings = {'env.use_skills_only_memory': True,
                'env.skills_only_memory.retrieval_mode': 'template',
                'env.skills_only_memory.task_specific_top_k': None}
    if method == 'skill0':
        settings.update({
            'env.skills_only_memory.enable_dynamic_update': False,
            'env.skills_only_memory.global_internalization_mode': 'skillzero',
            'env.skills_only_memory.global_top_k_schedule': [6, 6, 6, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0],
            'env.skills_only_memory.global_skill_top_k': 6,
            'env.skills_only_memory.skillzero_include_task_specific': benchmark == 'alfworld',
            'env.skills_only_memory.onpolicy_helpfulness_eval_enabled': True,
            'env.skills_only_memory.onpolicy_helpfulness_eval_auto_internalize': True,
            'env.skills_only_memory.onpolicy_helpfulness_eval_delta_threshold': .01,
            'env.skills_only_memory.onpolicy_helpfulness_eval_patience': 1,
            'env.skills_only_memory.onpolicy_helpfulness_eval_remove_per_round': 1,
            'env.skills_only_memory.onpolicy_helpfulness_eval_min_active_skills': 0,
            'env.skills_only_memory.eval_internalization_modes': True})
    elif method == 'skillrl':
        settings.update({'env.skills_only_memory.enable_dynamic_update': True,
                         'env.skills_only_memory.update_skills_from_train': True,
                         'env.skills_only_memory.update_threshold': .4,
                         'env.skills_only_memory.max_new_skills': 3})
    else:
        settings.update({f'env.skills_only_memory.{key}': True for key in (
            'enable_dynamic_update', 'update_skills_from_train', 'enable_coskill',
            'enable_hierarchy', 'enable_playbook', 'enable_playbook_evolve', 'enable_failure_analysis')})
        settings.update({'env.skills_only_memory.enable_tree_rl_internalize': method == 'coskill_tree_rl',
                         'env.skills_only_memory.enable_internalize': False,
                         'env.skills_only_memory.skill_update_freq': 1,
                         'env.skills_only_memory.tree_rl_order': 'root',
                         'env.skills_only_memory.tree_rl_min_updates': 5,
                         'env.skills_only_memory.tree_rl_min_train_episodes': 24,
                         'env.skills_only_memory.tree_rl_train_success_threshold': .7,
                         'env.skills_only_memory.tree_rl_min_probe_episodes': 24,
                         'env.skills_only_memory.tree_rl_probe_success_threshold': .7,
                         'env.skills_only_memory.tree_rl_state_save_freq': 1,
                         'env.skills_only_memory.playbook_evolve_min_samples': 6,
                         'env.skills_only_memory.max_new_skills': 3,
                         'env.skills_only_memory.min_calls': 10,
                         'env.traces_pool.min_samples': 16,
                         'env.traces_pool.capacity_watermark': 50000,
                         'env.traces_pool.perf_watermark': .6,
                         'env.traces_pool.cloud_evidence_mode': 'tree_only'})
    return settings


def schedule(frame, count: int, seed: int):
    """One shared deterministic task order, repeated without dropping the tail."""
    import random
    import pandas as pd
    order = list(range(len(frame)))
    if not order or count < 1:
        raise ValueError('Nonempty task data and positive task count required')
    random.Random(seed).shuffle(order)
    return pd.concat([frame.iloc[[order[i % len(order)]]] for i in range(count)], ignore_index=True)


def build(args) -> tuple:
    import pandas as pd
    from omegaconf import OmegaConf
    profile = resolve(args.group, args.method, args.benchmark, args.prompt_tokens)
    root = Path(args.repo).resolve()
    output = Path(args.output_root).resolve() / profile['output_namespace'] / args.run_id
    if output.exists():
        raise ValueError('Use a new run-id; existing runs are never overwritten')
    model = Path(args.model).resolve()
    provenance = [p for p in (model/'searchqa_sft_provenance.json', model/'sft_provenance.json') if p.is_file()]
    if not provenance:
        raise ValueError('Explicit merged SFT checkpoint provenance is required; no base-model fallback')
    model_record = json.loads(provenance[0].read_text(encoding='utf-8'))
    domain = {'alfworld': 'alfworld', 'webshop': 'webshop', 'searchqa_live': 'search', 'scienceworld': 'scienceworld'}[args.benchmark]
    if domain not in str(model_record.get('model_id', model_record.get('benchmark', ''))).lower():
        raise ValueError('SFT provenance does not identify the requested benchmark domain')
    skills = Path(args.skills).resolve()
    skill_data = json.loads(skills.read_text(encoding='utf-8'))
    has_skills = bool(skill_data.get('general_skills') or skill_data.get('common_mistakes') or
                      any(skill_data.get('task_specific_skills', {}).values()))
    if args.method == 'skill0' and not has_skills and not args.allow_empty_skill_ablation:
        raise ValueError('Skill0 needs skills to internalize. For an explicitly labeled ablation use --allow-empty-skill-ablation')
    train = pd.read_parquet(args.train_data)
    dev = pd.read_parquet(args.dev_data)
    if args.benchmark == 'alfworld':
        # Explicit reset seeds are shared across methods and frozen across validations.
        for frame, offset in ((train, 0), (dev, 1000000)):
            frame['env_kwargs'] = [dict(row, controlled_seed=offset+i)
                for i, row in enumerate(frame.get('env_kwargs', [{}]*len(frame)))]
    if args.benchmark == 'webshop':
        for frame, split in ((train, 'train'), (dev, 'dev')):
            if 'env_kwargs' not in frame or any('goal_index' not in r for r in frame.env_kwargs):
                raise ValueError('Controlled WebShop requires explicit goal_index in every env_kwargs row')
            if any((int(r['goal_index']) < 500) != (split == 'dev') for r in frame.env_kwargs):
                raise ValueError('WebShop uses goals 0..499 for held-out validation and >=500 for training')
    if args.benchmark in ('searchqa_live', 'scienceworld'):
        for frame, expected in ((train, 'train'), (dev, 'dev')):
            if any(row['split'] != expected for row in frame['env_kwargs']):
                raise ValueError('Training/evaluation data must declare train/dev respectively')
        def identity(row):
            return row['question'].strip().casefold() if args.benchmark == 'searchqa_live' else (row['task'], int(row['variation']))
        if {identity(r) for r in train.env_kwargs} & {identity(r) for r in dev.env_kwargs}:
            raise ValueError('Train/dev instance overlap')
    common = profile['common']
    requested_updates = getattr(args, 'training_updates', None)
    if requested_updates is not None and (requested_updates < 1 or args.smoke_updates):
        raise ValueError('Positive training-updates is separate from smoke mode')
    updates = args.smoke_updates or requested_updates or common['training_updates']
    tasks = args.smoke_tasks or common['tasks_per_update']
    repeats = args.smoke_repeats or common['rollouts_per_task']
    dev_count = min(len(dev), args.smoke_dev or common['dev_size'])
    if not args.smoke_updates and dev_count != common['dev_size']:
        raise ValueError('Formal comparison requires the complete fixed dev subset')
    scheduled = schedule(train, updates * tasks, common['seed'])
    fixed_dev = schedule(dev, dev_count, common['seed'] + 1000)
    cfg = OmegaConf.load(root/'verl/trainer/config/ppo_trainer.yaml')
    # no-RL uses the same environment/control config, but never invokes PPO.
    rl_plan = resolve(args.group, args.method if args.method != 'coskill_norl' else 'skillrl', args.benchmark, args.prompt_tokens)
    overrides = {item[2:].split('=', 1)[0]: json.loads(item.split('=', 1)[1]) for item in rl_plan['hydra_overrides']}
    overrides.update(method_settings(args.method, args.benchmark))
    overrides.update({'env.env_name': 'alfworld/AlfredTWEnv' if args.benchmark == 'alfworld' else args.benchmark,
                      'actor_rollout_ref.model.path': str(model),
                      'env.skills_only_memory.skills_json_path': str(skills),
                      'env.controlled_comparison': True,
                      'data.train_files': str(output/'scheduled_train.parquet'),
                      'data.val_files': str(output/'fixed_dev.parquet'),
                      'data.shuffle': False, 'data.return_raw_chat': True,
                      'data.train_batch_size': tasks, 'data.val_batch_size': dev_count,
                      'env.rollout.n': repeats,
                      'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': args.micro_batch,
                      'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': args.micro_batch,
                      'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': args.micro_batch,
                      'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
                      'actor_rollout_ref.rollout.gpu_memory_utilization': args.gpu_memory_utilization,
                      'actor_rollout_ref.rollout.max_model_len': args.prompt_tokens + common['response_tokens'],
                      'actor_rollout_ref.rollout.max_num_batched_tokens': args.prompt_tokens + common['response_tokens'],
                      'trainer.n_gpus_per_node': args.n_gpus, 'trainer.nnodes': 1,
                      'trainer.default_local_dir': str(output), 'trainer.logger': ['console', 'jsonl'],
                      'trainer.total_training_steps': updates, 'trainer.total_epochs': 1,
                      'trainer.max_actor_ckpt_to_keep': 1, 'trainer.max_critic_ckpt_to_keep': 1,
                      'trainer.save_freq': 5,
                      'env.searchqa_live.audit_dir': str(output/'search_audit')})
    if args.benchmark == 'searchqa_live':
        backend = getattr(args, 'search_backend', 'tavily')
        endpoint = getattr(args, 'retrieval_endpoint', '')
        corpus_id = getattr(args, 'retrieval_corpus_id', '')
        if backend == 'search_r1' and (not endpoint or not corpus_id):
            raise ValueError('Local search requires endpoint and pinned corpus/index identity')
        overrides.update({'env.searchqa_live.backend': backend,
                          'env.searchqa_live.retrieval_endpoint': endpoint,
                          'env.searchqa_live.retrieval_corpus_id': corpus_id})
        profile['retrieval'] = {'backend': backend, 'endpoint': endpoint,
                                'corpus_id': corpus_id, 'topk': 5}
    if args.method == 'skill0':
        # Skill0 writes a JSON array, unlike SkillRL's per-batch JSONL dumps.
        # Set only the trajectory path to avoid duplicating full prompt contexts.
        overrides['trainer.trajectory_log_path'] = str(output/'trajectories.json')
    if args.smoke_updates:
        overrides['actor_rollout_ref.actor.ppo_mini_batch_size'] = tasks * repeats
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    entry = root / ('examples/playbook_evolve/run_controlled_norl.py'
                    if args.method == 'coskill_norl' else 'verl/trainer/main_ppo.py')
    if not entry.is_file():
        raise RuntimeError(f'Required native entrypoint missing: {entry}')
    output.mkdir(parents=True)
    scheduled.to_parquet(output/'scheduled_train.parquet', index=False)
    fixed_dev.to_parquet(output/'fixed_dev.parquet', index=False)
    OmegaConf.save(cfg, output/'resolved_config.yaml')
    profile.update(output=str(output), checkpoint=str(model), checkpoint_manifest_sha256=sha(model/'release-manifest.json')
                   if (model/'release-manifest.json').is_file() else sha(provenance[0]),
                   tokenizer_sha256=sha(model/'tokenizer_config.json'), initial_skills_sha256=sha(skills),
                   train_sha256=sha(Path(args.train_data)), dev_sha256=sha(Path(args.dev_data)),
                   scheduled_train_sha256=sha(output/'scheduled_train.parquet'), fixed_dev_sha256=sha(output/'fixed_dev.parquet'),
                   effective_train_episodes=updates*tasks*repeats, effective_training_updates=updates,
                   budget_override=requested_updates, smoke=bool(args.smoke_updates),
                   empty_skill_ablation=not has_skills, launch_status='prepared',
                   blockers=[],
                   validation_scope='Prepared configuration; not evidence of a successful training run')
    profile['driver'] = 'controlled_norl' if args.method == 'coskill_norl' else 'native_main_ppo'
    (output/'comparison.json').write_text(json.dumps(profile, indent=2))
    return cfg, profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'group', 'method', 'benchmark', 'model', 'skills', 'train-data', 'dev-data', 'output-root', 'run-id'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--prompt-tokens', type=int, required=True)
    p.add_argument('--n-gpus', type=int, default=1)
    p.add_argument('--training-updates', type=int, help='Explicit pilot budget; leaves task, rollout and dev sizes unchanged')
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--gpu-memory-utilization', type=float, default=.4)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--search-backend', choices=['tavily', 'search_r1'], default='tavily')
    p.add_argument('--retrieval-endpoint', default='')
    p.add_argument('--retrieval-corpus-id', default='')
    p.add_argument('--allow-empty-skill-ablation', action='store_true')
    p.add_argument('--base-url', default=os.getenv('EDGE_API_BASE', 'http://127.0.0.1:8000/v1'))
    p.add_argument('--served-model', default=os.getenv('EDGE_MODEL'))
    for key in ('updates', 'tasks', 'repeats', 'dev'):
        p.add_argument('--smoke-'+key, type=int, default=0)
    args = p.parse_args()
    if any(getattr(args, 'smoke_'+k) for k in ('tasks', 'repeats', 'dev')) and not args.smoke_updates:
        p.error('Reduced budgets require --smoke-updates and are labeled smoke')
    cfg, report = build(args)
    print(json.dumps({'output': report['output'], 'episodes': report['effective_train_episodes'], 'execute': args.execute}))
    if not args.execute:
        return
    out = Path(report['output'])
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', '--config-path', str(out), '--config-name', 'resolved_config']
    if args.method == 'coskill_norl':
        if not args.served_model:
            p.error('no-RL execution requires explicit --served-model matching the SFT checkpoint')
        command = [sys.executable, '-m', 'examples.playbook_evolve.run_controlled_norl', '--config', str(out/'resolved_config.yaml'),
                   '--base-url', args.base_url, '--model', args.served_model]
    report['launch_status'] = 'running'
    (out/'comparison.json').write_text(json.dumps(report, indent=2))
    try:
        process_env = dict(os.environ, JSONL_PATH=str(out/'metrics.jsonl'),
                           JSONL_METRICS_PATH=str(out/'metrics.jsonl'))
        subprocess.run(command, cwd=args.repo, env=process_env, check=True)
    except BaseException:
        report['launch_status'] = 'failed'
        raise
    else:
        report['launch_status'] = 'completed'
    finally:
        (out/'comparison.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
