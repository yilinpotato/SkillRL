"""Render explicit comparison settings. Does not launch training or claim readiness."""
import argparse
import json
from pathlib import Path


def resolve(group: str, method: str, benchmark: str, prompt_tokens: int) -> dict:
    profile = json.loads(Path(__file__).with_name('standard_v1.json').read_text())
    if group not in profile['groups'] or method not in profile['groups'][group]:
        raise ValueError('Method does not belong to this independent comparison group')
    if benchmark not in profile['benchmarks'] or prompt_tokens <= 0:
        raise ValueError('Choose a supported benchmark and explicit shared context capacity')
    common, rl, env = profile['common'], profile['rl'], profile['benchmarks'][benchmark]
    result = dict(version=profile['version'], group=group, method=method, benchmark=benchmark,
                  output_namespace=f'{group}/{benchmark}/{method}', common=common, environment=env,
                  prompt_tokens=prompt_tokens, overflow='error; never silently truncate',
                  blockers=[], training_launched=False)
    reason = profile.get('readiness', {}).get(method, {}).get(benchmark)
    if reason:
        result['blockers'].append(reason)
    result['required_external_evidence'] = ['identical SFT weight/tokenizer hashes within group',
        'identical train/dev/test instance manifests and sampling schedule',
        'same initial skill source; no fabricated benchmark skills', 'frozen final test',
        'resolved runtime config and actual episode/token accounting']
    if method == 'coskill_norl':
        result['rl'] = None
        result['note'] = 'Frozen policy: optimizer, KL and LoRA-update settings are not applicable.'
        return result
    result['rl'] = rl
    overrides = {
        'algorithm.adv_estimator': rl['algorithm'],
        'algorithm.use_kl_in_reward': False,
        'data.train_batch_size': common['tasks_per_update'],
        'data.val_batch_size': common['dev_size'],
        'data.max_prompt_length': prompt_tokens,
        'data.max_response_length': common['response_tokens'],
        'data.truncation': 'error', 'data.filter_overlong_prompts': False,
        'actor_rollout_ref.model.lora_rank': rl['lora_rank'],
        'actor_rollout_ref.model.lora_alpha': rl['lora_alpha'],
        'actor_rollout_ref.model.target_modules': rl['target_modules'],
        'actor_rollout_ref.actor.optim.lr': rl['learning_rate'],
        'actor_rollout_ref.actor.optim.lr_warmup_steps': rl['warmup_steps'],
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio': 0.0,
        'actor_rollout_ref.actor.optim.warmup_style': rl['schedule'],
        'actor_rollout_ref.actor.optim.weight_decay': rl['weight_decay'],
        'actor_rollout_ref.actor.use_kl_loss': True,
        'actor_rollout_ref.actor.kl_loss_coef': rl['kl_coef'],
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.actor.entropy_coeff': rl['entropy_coef'],
        'actor_rollout_ref.actor.use_invalid_action_penalty': True,
        'actor_rollout_ref.actor.invalid_action_penalty_coef': rl['invalid_action_penalty'],
        'actor_rollout_ref.actor.ppo_mini_batch_size': rl['ppo_mini_batch_size'],
        'actor_rollout_ref.actor.ppo_epochs': rl['ppo_epochs'],
        'actor_rollout_ref.actor.clip_ratio': rl['clip_ratio'],
        'actor_rollout_ref.actor.clip_ratio_low': rl['clip_ratio'],
        'actor_rollout_ref.actor.clip_ratio_high': rl['clip_ratio'],
        'actor_rollout_ref.rollout.temperature': common['train_temperature'],
        'actor_rollout_ref.rollout.do_sample': True,
        'actor_rollout_ref.rollout.top_p': common['top_p'],
        'actor_rollout_ref.rollout.top_k': -1,
        'actor_rollout_ref.rollout.val_kwargs.temperature': common['eval_temperature'],
        'actor_rollout_ref.rollout.val_kwargs.do_sample': True,
        'actor_rollout_ref.rollout.val_kwargs.top_p': common['top_p'],
        'actor_rollout_ref.rollout.val_kwargs.top_k': -1,
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'env.seed': common['seed'], 'env.max_steps': env['max_steps'],
        'env.history_length': env['history_length'], 'env.rollout.n': common['rollouts_per_task'],
        'trainer.total_training_steps': common['training_updates'],
        'trainer.total_epochs': common['training_updates'],
        'trainer.test_freq': common['validate_every_episodes'] // (common['tasks_per_update'] * common['rollouts_per_task']),
        'trainer.val_before_train': True,
        'env.skills_only_memory.top_k': common['initial_general_skills'],
    }
    if method == 'skill0' and benchmark == 'alfworld':
        overrides.update({'env.alfworld.use_dense_reward': False,
                          'env.alfworld.train_task_type': None, 'env.alfworld.eval_task_type': None})
    if benchmark == 'webshop':
        overrides.update({'env.webshop.use_small': True, 'env.webshop.human_goals': False})
    # ++ works for both legacy populated keys and new optional configuration nodes.
    result['hydra_overrides'] = ['++' + k + '=' + json.dumps(v, separators=(',', ':')) for k, v in overrides.items()]
    result['note'] = 'Apply after original launcher arguments; preserve method-specific evolution/internalization flags. This is not a complete hardware launcher.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=['norl_primary', 'all_rl'])
    parser.add_argument('--method', required=True, choices=['skill0', 'skillrl', 'coskill_norl', 'coskill_tree_rl'])
    parser.add_argument('--benchmark', required=True, choices=['alfworld', 'webshop', 'searchqa_live', 'scienceworld'])
    parser.add_argument('--prompt-tokens', required=True, type=int)
    parser.add_argument('--require-ready', action='store_true', help='Fail if known driver alignment blockers remain')
    args = parser.parse_args()
    result = resolve(args.group, args.method, args.benchmark, args.prompt_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_ready and result['blockers']:
        raise SystemExit(2)
