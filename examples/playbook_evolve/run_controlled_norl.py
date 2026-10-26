"""Frozen-policy CoSkill using the native environment managers and shared schedule.

This runner never initializes an optimizer or a local inference server. An existing
OpenAI-compatible endpoint must serve the explicitly selected SFT checkpoint.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import urllib.request


def run_batch(env, rows, generate, max_steps, temperature, namespace):
    observations, infos = env.reset(rows)
    count = len(rows)
    done = [False] * count
    traces = [[] for _ in rows]
    usages = [[] for _ in rows]
    memories = copy.deepcopy(env.retrieved_memories or [{} for _ in rows])
    tasks = list(env.tasks)
    rewards = [0.] * count
    for step in range(max_steps):
        active = [i for i in range(count) if not done[i]]
        if not active:
            break
        actions = [''] * count
        for i in active:
            messages = (observations['messages'][i] if observations.get('messages') is not None
                        else [{'role': 'user', 'content': observations['text'][i]}])
            actions[i], usage = generate(messages, temperature)
            usages[i].append(usage)
        following, step_rewards, step_done, step_infos = env.step(actions)
        for i in active:
            info = step_infos[i]
            traces[i].append({'observation': observations['anchor'][i] if observations.get('anchor') is not None
                             else observations['text'][i],
                             'action': info.get('executed_action', actions[i]),
                             'model_response': actions[i], 'step': step + 1})
            rewards[i] += float(step_rewards[i])
            infos[i] = info
            done[i] = bool(step_done[i])
        observations = following
    results = []
    for i in range(count):
        info = infos[i]
        won = bool(info.get('won', info.get('success', False)))
        task_type = env.retrieval_memory._detect_task_type(tasks[i])
        ids = memories[i].get('injected_skill_ids', [])
        results.append({'traj_uid': f'{namespace}/{i}', 'task': tasks[i], 'task_type': task_type,
                        'outcome': 'success' if won else 'failure', 'episode_reward': rewards[i],
                        'task_score': float(info.get('task_score', won)), 'steps': traces[i],
                        'meta': {'skill_ids_used': ids, 'model_version': 'frozen'},
                        'provider_usage': usages[i], 'terminal_info': info})
    return results


def execute(cfg, train, dev, envs, generate, pool, cloud, output):
    """Injectable execution core; evaluation never enters the learning pool."""
    train_env, val_env = envs
    tasks = int(cfg.data.train_batch_size)
    repeats = int(cfg.env.rollout.n)
    updates = int(cfg.trainer.total_training_steps)
    if len(train) != tasks * updates:
        raise ValueError('Scheduled task count must exactly match tasks * updates')
    completed = 0
    output = Path(output)
    def save(kind, update, results):
        with (output / (kind + '.jsonl')).open('a', encoding='utf-8') as handle:
            for result in results:
                handle.write(json.dumps(dict(result, update=update), ensure_ascii=False, default=str) + '\n')
        usage = [u for r in results for u in r['provider_usage']]
        complete = all(u is not None and 'prompt_tokens' in u and 'completion_tokens' in u for u in usage)
        prompt = sum(u['prompt_tokens'] for u in usage) if complete else None
        response = sum(u['completion_tokens'] for u in usage) if complete else None
        metrics = {'step': update, 'split': kind, 'episodes': len(results),
            'success_rate': sum(r['outcome'] == 'success' for r in results)/len(results),
            'task_score': sum(r['task_score'] for r in results)/len(results),
            'tokens/small_model/prompt': prompt, 'tokens/small_model/response': response,
            'tokens/small_model/total': prompt + response if complete else None,
            'token_source': 'provider_usage_active_decisions', 'usage_complete': complete}
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics) + '\n')
    def evaluate(update):
        # Retrieval may update counters on its own memory. Discard that copy after eval.
        val_env.retrieval_memory = copy.deepcopy(train_env.retrieval_memory)
        results = run_batch(val_env, dev, generate, cfg.env.max_steps,
                            cfg.actor_rollout_ref.rollout.val_kwargs.temperature, f'dev/{update}')
        save('dev', update, results)
    if cfg.trainer.val_before_train:
        evaluate(0)
    for update in range(1, updates + 1):
        rows = [row for row in train[(update-1)*tasks:update*tasks] for _ in range(repeats)]
        results = run_batch(train_env, rows, generate, cfg.env.max_steps,
                            cfg.actor_rollout_ref.rollout.temperature, f'train/{update}')
        for result in results:
            pool.add_trace(result)
            memory = train_env.retrieval_memory
            memory.record_usage(result['meta']['skill_ids_used'], result['outcome'] == 'success', result['task_type'])
            memory.record_playbook_usage(result['task_type'], result['outcome'] == 'success')
        completed += len(results)
        save('train', update, results)
        cloud.maybe_update(pool, train_env.retrieval_memory, completed)
        if getattr(cloud, '_analyzer_init_failed', False):
            raise RuntimeError('Cloud evolution initialization failed; this is not a valid CoSkill run')
        train_env.retrieval_memory.save_skills(str(output / 'skills_latest.json'))
        if update % cfg.trainer.test_freq == 0 or update == updates:
            evaluate(update)
        analyzer = getattr(cloud, 'cloud_analyzer', None)
        (output / 'progress.json').write_text(json.dumps({'updates': update, 'train_episodes': completed,
            'cloud_prompt_tokens': getattr(analyzer, 'total_prompt_tokens', None),
            'cloud_completion_tokens': getattr(analyzer, 'total_completion_tokens', None)}), encoding='utf-8')
    return completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from agent_system.environments.env_manager import make_envs
    from agent_system.memory import TracesPool, CoSkillCloudLoop
    cfg = OmegaConf.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    def generate(messages, temperature):
        length = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if length > cfg.data.max_prompt_length:
            raise ValueError(f'Prompt exceeds shared capacity: {length} > {cfg.data.max_prompt_length}')
        payload = {'model': args.model, 'messages': messages, 'temperature': float(temperature),
                   'top_p': float(cfg.actor_rollout_ref.rollout.top_p), 'max_tokens': int(cfg.data.max_response_length)}
        request = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.getenv('EDGE_API_KEY', os.getenv('OPENAI_API_KEY', 'EMPTY'))})
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        message = result['choices'][0]['message']
        content = message.get('content') or ''
        if message.get('reasoning_content') and '<think>' not in content:
            content = '<think>' + message['reasoning_content'] + '</think>\n' + content
        return content, result.get('usage')  # Missing usage stays null, never fabricated as zero.
    settings = cfg.env.skills_only_memory
    output = cfg.trainer.default_local_dir
    pool = TracesPool(output_dir=output, **OmegaConf.to_container(cfg.env.traces_pool, resolve=True))
    cloud = CoSkillCloudLoop(output_dir=output, enable_coskill=True, enable_playbook_evolve=True,
        enable_failure_analysis=True, max_new_skills=settings.max_new_skills,
        playbook_evolve_min_samples=settings.playbook_evolve_min_samples, environment_name=cfg.env.env_name)
    envs = make_envs(cfg)
    try:
        execute(cfg, list(pd.read_parquet(cfg.data.train_files).env_kwargs),
                list(pd.read_parquet(cfg.data.val_files).env_kwargs), envs, generate, pool, cloud, output)
    finally:
        for env in envs:
            close = getattr(env, 'close', None) or getattr(getattr(env, 'envs', None), 'close', None)
            if close:
                close()


if __name__ == '__main__':
    main()
