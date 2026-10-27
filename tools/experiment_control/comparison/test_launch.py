import argparse
import json
from pathlib import Path
import os

import pandas as pd
import pytest
from launch import build, schedule


def test_schedule_wraps_without_losing_last_task():
    frame = pd.DataFrame({'id': range(5)})
    result = schedule(frame, 12, 0)
    assert len(result) == 12
    assert set(result.id[:5]) == set(range(5))
    assert list(result.id[:5]) == list(result.id[5:10])


@pytest.mark.parametrize('group,method', [('norl_primary', 'skill0'), ('norl_primary', 'skillrl'),
    ('norl_primary', 'coskill_norl'), ('all_rl', 'skill0'), ('all_rl', 'skillrl'), ('all_rl', 'coskill_tree_rl')])
@pytest.mark.parametrize('backend', ['tavily', 'search_r1'])
def test_native_resolved_config(tmp_path, group, method, backend):
    repo = os.environ.get('CONTROLLED_' + ('COSKILL' if method.startswith('coskill') else method.upper()))
    if not repo:
        pytest.skip('Set CONTROLLED_SKILL0/SKILLRL/COSKILL to test actual repository configs')
    model = tmp_path/'model'; model.mkdir()
    for name in ('sft_provenance.json', 'tokenizer_config.json'):
        (model/name).write_text('{"model_id":"fixture-search-sft"}')
    skills = tmp_path/'skills.json'
    skills.write_text(json.dumps({'general_skills': [{'id': 'fixture', 'content': 'Test only'}]}))
    for split in ('train', 'dev'):
        pd.DataFrame({'env_kwargs': [{'question': f'{split}-{i}', 'answers': ['A'], 'split': split}
                                    for i in range(40)]}).to_parquet(tmp_path/f'{split}.parquet')
    args = argparse.Namespace(repo=repo, group=group, method=method, benchmark='searchqa_live',
        prompt_tokens=16000, output_root=str(tmp_path/'out'), run_id='test', model=str(model),
        skills=str(skills), train_data=str(tmp_path/'train.parquet'), dev_data=str(tmp_path/'dev.parquet'),
        allow_empty_skill_ablation=False, smoke_updates=0, smoke_tasks=0, smoke_repeats=0, smoke_dev=0,
        micro_batch=1, n_gpus=1, gpu_memory_utilization=.4,
        search_backend=backend, retrieval_endpoint='http://127.0.0.1:8001/retrieve',
        retrieval_corpus_id='wiki18-bm25-fixture-revision')
    cfg, report = build(args)
    assert cfg.env.searchqa_live.backend == backend
    assert report['retrieval']['topk'] == 5
    assert report['retrieval']['corpus_id'] == args.retrieval_corpus_id
    if method == 'skill0':
        assert Path(cfg.trainer.trajectory_log_path).name == 'trajectories.json'
    assert report['effective_train_episodes'] == 7200
    assert cfg.algorithm.adv_estimator == 'grpo'
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == .4
    assert cfg.env.history_length == 0 and cfg.env.rollout.n == 6
    assert cfg.trainer.test_freq == 5 and cfg.trainer.total_training_steps == 100
    assert len(pd.read_parquet(cfg.data.train_files)) == 1200
    assert len(pd.read_parquet(cfg.data.val_files)) == 32
    assert cfg.data.truncation == 'error'
    assert cfg.trainer.max_actor_ckpt_to_keep == 1
    assert cfg.actor_rollout_ref.rollout.max_num_batched_tokens >= cfg.actor_rollout_ref.rollout.max_model_len
    if method == 'coskill_tree_rl':
        assert cfg.env.skills_only_memory.enable_tree_rl_internalize
    if method == 'coskill_norl':
        assert report['rl'] is None and report['driver'] == 'controlled_norl'
    with pytest.raises(ValueError, match='never overwritten'):
        build(args)
    args.run_id = 'pilot30'
    args.training_updates = 30
    cfg30, report30 = build(args)
    assert cfg30.trainer.total_training_steps == 30
    assert report30['effective_train_episodes'] == 2160
    assert not report30['smoke'] and cfg30.env.rollout.n == 6
    assert len(pd.read_parquet(cfg30.data.val_files)) == 32
