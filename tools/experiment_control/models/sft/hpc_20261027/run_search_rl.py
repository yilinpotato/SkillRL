"""Run one controlled SearchQA pilot/smoke in its assigned HPC container."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--method', choices=['skill0', 'skillrl'], required=True)
p.add_argument('--run-id', required=True)
p.add_argument('--smoke', action='store_true')
p.add_argument('--prepare-only', action='store_true')
p.add_argument('--search-backend', choices=['tavily', 'search_r1'], default='search_r1')
p.add_argument('--retrieval-endpoint', default='')
p.add_argument('--retrieval-corpus-id', default='')
a = p.parse_args()
if a.search_backend == 'search_r1' and (not a.retrieval_endpoint or not a.retrieval_corpus_id):
    p.error('Specify verified Search-R1 endpoint and pinned corpus/index identity')
root = a.root
lock = (root/(a.method+'-rl.lock')).open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = root/(a.method+'-rl.status.json')
state = {'method': a.method, 'run_id': a.run_id, 'smoke': a.smoke, 'status': 'starting'}
def save():
    state['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2)); tmp.replace(state_path)
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
           PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1',
           RESOURCE_DIAGNOSTICS_DIR=str(root/'results'/f'{a.method}-{a.run_id}-diagnostics'),
           TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
           DEEPSEEK_MODEL='deepseek-v4-flash', DEEPSEEK_API_BASE='https://api.deepseek.com/v1',
           SKILL_UPDATER_BACKEND='deepseek')
cmd = [sys.executable, str(Path(__file__).resolve().parents[3]/'comparison/launch.py'),
       '--repo', str(root/'code'/a.method), '--group', 'norl_primary', '--method', a.method,
       '--benchmark', 'searchqa_live', '--model', str(root/'releases/CoSkill-Qwen3-4B-Search-SFT-v2'),
       '--skills', str(Path(__file__).resolve().parents[3]/'comparison/initial_skills/searchqa_live.json'),
       '--train-data', str(root/'data/searchqa-native-20261024/train.parquet'),
       '--dev-data', str(root/'data/searchqa-native-20261024/dev.parquet'),
       '--output-root', str(root/'results'), '--run-id', a.run_id, '--prompt-tokens', '16384',
       '--search-backend', a.search_backend, '--retrieval-endpoint', a.retrieval_endpoint,
       '--retrieval-corpus-id', a.retrieval_corpus_id]
if a.smoke:
    cmd += ['--smoke-updates', '2', '--smoke-tasks', '2', '--smoke-repeats', '2', '--smoke-dev', '2']
else:
    cmd += ['--training-updates', '30']
if not a.prepare_only:
    cmd += ['--execute']
save()
try:
    state['status'] = 'running'; save()
    with (root/'logs'/f'{a.method}-{a.run_id}.log').open('x') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    state['status'] = 'prepared' if a.prepare_only else 'completed'
except Exception as exc:
    state.update(status='failed', error_type=type(exc).__name__)
    raise
finally:
    save()
