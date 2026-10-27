"""Finite gate: require two real optimizer steps and a free assigned GPU before pilot."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--method', choices=['skill0', 'skillrl'], required=True)
p.add_argument('--smoke-id', required=True)
p.add_argument('--pilot-id', required=True)
a = p.parse_args()
lock = (a.root/(a.method+'-continuation.lock')).open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
run = a.root/'results/norl_primary/searchqa_live'/a.method/a.smoke_id
deadline = time.monotonic() + 7200
while time.monotonic() < deadline:
    status = json.loads((a.root/(a.method+'-rl.status.json')).read_text())
    if status['run_id'] != a.smoke_id:
        raise RuntimeError('Another run changed status; require manual review')
    if status['status'] == 'failed':
        raise RuntimeError('Smoke failed; do not launch formal pilot')
    if status['status'] == 'completed':
        metrics = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines() if s.strip()]
        trained = [m for m in metrics if any(k.startswith('actor/') for k in m)]
        if not trained or max(m['step'] for m in trained) < 2:
            raise RuntimeError('No evidence of two optimizer steps')
        # Mechanical updates alone do not prove a usable environment loop.
        # Check integration, never gate continuation on benchmark success score.
        records = [json.loads(line) for path in (run/'trajectories').glob('*.jsonl')
                   for line in path.read_text().splitlines() if line.strip()]
        if (run/'trajectories.json').exists():
            records.extend(json.loads((run/'trajectories.json').read_text()))
        steps = [step for record in records for step in record.get('steps', [])]
        if not any(step.get('is_action_valid') for step in steps):
            raise RuntimeError('No executed valid action in smoke trajectories')
        for step in steps:
            response = step.get('model_response', step.get('raw_response', ''))
            if 'Choose one admissible action.' in response and response.count('<action>') > 1:
                raise RuntimeError('Duplicate forced action remains; do not launch pilot')
        searches = [json.loads(line) for path in (run/'search_audit').glob('*.jsonl')
                    for line in path.read_text().splitlines() if line.strip()]
        if not any(isinstance(row.get('response'), dict) and row['response'].get('results') for row in searches):
            raise RuntimeError('No real successful search response in smoke')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if not active:
            break
    time.sleep(30)
else:
    raise TimeoutError('Smoke gate timed out; review run before continuation')
print('Two optimizer steps verified; launching independent 30-update pilot', flush=True)
retrieval = json.loads((run/'comparison.json').read_text()).get('retrieval')
if not retrieval:
    raise RuntimeError('Smoke has no recorded retrieval backend; do not infer a paid default')
subprocess.run([sys.executable, str(Path(__file__).with_name('run_search_rl.py')),
    '--root', str(a.root), '--method', a.method, '--run-id', a.pilot_id,
    '--search-backend', retrieval['backend'], '--retrieval-endpoint', retrieval.get('endpoint', ''),
    '--retrieval-corpus-id', retrieval.get('corpus_id', '')], check=True)
