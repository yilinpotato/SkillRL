"""Wait for this container's finite SFT queue, then run a gated SkillRL pilot."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
lock = (a.root/'skillrl-after-sft.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
deadline = time.monotonic() + 12 * 3600
while time.monotonic() < deadline:
    state = json.loads((a.root/'sft-b.status.json').read_text())
    if state['status'] == 'failed':
        raise RuntimeError('SFT queue failed; review before using GPU')
    if state['status'] == 'completed':
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if not active:
            break
    time.sleep(30)
else:
    raise TimeoutError('SFT queue still incomplete; no RL launched')
scripts = Path(__file__).parent
subprocess.run([sys.executable, str(scripts/'run_search_rl.py'), '--root', str(a.root),
    '--method', 'skillrl', '--run-id', 'smoke01', '--smoke'], check=True)
subprocess.run([sys.executable, str(scripts/'continue_after_smoke.py'), '--root', str(a.root),
    '--method', 'skillrl', '--smoke-id', 'smoke01', '--pilot-id', 'pilot30-01'], check=True)
