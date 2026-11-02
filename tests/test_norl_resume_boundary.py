import importlib.util
import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, deque

def test_restore_boundary():
    p = Path(__file__).parents[1] / 'examples/playbook_evolve/resume_norl.py'
    s = importlib.util.spec_from_file_location('resume_test', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); raw = b'{"general_skills":[{"lifecycle":{"protected":false,"layer":"L0"}}],"skill_trees":{"x":{"content":"retained tree","version":12}}}'
        (root/'skills.json').write_bytes(raw)
        state = dict(pool_empty=True, skills_path=str(root/'skills.json'), skills_sha256=hashlib.sha256(raw).hexdigest(), cycle=16, update=16, train_episodes=1152, recent_outcomes={'x':['success','failure']})
        (root/'state.json').write_text(json.dumps(state))
        memory = SimpleNamespace(skills={}); pool = SimpleNamespace(_recent_outcomes=defaultdict(lambda:deque(maxlen=2)))
        assert m.restore_boundary(root/'state.json',memory,pool)==16
        assert memory.skills==json.loads(raw) and memory.cycle==16
        assert memory.task_playbooks['x']['version']==12
        assert memory.task_playbooks['x'] is memory.skills['skill_trees']['x']
        assert list(pool._recent_outcomes['x'])==['success','failure']
        (root/'skills.json').write_text('{}')
        try: m.restore_boundary(root/'state.json',memory,pool)
        except ValueError: pass
        else: raise AssertionError('checksum must fail')
