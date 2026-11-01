"""Restore an audited, empty-pool noRL boundary without reclassifying skills."""
import hashlib
import json
from pathlib import Path


def restore_boundary(path, memory, pool):
    state = json.loads(Path(path).read_text(encoding='utf-8'))
    if not state['pool_empty']:
        raise ValueError('Only an audited exported-pool boundary is supported')
    raw = Path(state['skills_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != state['skills_sha256']:
        raise ValueError('Resume skills checksum mismatch')
    # Constructor treats all loaded entries as seeds. Replace its modified copy.
    memory.skills = json.loads(raw)
    memory.cycle = state['cycle']
    memory._skill_embeddings_cache = None
    for task_type, outcomes in state['recent_outcomes'].items():
        pool._recent_outcomes[task_type].extend(outcomes)
    pool._total_added = state['train_episodes']
    pool._total_dropped_loops = state.get('total_dropped_loops', 0)
    return state['update']
