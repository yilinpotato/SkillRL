"""Raw trajectory persistence must retain nested numerical structure."""
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


def test_numpy_raw_trace_roundtrip():
    source = Path(__file__).resolve().parents[3] / 'agent_system/memory/traces_pool.py'
    spec = importlib.util.spec_from_file_location('trace_json_test_module', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = {'steps': [{'observation': np.array([[1, 2], [3, 4]]),
                        'reward': np.float32(0.5), 'valid': np.bool_(True)}]}
    with tempfile.TemporaryDirectory() as directory:
        pool = module.TracesPool.__new__(module.TracesPool)
        pool._raw_log_path = str(Path(directory) / 'raw.jsonl')
        pool._append_raw_log(trace)
        pool._append_raw_log({'empty': np.array([])})
        rows = [json.loads(line) for line in Path(pool._raw_log_path).read_text().splitlines()]
    assert rows == [{'steps': [{'observation': [[1, 2], [3, 4]], 'reward': 0.5, 'valid': True}]}, {'empty': []}]
    assert isinstance(trace['steps'][0]['observation'], np.ndarray)
    try:
        json.dumps({'unsupported': object()}, default=module._raw_json_default)
    except TypeError:
        pass
    else:
        raise AssertionError('Unknown objects must not be silently stringified')


if __name__ == '__main__':
    test_numpy_raw_trace_roundtrip()
    print('RAW_TRACE_ROUNDTRIP_OK')
