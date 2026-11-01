import importlib.util
import json
import tempfile
from pathlib import Path

def test_guard_latches_and_redacts():
    path = Path(__file__).parents[1] / 'examples/playbook_evolve/cloud_api_guard.py'
    spec = importlib.util.spec_from_file_location('guard_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for code in (401, 402, 429, 500):
        calls = []
        class ProviderError(Exception):
            status_code = code
        def fail(**kwargs):
            calls.append(1)
            raise ProviderError('SECRET response body')
        with tempfile.TemporaryDirectory() as output:
            guard = module.CloudAPIGuard(fail, output)
            for _ in range(2):
                try:
                    guard(model='test')
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('must fail')
            assert len(calls) == 1
            text = (Path(output) / 'cloud_api_failure.json').read_text()
            assert 'SECRET' not in text
            assert json.loads(text)['http_status'] == code
            assert module.CloudAPIGuard(lambda **kw: 'ok', output)() == 'ok'
