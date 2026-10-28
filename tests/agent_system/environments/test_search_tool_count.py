"""CPU regression: per-step tool flags match actual provider calls."""
import importlib.util
from pathlib import Path
import sys
import types


def load_session():
    root = Path(__file__).resolve().parents[3] / 'agent_system/environments'
    name = '_search_count_test'
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + '.benchmark_sessions', root / 'benchmark_sessions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SearchSession


class Provider:
    def __init__(self):
        self.calls = 0
    def search(self, query):
        self.calls += 1
        return []  # Empty evidence still counts as a real tool invocation.


def test_search_flags_count_calls_not_cumulative_or_rejected_actions():
    provider = Provider()
    session = load_session()({'question': 'Who?', 'answers': ['X']}, provider, 15, 2)
    actions = ['search[a]', 'search[b]', 'search[c]', 'open[url]', 'search[]', 'answer[X]']
    flags = [session.step('<action>' + a + '</action>')[2]['tool_calling'] for a in actions]
    assert flags == [True, True, False, False, False, False]
    assert sum(flags) == provider.calls == 2
    assert session.step('<action>search[after done]</action>')[2]['tool_calling'] is False


def test_last_decision_search_counts_once_then_done_is_zero():
    session = load_session()({'question': 'Who?', 'answers': ['X']}, Provider(), 1, 8)
    _, done, info = session.step('<action>search[a]</action>')
    assert done and info['tool_calling']
    assert not session.step('<action>search[b]</action>')[2]['tool_calling']


def test_malformed_and_reasoning_only_actions_do_not_count():
    session = load_session()({'question': 'Who?', 'answers': ['X']}, Provider(), 15, 8)
    for response in ['<action>search[x]', '<think><action>search[x]</action>', '<action>search[x]</action><action>search[y]</action>']:
        assert not session.step(response)[2]['tool_calling']
