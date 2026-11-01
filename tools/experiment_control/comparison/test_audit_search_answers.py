import importlib.util
from pathlib import Path

from audit_search_answers import prediction, question

spec = importlib.util.spec_from_file_location('protocol', Path(__file__).parents[3] / 'agent_system/environments/benchmark_protocol.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_invalid_answer_is_not_scored():
    trace = {'steps': [dict(is_action_valid=False, protocol_response='<action>answer[gold]</action>'), dict(is_action_valid=True, protocol_response='<action>search[gold]</action>')]}
    assert prediction(trace, protocol.project_action) == ('', 1)


def test_first_executed_answer_and_alias_scoring():
    trace = {'steps': [dict(is_action_valid=True, protocol_response='<action>answer[The Cat]</action>'), dict(is_action_valid=True, protocol_response='<action>answer[dog]</action>')]}
    answer, searches = prediction(trace, protocol.project_action)
    assert searches == 0
    assert protocol.score_answer(answer, ['cat', 'feline']) == {'exact_match': 1., 'f1': 1.}
    assert protocol.score_answer('cat dog', ['cat'])['f1'] == 2/3


def test_question_ignores_skill_prompt():
    assert question({'steps': [{'observation': 'Skills\nQuestion: Who?\nMaximum actions: 15; searches: 8'}]}) == 'Who?'
