import ast
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]

def method(relative, cls, name, globals):
    module=ast.parse((ROOT/relative).read_text())
    node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=ast.Module(body=[fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(relative),'exec'),globals)
    return globals[name]


def test_webshop_explicit_goals_and_legacy_reset():
    calls=[]
    reset=method('agent_system/environments/env_package/webshop/envs.py','WebshopMultiProcessEnv','reset',{'np':np,'ray':NS(get=lambda x:x)})
    workers=[NS(reset=NS(remote=lambda index: (str(index),{}))) for _ in range(2)]
    env=NS(_workers=workers,goal_idxs=range(500),env_num=1,group_n=2,_rng=np.random.RandomState(0))
    obs,_=reset(env,[{'goal_index':7},{'goal_index':7}])
    assert obs==['7','7']
    with pytest.raises(ValueError): reset(env,[{'goal_index':500}]*2)
    obs,_=reset(env)
    assert len(obs)==2 and obs[0]==obs[1]


def test_alfworld_worker_explicit_seed_only():
    calls=[]
    reset=method('agent_system/environments/env_package/alfworld/envs.py','AlfworldWorker','reset',{})
    underlying=NS(seed=calls.append,reset=lambda: (['room'],{}))
    env=NS(env=underlying,task_type=None)
    reset(env)
    assert calls==[]
    reset(env,123)
    assert calls==[123]


def test_benchmark_skill_attribution_does_not_use_household_keywords():
    from agent_system.memory.skills_only_memory import SkillsOnlyMemory
    memory = SkillsOnlyMemory.__new__(SkillsOnlyMemory)
    memory.controlled_benchmark = 'searchqa_live'
    assert memory._detect_task_type('Who wrote Two Cities?') == 'searchqa'
    memory.controlled_benchmark = 'scienceworld'
    memory.benchmark_task_types = {'Heat water': 'boil'}
    assert memory._detect_task_type('Task: Heat water') == 'boil'
