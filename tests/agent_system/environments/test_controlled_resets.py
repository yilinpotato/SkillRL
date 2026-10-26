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


def test_webshop_manager_forwards_manifest_only_in_controlled_mode():
    from typing import Dict, Any
    reset = method('agent_system/environments/env_manager.py', 'WebshopEnvironmentManager',
                   'reset', {'Dict': Dict, 'Any': Any})
    calls = []
    def underlying(**kwargs):
        calls.append(kwargs)
        return ['room'], [{}]
    env = NS(envs=NS(reset=underlying), config=NS(env={'controlled_comparison': True}),
             extract_task=lambda obs: ['task'], format_obs=lambda obs: obs,
             retrieval_memory=None, build_text_obs=lambda *args, **kwargs: ['prompt'],
             memory=NS(reset=lambda **kwargs: None))
    rows = [{'goal_index': 7}]
    reset(env, rows)
    assert calls[-1] == {'controlled_rows': rows}
    env.config.env['controlled_comparison'] = False
    reset(env, rows)
    assert calls[-1] == {}
