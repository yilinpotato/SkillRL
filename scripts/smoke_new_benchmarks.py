"""Real CPU environment smoke, no model, no RL update, no GPU. Search costs one API call."""
import argparse
import json
from omegaconf import OmegaConf
from agent_system.environments.env_manager import make_envs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['searchqa_live', 'scienceworld'], required=True)
    args = p.parse_args()
    cfg = OmegaConf.create({'env': {'env_name': args.benchmark, 'max_steps': 15 if args.benchmark == 'searchqa_live' else 100,
                                    'history_length': 0, 'rollout': {'n': 1}, 'resources_per_worker': {'num_cpus': 1}},
                            'data': {'truncation': 'error'}})
    train, evaluation = make_envs(cfg)
    try:
        if args.benchmark == 'scienceworld':
            from scienceworld import ScienceWorldEnv
            probe = ScienceWorldEnv('', envStepLimit=100)
            try:
                task = probe.get_task_names()[0]
                probe.load(task, 0, '', generateGoldPath=False)
                row = {'task': task, 'variation': int(probe.get_variations_test()[0]), 'split': 'test'}
            finally:
                probe.close()
            obs, _ = evaluation.reset([row])
            initial = evaluation.sessions[0].current
            assert obs['messages'][0][-1]['content'] == initial
            _, _, _, infos = evaluation.step(['<think>Inspect the room.</think><action>look around</action>'])
            assert infos[0]['executed_action'] == 'look around'
        else:
            row = {'id': 'smoke-only', 'question': 'What is the boiling point of water at standard pressure in Celsius?',
                   'answers': ['100'], 'split': 'test'}
            evaluation.reset([row])
            evaluation.step(['<think>Find a source.</think><action>search[water boiling point standard pressure Celsius]</action>'])
            assert evaluation.search.calls == 1
            assert evaluation.sessions[0].env.urls
            _, rewards, dones, infos = evaluation.step(['<think>Submit the test answer.</think><action>answer[100]</action>'])
            assert rewards[0] == 1 and dones[0]
        print(json.dumps({'benchmark': args.benchmark, 'environment_smoke': 'passed', 'model_called': False,
                          'benchmark_score_claim': False, 'info': infos[0]}))
    finally:
        train.close(); evaluation.close()


if __name__ == '__main__':
    main()
