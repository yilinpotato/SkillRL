from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import get_effective_train_batch_size


def _config(*, actor_rollout_n: int, env_rollout_n: int):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 12},
            "actor_rollout_ref": {"rollout": {"n": actor_rollout_n}},
            "env": {"rollout": {"n": env_rollout_n}},
        }
    )


def test_multiturn_grpo_uses_env_expansion_for_eight_gpu_divisibility():
    # CoSkill keeps actor_rollout_ref.rollout.n=1 and lets the environment
    # expand each of 12 goals into six on-policy GRPO trajectories.
    assert get_effective_train_batch_size(_config(actor_rollout_n=1, env_rollout_n=6)) == 72
    assert get_effective_train_batch_size(_config(actor_rollout_n=1, env_rollout_n=6)) % 8 == 0


def test_standard_verl_actor_rollout_expansion_is_unchanged():
    assert get_effective_train_batch_size(_config(actor_rollout_n=4, env_rollout_n=1)) == 48
