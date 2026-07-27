from pathlib import Path

from verl.trainer.main_ppo import _quiet_training_logs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "examples/grpo_trainer/run_coskill_tree_rl.sh"
OVERLAY = PROJECT_ROOT / "docker/coskill/Dockerfile.tree-rl-runtime-env-overlay"


def test_tree_rl_launcher_enables_quiet_logs_by_default():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'export COSKILL_QUIET_LOGS="${COSKILL_QUIET_LOGS:-1}"' in text
    assert 'export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"' in text
    assert 'export TQDM_DISABLE="${TQDM_DISABLE:-1}"' in text


def test_quiet_logs_environment_switch(monkeypatch):
    monkeypatch.setenv("COSKILL_QUIET_LOGS", "1")
    assert _quiet_training_logs() is True
    monkeypatch.setenv("COSKILL_QUIET_LOGS", "0")
    assert _quiet_training_logs() is False


def test_overlay_copies_every_runtime_file_with_quiet_log_gates():
    text = OVERLAY.read_text(encoding="utf-8")
    for source in (
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "verl/trainer/main_ppo.py",
        "verl/trainer/ppo/ray_trainer.py",
        "verl/workers/fsdp_workers.py",
        "verl/workers/rollout/vllm_rollout/vllm_rollout.py",
        "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    ):
        assert f"COPY {source} /workspace/CoSkill/{source}" in text
