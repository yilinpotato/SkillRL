from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "examples/grpo_trainer/run_coskill_tree_rl.sh"
EIGHT_GPU_LAUNCHER = (
    PROJECT_ROOT / "examples/grpo_trainer/run_coskill_tree_rl_8xa800.sh"
)


def test_run_config_dump_is_safe_under_nounset_for_optional_settings():
    """Optional exports must not abort the launcher while writing run_config.env."""
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'echo "$key=${!key-}"' in text
    assert 'echo "$key=${!key}"' not in text


def test_formal_launcher_rejects_silent_per_trajectory_padding_for_grpo():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "for candidate in 4 3 2 1" in text
    assert "BATCH_ADJUST_DIVISOR" in text
    assert "alter GRPO group weights" in text


def test_eight_gpu_throughput_profile_is_explicit_and_reversible():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'TREE_RL_8GPU_THROUGHPUT_MODE="${TREE_RL_8GPU_THROUGHPUT_MODE:-1}"' in text
    assert "DEFAULT_TRAIN_DATA_SIZE=16" in text
    assert "DEFAULT_TRAIN_ROLLOUT_BUDGET=7200" in text
    assert "DEFAULT_TOTAL_TRAINING_STEPS=$((DEFAULT_TRAIN_ROLLOUT_BUDGET / ROLLOUTS_PER_STEP))" in text
    assert "DEFAULT_TEST_FREQ=10" in text
    assert "TRAINING_ROLLOUTS_TOTAL=$((TOTAL_TRAINING_STEPS * ROLLOUTS_PER_STEP))" in text
    assert "DEFAULT_PPO_MINI_BATCH=48" in text
    assert "DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS=32768" in text
    assert "not a 72-rollout learning-curve replicate" in text


def test_exact_72_rollout_profile_remains_available():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'TREE_RL_8GPU_THROUGHPUT_MODE" == "1"' in text
    assert "DEFAULT_TRAIN_DATA_SIZE=12" in text
    assert "DEFAULT_PPO_MINI_BATCH=72" in text


def test_eight_gpu_entrypoint_selects_consistent_throughput_geometry():
    text = EIGHT_GPU_LAUNCHER.read_text(encoding="utf-8")
    assert 'TREE_RL_8GPU_THROUGHPUT_MODE:-1' in text
    assert 'TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"' in text
    assert 'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-75}"' in text
    assert 'TEST_FREQ="${TEST_FREQ:-10}"' in text
    assert 'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-48}"' in text
    assert 'PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"' in text
    assert 'LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-4}"' in text
    assert 'VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"' in text
