from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "examples/grpo_trainer/run_coskill_tree_rl.sh"


def test_run_config_dump_is_safe_under_nounset_for_optional_settings():
    """Optional exports must not abort the launcher while writing run_config.env."""
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'echo "$key=${!key-}"' in text
    assert 'echo "$key=${!key}"' not in text
