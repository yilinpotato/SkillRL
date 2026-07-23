"""Static contract tests for the no-RL cloud preflight launcher."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "examples/playbook_evolve/run_alfworld_playbook_evolve_norl.sh"
CONTAINER_ENTRYPOINT = PROJECT_ROOT / "docker/alfworld-ablation/entrypoint.sh"


def test_norl_launcher_checks_cloud_before_cuda_allocation() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    preflight = text.index("Checking cloud API before CUDA/vLLM allocation")
    gpu_selection = text.index("# GPU selection.")
    assert preflight < gpu_selection
    assert 'DEEPSEEK_API_KEY is required before a CoSkill no-RL rollout.' in text
    assert 'python3 "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py"' in text
    assert 'CLOUD_BOOTSTRAP_PROBE="${CLOUD_BOOTSTRAP_PROBE:-1}"' in text
    assert 'COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE:-0' in text


def test_norl_launcher_preflight_uses_the_effective_skills_json_override() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'CLOUD_SKILLS_JSON="memory_data/alfworld/claude_style_skills.json"' in text
    assert '--skills_json)' in text
    assert '--skills_json=*)' in text
    assert '--skills-json "$CLOUD_SKILLS_JSON"' in text


def test_trace_container_checks_cloud_before_model_or_gpu_setup() -> None:
    text = CONTAINER_ENTRYPOINT.read_text(encoding="utf-8")

    preflight = text.index("Checking cloud API before container model/data/GPU setup")
    model_setup = text.index('if [[ -f "$BAKED_MODEL_PATH/config.json" ]]')
    gpu_setup = text.index('GPU_COUNT="$(detect_gpu_count)"')
    assert preflight < model_setup < gpu_setup
    assert 'export COSKILL_INTERNAL_CLOUD_PREFLIGHT_DONE=1' in text
