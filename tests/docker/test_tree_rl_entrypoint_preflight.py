from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "docker/coskill/entrypoint.sh"
OVERLAY = PROJECT_ROOT / "docker/coskill/Dockerfile.tree-rl-runtime-env-overlay"


def test_preflight_performs_the_same_mandatory_real_cloud_probe_as_training():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    preflight_block = text.split("    preflight)", 1)[1].split("    alfworld-root", 1)[0]
    assert 'cloud_preflight "$PREFLIGHT_BENCHMARK"' in preflight_block
    assert "CLOUD_BOOTSTRAP_PROBE:-0" not in preflight_block


def test_overlay_copies_the_coherent_tree_evidence_memory_protocol():
    """Do not pair a new trainer with the base image's old TracesPool API."""
    text = OVERLAY.read_text(encoding="utf-8")
    assert "COPY agent_system/memory/ /workspace/CoSkill/agent_system/memory/" in text
    assert (
        "COPY scripts/preflight_tree_rl_runtime.py "
        "/workspace/CoSkill/scripts/preflight_tree_rl_runtime.py"
    ) in text
    assert "python /workspace/CoSkill/scripts/preflight_tree_rl_runtime.py" in text
