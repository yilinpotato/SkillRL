from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "docker/coskill/entrypoint.sh"


def test_preflight_performs_the_same_mandatory_real_cloud_probe_as_training():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    preflight_block = text.split("    preflight)", 1)[1].split("    alfworld-root", 1)[0]
    assert 'cloud_preflight "$PREFLIGHT_BENCHMARK"' in preflight_block
    assert "CLOUD_BOOTSTRAP_PROBE:-0" not in preflight_block
