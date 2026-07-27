from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_parallel_launcher_uses_disjoint_single_gpu_arm_workers():
    launcher = (
        PROJECT_ROOT
        / "examples"
        / "playbook_evolve"
        / "run_alfworld_skill_tree_depth_v4_extend_validation_2x1a800.sh"
    ).read_text()
    assert "skill_level_l0,skill_level_l2,skill_level_l4" in launcher
    assert "skill_level_l1,skill_level_l3,skill_level_l5" in launcher
    assert "DATA_PARALLEL_WORKERS=1" in launcher
    assert "--phase prepare" in launcher
    assert "--phase evaluate" in launcher
    assert "--phase summary" in launcher
    assert "--resume 1" in launcher
    assert "V4_EXTENSION_CUDA_VISIBLE_DEVICES" in launcher
