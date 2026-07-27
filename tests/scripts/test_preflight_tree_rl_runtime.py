from scripts.preflight_tree_rl_runtime import (
    _check_signatures,
    _check_trainer_ingest_contract,
    _check_tree_codec,
)


def test_tree_rl_cross_module_signatures_are_compatible():
    _check_signatures()


def test_tree_only_codec_projects_without_flat_steps():
    _check_tree_codec()


def test_trainer_can_create_and_feed_tree_only_traces_pool():
    _check_trainer_ingest_contract()
