from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.memory.traces_pool import TracesPool


def _trace(uid, outcome, suffix):
    return {
        "traj_uid": uid,
        "task_type": "clean",
        "task": "clean the object",
        "outcome": outcome,
        "steps": [
            {"action": "go to cabinet 1", "observation": "cabinet is closed"},
            {"action": "open cabinet 1", "observation": "you see a mug"},
            {"action": suffix, "observation": "state after decision"},
        ],
    }


def test_tree_evidence_codec_replaces_nested_tree_with_node_paths():
    pool = TracesPool(min_samples=1, enable_prefix_tree=True)
    for index in range(8):
        pool.add_trace(_trace(f"success-{index}", "success", "take mug 1"))
        pool.add_trace(_trace(f"failure-{index}", "failure", "go to drawer 1"))

    batch = pool.export_batch(trigger_reason="test")
    codec = batch["tree_evidence"]

    assert "prefix_tree" not in batch
    assert codec["version"] == 1
    assert len(codec["records"]) == 16
    assert all(len(node) == 4 for node in codec["nodes"])
    assert all(isinstance(record["q"], list) for record in codec["records"])
    serialized = str(codec)
    assert "children" not in serialized
    assert "example_actions" not in serialized
    assert "n_variants" not in serialized


def test_cloud_renderer_uses_node_table_and_paths_not_repeated_actions():
    pool = TracesPool(min_samples=1, enable_prefix_tree=True)
    pool.add_trace(_trace("success", "success", "take mug 1"))
    pool.add_trace(_trace("failure", "failure", "go to drawer 1"))
    batch = pool.export_batch(trigger_reason="test")

    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.evidence_render_limits = {
        "steps_per_trace": 120,
        "observation_chars_per_step": 4000,
    }
    rendered = analyzer._format_difftraces(
        batch["success_samples"] + batch["failure_samples"],
        limit=2,
        consensus=[],
        tree_evidence=batch["tree_evidence"],
    )

    assert "ACTION NODE TABLE" in rendered
    assert "path: N" in rendered
    assert "N1 |" in rendered
    assert "action: go to cabinet" not in rendered
