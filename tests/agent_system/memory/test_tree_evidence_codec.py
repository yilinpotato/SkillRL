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
            {"action": suffix, "observation": "state after decision", "reward": 1 if outcome == "success" else 0},
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
    assert codec["version"] == 2
    assert codec["actions"] == [
        "go to cabinet #",
        "open cabinet #",
        "take mug #",
        "go to drawer #",
    ]
    assert len(codec["records"]) == 16
    assert all(len(node) == 4 for node in codec["nodes"])
    assert all(isinstance(node[1], int) for node in codec["nodes"])
    assert all(isinstance(record["q"], list) for record in codec["records"])
    serialized = str(codec)
    assert "children" not in serialized
    assert "example_actions" not in serialized
    assert "n_variants" not in serialized


def test_cloud_renderer_uses_action_vocabulary_and_paths_not_repeated_actions():
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

    assert "ACTION KEY" in rendered
    assert "actions: A" in rendered
    assert "s1 A1" in rendered
    assert "[ref=success]" in rendered
    assert "nonzero_rewards: s3=1" in rendered
    assert "Unlisted rewards are 0" in rendered
    assert "action: go to cabinet" not in rendered
    assert rendered.count("go to cabinet #") == 1


def test_cloud_renderer_stays_smaller_when_trie_has_many_unique_nodes():
    pool = TracesPool(min_samples=1, enable_prefix_tree=True)
    for index in range(20):
        steps = []
        for depth in range(12):
            # The first five decisions encode the trace index, producing many
            # unique prefix nodes from only two normalized action types.  This
            # mirrors the real ALFWorld failure mode (882 nodes, 46 actions).
            receptacle = "cabinet" if (index >> (depth % 5)) & 1 else "drawer"
            steps.append({
                "action": f"go to {receptacle} {index + depth + 1}",
                "observation": f"state {index}-{depth}",
            })
        pool.add_trace({
            "traj_uid": f"trace-{index}",
            "task_type": "clean",
            "task": "clean the object",
            "outcome": "success" if index % 2 else "failure",
            "steps": steps,
        })
    batch = pool.export_batch(trigger_reason="test")

    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.evidence_render_limits = {
        "steps_per_trace": 120,
        "observation_chars_per_step": 4000,
    }
    traces = batch["success_samples"] + batch["failure_samples"]
    compact = analyzer._format_difftraces(
        traces,
        limit=len(traces),
        consensus=[],
        tree_evidence=batch["tree_evidence"],
    )
    flat = analyzer._format_difftraces_flat(traces, consensus=[])

    assert len(batch["tree_evidence"]["nodes"]) > 100
    assert len(batch["tree_evidence"]["actions"]) == 2
    assert len(compact) < len(flat)
    assert compact.count("go to cabinet #") == 1


def test_cloud_renderer_remains_compatible_with_v1_codec():
    pool = TracesPool(min_samples=1, enable_prefix_tree=True)
    pool.add_trace(_trace("success", "success", "take mug 1"))
    batch = pool.export_batch(trigger_reason="test")
    codec_v2 = batch["tree_evidence"]
    codec_v1 = {
        "version": 1,
        "nodes": [
            [node[0], codec_v2["actions"][node[1] - 1], node[2], node[3]]
            for node in codec_v2["nodes"]
        ],
        "records": codec_v2["records"],
    }

    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.evidence_render_limits = {
        "steps_per_trace": 120,
        "observation_chars_per_step": 4000,
    }
    rendered = analyzer._format_difftraces(
        batch["success_samples"],
        limit=1,
        consensus=[],
        tree_evidence=codec_v1,
    )

    assert "ACTION KEY" in rendered
    assert "A1=go to cabinet #" in rendered
