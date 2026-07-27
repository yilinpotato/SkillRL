import pytest

from agent_system.memory.cloud_analyzer import CloudAnalyzer
from agent_system.memory.coskill_loop import CoSkillCloudLoop
from agent_system.memory.traces_pool import TracesPool, _normalize_action_for_merge


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


def test_webshop_action_normalization_preserves_numeric_purchase_options():
    assert _normalize_action_for_merge("click[size 10]") == "click[size 10]"
    assert _normalize_action_for_merge("click[size 11]") == "click[size 11]"
    assert (
        _normalize_action_for_merge("search[iphone 15 case]")
        == "search[iphone 15 case]"
    )
    assert _normalize_action_for_merge("go to cabinet 17") == "go to cabinet #"


def test_webshop_tree_keeps_options_separate_and_preserves_task_score():
    pool = TracesPool(min_samples=1, enable_prefix_tree=True)
    for size, score in ((10, 1.0), (11, 0.5)):
        pool.add_trace(
            {
                "traj_uid": f"size-{size}",
                "task_type": "footwear",
                "task": "buy shoes in the requested size",
                "outcome": "success" if score == 1.0 else "failure",
                "steps": [
                    {
                        "action": "search[running shoes]",
                        "observation": "results",
                    },
                    {
                        "action": f"click[size {size}]",
                        "observation": f"size {size} selected",
                    },
                ],
                "meta": {
                    "environment": "WebShop",
                    "task_score": score,
                },
            }
        )

    codec = pool.export_batch(trigger_reason="test")["tree_evidence"]
    assert "click[size 10]" in codec["actions"]
    assert "click[size 11]" in codec["actions"]
    assert {record["r"] for record in codec["records"]} == {0.5, 1.0}


def test_webshop_pagination_is_not_filtered_when_the_page_changes():
    pool = TracesPool(loop_threshold=3)
    steps = [
        {
            "action": "click[next >]",
            "observation": f"result page {page}",
            "reward": 0,
        }
        for page in range(1, 6)
    ]
    cleaned, dropped = pool._filter_loops(steps)
    assert cleaned == steps
    assert dropped == 0


def test_tree_evidence_codec_replaces_nested_tree_with_node_paths():
    pool = TracesPool(min_samples=1, enable_prefix_tree=True)
    for index in range(8):
        pool.add_trace(_trace(f"success-{index}", "success", "take mug 1"))
        pool.add_trace(_trace(f"failure-{index}", "failure", "go to drawer 1"))

    batch = pool.export_batch(trigger_reason="test")
    codec = batch["tree_evidence"]

    assert "prefix_tree" not in batch
    assert codec["version"] == 3
    assert codec["mode"] == "tree_only"
    assert codec["tasks"] == ["clean the object"]
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


def test_tree_only_cloud_projection_removes_flat_steps_but_stays_renderable():
    pool = TracesPool(min_samples=1)
    pool.add_trace(_trace("success", "success", "take mug 1"))
    pool.add_trace(_trace("failure", "failure", "go to drawer 1"))
    local_batch = pool.export_batch(trigger_reason="test")
    cloud_batch = pool.project_cloud_batch(local_batch)

    assert "steps" in local_batch["success_samples"][0]
    assert "steps" not in cloud_batch["success_samples"][0]
    assert cloud_batch["cloud_projection"] == {
        "mode": "tree_only",
        "flat_steps_uploaded": False,
        "codec_version": 3,
    }

    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.evidence_render_limits = {
        "steps_per_trace": 120,
        "observation_chars_per_step": 4000,
    }
    rendered = analyzer._format_difftraces(
        cloud_batch["success_samples"] + cloud_batch["failure_samples"],
        limit=2,
        consensus=[],
        tree_evidence=cloud_batch["tree_evidence"],
    )

    assert "task: clean the object" in rendered
    assert "state after decision" in rendered
    assert "nonzero_rewards: s3=1" in rendered


def test_tree_only_cloud_renderer_never_falls_back_to_flat_steps():
    pool = TracesPool(min_samples=1)
    pool.add_trace(_trace("success", "success", "take mug 1"))
    cloud_batch = pool.project_cloud_batch(pool.export_batch(trigger_reason="test"))
    cloud_batch["tree_evidence"]["records"] = []

    analyzer = CloudAnalyzer.__new__(CloudAnalyzer)
    analyzer.evidence_render_limits = {
        "steps_per_trace": 120,
        "observation_chars_per_step": 4000,
    }
    with pytest.raises(ValueError, match="missing codec record"):
        analyzer._format_difftraces(
            cloud_batch["success_samples"],
            limit=1,
            consensus=[],
            tree_evidence=cloud_batch["tree_evidence"],
        )


def test_coskill_loop_passes_only_projected_tree_evidence_to_cloud(tmp_path):
    class RecordingAnalyzer:
        def __init__(self):
            self.received = None

        def contrastive_distill(self, batch, _current_skills):
            self.received = batch
            return []

        def get_update_summary(self):
            return {}

    class SkillLib:
        skills = {}

        def advance_lifecycle(self, modified_ids):
            assert modified_ids == []

        def save_skills(self, path):
            with open(path, "w") as handle:
                handle.write("{}")

    pool = TracesPool(min_samples=1)
    pool.add_trace(_trace("success", "success", "take mug 1"))
    analyzer = RecordingAnalyzer()
    loop = CoSkillCloudLoop(str(tmp_path), enable_coskill=True)
    loop.cloud_analyzer = analyzer

    assert loop.maybe_update(
        pool,
        SkillLib(),
        global_step=1,
        force_reason="test",
    )
    assert analyzer.received["cloud_projection"]["mode"] == "tree_only"
    assert "steps" not in analyzer.received["success_samples"][0]
    assert analyzer.received["tree_evidence"]["records"]

    audit_files = list((tmp_path / "cloud_io").glob("cloud_batch_*.json"))
    assert len(audit_files) == 1


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
    codec_current = batch["tree_evidence"]
    codec_v1 = {
        "version": 1,
        "nodes": [
            [node[0], codec_current["actions"][node[1] - 1], node[2], node[3]]
            for node in codec_current["nodes"]
        ],
        "records": codec_current["records"],
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
