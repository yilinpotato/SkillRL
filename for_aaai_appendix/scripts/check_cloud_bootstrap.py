#!/usr/bin/env python3
""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _tree_count(skills: dict[str, Any]) -> int:
    ""
    trees = skills.get("skill_trees") or skills.get("task_playbooks") or {}
    return len(trees) if isinstance(trees, dict) else 0


def _skill_counts(skills: dict[str, Any]) -> dict[str, int]:
    task_specific = skills.get("task_specific_skills", {})
    return {
        "general_skills": len(skills.get("general_skills", []) or []),
        "task_specific_skill_groups": (
            len(task_specific) if isinstance(task_specific, dict) else 0
        ),
        "common_mistakes": len(skills.get("common_mistakes", []) or []),
        "stored_skill_trees": _tree_count(skills),
    }


def _probe(analyzer) -> None:
    ""
    analyzer.client.chat.completions.create(
        model=analyzer.model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_completion_tokens=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=("alfworld", "webshop"))
    parser.add_argument("--skills-json", required=True, type=Path)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="make one tiny cloud request after local client initialization",
    )
    args = parser.parse_args()

    try:
        with args.skills_json.open("r", encoding="utf-8") as handle:
            skills = json.load(handle)
        if not isinstance(skills, dict):
            raise ValueError("skill JSON root must be an object")
    except Exception as exc:
        print(
            f"[cloud-bootstrap] FAIL skills_json={args.skills_json}: {exc}",
            file=sys.stderr,
        )
        return 2




    try:
        from agent_system.memory.cloud_analyzer import CloudAnalyzer

        analyzer = CloudAnalyzer(environment_name=args.environment)
    except Exception as exc:
        print(
            "[cloud-bootstrap] FAIL CloudAnalyzer initialization: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    if args.probe:
        try:
            _probe(analyzer)
        except Exception as exc:
            print(
                "[cloud-bootstrap] FAIL remote probe: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 4

    backend = os.environ.get("SKILL_UPDATER_BACKEND", "deepseek").lower()
    report = {
        "status": "ok",
        "environment": args.environment,
        "backend": backend,
        "model": analyzer.model,
        "skills_json": str(args.skills_json.resolve()),
        "remote_probe": bool(args.probe),
        **_skill_counts(skills),
    }
    print("[cloud-bootstrap] " + json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["stored_skill_trees"] == 0:
        print(
            "[cloud-bootstrap] no seed skill tree: expected. Trees must be created "
            "from successful on-policy rollout references after a pool watermark fires."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
