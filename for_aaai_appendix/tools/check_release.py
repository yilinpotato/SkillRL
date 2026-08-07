#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".cache", "outputs", "__pycache__"}
TEXT_SUFFIXES = {
    ".c",
    ".cfg",
    ".h",
    ".ini",
    ".json",
    ".l",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".y",
    ".yaml",
    ".yml",
}
PATTERNS = {
    "credential": re.compile(r"(?:ghp_[A-Za-z0-9]+|sk-[A-Za-z0-9]{16,})"),
    "personal path": re.compile(r"/(?:home|data2)/[A-Za-z0-9_.-]+"),
    "authoring trace": re.compile(
        r"\b(?:" + "Co" + r"dex|Chat" + r"GPT|AI-" + r"generated)\b",
        re.I,
    ),
}
AUTHORING_NAME = re.compile(
    r"(?:" + "Co" + r"dex|Chat" + r"GPT|AI[-_ ]generated)",
    re.I,
)
HAN_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def main() -> None:
    failures: list[str] = []
    expected_paths = (
        ROOT / "memory_data/alfworld/initial_skills.json",
        ROOT / "agent_system/environments/env_package/webshop/webshop",
    )
    for path in expected_paths:
        if not path.exists():
            failures.append(f"missing runtime path: {path.relative_to(ROOT)}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        relative_path = path.relative_to(ROOT)
        if AUTHORING_NAME.search(str(relative_path)):
            failures.append(f"authoring filename: {relative_path}")
        if HAN_TEXT.search(str(relative_path)):
            failures.append(f"Chinese filename: {relative_path}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitignore",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if HAN_TEXT.search(text):
            failures.append(f"Chinese text: {relative_path}")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative_path}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("release scan passed")


if __name__ == "__main__":
    main()
