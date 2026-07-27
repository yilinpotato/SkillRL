#!/usr/bin/env python3
"""Load a mounted cloud dotenv file into a container shell safely.

Docker's ``--env-file`` is deliberately not used for CoSkill credentials:
users commonly keep shell-style quoted values in ``.env`` and Docker's parser
does not have the same semantics as the launcher.  This helper accepts the
small dotenv subset needed by the cloud client and emits shell-quoted exports
to stdout for the entrypoint to evaluate without logging secrets.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path


DEFAULT_PATH = Path("/run/secrets/coskill.env")
ALLOWED_KEYS = {
    "SKILL_UPDATER_BACKEND",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_MODEL",
}
_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _parse_value(raw: str, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise ValueError(f"line {line_number}: unterminated quoted value")
        return value[1:-1]
    # Inline comments are supported only when separated by whitespace.  A '#'
    # inside a key/URL remains literal.
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            raise ValueError(f"line {line_number}: expected KEY=VALUE dotenv assignment")
        key, raw_value = match.groups()
        if key in ALLOWED_KEYS:
            values[key] = _parse_value(raw_value, line_number)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--emit-shell", action="store_true")
    args = parser.parse_args()
    path = args.file
    if not path.is_file():
        print(f"[container-dotenv] missing mounted dotenv file: {path}", file=sys.stderr)
        return 2
    try:
        values = parse_dotenv(path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[container-dotenv] invalid dotenv file: {exc}", file=sys.stderr)
        return 3
    if not values:
        print("[container-dotenv] no supported cloud keys found", file=sys.stderr)
        return 4
    if args.emit_shell:
        for key in sorted(values):
            print(f"export {key}={shlex.quote(values[key])}")
    else:
        print("[container-dotenv] loaded keys: " + ", ".join(sorted(values)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
