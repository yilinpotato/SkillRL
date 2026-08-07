#!/usr/bin/env python3
""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "prompts" / "PROMPTS.md"


@dataclass(frozen=True)
class PromptSpec:
    title: str
    path: str
    symbol: str
    kind: str = "constant"


ONLINE_PROMPTS = (
    PromptSpec("ALFWorld: initial step", "agent_system/environments/prompts/alfworld.py", "ALFWORLD_TEMPLATE_NO_HIS"),
    PromptSpec("ALFWorld: history without retrieved skills", "agent_system/environments/prompts/alfworld.py", "ALFWORLD_TEMPLATE"),
    PromptSpec("ALFWorld: history with retrieved skills", "agent_system/environments/prompts/alfworld.py", "ALFWORLD_TEMPLATE_WITH_MEMORY"),
    PromptSpec("WebShop: initial step", "agent_system/environments/prompts/webshop.py", "WEBSHOP_TEMPLATE_NO_HIS"),
    PromptSpec("WebShop: history without retrieved skills", "agent_system/environments/prompts/webshop.py", "WEBSHOP_TEMPLATE"),
    PromptSpec("WebShop: history with retrieved skills", "agent_system/environments/prompts/webshop.py", "WEBSHOP_TEMPLATE_WITH_MEMORY"),
    PromptSpec("Cloud: contrastive skill distillation", "agent_system/memory/cloud_analyzer.py", "_build_contrastive_prompt", "function_return"),
    PromptSpec("Cloud: failure diagnosis", "agent_system/memory/cloud_analyzer.py", "_build_diagnose_prompt", "function_return"),
    PromptSpec("Cloud: skill-tree evolution", "agent_system/memory/cloud_analyzer.py", "_build_evolve_prompt", "function_return"),
)

def _render_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                expression = ast.unparse(value.value)
                if value.conversion != -1:
                    expression += f"!{chr(value.conversion)}"
                if value.format_spec is not None:
                    expression += f":{_render_string(value.format_spec)}"
                parts.append("{{ " + expression + " }}")
        return "".join(parts)
    raise TypeError(f"Unsupported prompt expression: {type(node).__name__}")


def _extract(spec: PromptSpec) -> tuple[str, int]:
    source_path = ROOT / spec.path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    if spec.kind == "constant":
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = []
                if isinstance(node, ast.Assign):
                    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                    value = node.value
                else:
                    names = [node.target.id] if isinstance(node.target, ast.Name) else []
                    value = node.value
                if spec.symbol in names and value is not None:
                    return _render_string(value).strip(), node.lineno
    elif spec.kind == "function_return":
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == spec.symbol:
                returns = [
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.Return)
                    and isinstance(child.value, (ast.Constant, ast.JoinedStr))
                ]
                if not returns:
                    break
                final_return = max(returns, key=lambda item: item.lineno)
                return _render_string(final_return.value).strip(), node.lineno
    raise LookupError(f"Cannot extract {spec.symbol} from {spec.path}")


def _extract_domain_contract(environment: str) -> tuple[str, int]:
    path = ROOT / "agent_system/memory/cloud_analyzer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_domain_context":
            continue
        for branch in node.body:
            if not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Compare):
                continue
            test = branch.test
            if (
                isinstance(test.left, ast.Name)
                and test.left.id == "env"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == environment
            ):
                for statement in branch.body:
                    if isinstance(statement, ast.Return) and statement.value is not None:
                        return _render_string(statement.value).strip(), branch.lineno
    raise LookupError(f"Cannot extract {environment} contract from {path}")


def _section(specs: Iterable[PromptSpec]) -> str:
    chunks: list[str] = []
    for index, spec in enumerate(specs, start=1):
        prompt, line = _extract(spec)
        chunks.extend(
            (
                f"### {index}. {spec.title}",
                "",
                f"Source: [`{spec.path}:{line}`](../{spec.path}#L{line})",
                "",
                "~~~~text",
                prompt,
                "~~~~",
                "",
            )
        )
    return "\n".join(chunks)


def _domain_contract_section() -> str:
    chunks = [
        "## B. Environment contracts injected into every cloud prompt",
        "",
        "These strings replace the `{{ self._domain_context() }}` or",
        "`{{ domain_context }}` placeholders above at runtime.",
        "",
    ]
    for index, environment in enumerate(("alfworld", "webshop"), start=1):
        contract, line = _extract_domain_contract(environment)
        chunks.extend(
            (
                f"### B.{index}. {environment.upper()} contract",
                "",
                f"Source: [`agent_system/memory/cloud_analyzer.py:{line}`]"
                f"(../agent_system/memory/cloud_analyzer.py#L{line})",
                "",
                "~~~~text",
                contract,
                "~~~~",
                "",
            )
        )
    return "\n".join(chunks)


def build_document() -> str:
    text = """# CoSkill Prompt Appendix

This file is exported from the runtime source with
`python tools/export_prompts.py`. Dynamic values appear as
`{{ expression }}`.

## Scope and notation

This appendix contains the exact repository-owned prompt templates used by the
online edge executor and cloud analyzer. Runtime values—task descriptions,
observations, admissible actions, trajectory evidence, retrieved skills, and
the current skill tree—are represented by placeholders. Qwen's tokenizer-owned
chat template is applied after these strings are rendered and is not duplicated
here.

The retrieved-memory placeholder is rendered at runtime in this order:

1. learned skill tree;
2. general principles;
3. task-relevant skills;
4. mistakes to avoid.

The cloud templates also receive an explicit environment contract from
`CloudAnalyzer._domain_context()`. The contract is included through the
`{{ self._domain_context() }}` placeholder and its source is documented in the
companion provenance file.

## A. Online experiment prompts

"""
    text += _section(ONLINE_PROMPTS)
    text += "\n" + _domain_contract_section()
    return text.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_document(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
