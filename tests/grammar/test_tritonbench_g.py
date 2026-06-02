from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITONBENCH_G_ROOT = PROJECT_ROOT / "vendor" / "TritonBench" / "data" / "TritonBench_G_v1"
TRITONBENCH_G_INDEX = PROJECT_ROOT / "vendor" / "TritonBench" / "data" / "TritonBench_G_v1.json"
TEST_DELIMITER = "#" * 146


@dataclass(frozen=True)
class JitBlock:
    filename: str
    function_name: str
    line_number: int
    source: str


def kernel_section(path: Path) -> str:
    code, delimiter, _tests = path.read_text(encoding="utf-8").partition(TEST_DELIMITER)
    assert delimiter, f"{path} does not contain the TritonBench test delimiter"
    return code.rstrip() + "\n"


def tritonbench_g_files() -> list[str]:
    data = json.loads(TRITONBENCH_G_INDEX.read_text(encoding="utf-8"))
    return [row["file"] for row in data]


def is_triton_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "jit"
        and isinstance(node.value, ast.Name)
        and node.value.id == "triton"
    )


def normalized_jit_blocks(filename: str) -> list[JitBlock]:
    code = kernel_section(TRITONBENCH_G_ROOT / filename)
    tree = ast.parse(code, filename=filename)
    blocks: list[JitBlock] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        jit_decorators = [
            decorator for decorator in node.decorator_list if is_triton_jit_decorator(decorator)
        ]
        if not jit_decorators:
            continue
        jit_decorator = jit_decorators[-1]
        normalized_node = copy.copy(node)
        normalized_node.decorator_list = [jit_decorator]
        blocks.append(
            JitBlock(
                filename=filename,
                function_name=node.name,
                line_number=jit_decorator.lineno,
                source=ast.unparse(normalized_node).rstrip() + "\n",
            )
        )

    assert blocks, f"{filename} does not define any @triton.jit functions"
    return blocks


@pytest.mark.corpus
@pytest.mark.parametrize("filename", tritonbench_g_files(), ids=lambda filename: filename)
def test_tritonbench_g_file_acceptance(triton_xgrammar_jit_block, filename: str):
    for jit_block in normalized_jit_blocks(filename):
        result = triton_xgrammar_jit_block.match(jit_block.source)
        assert result.accepted, (
            f"{jit_block.filename}::{jit_block.function_name}:{jit_block.line_number}\n"
            f"{result.error}\n\n{jit_block.source}"
        )
