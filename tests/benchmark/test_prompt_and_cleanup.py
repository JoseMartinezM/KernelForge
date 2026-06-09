from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from kernelforge.benchmark.tritonbench import (
    _instrument_test_code_for_benchmark,
    _summarize_benchmarks,
    cleanup_generated_code,
    make_prompt,
)


def test_make_prompt_omits_empty_optional_sections():
    entry = {
        "description": "Element-wise addition.",
        "func_inputs": "add(input, other) -> Tensor",
        "torch_code": "return input + other",
        "math": "N/A",
        "other": None,
    }

    prompt = make_prompt(entry)

    assert "## Functional Description" in prompt
    assert "Element-wise addition." in prompt
    assert "## Reference PyTorch Implementation" in prompt
    assert "return input + other" in prompt
    assert "Mathematical Formulation" not in prompt
    assert "Additional Information" not in prompt


def test_make_prompt_includes_math_and_other_blocks_when_present():
    entry = {
        "description": "Scaled dot-product attention.",
        "func_inputs": "attention(query, key, value)",
        "torch_code": "return torch.softmax(scores, dim=-1) @ value",
        "math": "softmax(QK^T / sqrt(d)) V",
        "other": "Uses causal masking in decoder paths.",
    }

    prompt = make_prompt(entry)

    assert "## Mathematical Formulation" in prompt
    assert "softmax(QK^T" in prompt
    assert "## Additional Information" in prompt
    assert "causal masking" in prompt


def test_make_prompt_for_vendor_tanh_entry(tritonbench_root: Path):
    from kernelforge.benchmark.tritonbench import load_t_simple_entries

    entries, errors, _, _ = load_t_simple_entries(tritonbench_root)
    assert not errors

    tanh = next(entry for entry in entries if entry["file"] == "tanh.py")
    prompt = make_prompt(tanh)

    assert "hyperbolic tangent" in prompt.lower()
    assert "tanh(" in prompt
    assert "```python" in prompt


def test_cleanup_generated_code_strips_markdown_fence():
    raw = textwrap.dedent("""
        Here is the kernel:

        ```python
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            tl.store(x_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)
        ```
    """)

    code = cleanup_generated_code(raw)

    assert "```" not in code
    assert code.startswith("import triton")
    ast.parse(code)


def test_cleanup_generated_code_strips_redacted_tokens():
    raw = "import triton<|im_end|>\n"

    assert cleanup_generated_code(raw) == "import triton"


def test_cleanup_generated_code_returns_invalid_syntax_unchanged():
    raw = "def broken(:\n    pass"

    assert cleanup_generated_code(raw) == raw


def test_instrument_test_code_for_benchmark_wraps_result_calls():
    source = textwrap.dedent("""
        import torch

        def test_div():
            results = {}
            input1 = torch.tensor([6.0], device='cuda')
            other1 = torch.tensor([3.0], device='cuda')
            results["test_case_1"] = div(input1, other1)
            return results

        test_results = test_div()
    """)

    instrumented = _instrument_test_code_for_benchmark(source)

    ast.parse(instrumented)
    assert "_tb_record_benchmark('test_case_1', lambda: div(input1, other1))" in instrumented
    assert "test_results = test_div()" in instrumented


def test_summarize_benchmarks_reports_total_speedup_for_measured_cases():
    summary = _summarize_benchmarks(
        pred_cases={"case_1": {"ms": 2.0}, "case_2": {"error": "failed"}},
        ref_cases={"case_1": {"ms": 5.0}, "case_2": {"ms": 1.0}},
    )

    assert summary["pred_ms"] == 2.0
    assert summary["ref_ms"] == 5.0
    assert summary["speedup"] == 2.5
    assert summary["measured_cases"] == 1
    assert summary["cases"]["case_1"]["speedup"] == 2.5
    assert summary["cases"]["case_2"]["speedup"] is None
