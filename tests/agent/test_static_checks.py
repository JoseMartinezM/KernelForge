from __future__ import annotations

from kernelforge.agent.static_checks import (
    has_incomplete_marker,
    jit_only_source,
    run_static_checks,
    static_guardrail_flags,
    torch_call_names,
)


def test_static_checks_capture_semantic_and_torch_signals():
    source = """
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


def wrapper(x):
    return torch.empty_like(x)
"""

    result = run_static_checks(source)

    assert result.syntax_ok
    assert result.triton_jit_count == 1
    assert result.semantic_warnings == []
    assert result.torch_calls == ["torch.empty_like"]
    assert "torch_calls_present" in result.flags


def test_static_checks_flag_missing_jit_and_memory_ops():
    result = run_static_checks("def add(x, y):\n    return x + y\n")

    assert result.syntax_ok
    assert "no_triton_jit" in result.flags
    assert "missing_tl_load_or_store" in result.flags


def test_torch_call_names_handles_syntax_errors():
    assert torch_call_names("def broken(:\n") == []


def test_static_checks_do_not_treat_wrapper_block_size_as_kernel_warning():
    source = """
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


def add(x):
    BLOCK_SIZE = 1024
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), BLOCK_SIZE),)](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out
"""

    result = run_static_checks(source)

    assert result.semantic_warnings == []
    assert "def add(x):" not in jit_only_source(source)


def test_incomplete_marker_ignores_pass_in_prose():
    assert not has_incomplete_marker('''"""forward pass"""\ndef add(x):\n    return x\n''')
    assert has_incomplete_marker("def add(x):\n    pass\n")


def test_static_guardrails_flag_common_triton_api_hallucinations():
    source = """
import torch
import triton
import triton.language as tl


@triton.jit
def bad_kernel(x_ptr, out_ptr, D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    values = tl.zeros([D], dtype=tl.float32)
    offsets = tl.arange(0, D)
    tl.store(out_ptr + pid + offsets, values)


def wrapper(x):
    out = torch.empty_like(x)
    _ = tl.program_id(0)
    bad_kernel[(1,)](x, out, x.shape[0], stream=torch.cuda.Stream(), BLOCK_SIZE=1024)
    stride = x.stride[0]
    return out + stride
"""

    flags = static_guardrail_flags(source)

    assert "kernel_launch_stream_arg" in flags
    assert "runtime_triton_vector_shape" in flags
    assert "triton_intrinsic_outside_jit" in flags
    assert "stride_method_subscript" in flags


def test_static_guardrails_allow_constexpr_triton_vector_shapes():
    source = """
import triton
import triton.language as tl


@triton.jit
def good_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loaded = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values + loaded, mask=mask)
"""

    assert static_guardrail_flags(source) == []
