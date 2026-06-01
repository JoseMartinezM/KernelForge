"""
Modal evaluation function for KernelForge.

Runs a generated Triton kernel against the TritonBench reference on a T4 GPU
and returns call@1, exe@1, and mismatch details. Model-agnostic: works with
any kernel regardless of which model generated it.

Standalone smoke test:
    uv run modal run scripts/modal_eval.py

Test a specific entry:
    uv run modal run scripts/modal_eval.py --entry-file softmax.py
"""

import sys

import modal

eval_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("gcc")
    .pip_install("triton", "numpy")
    .add_local_dir("src/kernelforge", remote_path="/app/src/kernelforge")
    .add_local_dir("vendor/TritonBench/data", remote_path="/app/vendor/TritonBench/data")
)

app = modal.App("kernelforge-eval")


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=300,
)
def evaluate_kernel(kernel_code: str, entry_file: str) -> dict:
    sys.path.insert(0, "/app/src")
    from kernelforge.benchmark.semantic_checker import check_kernel
    from kernelforge.benchmark.tritonbench import evaluate_entry, load_t_simple_entries

    semantic_warnings = check_kernel(kernel_code)

    entries, errors, _, _ = load_t_simple_entries("/app/vendor/TritonBench")
    if errors:
        return {
            "file": entry_file,
            "call@1": False,
            "exe@1": False,
            "mismatches": [f"dataset load errors: {errors}"],
            "semantic_warnings": semantic_warnings,
            "execution_time": None,
        }

    entry = next((e for e in entries if e["file"] == entry_file), None)
    if entry is None:
        return {
            "file": entry_file,
            "call@1": False,
            "exe@1": False,
            "mismatches": [f"entry_file '{entry_file}' not found in dataset"],
            "semantic_warnings": semantic_warnings,
            "execution_time": None,
        }

    result = evaluate_entry(entry, pred_code=kernel_code, timeout=180)
    result["semantic_warnings"] = semantic_warnings
    result.setdefault("execution_time", None)
    return result


@app.local_entrypoint()
def test(entry_file: str = "tanh.py", kernel_file: str = ""):
    """
    Run evaluation on Modal.

    Smoke test (no kernel_file):
        uv run modal run scripts/modal_eval.py

    From agent (pass a kernel file):
        uv run modal run scripts/modal_eval.py --entry-file softmax.py --kernel-file /tmp/kernel.py
    """
    import json

    if kernel_file:
        kernel_code = open(kernel_file, encoding="utf-8").read()
    else:
        kernel_code = """\
import triton
import triton.language as tl
import torch


@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    exp2x = tl.exp(2.0 * x)
    result = (exp2x - 1.0) / (exp2x + 1.0)
    tl.store(output_ptr + offsets, result.to(tl.float16) if x.dtype == tl.float16 else result, mask=mask)


def tanh(input, *, out=None):
    output = torch.empty_like(input)
    n = input.numel()
    BLOCK_SIZE = 1024
    tanh_kernel[triton.cdiv(n, BLOCK_SIZE),](input, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
"""

    result = evaluate_kernel.remote(kernel_code=kernel_code, entry_file=entry_file)
    print(json.dumps(result))
