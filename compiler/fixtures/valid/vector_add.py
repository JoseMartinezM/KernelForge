"""
Fixture VÁLIDO — Kernel de suma de vectores (el "Hello World" de Triton).
El translator debe:
  - Detectar 1 kernel con @triton.jit
  - Registrar BLOCK_SIZE como constexpr
  - Detectar tl.program_id, tl.arange, tl.load, tl.store
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
