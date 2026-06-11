"""
INVALID fixture - two kernels with the same name.
The translator should report a semantic error for the duplicate function.
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid + tl.arange(0, n)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid + tl.arange(0, n)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))
