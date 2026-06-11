"""
INVALID fixture - pure syntax error.
The ':' is missing at the end of the def line, so the parser should report an
error.
"""

import triton
import triton.language as tl


@triton.jit
def broken_kernel(x_ptr, BLOCK_SIZE: tl.constexpr)  # missing ':'
    pid = tl.program_id(axis=0)
    tl.store(x_ptr + pid, 0)
