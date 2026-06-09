"""
Fixture INVÁLIDO — error de sintaxis puro.
Falta el ':' al final del def → el parser debe reportar error.
"""

import triton
import triton.language as tl


@triton.jit
def broken_kernel(x_ptr, BLOCK_SIZE: tl.constexpr)  # falta ':'
    pid = tl.program_id(axis=0)
    tl.store(x_ptr + pid, 0)
