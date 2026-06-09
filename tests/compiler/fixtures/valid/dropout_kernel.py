"""
Fixture VÁLIDO — Dropout kernel.
Prueba: comentario como primera línea del cuerpo (bug #1 corregido),
        función con tipo de retorno -> None (bug #4 corregido),
        operadores bitwise, tl.where, tl.rand.
"""

import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # offset del bloque actual
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p

    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
