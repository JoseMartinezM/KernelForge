"""
VALID fixture - Layer Normalization kernel.
Checks: multiple functions in one file (helper + kernel), a helper function
        without @triton.jit, assert, return with a value, and chained
        comparison operators.
"""

import triton
import triton.language as tl


def _check_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@triton.jit
def layer_norm_kernel(
    X,
    W,
    B,
    Y,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Row index.
    row = tl.program_id(axis=0)
    X_ptr = X + row * stride
    Y_ptr = Y + row * stride

    # Compute mean.
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N

    # Compute variance.
    xmean = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale.
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    xhat = xmean * rstd
    y = xhat * w + b
    tl.store(Y_ptr + cols, y, mask=mask)


@triton.jit
def layer_norm_bwd_dx_fused(
    DX,
    DY,
    DW,
    DB,
    X,
    W,
    Mean,
    Rstd,
    Lock,
    stride,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Backward-pass kernel.
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat_wdy = xhat * wdy

    c1 = tl.sum(xhat_wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + row * stride + cols, dx, mask=mask)
