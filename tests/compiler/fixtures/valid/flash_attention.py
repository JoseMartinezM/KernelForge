"""
VALID fixture - Simplified Flash Attention kernel.
Checks: multiple tl.constexpr parameters, bitwise operators, an accumulator
        for loop, 2D slices, and several tl.* calls in one arithmetic
        expression.
"""

import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Block identifiers.
    start_m = tl.program_id(axis=0)
    off_hz = tl.program_id(axis=1)

    off_z = off_hz // H
    off_h = off_hz % H

    # Offset ranges.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Q, K, and V pointers.
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn

    # Accumulators.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs)

    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K and V.
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Incremental softmax.
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)

        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        acc_scale = l_i / l_i_new * alpha

        acc = acc * acc_scale[:, None]
        p = p * p_scale[:, None]
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    # Write output.
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc)
