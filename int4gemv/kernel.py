"""Triton int4 dequantize + GEMV for decode (y = x @ W_hat.T, x has a few rows).

Each program computes BLOCK_N outputs for one row of x, walking K in BLOCK_K steps. A step
never crosses a quantization group, so the scale and zero point are one value per output row,
and the zero point comes out of the inner sum:

    sum_k (q_k - z) * s * x_k  =  s * (sum_k q_k * x_k  -  z * sum_k x_k)

so the inner loop is unpack, convert, multiply-add, with the dequantization folded into one
multiply and one subtract per row per step.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _int4_gemv_kernel(
    x_ptr, qw_ptr, s_ptr, z_ptr, y_ptr,
    N, K,
    stride_xm, stride_qn, stride_sn, stride_zn, stride_ym,
    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N
    words = tl.arange(0, BLOCK_K // 8)
    nib = tl.arange(0, 8)
    shifts = nib * 4

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kw = k0 // 8 + words
        packed = tl.load(qw_ptr + rn[:, None] * stride_qn + kw[None, :], mask=mask_n[:, None], other=0)
        # [BLOCK_N, BLOCK_K // 8, 8]: nibble j of word w is column 8 * w + j
        q = ((packed[:, :, None] >> shifts[None, None, :]) & 0xF).to(tl.float32)
        x = tl.load(x_ptr + m * stride_xm + kw[:, None] * 8 + nib[None, :]).to(tl.float32)
        qx = tl.sum(tl.sum(q * x[None, :, :], axis=2), axis=1)
        xs = tl.sum(tl.sum(x, axis=1), axis=0)
        g = k0 // GROUP
        s = tl.load(s_ptr + rn * stride_sn + g, mask=mask_n, other=0).to(tl.float32)
        z = tl.load(z_ptr + rn * stride_zn + g, mask=mask_n, other=0).to(tl.float32)
        acc += s * (qx - z * xs)

    tl.store(y_ptr + m * stride_ym + rn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def int4_gemv(x, qweight, scales, zeros, group_size=128, block_n=64, block_k=128, num_warps=4):
    """y [M, N] = x [M, K] @ W_hat.T for the format in quant.py. Meant for small M (decode)."""
    m, k = x.shape
    n = qweight.shape[0]
    assert qweight.shape == (n, k // 8) and scales.shape == zeros.shape == (n, k // group_size)
    assert k % block_k == 0 and group_size % block_k == 0, "a K step must stay inside one group"
    x = x.contiguous()
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, block_n), m)
    _int4_gemv_kernel[grid](
        x, qweight, scales, zeros, y, n, k,
        x.stride(0), qweight.stride(0), scales.stride(0), zeros.stride(0), y.stride(0),
        GROUP=group_size, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps,
    )
    return y
