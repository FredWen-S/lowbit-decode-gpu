"""Triton int4 dequantize + GEMV for decode (y = x @ W_hat.T, x has a few rows).

Each program computes BLOCK_N outputs for one row of x, walking K in BLOCK_K steps. A step
never crosses a quantization group, so the scale and zero point are one value per output row,
and the zero point comes out of the inner sum:

    sum_k (q_k - z) * s * x_k  =  s * (sum_k q_k * x_k  -  z * sum_k x_k)

so the inner loop is unpack, convert, multiply-add, with the dequantization folded into one
multiply and one subtract per row per step.

Split-K (v2): with few output rows there are too few programs to keep the GPU's memory system
busy (see results/L4_v1.md). split_k > 1 also splits K across programs; each writes an fp32
partial sum and a second small kernel adds the partials and casts to the output dtype.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _int4_gemv_kernel(
    x_ptr, qw_ptr, s_ptr, z_ptr, out_ptr,
    N, K, k_per_split,
    stride_xm, stride_qn, stride_sn, stride_zn, stride_os, stride_om,
    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    m = tl.program_id(2)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N
    words = tl.arange(0, BLOCK_K // 8)
    nib = tl.arange(0, 8)
    shifts = nib * 4

    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
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

    # out is y itself when split_k == 1 (stride_os = 0), else the fp32 partials [split, M, N]
    tl.store(out_ptr + pid_k * stride_os + m * stride_om + rn, acc.to(out_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _sum_splits_kernel(p_ptr, y_ptr, total, n_split, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n_split):
        acc += tl.load(p_ptr + i * total + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


def int4_gemv(x, qweight, scales, zeros, group_size=128, block_n=64, block_k=128, num_warps=4, split_k=1):
    """y [M, N] = x [M, K] @ W_hat.T for the format in quant.py. Meant for small M (decode)."""
    m, k = x.shape
    n = qweight.shape[0]
    assert qweight.shape == (n, k // 8) and scales.shape == zeros.shape == (n, k // group_size)
    assert k % block_k == 0 and group_size % block_k == 0, "a K step must stay inside one group"
    x = x.contiguous()
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    steps = k // block_k
    steps_per_split = triton.cdiv(steps, split_k)
    split_k = triton.cdiv(steps, steps_per_split)  # no empty splits
    if split_k == 1:
        out, stride_os = y, 0
    else:
        out = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        stride_os = out.stride(0)
    grid = (triton.cdiv(n, block_n), split_k, m)
    _int4_gemv_kernel[grid](
        x, qweight, scales, zeros, out, n, k, steps_per_split * block_k,
        x.stride(0), qweight.stride(0), scales.stride(0), zeros.stride(0), stride_os, out.stride(-2),
        GROUP=group_size, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps,
    )
    if split_k > 1:
        total = m * n
        _sum_splits_kernel[(triton.cdiv(total, 1024),)](out, y, total, split_k, BLOCK=1024)
    return y
