"""Group-wise asymmetric int4 weight format and a PyTorch reference.

Layout for a weight W of shape [N, K] (y = x @ W.T), group size G along K:
  qweight  int32 [N, K // 8]  eight 4-bit codes per word; code for column k sits in
                              word k // 8 at bits 4 * (k % 8) .. 4 * (k % 8) + 3
  scales   fp16  [N, K // G]
  zeros    uint8 [N, K // G]  zero point in 0..15
  W_hat[n, k] = (q[n, k] - zeros[n, k // G]) * scales[n, k // G]

This is the GPTQ/AWQ idea (4-bit codes, per-group scale and zero point) with the simplest
packing, packed along K so a GEMV row reads contiguous words.
"""
import torch


def quantize(w, group_size=128):
    """Round-to-nearest min-max quantization of w [N, K] to int4. Returns (qweight, scales, zeros)."""
    n, k = w.shape
    assert k % group_size == 0 and group_size % 8 == 0, "K must be a multiple of the group size"
    g = w.float().reshape(n, k // group_size, group_size)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scales = ((hi - lo) / 15).clamp_min(1e-8)
    zeros = (-lo / scales).round().clamp(0, 15)
    q = ((g / scales).round() + zeros).clamp(0, 15).to(torch.uint8).reshape(n, k)
    return pack(q), scales.squeeze(-1).half(), zeros.squeeze(-1).to(torch.uint8)


def pack(q):
    """Pack uint8 codes [N, K] (values 0..15) into int32 [N, K // 8]."""
    n, k = q.shape
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int64)
    words = (q.reshape(n, k // 8, 8).to(torch.int64) << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def unpack(qweight):
    """Inverse of pack: int32 [N, K // 8] -> uint8 [N, K]."""
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = (qweight.unsqueeze(-1) >> shifts) & 0xF
    return q.reshape(qweight.shape[0], -1).to(torch.uint8)


def dequantize(qweight, scales, zeros, group_size=128):
    """W_hat [N, K] in fp32."""
    q = unpack(qweight).float()
    n, k = q.shape
    q = q.reshape(n, k // group_size, group_size)
    w = (q - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)
    return w.reshape(n, k)


def gemv_reference(x, qweight, scales, zeros, group_size=128):
    """y = x @ W_hat.T in fp32, x [M, K]. The ground truth for the kernels."""
    return x.float() @ dequantize(qweight, scales, zeros, group_size).T
