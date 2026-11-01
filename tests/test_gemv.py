"""Correctness tests. On a machine without a GPU, run with TRITON_INTERPRET=1 (CPU interpreter)."""
import os

import pytest
import torch

from int4gemv import dequantize, gemv_reference, pack, quantize, unpack
from int4gemv.kernel import int4_gemv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("no GPU: set TRITON_INTERPRET=1 to run the kernel on the CPU interpreter",
                allow_module_level=True)


def test_pack_roundtrip():
    q = torch.randint(0, 16, (5, 64), dtype=torch.uint8)
    assert torch.equal(unpack(pack(q)), q)


def test_quantization_error_is_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    qw, s, z = quantize(w, 128)
    err = (dequantize(qw, s, z, 128) - w).abs().reshape(32, 2, 128)
    # rounding error is at most half a step, plus fp16 rounding of the scale
    assert (err <= s.float().unsqueeze(-1) * 0.5 * 1.01 + 1e-3).all()


@pytest.mark.parametrize("m,n,k,group,block_n,block_k,split_k", [
    (1, 64, 256, 128, 64, 128, 1),
    (1, 100, 512, 128, 32, 64, 1),    # N not a multiple of BLOCK_N
    (3, 48, 256, 64, 16, 32, 1),      # several rows, smaller groups
    (1, 256, 1024, 128, 128, 128, 1),
    (1, 100, 1024, 128, 32, 128, 4),  # split-K
    (1, 64, 640, 128, 16, 128, 2),    # 5 K steps over 2 splits: uneven
    (3, 48, 512, 64, 16, 64, 16),     # more splits than needed, several rows
])
def test_kernel_matches_reference(m, n, k, group, block_n, block_k, split_k):
    torch.manual_seed(m * 1000 + n + k)
    w = torch.randn(n, k)
    qw, s, z = quantize(w, group)
    x = torch.randn(m, k)
    ref = gemv_reference(x, qw, s, z, group)
    qw, s, z, x = (t.to(DEVICE) for t in (qw, s, z, x))
    y = int4_gemv(x.half(), qw, s, z, group, block_n, block_k, split_k=split_k).float().cpu()
    # x is rounded to fp16 and y is stored in fp16, so compare at fp16 precision
    ref16 = gemv_reference(x.half().float().cpu(), qw.cpu(), s.cpu(), z.cpu(), group)
    torch.testing.assert_close(y, ref16, rtol=2e-3, atol=2e-3 * ref.abs().max().item())
