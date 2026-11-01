#!/usr/bin/env python3
"""Decode GEMV (M=1) on a GPU: fp16 cuBLAS vs PyTorch dequantize-then-matmul vs the Triton int4 kernel.

For each linear layer shape of Qwen2.5-1.5B and -7B, reports time, bytes moved and effective
bandwidth as a fraction of the measured copy bandwidth of this GPU (the practical roofline
for a memory-bound GEMV).

usage: python bench/bench_gemv.py [--models 1.5b 7b] [--m 1] [--out results/gemv_<gpu>.json]
"""
import argparse
import itertools
import json
import os
import sys

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from int4gemv import dequantize, gemv_reference, quantize  # noqa: E402
from int4gemv.kernel import int4_gemv  # noqa: E402

# (name, N, K) of each linear layer used in decode; q/k/v and gate/up are listed separately
SHAPES = {
    "1.5b": [("q_proj", 1536, 1536), ("kv_proj", 256, 1536), ("o_proj", 1536, 1536),
             ("gate_up", 8960, 1536), ("down", 1536, 8960)],
    "7b": [("q_proj", 3584, 3584), ("kv_proj", 512, 3584), ("o_proj", 3584, 3584),
           ("gate_up", 18944, 3584), ("down", 3584, 18944)],
}
GROUP = 128
CONFIGS = [dict(block_n=bn, block_k=bk, num_warps=nw, split_k=sk)
           for bn, bk, nw, sk in itertools.product([8, 16, 32, 64, 128], [64, 128], [2, 4, 8], [1, 2, 4, 8, 16])]


def copy_bandwidth():
    """Achievable DRAM bandwidth: read + write of a 1 GiB buffer."""
    src = torch.empty(2**29, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    ms = do_bench(lambda: dst.copy_(src), warmup=50, rep=300)
    return 2 * src.numel() * 2 / (ms * 1e-3) / 1e9


def bench_shape(n, k, m):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    qw, s, z = quantize(w, GROUP)
    ref = gemv_reference(x, qw, s, z, GROUP)

    def gbps(nbytes, ms):
        return nbytes / (ms * 1e-3) / 1e9

    io = (m * k + m * n) * 2
    fp16_bytes = n * k * 2 + io
    int4_bytes = qw.numel() * 4 + s.numel() * 2 + z.numel() + io
    out = {"N": n, "K": k, "M": m}

    ms = do_bench(lambda: torch.nn.functional.linear(x, w))
    out["fp16_cublas"] = dict(us=ms * 1e3, GBps=gbps(fp16_bytes, ms))
    ms = do_bench(lambda: torch.nn.functional.linear(x, dequantize(qw, s, z, GROUP).half()))
    out["torch_dequant_matmul"] = dict(us=ms * 1e3, GBps=gbps(int4_bytes, ms))

    # short timing for the sweep, then a full-length timing of the winners
    sweep = []
    for cfg in CONFIGS:
        if GROUP % cfg["block_k"] or k % cfg["block_k"]:
            continue
        try:
            sweep.append((do_bench(lambda: int4_gemv(x, qw, s, z, GROUP, **cfg), warmup=5, rep=20), cfg))
        except Exception as e:  # out of shared memory / registers for big tiles
            print(f"  skip {cfg}: {type(e).__name__}", file=sys.stderr)
    # v1 = best without split-K, v2 = best overall; both from this run on this GPU
    for key, pool in (("triton_int4_nosplit", [c for c in sweep if c[1]["split_k"] == 1]),
                      ("triton_int4", sweep)):
        cfg = min(pool, key=lambda c: c[0])[1]
        ms = do_bench(lambda: int4_gemv(x, qw, s, z, GROUP, **cfg))
        y = int4_gemv(x, qw, s, z, GROUP, **cfg).float()
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        steps = k // cfg["block_k"]
        split = triton.cdiv(steps, triton.cdiv(steps, cfg["split_k"]))  # as int4_gemv clamps it
        out[key] = dict(us=ms * 1e3, GBps=gbps(int4_bytes, ms), config=cfg, max_rel_err=err,
                        effective_split_k=split, programs=triton.cdiv(n, cfg["block_n"]) * split * m)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["1.5b", "7b"])
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--tag", default="v2", help="kernel version, used in the output file name")
    a = p.parse_args()

    gpu = torch.cuda.get_device_name()
    bw = copy_bandwidth()
    print(f"{gpu}: copy bandwidth {bw:.0f} GB/s, torch {torch.__version__}, triton {triton.__version__}")
    result = dict(gpu=gpu, copy_GBps=bw, torch=torch.__version__, triton=triton.__version__,
                  group_size=GROUP, layers={})
    print(f"{'layer':<14}{'N x K':>13}{'fp16 us':>9}{'nosplit us':>11}{'split us':>9}"
          f"{'split %BW':>10}{'vs fp16':>8}{'split_k':>8}")
    for model in a.models:
        for name, n, k in SHAPES[model]:
            r = bench_shape(n, k, a.m)
            result["layers"][f"{model}/{name}"] = r
            f, v1, t = r["fp16_cublas"], r["triton_int4_nosplit"], r["triton_int4"]
            print(f"{model + '/' + name:<14}{f'{n}x{k}':>13}{f['us']:>9.1f}{v1['us']:>11.1f}{t['us']:>9.1f}"
                  f"{100 * t['GBps'] / bw:>9.0f}%{f['us'] / t['us']:>7.2f}x{t['effective_split_k']:>8}", flush=True)

    out = a.out or f"results/gemv_{gpu.replace(' ', '_')}_m{a.m}_{a.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
