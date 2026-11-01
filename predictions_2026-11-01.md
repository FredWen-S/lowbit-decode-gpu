# Predictions before the first GPU run (2026-11-01)

Written before running `bench/bench_gemv.py` on any GPU. The kernel is v1: one program per BLOCK_N output rows, no split-K. The target is an NVIDIA L4 on Colab (spec 300 GB/s, 58 SMs).

**Question:** on the A53, int4 unpacking made sub-4-bit formats compute-bound. On a GPU, with far more compute per byte, is the int4 GEMV bandwidth-bound, so that it gets close to the 3.8× byte saving over fp16?

Byte budget per weight:
- fp16: 2 B.
- int4 in this format: 0.5 B for the code, plus scale and zero point, (2 + 1) / 128 B.
- Total 0.523 B, so the ideal speedup is about **3.8×**.

| # | Prediction | Would be wrong if |
|---|---|---|
| P1 | fp16 cuBLAS reaches ≥ 75% of measured copy bandwidth on the large 7B layers (gate_up, down, q/o_proj) | below 75% |
| P2 | Triton int4 v1 is **2–3×** faster than fp16 on the large 7B layers. It is short of 3.8× because v1 is not tuned, but unpacking is not the limit | < 1.5× (compute-bound like the A53) or > 3.4× |
| P3 | The 7B `down` layer (N = 3584, K = 18944) has the lowest int4 %bandwidth of the large layers. With BLOCK_N = 64 there are only 56 programs for 58 SMs, so split-K is the fix | another large layer is lower |
| P4 | Small layers (`kv_proj`, every 1.5B layer) gain **< 1.5×**, because launch and latency dominate below ~5 µs | ≥ 1.5× |
| P5 | PyTorch dequantize-then-matmul is **slower than fp16**, because it writes and rereads a full fp16 weight | faster than fp16 |

If P2 holds and P3 holds, the next step is split-K. If P2 fails low, profile the inner loop (unpack versus FMA) with Nsight Compute before touching the launch shape.
