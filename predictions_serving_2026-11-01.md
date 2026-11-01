# Serving predictions before the first run (2026-11-01)

These were written before running `bench/serve_sweep.py` on any GPU.

**Setup:**
- Qwen2.5-1.5B-Instruct in bf16 on a Colab L4. Measured copy bandwidth is 231 GB/s.
- vLLM and SGLang, each with 85% of GPU memory and a 4096-token context.
- Prompt lengths are nominal. Random token ids re-encode about 4% longer (512 becomes about 533), and the server-reported lengths are recorded.

**Back-of-envelope:**
- **Decode step at batch 1:** the weights are about 3.1 GB, all read once per step, so the floor is 3.1 GB / 231 GB/s ≈ **13.4 ms**.
- **KV cache per token:** 28 layers × 2 (K, V) × 2 KV heads × 128 dims × 2 bytes = 28.7 KB.
- **At C = 64:** each step reads 64 × ~660 tokens × 28.7 KB ≈ 1.2 GB of KV cache, which adds about 5 ms.
- **Compute:** about 3 GFLOP per token, so one 3,000-token prefill is about 9 TFLOP. On the L4 (121 dense bf16 TFLOPS, maybe 60% achieved) that is about 120 ms.

| # | Prediction | Would be wrong if |
|---|---|---|
| S1 | At C = 1, TPOT p50 is 14–20 ms for both engines, close to the 13.4 ms weight-read floor | outside 14–20 ms |
| S2 | Batching is nearly free up to C = 16: TPOT(16) ≤ 1.4 × TPOT(1), and throughput(16) ≥ 10 × throughput(1) | either bound fails |
| S3 | At C = 64, TPOT is 1.3–2.0 × TPOT(1), and throughput is 2,000–3,500 output tok/s | outside either range |
| S4 | vLLM and SGLang are within ±15% of each other in throughput at every concurrency | any point differs by more than 15% |
| S5 | With prefix caching, TTFT p50 on the 1.5k-token shared-prefix workload is ≥ 3× lower than without it, on both engines | < 3× on either engine |
| S6 | In the mixed workload, 512-token prefill chunks keep the short streams' inter-token p99 ≤ 50 ms. 4096-token chunks let a 3k prompt stall them: inter-token p99 ≥ 100 ms | either bound fails on either engine |
| S7 | At C = 16, TTFT p50 for 2048-token prompts is ≥ 5× that for 128-token prompts | < 5× |
