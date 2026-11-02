# Quantization predictions before the first run (2026-11-02)

Written before running `bench/quant_sweep.py` on any GPU. The setup is Qwen2.5-7B-Instruct on vLLM 0.32.0 on a Colab L4, with GPU memory utilization 0.90 and a 4096-token context.

## Bytes read per decode step at batch 1

Parameter counts:
- 6.53 B non-embedding parameters.
- The untied LM head adds 0.545 B and is read every step. The input embedding table is not: only one row is used.

| Format | Linear layers | LM head (bf16) | Total | Floor at the 1.5B run's 236 GB/s | Ideal speedup vs BF16 |
|---|---|---|---|---|---|
| BF16 | 13.06 GB | 1.09 GB | 14.15 GB | 60 ms | 1× |
| FP8 (vLLM dynamic, W8A8) | 6.53 GB | 1.09 GB | 7.62 GB | 32 ms | 1.86× |
| int4, group 128 (AWQ / GPTQ) | 3.27 GB + 0.13 GB scales and zeros | 1.09 GB | 4.49 GB | 19 ms | 3.15× |

Two things follow from the table:
- The LM head stays in bf16 in all three formats. Its 1.09 GB is a quarter of the int4 step, so int4 cannot reach 4×.
- My Triton GEMV reached 47–57% of copy bandwidth on these 7B layer shapes. vLLM's Marlin kernels are reported to be close to the roofline at batch 1, so the AWQ/GPTQ results also benchmark my kernel against a production one.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| R1 | BF16 TPOT p50 at C = 1 is 60–66 ms, at the bandwidth floor as the 1.5B model was | outside 60–66 ms |
| R2 | FP8 TPOT at C = 1 is 33–38 ms (1.6–1.85× faster than BF16) | outside 33–38 ms |
| R3 | AWQ and GPTQ (Marlin) TPOT at C = 1 is 20–25 ms: ≥ 2.5× faster than BF16 and ≥ 75% of their 19 ms floor | outside 20–25 ms on either |
| R4 | The same AWQ weights on vLLM's non-Marlin AWQ kernel are ≥ 1.5× slower at C = 1 than with Marlin | < 1.5× (or untested if that config fails to start) |
| R5 | The int4 advantage shrinks with batching: AWQ/BF16 throughput ratio at C = 64 is ≤ 1.8×, below its C = 1 ratio | > 1.8× |
| R6 | MMLU, paired against BF16 on 4,681 items: FP8 is within ±0.5 pp, and AWQ and GPTQ each lose 0.3–2.0 pp | any outside its range |
| R7 | WikiText-2 perplexity vs BF16: FP8 +0–1.5%, AWQ and GPTQ +2–8% | any outside its range |
| R8 | AWQ and GPTQ are within 1 pp of each other on MMLU | > 1 pp apart |
