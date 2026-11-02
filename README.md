# Low-bit decode on a GPU: a Triton int4 GEMV and LLM serving benchmarks

This project asks the same question as [lowbit-decode-a53](https://github.com/FredWen-S/lowbit-decode-a53), this time on a datacenter GPU: **when do fewer bits make decode faster?**

On a Cortex-A53 phone the answer was "not below 4 bits", because unpacking int4 made decode compute-bound. A GPU has far more compute per byte of bandwidth, so the expected answer is different.

**Status (work in progress):**

1. **Triton int4 dequantize + GEMV kernel.** Batch-1 decode on an NVIDIA L4:
   - **v1** was 1.5–2.1× faster than fp16 cuBLAS on the large 7B layers, short of the predicted 2–3×. On kv_proj it was slower than fp16. The limit was memory parallelism, not unpacking ([`results/L4_v1.md`](results/L4_v1.md)).
   - **v2** adds split-K. It beats fp16 on every layer: **2.1–2.5× on the 7B layers**, and 1.2–1.5× on kv_proj ([`results/L4_v2.md`](results/L4_v2.md)).
   - **What explains it:** bytes in flight. More programs and wider tiles both raise it.
   - **Still open:** large layers stop at about 50–57% of copy bandwidth, against 3.8× ideal. Nsight Compute is next.
   - Predictions were written before each run, and the results were scored against them.
2. **vLLM 0.32 vs SGLang 0.5.23 serving benchmark.** Qwen2.5-1.5B bf16 on an L4, both engines measured by the same load generator ([`results/L4_serving.md`](results/L4_serving.md)):
   - **Batch-1 decode is at the roofline.** It runs at 13.1–13.5 ms/token in both engines, which is 99–102% of copy bandwidth for the 3.09 GB of weights.
   - **Throughput is equal up to concurrency 32.** At 64, vLLM is 25% ahead.
   - **Default schedulers make opposite trade-offs.** vLLM keeps TTFT low (p50 ≤ 421 ms) but stalls running streams (inter-token p99 113–142 ms). SGLang keeps streams smooth (24–64 ms) and makes new prompts wait (TTFT p50 up to 1.2 s).
   - **Prefix caching** cuts TTFT 5.5–6× on both.
   - **Chunked prefill protects running streams only when chunks share a step with decode.** A new 3.2k-token prompt every 2 s, with every gap recorded:
     - vLLM with 512-token chunks: max gap 41 ms.
     - 4096-token chunks: the stall is about 200 ms on both engines.
     - SGLang with 512-token chunks (`enable_mixed_chunk` is off by default): 7 back-to-back prefill batches, so the max gap is 311 ms. That is worse than not chunking.
     - p99 hid all of these stalls, because they touch under 1% of gaps.

     ![Gaps](results/serving_L4_mixed3/gaps.png)
   - **Predictions:** 3 of 7 held. The chunked-prefill test then took three runs: the first repeated one prompt (prefix-cache hits), and the second used p99, which cannot see rare stalls. The third run's predictions held for vLLM and failed for SGLang, for the reason above.

   ![Concurrency sweep](results/serving_L4/concurrency.png)
3. **BF16 vs FP8 vs AWQ vs GPTQ int4 on Qwen2.5-7B (vLLM 0.32)** ([`results/L4_quant.md`](results/L4_quant.md)):
   - **Batch 1 runs at the roofline in every format.** Marlin int4 reads weights at 237–239 GB/s, the same as BF16. That is 2.98–3.01× faster than BF16, against the ideal 3.15× from bytes (the bf16 LM head caps it). My Triton GEMV reached about half of that efficiency.
   - **The advantage fades with batching.** int4 is 1.8× faster than BF16 at concurrency 16 but only 1.2× at 64, where compute-bound prefill dominates. This is the A53 lesson on a GPU: fewer bits help only while decode is bound by weight reads.
   - **Accuracy:** GPTQ has the lower perplexity cost (+4.5% vs +6.3%), but AWQ the smaller MMLU loss (−0.7 vs −1.2 ± 0.4 pp, paired over 4,681 items).
   - **FP8 costs nothing in accuracy** (MMLU +0.19 ± 0.24 pp, perplexity +0.8%). It is 1.65× faster at batch 1, and **it overtakes int4 at concurrency 64** (1.46× vs 1.2×): W8A8 runs matmuls on FP8 tensor cores and cuts compute, while weight-only int4 cuts only bytes. The fastest format depends on load.
   - **Predictions:** 5 of 8 held. Two failed on the fast side: the copy-bandwidth floor was conservative. One was not testable: vLLM 0.32 runs AWQ on Marlin even when asked not to.

   ![Quantization summary](results/quant_L4/summary.png)

## Layout

| Path | Contents |
|---|---|
| [`int4gemv/quant.py`](int4gemv/quant.py) | int4 format: group-wise asymmetric, 8 codes per int32 packed along K. Quantize / pack / unpack, plus a PyTorch reference GEMV |
| [`int4gemv/kernel.py`](int4gemv/kernel.py) | Triton kernel. The zero point is factored out of the inner sum, so dequantization costs one multiply and one subtract per row per K step |
| [`tests/`](tests/) | Correctness tests: pack round-trip, quantization error bound, and kernel vs reference over several shapes |
| [`bench/bench_gemv.py`](bench/bench_gemv.py) | Times fp16 cuBLAS, PyTorch dequantize-then-matmul, and the Triton kernel (with a config sweep) on every decode linear layer of Qwen2.5-1.5B and -7B. Reports bandwidth against the GPU's measured copy bandwidth |
| [`colab/gemv_bench.ipynb`](colab/gemv_bench.ipynb) | Runs the tests and the benchmark on a Colab GPU |
| [`bench/serve_client.py`](bench/serve_client.py) | Engine-neutral streaming load generator: closed-loop concurrency plus an optional periodic background request. Records TTFT, TPOT, inter-token gaps and server-reported token counts |
| [`bench/serve_sweep.py`](bench/serve_sweep.py) | Starts vLLM or SGLang (each from its own virtualenv) with each config, runs the experiments and stops the server. Writes one JSON per experiment point |
| [`tests/fake_engine/`](tests/fake_engine/) | A stand-in `vllm` server for dry-running the sweep without a GPU |
| [`colab/serving_bench.ipynb`](colab/serving_bench.ipynb) | Installs both engines and runs the smoke test and the full sweep on Colab. [`serving_mixed_rerun.ipynb`](colab/serving_mixed_rerun.ipynb) reruns only the chunked-prefill configs |
| [`bench/plot_serving.py`](bench/plot_serving.py) | Draws the concurrency figure from the result JSON |
| [`bench/quant_sweep.py`](bench/quant_sweep.py), [`bench/accuracy_http.py`](bench/accuracy_http.py) | Quantization comparison: one vLLM server per format, speed through the load generator, MMLU and WikiText-2 through the same server. [`colab/quant_bench.ipynb`](colab/quant_bench.ipynb) runs it |
| [`bench/plot_quant.py`](bench/plot_quant.py) | Draws the quantization summary (throughput vs BF16 and accuracy cost) |
| [`bench/plot_gaps.py`](bench/plot_gaps.py) | Draws every inter-token gap over time for the mixed workload |

## Running

Without a GPU, the kernel tests run in Triton's CPU interpreter. Triton needs Linux; on Windows use WSL.

```sh
pip install torch triton pytest
TRITON_INTERPRET=1 python -m pytest -q tests
```

On a GPU:

```sh
python -m pytest -q tests
python bench/bench_gemv.py
```

Or open the notebook in Colab with an L4 runtime.

## License

MIT
