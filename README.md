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
2. **vLLM vs SGLang serving benchmark.** Will measure TTFT, TPOT, throughput and memory, sweeping concurrency and prompt length, with prefix caching and chunked prefill on and off. Not started.
3. **FP16 vs AWQ / GPTQ int4 (FP8 on L4).** Will cover speed and accuracy, reusing the A53 eval sets. Not started.

## Layout

| Path | Contents |
|---|---|
| [`int4gemv/quant.py`](int4gemv/quant.py) | int4 format: group-wise asymmetric, 8 codes per int32 packed along K. Quantize / pack / unpack, plus a PyTorch reference GEMV |
| [`int4gemv/kernel.py`](int4gemv/kernel.py) | Triton kernel. The zero point is factored out of the inner sum, so dequantization costs one multiply and one subtract per row per K step |
| [`tests/`](tests/) | Correctness tests: pack round-trip, quantization error bound, and kernel vs reference over several shapes |
| [`bench/bench_gemv.py`](bench/bench_gemv.py) | Times fp16 cuBLAS, PyTorch dequantize-then-matmul, and the Triton kernel (with a config sweep) on every decode linear layer of Qwen2.5-1.5B and -7B. Reports bandwidth against the GPU's measured copy bandwidth |
| [`colab/gemv_bench.ipynb`](colab/gemv_bench.ipynb) | Runs the tests and the benchmark on a Colab GPU |

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
