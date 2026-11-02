#!/usr/bin/env python3
"""Qwen2.5-7B-Instruct on vLLM: BF16 vs FP8 vs AWQ int4 vs GPTQ int4 -- speed and accuracy.

One vLLM server per config, all measured by the same client:
  speed     in 512 / out 128, closed loop at C = 1, 4, 16, 64 (serve_client)
  wikitext  WikiText-2 test, 100 chunks of 1024 tokens -> perplexity (accuracy_http)
  mmlu      MMLU test, every 3rd question (4,681), zero-shot letter choice (accuracy_http)
The two AWQ configs load the same weights; `awq_nomarlin` forces vLLM's older AWQ kernel
instead of Marlin, so the difference between them is the kernel alone.

usage:
  python bench/quant_sweep.py --models-dir /content/models --python /content/venv-vllm/bin/python
  add --quick for a short smoke run; --configs to pick a subset
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy_http  # noqa: E402
import serve_client  # noqa: E402
from serve_sweep import PORT, Prompts, engine_version, gpu_mem_used_mib, stop, venv_env, wait_ready  # noqa: E402

MEM = 0.90      # BF16 weights take ~15 GB of the L4's 22.5 GB; leave the rest for KV cache
MAX_LEN = 4096
BF16 = "Qwen2.5-7B-Instruct"
CONFIGS = {  # name -> (model directory under --models-dir, extra vLLM flags)
    "bf16": (BF16, ["--dtype", "bfloat16"]),
    "fp8": (BF16, ["--quantization", "fp8"]),
    "awq": ("Qwen2.5-7B-Instruct-AWQ", []),
    "awq_nomarlin": ("Qwen2.5-7B-Instruct-AWQ", ["--quantization", "awq"]),
    "gptq": ("Qwen2.5-7B-Instruct-GPTQ-Int4", []),
}
KERNEL_LOG = re.compile(r"marlin|quantization|kernel", re.I)


def load_eval_data(tokenizer, quick):
    from datasets import load_dataset
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    items = [mmlu[i] for i in range(0, len(mmlu), 3)]
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("".join(wiki["text"]), add_special_tokens=False).input_ids
    chunks = [ids[i * 1024:(i + 1) * 1024] for i in range(100)]
    if quick:
        items, chunks = items[:100], chunks[:10]
    return items, chunks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--python", required=True, help="the vLLM virtualenv's python")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--out", default="results/quant")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    version = engine_version("vllm", a.python)
    if version is None:
        sys.exit(1)
    P = Prompts(os.path.join(a.models_dir, BF16))
    items, chunks = load_eval_data(P.tok, a.quick)
    url = f"http://127.0.0.1:{PORT}/v1/completions"
    conc = [1, 16] if a.quick else [1, 4, 16, 64]
    print(f"vllm {version}; MMLU items {len(items)}, WikiText chunks {len(chunks)}", flush=True)

    meta = {"engine": "vllm", "version": version, "mem_fraction": MEM, "max_len": MAX_LEN,
            "quick": a.quick, "mmlu_items": len(items), "wikitext_chunks": len(chunks), "configs": {}}
    failures = []
    for name in a.configs:
        model_dir, flags = CONFIGS[name]
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "server.log")
        cmd = [a.python, "-m", "vllm.entrypoints.openai.api_server", "--model", os.path.join(a.models_dir, model_dir),
               "--served-model-name", "model", "--port", str(PORT), "--gpu-memory-utilization", str(MEM),
               "--max-model-len", str(MAX_LEN)] + flags
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env=venv_env(a.python))
        try:
            startup = wait_ready(proc, log_path, timeout=1200)
            asyncio.run(serve_client.run(url, [(x, 32) for x in P.make(4, 256)], 4))
            with open(log_path) as f:
                kernel_lines = [ln.strip()[:300] for ln in f if KERNEL_LOG.search(ln)][:20]
            meta["configs"][name] = {"model": model_dir, "flags": flags, "startup_s": startup,
                                     "gpu_mem_used_mib": gpu_mem_used_mib(), "kernel_log": kernel_lines}
            print(f"-- {name}: ready in {startup:.0f} s", flush=True)

            for c in conc:
                n = max(4, max(16, 4 * c) // (4 if a.quick else 1))
                records, wall = asyncio.run(serve_client.run(url, [(x, 128) for x in P.make(n, 512)], c))
                for r in records:
                    r["itl"] = [round(g, 5) for g in r["itl"]]
                s = serve_client.summarize(records, wall)
                with open(os.path.join(out_dir, f"speed_c{c}.json"), "w") as f:
                    json.dump({"config": name, "concurrency": c, "summary": s, "records": records}, f, indent=1)
                print(f"   speed c{c:<3} {s['output_tok_per_s']:7.0f} tok/s  TPOT p50 {s['tpot_p50_ms']:6.1f} ms  "
                      f"TTFT p50 {s['ttft_p50_ms']:6.0f} ms", flush=True)

            nll = asyncio.run(accuracy_http.chunk_nll(url, chunks))
            ppl = float(__import__("math").exp(sum(nll) / len(nll)))
            with open(os.path.join(out_dir, "wikitext.json"), "w") as f:
                json.dump({"config": name, "ppl": ppl, "chunk_nll": nll}, f, indent=1)
            res = asyncio.run(accuracy_http.mmlu(url, items))
            acc = sum(r["correct"] for r in res) / len(res)
            no_letter = sum(r["pred"] is None for r in res)
            with open(os.path.join(out_dir, "mmlu.json"), "w") as f:
                json.dump({"config": name, "accuracy": acc, "no_letter": no_letter, "items": res}, f)
            print(f"   WikiText-2 PPL {ppl:.3f}   MMLU {100 * acc:.2f}% (no letter in top-20: {no_letter})", flush=True)
        except Exception as e:  # one bad config should not cost the rest of the run
            msg = f"{name}: {type(e).__name__}: {e}"
            print(f"!! {msg}\n   last lines of {log_path}:", flush=True)
            with open(log_path) as f:
                print("".join(f.readlines()[-15:]), flush=True)
            meta["configs"].setdefault(name, {})["error"] = msg
            failures.append(msg)
        finally:
            stop(proc)
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
