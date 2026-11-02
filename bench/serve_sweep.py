#!/usr/bin/env python3
"""vLLM vs SGLang on one GPU: start each engine with each config, run the experiments, stop it.

Each engine lives in its own virtualenv (their torch pins conflict); this script and the load
generator run in a third Python and talk to the servers over HTTP, so both engines are measured
by the same client.

Experiments (prompt / output lengths in tokens; prompts are random token ids):
  concurrency  in 512, out 128, closed loop at C = 1..64           -> TTFT, TPOT, throughput
  input_len    in 128 and 2048, out 128, C = 16
  prefix       1536-token shared prefix + 64 unique, out 64, C = 8 -> TTFT with/without prefix cache
  mixed        8 streams of in 128 / out 512, plus a new 3072-token prompt every 2 s
               -> decode stalls (TPOT, inter-token p99) under small vs large prefill chunks

usage:
  python bench/serve_sweep.py --model /content/models/qwen2.5-1.5b-instruct \
      --python vllm=/content/venv-vllm/bin/python --python sglang=/content/venv-sglang/bin/python
  add --quick for a 5-minute smoke run
"""
import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_client  # noqa: E402

PORT = 8100
MEM = 0.85          # fraction of GPU memory each engine may take for weights + KV cache
MAX_LEN = 4096

# engine -> config -> extra server flags
FLAGS = {
    "vllm": {
        "default": [],
        "nocache": ["--no-enable-prefix-caching"],
        "chunk512": ["--max-num-batched-tokens", "512"],
        "chunk4096": ["--max-num-batched-tokens", "4096"],
    },
    "sglang": {
        "default": [],
        "nocache": ["--disable-radix-cache"],
        "chunk512": ["--chunked-prefill-size", "512"],
        "chunk4096": ["--chunked-prefill-size", "4096"],
    },
}
CONFIG_EXPERIMENTS = {
    "default": ["concurrency", "input_len", "prefix"],
    "nocache": ["prefix"],
    "chunk512": ["mixed"],
    "chunk4096": ["mixed"],
}


def server_cmd(engine, python, model):
    if engine == "vllm":
        return [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
                "--served-model-name", "model", "--port", str(PORT),
                "--gpu-memory-utilization", str(MEM), "--max-model-len", str(MAX_LEN)]
    return [python, "-m", "sglang.launch_server", "--model-path", model,
            "--served-model-name", "model", "--port", str(PORT),
            "--mem-fraction-static", str(MEM), "--context-length", str(MAX_LEN)]


def venv_env(python):
    """Environment as if the engine's virtualenv were activated. Engines JIT-compile kernels at
    startup and call tools installed next to their Python (flashinfer runs `ninja`)."""
    env = dict(os.environ)
    bin_dir = os.path.dirname(os.path.abspath(python))
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.dirname(bin_dir)
    return env


def engine_version(engine, python):
    """The engine's version string, or None if it cannot be imported from that Python."""
    out = subprocess.run([python, "-c", f"import {engine}; print({engine}.__version__)"],
                         capture_output=True, text=True, env=venv_env(python))
    if out.returncode != 0:
        print(f"!! {engine} cannot be imported from {python}:\n{out.stderr.strip()[-1500:]}", flush=True)
        return None
    return out.stdout.strip()


def wait_ready(proc, log_path, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}, see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server not ready after {timeout} s, see {log_path}")


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    time.sleep(10)  # let the driver release GPU memory before the next server starts


def gpu_mem_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return int(out.stdout.split()[0]) if out.returncode == 0 else None


class Prompts:
    def __init__(self, model, seed=0):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.rng = random.Random(seed)

    def ids(self, n):
        # ordinary-token range; random ids keep the engines from benefiting from any real text
        return [self.rng.randrange(1000, 30000) for _ in range(n)]

    def make(self, n, length, prefix=None):
        return [self.tok.decode((prefix or []) + self.ids(length)) for _ in range(n)]


def experiment_points(name, P, quick):
    """Yield (point_name, requests, concurrency, background, interval)."""
    def n(full):
        return max(4, full // 4) if quick else full

    if name == "concurrency":
        for c in ([1, 8, 32] if quick else [1, 2, 4, 8, 16, 32, 64]):
            k = n(max(16, 4 * c))
            yield f"c{c}", [(p, 128) for p in P.make(k, 512)], c, None, None
    elif name == "input_len":
        for length in (128, 2048):
            yield f"in{length}", [(p, 128) for p in P.make(n(64), length)], 16, None, None
    elif name == "prefix":
        prefix = P.ids(1536)
        yield "shared1536", [(p, 64) for p in P.make(n(64), 64, prefix=prefix)], 8, None, None
    elif name == "mixed":
        shorts = [(p, 512) for p in P.make(n(24), 128)]
        # a fresh long prompt per arrival; one repeated prompt is a prefix-cache hit after the first
        longs = [(p, 16) for p in P.make(40, 3072)]
        yield "mixed", shorts, 8, longs, 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--python", action="append", required=True, help="engine=path/to/python")
    p.add_argument("--configs", nargs="+", default=list(CONFIG_EXPERIMENTS))
    p.add_argument("--out", default="results/serving")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    pythons = dict(x.split("=", 1) for x in a.python)
    P = Prompts(a.model)
    url = f"http://127.0.0.1:{PORT}/v1/completions"

    failures = []
    for engine, python in pythons.items():
        meta = {"engine": engine, "version": engine_version(engine, python), "model": a.model,
                "mem_fraction": MEM, "max_len": MAX_LEN, "quick": a.quick, "configs": {}}
        os.makedirs(os.path.join(a.out, engine), exist_ok=True)
        if meta["version"] is None:
            failures.append(f"{engine}: not importable")
            continue
        print(f"== {engine} {meta['version']}", flush=True)
        for config in a.configs:
            out_dir = os.path.join(a.out, engine, config)
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, "server.log")
            cmd = server_cmd(engine, python, a.model) + FLAGS[engine][config]
            with open(log_path, "w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env=venv_env(python))
            try:
                startup = wait_ready(proc, log_path)
                warm = [(x, 32) for x in P.make(4, 256)]
                asyncio.run(serve_client.run(url, warm, 4))
                meta["configs"][config] = {"flags": FLAGS[engine][config], "startup_s": startup,
                                           "gpu_mem_used_mib": gpu_mem_used_mib()}
                print(f"-- {config}: ready in {startup:.0f} s", flush=True)
                for exp in CONFIG_EXPERIMENTS[config]:
                    for point, reqs, c, bg, interval in experiment_points(exp, P, a.quick):
                        records, wall = asyncio.run(serve_client.run(url, reqs, c, bg, interval))
                        res = {"engine": engine, "config": config, "experiment": exp, "point": point,
                               "concurrency": c, "summary": serve_client.summarize(records, wall)}
                        if bg is not None:
                            res["background_summary"] = serve_client.summarize(records, wall, kind="periodic")
                        for r in records:  # keep every gap so tails can be re-read later
                            r["itl"] = [round(g, 5) for g in r["itl"]]
                        res["records"] = records
                        with open(os.path.join(out_dir, f"{exp}_{point}.json"), "w") as f:
                            json.dump(res, f, indent=1)
                        s = res["summary"]
                        print(f"   {exp:<12}{point:<11} {s['output_tok_per_s']:8.0f} tok/s  "
                              f"TTFT p50 {s['ttft_p50_ms']:7.0f} ms  TPOT p50 {s['tpot_p50_ms']:6.1f} ms  "
                              f"ITL p99 {s['itl_p99_ms']:6.1f} max {s['itl_max_ms']:6.1f} ms", flush=True)
            except Exception as e:  # one bad config should not cost the rest of the run
                msg = f"{engine}/{config}: {type(e).__name__}: {e}"
                print(f"!! {msg}\n   last lines of {log_path}:", flush=True)
                with open(log_path) as f:
                    print("".join(f.readlines()[-15:]), flush=True)
                meta["configs"].setdefault(config, {})["error"] = msg
                failures.append(msg)
            finally:
                stop(proc)
        with open(os.path.join(a.out, engine, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
