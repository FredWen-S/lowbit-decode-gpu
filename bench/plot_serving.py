#!/usr/bin/env python3
"""Concurrency sweep figure: vLLM vs SGLang, from results/serving_L4/<engine>/default/concurrency_c*.json.

usage: python bench/plot_serving.py [results/serving_L4]   (needs matplotlib) -> <dir>/concurrency.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/serving_L4"
CONC = [1, 2, 4, 8, 16, 32, 64]
ENGINES = {"vllm": ("vLLM", "#2a78d6"), "sglang": ("SGLang", "#eb6834")}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PANELS = [  # key, title, y label
    ("output_tok_per_s", "Same throughput to C = 32; vLLM +25% at 64", "output tokens / s"),
    ("tpot_p50_ms", "Per-token time doubles past C = 16", "TPOT p50 (ms)"),
    ("ttft_p50_ms", "SGLang queues prefills: higher TTFT", "TTFT p50 (ms)"),
    ("itl_p99_ms", "vLLM stalls decode: higher gap p99", "inter-token gap p99 (ms)"),
]

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

data = {e: [json.load(open(f"{ROOT}/{e}/default/concurrency_c{c}.json"))["summary"] for c in CONC]
        for e in ENGINES}
versions = {e: json.load(open(f"{ROOT}/{e}/meta.json"))["version"] for e in ENGINES}

fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), dpi=200)
for ax, (key, title, ylabel) in zip(axes, PANELS):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for e, (name, color) in ENGINES.items():
        ax.plot(CONC, [s[key] for s in data[e]], color=color, lw=2, marker="o", ms=5, mec=SURF, mew=1.2,
                label=f"{name} {versions[e]}")
    ax.set_xscale("log", base=2); ax.set_xticks(CONC); ax.set_xticklabels(CONC)
    ax.set_xlabel("concurrent requests"); ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", color=INK, fontsize=9.5)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Qwen2.5-1.5B bf16 on an NVIDIA L4: 512-token prompts, 128 output tokens, closed loop",
             x=0.01, ha="left", fontsize=9, color=INK2)
fig.tight_layout(w_pad=2.2)
out = f"{ROOT}/concurrency.png"
fig.savefig(out)
print("wrote", out)
