#!/usr/bin/env python3
"""Quantization summary: throughput relative to BF16 vs concurrency, and accuracy cost.

Reads results/quant_L4/<cfg>/ (BF16, GPTQ; AWQ from the complete `awq_nomarlin` run, which
vLLM 0.32 also ran on Marlin) and results/quant_L4_fp8/fp8/.
usage: python bench/plot_quant.py   (needs matplotlib) -> results/quant_L4/summary.png
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
DIRS = {"bf16": f"{R}/quant_L4/bf16", "fp8": f"{R}/quant_L4_fp8/fp8",
        "awq": f"{R}/quant_L4/awq_nomarlin", "gptq": f"{R}/quant_L4/gptq"}
STYLE = {"fp8": ("FP8 (W8A8)", "#1baf7a"), "awq": ("AWQ int4", "#2a78d6"), "gptq": ("GPTQ int4", "#eb6834")}
CONC = [1, 4, 16, 64]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def tput(cfg, c):
    return json.load(open(f"{DIRS[cfg]}/speed_c{c}.json"))["summary"]["output_tok_per_s"]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    return mu, math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1) / n)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=200)

ax1.grid(axis="y", color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
for cfg, (name, color) in STYLE.items():
    ys = [tput(cfg, c) / tput("bf16", c) for c in CONC]
    ax1.plot(CONC, ys, color=color, lw=2, marker="o", ms=5, mec=SURF, mew=1.2, label=name)
    dy = {"gptq": 5, "awq": -6}.get(cfg, 0)  # the two int4 lines end 0.03 apart
    ax1.annotate(f"{ys[-1]:.2f}×", (CONC[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                 va="center", fontsize=7.5, color=color)
ax1.set_xscale("log", base=2); ax1.set_xticks(CONC); ax1.set_xticklabels(CONC); ax1.set_xlim(0.8, 100)
ax1.set_ylim(0, 3.3)
ax1.set_xlabel("concurrent requests"); ax1.set_ylabel("output throughput vs BF16")
ax1.set_title("int4 wins at batch 1, FP8 once prefill dominates", loc="left", color=INK, fontsize=9.5)
ax1.legend(frameon=False, fontsize=8, loc="upper right")

ax2.grid(color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
bw = json.load(open(f"{DIRS['bf16']}/wikitext.json"))["chunk_nll"]
bm = [r["correct"] for r in json.load(open(f"{DIRS['bf16']}/mmlu.json"))["items"]]
for cfg, (name, color) in STYLE.items():
    dn, sn = paired(json.load(open(f"{DIRS[cfg]}/wikitext.json"))["chunk_nll"], bw)
    dm, sm = paired([r["correct"] for r in json.load(open(f"{DIRS[cfg]}/mmlu.json"))["items"]], bm)
    x, xe = 100 * math.expm1(dn), 100 * sn * math.exp(dn)
    ax2.errorbar(x, 100 * dm, xerr=1.96 * xe, yerr=1.96 * 100 * sm, fmt="o", color=color, ms=6,
                 mec=SURF, mew=1.2, elinewidth=1, capsize=2)
    ax2.annotate(name, (x, 100 * dm), xytext=(7, 4), textcoords="offset points", fontsize=8, color=color)
ax2.set_xlim(-0.5, 8); ax2.set_ylim(-2.2, 0.9)
ax2.set_xlabel("WikiText-2 perplexity vs BF16 (%)"); ax2.set_ylabel("MMLU vs BF16 (pp, paired)")
ax2.set_title("FP8 is free; int4 methods trade PPL vs MMLU", loc="left", color=INK, fontsize=9.5)

fig.suptitle("Qwen2.5-7B-Instruct on vLLM 0.32, NVIDIA L4 (95% intervals; MMLU over 4,681 items)",
             x=0.01, ha="left", fontsize=9, color=INK2)
fig.tight_layout(w_pad=3)
fig.savefig(f"{R}/quant_L4/summary.png")
print("wrote", f"{R}/quant_L4/summary.png")
