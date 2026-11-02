#!/usr/bin/env python3
"""Mixed workload: every inter-token gap of the 8 short streams over time, per engine and chunk size.

usage: python bench/plot_gaps.py [results/serving_L4_mixed3]   (needs matplotlib) -> <dir>/gaps.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/serving_L4_mixed3"
ENGINES = {"vllm": ("vLLM", "#2a78d6"), "sglang": ("SGLang", "#eb6834")}
CHUNKS = {"chunk512": "512-token prefill chunks", "chunk4096": "4096-token chunks (one step)"}
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b3"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), dpi=200, sharey=True)
for row, (engine, (name, color)) in enumerate(ENGINES.items()):
    for col, (cfg, label) in enumerate(CHUNKS.items()):
        ax = axes[row][col]
        d = json.load(open(f"{ROOT}/{engine}/{cfg}/mixed_mixed.json"))
        t0 = min(r["start"] for r in d["records"])
        for r in d["records"]:
            if r["kind"] != "main":
                ax.axvline(r["start"] - t0, color=MUTED, lw=0.8, zorder=0)
                continue
            t = r["start"] + r["ttft"] - t0
            xs, ys = [], []
            for g in r["itl"]:
                t += g
                xs.append(t); ys.append(1e3 * g)
            ax.scatter(xs, ys, s=2, color=color, alpha=0.5, linewidths=0)
        s = d["summary"]
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_ylim(0, 340)
        ax.set_title(f"{name}, {label}: max {s['itl_max_ms']:.0f} ms, "
                     f"{s['gaps_over_100ms']} gaps > 100 ms", loc="left", color=INK, fontsize=9)
        if col == 0:
            ax.set_ylabel("inter-token gap (ms)")
        if row == 1:
            ax.set_xlabel("time (s); grey lines = a new 3.2k-token prompt arrives")
fig.suptitle("Do long prefills stall running streams? 8 streams decoding, plus a fresh 3.2k-token prompt every 2 s "
             "(Qwen2.5-1.5B, L4)", x=0.01, ha="left", fontsize=9, color=INK2)
fig.tight_layout(h_pad=1.5, w_pad=2)
out = f"{ROOT}/gaps.png"
fig.savefig(out)
print("wrote", out)
