"""
Plot evaluation results from eval_results.json as a grouped bar chart.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
results_path = BASE_DIR / "eval_results.json"

with open(results_path) as f:
    data = json.load(f)

overall = data["overall"]
top_k = data["config"]["top_k"]

conditions = list(overall.keys())
metrics    = [f"Recall@{top_k}", f"Precision@{top_k}", f"MRR@{top_k}"]
metric_labels = [f"Recall@{top_k}", f"Precision@{top_k}", f"MRR@{top_k}"]

# Short labels for the legend
short_labels = {
    "BM25 only":           "BM25",
    "Dense only":          "Dense",
    "Hybrid (2:1 D:B RRF)":"Hybrid",
    "Hybrid + Reranker":   "Hybrid + Reranker",
}

values = {c: [overall[c][m] for m in metrics] for c in conditions}

# ── Plot ──────────────────────────────────────────────────────────────────────
x      = np.arange(len(metrics))
n      = len(conditions)
width  = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(10, 6))

bars = []
for i, (cond, color) in enumerate(zip(conditions, colors)):
    b = ax.bar(x + offsets[i], values[cond], width,
               label=short_labels.get(cond, cond), color=color,
               edgecolor="white", linewidth=0.5)
    bars.append(b)
    for rect, val in zip(b, values[cond]):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    f"Legal RAG — Retrieval Evaluation  (N=50, K={top_k})\n"
    "BM25  |  Dense  |  Hybrid (2:1 Dense:BM25 RRF)  |  Hybrid + Reranker",
    fontsize=12, pad=14,
)
ax.set_ylim(0, max(v for vs in values.values() for v in vs) * 1.25)
ax.legend(fontsize=10, framealpha=0.9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = BASE_DIR / "eval_results.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

# ── Per-dataset subplots ──────────────────────────────────────────────────────
per_ds = data.get("per_dataset", {})
if per_ds:
    datasets = sorted(per_ds.keys())
    fig2, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 6), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax2, ds in zip(axes, datasets):
        ds_data = per_ds[ds]
        ds_vals = {c: [ds_data[c][m] for m in metrics] for c in conditions}
        for i, (cond, color) in enumerate(zip(conditions, colors)):
            b = ax2.bar(x + offsets[i], ds_vals[cond], width,
                        label=short_labels.get(cond, cond), color=color,
                        edgecolor="white", linewidth=0.5)
            for rect, val in zip(b, ds_vals[cond]):
                ax2.text(rect.get_x() + rect.get_width() / 2,
                         rect.get_height() + 0.005,
                         f"{val:.2f}", ha="center", va="bottom",
                         fontsize=7, color="#333333")

        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels, fontsize=10)
        ax2.set_title(ds, fontsize=12)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.set_axisbelow(True)
        ax2.set_ylim(0, 1.1)

    axes[0].set_ylabel("Score", fontsize=12)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig2.legend(handles, lbls, loc="lower center", ncol=4,
                fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig2.suptitle(f"Per-Dataset Breakdown  (K={top_k})", fontsize=13, y=1.02)
    plt.tight_layout()
    out2 = BASE_DIR / "eval_results_per_dataset.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
