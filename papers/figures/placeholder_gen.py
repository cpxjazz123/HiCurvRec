"""Generate placeholder figures for §1.2 experiment-backed expansion.
- residual_norm_distribution.pdf   (Issue A residual norm histogram)
- per_layer_imbalance.pdf          (Issue C fixed-κ imbalance)
- poincare_volume_curves.pdf      (Issue F volume growth)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Configure CJK font fallback
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                                    "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# Figure 1: residual norm per-layer histogram (Issue A)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
np.random.seed(0)

# Placeholder distributions (pre-experiment estimates, single seed)
L0 = np.random.normal(12.4, 1.2, 500)
L1 = np.random.normal(7.4, 0.8, 500)
L2 = np.random.normal(3.6, 0.4, 500)

bins = np.linspace(0, 16, 40)
ax.hist(L0, bins=bins, alpha=0.5, color="#1f77b4", label="L0 (coarse)")
ax.hist(L1, bins=bins, alpha=0.5, color="#2ca02c", label="L1 (medium)")
ax.hist(L2, bins=bins, alpha=0.5, color="#d62728", label="L2 (fine)")
ax.set_xlabel(r"$\|r_\ell\|$ after training", fontsize=10)
ax.set_ylabel("Frequency (placeholder)", fontsize=10)
ax.set_title("Per-layer residual norm distribution (placeholder, single seed=42)",
             fontsize=10)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig("/root/GeneRec/generec/papers/figures/residual_norm_distribution.pdf",
            bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: per-layer imbalance under fixed κ (Issue C)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=False)
kappas = [0.1, 0.5, 1.0, 2.0, 5.0]
kappa_labels = ["κ=0.1", "κ=0.5", "κ=1.0", "κ=2.0", "κ=5.0"]
metrics = ["Codebook util. (%)", "Max load",
           "Mean codeword dist", "SID prefix share (%)"]

# Placeholder values per (metric, κ)
# rows: metrics, cols: κ values
placeholder_data = {
    "Codebook util. (%)": {
        "L0": [78, 88, 92, 96, 98],
        "L1": [82, 90, 94, 97, 99],
        "L2": [45, 78, 88, 92, 96],
    },
    "Max load": {
        "L0": [40, 26, 20, 14, 10],
        "L1": [32, 22, 16, 12,  8],
        "L2": [55, 28, 19, 13,  9],
    },
    "Mean codeword dist": {
        "L0": [3.2, 2.4, 1.8, 1.3, 1.0],
        "L1": [2.6, 1.9, 1.4, 1.0, 0.8],
        "L2": [2.0, 1.4, 1.0, 0.7, 0.5],
    },
    "SID prefix share (%)": {
        "L0": [62, 42, 28, 19, 14],
        "L1": [58, 40, 27, 18, 13],
        "L2": [55, 38, 25, 17, 12],
    },
}

x = np.arange(len(kappa_labels))
width = 0.27
colors = {"L0": "#1f77b4", "L1": "#2ca02c", "L2": "#d62728"}

for i, (ax, metric) in enumerate(zip(axes, metrics)):
    data = placeholder_data[metric]
    ax.bar(x - width, data["L0"], width, color=colors["L0"], label="L0")
    ax.bar(x,         data["L1"], width, color=colors["L1"], label="L1")
    ax.bar(x + width, data["L2"], width, color=colors["L2"], label="L2")
    ax.set_xticks(x)
    ax.set_xticklabels(kappa_labels, fontsize=8)
    ax.set_title(metric, fontsize=9)
    ax.set_xlabel("Fixed κ", fontsize=9)
    if i == 0:
        ax.set_ylabel("Value (placeholder)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Per-layer imbalance under fixed κ (placeholder data, single seed=42)",
             fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/root/GeneRec/generec/papers/figures/per_layer_imbalance.pdf",
            bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: Poincaré volume growth curves (Issue F)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
dims = [32, 64, 128]
r = np.linspace(0, 5, 200)
kappas = [0.1, 0.5, 1.0, 2.0, 5.0]
cmap = plt.cm.viridis
colors = [cmap(i / (len(kappas) - 1)) for i in range(len(kappas))]

for ax, d in zip(axes, dims):
    for κ, col in zip(kappas, colors):
        # Approximation V_c(r) ∝ exp((d-1) * sqrt(κ) * r) (normalized)
        v = np.exp((d - 1) * np.sqrt(κ) * r)
        v = v / v[0]  # normalize at r=0
        ax.plot(r, v, label=f"κ={κ}", color=col, lw=1.4)
    ax.set_xlabel("r", fontsize=10)
    ax.set_title(f"d = {d}", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("V_c(r) (log scale, normalized)", fontsize=9)
axes[-1].legend(fontsize=8, loc="upper left")

fig.suptitle("Hyperbolic volume growth rate V_c(r) (placeholder analytical curves)",
             fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/root/GeneRec/generec/papers/figures/poincare_volume_curves.pdf",
            bbox_inches="tight")
plt.close(fig)

print("Generated 3 placeholder figures in /root/GeneRec/generec/papers/figures/")