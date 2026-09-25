# codebook_geometry.py
# Fig: RQ-VAE three-layer codebook geometry heterogeneity (SS 1.2 evidence)
# Panels: a) mean pairwise distance across unified codebook sizes
#         b) K=128: NN distance (down) + local density k=5 (up)
#         c) K=128: effective rank (up) + PC1 variance share (down)
# Export: SVG + PDF + TIFF (editable text, Arial, 5pt floor)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

BLUE_DARK = "#0F4D92"
BLUE_MID = "#3775BA"
BLUE_SOFT = "#B4C0E4"
BLACK = "#272727"
RED = "#B64342"
TEAL = "#42949E"

LAYER_COLORS = [BLUE_DARK, BLUE_MID, BLUE_SOFT]

ks = ["K = 64", "K = 128", "K = 256"]
pairwise = {
    "L0": [0.1922, 0.1946, 0.1993],
    "L1": [0.1059, 0.1060, 0.1035],
    "L2": [0.0856, 0.0839, 0.0799],
}

layers = ["L0", "L1", "L2"]
nn_dist = [0.1044, 0.0767, 0.0641]
local_density = [8.47, 12.46, 14.96]
eff_rank = [15.65, 22.17, 23.64]
pc1_var = [20.54, 8.36, 6.33]

fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.35), constrained_layout=True)

# ---- Panel a: pairwise distance (grouped bars) ----
ax = axes[0]
x = np.arange(3)
w = 0.26
for i, (layer, color) in enumerate(zip(layers, LAYER_COLORS)):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, pairwise[layer], width=w, color=color,
                  edgecolor="black", linewidth=0.5, label=layer)
    for bar, val in zip(bars, pairwise[layer]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=5.5)
ax.set_xticks(x)
ax.set_xticklabels(ks)
ax.set_ylim(0, 0.26)
ax.set_ylabel("Mean pairwise distance")
ax.set_title("Global scale", fontsize=7, pad=3)
ax.legend(loc="upper right", fontsize=6.2, ncols=1, handlelength=1.2, handletextpad=0.5, columnspacing=1.0)
ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontsize=9, fontweight="bold",
        ha="left", va="top")

# ---- Panel b: NN distance (down) + local density (up) ----
ax = axes[1]
ax.plot(layers, nn_dist, color=BLUE_DARK, marker="o", markersize=4.5, lw=1.4)
for xi, yi in zip(range(3), nn_dist):
    ax.text(xi, yi - 0.004, f"{yi:.4f}", ha="center", va="top", fontsize=5.5,
            color=BLUE_DARK)
ax.set_ylim(0.05, 0.115)
ax.set_ylabel("Mean NN distance")
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(layers, local_density, color=RED, marker="s", markersize=4.0, lw=1.4,
         ls="--")
for xi, yi in zip(range(3), local_density):
    ax2.text(xi, yi + 0.45, f"{yi:.2f}", ha="center", va="bottom", fontsize=5.5,
             color=RED)
ax2.set_ylim(6, 17)
ax2.set_ylabel("Local density (k = 5)", color=RED)
ax2.tick_params(axis="y", colors=RED)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(RED)
ax2.spines["right"].set_linewidth(0.8)
ax.set_title("Local structure", fontsize=7, pad=3)
ax.text(0.02, 0.97, "b", transform=ax.transAxes, fontsize=9, fontweight="bold",
        ha="left", va="top")

# ---- Panel c: effective rank (up) + PC1 variance (down) ----
ax = axes[2]
ax.plot(layers, eff_rank, color=TEAL, marker="o", markersize=4.5, lw=1.4)
for xi, yi in zip(range(3), eff_rank):
    ax.text(xi, yi + 0.45, f"{yi:.2f}", ha="center", va="bottom", fontsize=5.5,
            color=TEAL)
ax.set_ylim(13, 26)
ax.set_ylabel("Effective rank")
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(layers, pc1_var, color=BLACK, marker="^", markersize=4.5, lw=1.4,
         ls=":")
for xi, yi in zip(range(3), pc1_var):
    ax2.text(xi, yi - 0.85, f"{yi:.2f}%", ha="center", va="top", fontsize=5.5,
             color=BLACK)
ax2.set_ylim(2, 25)
ax2.set_ylabel("Var. explained by PC1 (%)", color=BLACK)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(0.8)
ax.set_title("Spectral shape", fontsize=7, pad=3)
ax.text(0.02, 0.97, "c", transform=ax.transAxes, fontsize=9, fontweight="bold",
        ha="left", va="top")

fig.savefig("codebook_geometry.svg", bbox_inches="tight")
fig.savefig("codebook_geometry.pdf", bbox_inches="tight")
fig.savefig("codebook_geometry.tiff", dpi=600, bbox_inches="tight")
print("saved svg/pdf/tiff")
