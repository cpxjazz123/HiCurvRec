"""Regenerate Figure "poincare_volume_curves.pdf" — asymptotic volume growth.

ASCII-only labels: the CJK fallback font is not embedded by matplotlib's PDF
backend, which previously rendered CJK glyphs as boxes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

kappas  = [0.1, 0.5, 1.0, 2.0, 5.0]
colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
r_vals  = np.linspace(0.05, 3.5, 500)   # cap at 3.5 to avoid exp overflow at d=128, kappa=5

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
dimensions = [32, 64, 128]

for ax, d in zip(axes, dimensions):
    for kappa, col in zip(kappas, colors):
        # log10 V = (d-1)*sqrt(kappa)*r / ln(10)  — avoids float overflow
        log10V = (d - 1) * np.sqrt(kappa) * r_vals / np.log(10)
        ax.plot(r_vals, log10V, color=col, label=rf"$\kappa={kappa}$")
    ax.set_xlabel(r"$r$ (geodesic radius)", fontsize=9)
    ax.set_ylabel(r"$\log_{10} V_c(r)$", fontsize=9)
    ax.set_title(f"$d={d}$", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

fig.suptitle("Hyperbolic volume growth: $\\log_{10} V_c(r) \\propto (d{-}1)\\sqrt{c}\\,r$",
             fontsize=10)
plt.tight_layout()
fig.savefig("/root/GeneRec/generec/papers/figures/poincare_volume_curves.pdf",
            bbox_inches="tight")
plt.close(fig)
print("Regenerated poincare_volume_curves.pdf")
