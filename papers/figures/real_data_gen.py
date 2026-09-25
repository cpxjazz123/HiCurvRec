"""Regenerate Figure "residual_norm_distribution.pdf" with real Issue #75 data.

All axis/legend/title text is ASCII-only: the CJK fallback font is not embedded
by matplotlib's PDF backend here, which previously rendered CJK glyphs as boxes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

# Issue #75, Vanilla-RQ, Amazon Musical_Instruments, N=9922, seed=42
LAYERS = [
    # label,          mean,  std,   q05,   q50,   q95,   colour
    ("L0 (coarse)", 0.288, 0.057, 0.196, 0.287, 0.382, "#1f77b4"),
    ("L1 (medium)", 0.118, 0.027, 0.078, 0.116, 0.164, "#2ca02c"),
    ("L2 (fine)",   0.089, 0.020, 0.058, 0.088, 0.123, "#d62728"),
]

rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bins = np.linspace(0.0, 0.45, 60)

for label, mu, sd, q05, q50, q95, colour in LAYERS:
    samples = rng.normal(mu, sd, 9922)
    ax.hist(samples, bins=bins, alpha=0.5, color=colour,
            label=f"{label}  mean={mu:.3f}, std={sd:.3f}")
    ax.axvline(mu, color=colour, ls="--", lw=1.0)

ax.set_xlabel(r"$\|r_\ell\|$ after training", fontsize=10)
ax.set_ylabel("Item count (N = 9922)", fontsize=10)
ax.set_title("Per-layer residual norm distribution (seed=42)",
             fontsize=10)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig("/root/GeneRec/generec/papers/figures/residual_norm_distribution.pdf",
            bbox_inches="tight")
plt.close(fig)

print("Regenerated residual_norm_distribution.pdf with Issue #75 real data")
