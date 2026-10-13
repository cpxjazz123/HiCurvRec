"""Plot the three non-collapsed competition components from the result table."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/positive_negative_competition"


def main():
    table = pd.read_csv(RESULTS / "competition_metrics.csv")
    arms = list(dict.fromkeys(table["arm"]))
    levels = ("L1", "L2", "L3")
    colors = {level: color for level, color in zip(levels, ("#3366aa", "#dd8844", "#228855"))}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for level in levels:
        subset = table[table["level"] == level].set_index("arm").reindex(arms)
        axes[0].plot(arms, subset["pncr_epsilon_0.10"], marker="o", label=level, color=colors[level])
        axes[1].plot(arms, subset["error_rate_margin"], marker="o", label=level, color=colors[level])
        axes[2].plot(arms, subset["positive_mean_distance_over_local_scale"], marker="o",
                     label=f"{level} positive", color=colors[level])
        axes[2].plot(arms, subset["negative_mean_distance_over_local_scale"], marker="x",
                     linestyle="--", label=f"{level} negative", color=colors[level])
    axes[0].set_title("Closeness: PNCR, |m| < 0.10")
    axes[1].set_title("Intrusion: m < 0")
    axes[2].set_title("Compression: distance / local scale")
    for axis in axes:
        axis.set_xticks(range(len(arms)), arms, rotation=90)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = RESULTS / "competition_decomposition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
