#!/usr/bin/env python3
"""Issue #116 Task 5 (2026-08-10): 从 issue116_audit.json 生成 5 张 curve plot.

用法:
  python3 scripts/plot_issue116_curves.py <product_dir>

输入: <product_dir>/issue116_audit.json (由 taskA/stage2.py 末尾自动生成)
输出: <product_dir>/plots/{kappa,util,entropy,saturation,margin}_curve.png

5 张图:
  1. kappa_curve.png       — κ per-layer per-epoch (含 anchor baseline)
  2. util_curve.png        — util_3digit per-layer per-epoch
  3. entropy_curve.png     — assign_entropy per-layer per-epoch
  4. saturation_curve.png  — S_top1 + S_all per-layer per-epoch (Task 2)
  5. margin_curve.png      — top1-top2 margin per-layer per-epoch
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI, 后台跑
import matplotlib.pyplot as plt


PLOT_KEYS = {
    "kappa_curve.png": "kappa",
    "util_curve.png": "util_3digit",
    "entropy_curve.png": "assign_entropy",
    "saturation_curve.png": ("top1_clip_ratio", "all_pair_clip_ratio"),
    "margin_curve.png": "top1_top2_margin_median",
}


def main():
    if len(sys.argv) != 2:
        print("用法: python3 scripts/plot_issue116_curves.py <product_dir>")
        sys.exit(1)

    product_dir = Path(sys.argv[1])
    audit_path = product_dir / "issue116_audit.json"
    if not audit_path.exists():
        print(f"❌ 找不到 {audit_path}")
        print("   请先跑 taskA/stage2.py 生成 issue116_audit.json")
        sys.exit(2)

    with open(audit_path) as f:
        audit = json.load(f)

    epochs = audit["epochs"]
    if not epochs:
        print("❌ issue116_audit.json 中 epochs 为空")
        sys.exit(3)

    n_layers = len(epochs[0]["layers"])
    epoch_ids = [e["epoch"] for e in epochs]

    out_dir = product_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    layer_labels = [f"L{i} (K={epochs[0]['layers'][i]['n_e']})" for i in range(n_layers)]

    for fname, attr in PLOT_KEYS.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        if attr == "kappa":
            for l in range(n_layers):
                vals = [e["layers"][l]["kappa"] for e in epochs]
                ax.plot(epoch_ids, vals, marker="o", color=colors[l], label=layer_labels[l])
            anchor = audit["config"]["kappa_anchors"]
            if anchor:
                for l, a in enumerate(anchor):
                    ax.axhline(y=a, color=colors[l], linestyle=":", alpha=0.4,
                               label=f"anchor L{l}={a:.2f}")
            ax.set_ylabel(r"$\kappa_{eff}$")
            ax.set_title("Issue #116 κ trajectory (per-layer)")

        elif attr == "util_3digit":
            for l in range(n_layers):
                vals = [e["layers"][l]["util_3digit"] for e in epochs]
                ax.plot(epoch_ids, vals, marker="o", color=colors[l], label=layer_labels[l])
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="collapse < 0.5")
            ax.set_ylabel("util_3digit")
            ax.set_ylim(0, 1.05)
            ax.set_title("Issue #116 codebook utilization (per-layer)")

        elif attr == "assign_entropy":
            for l in range(n_layers):
                vals = [e["layers"][l]["assign_entropy"] for e in epochs]
                ax.plot(epoch_ids, vals, marker="o", color=colors[l], label=layer_labels[l])
            ax.set_ylabel("H(assign)")
            ax.set_title("Issue #116 assignment entropy (per-layer)")

        elif isinstance(attr, tuple) and attr == ("top1_clip_ratio", "all_pair_clip_ratio"):
            linestyles = ["-", "--"]
            for l in range(n_layers):
                for s, ls in zip(attr, linestyles):
                    vals = [e["layers"][l][s] for e in epochs]
                    ax.plot(epoch_ids, vals, marker="o", linestyle=ls,
                            color=colors[l], alpha=0.7,
                            label=f"L{l} {s}")
            ax.set_ylabel("S_top1 / S_all")
            ax.set_ylim(0, 1.05)
            ax.set_title("Issue #116 saturation (top-1 vs all-pair)")

        elif attr == "top1_top2_margin_median":
            for l in range(n_layers):
                vals = [e["layers"][l]["top1_top2_margin_median"] for e in epochs]
                ax.plot(epoch_ids, vals, marker="o", color=colors[l], label=layer_labels[l])
            ax.axhline(y=0.0, color="red", linestyle="--", alpha=0.4, label="margin=0 (collapse)")
            ax.set_ylabel("median(top1-top2 dist)")
            ax.set_title("Issue #116 top1-top2 margin (per-layer)")

        ax.set_xlabel("epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        out_path = out_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved {out_path}")

    # 额外画 C1 vs C2 κ gradient 比较图 (Task 4)
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in range(n_layers):
        c1 = [e["layers"][l].get("c1_vq_kappa_grad", 0.0) for e in epochs]
        c2 = [e["layers"][l].get("c2_rel_kappa_grad", 0.0) for e in epochs]
        ax.plot(epoch_ids, c2, marker="o", linestyle="-", color=colors[l],
                label=f"L{l} C2 (relational)")
        ax.plot(epoch_ids, c1, marker="x", linestyle="--", color=colors[l], alpha=0.5,
                label=f"L{l} C1 (VQ)")
    ax.set_yscale("log")
    ax.set_ylabel("|∂L/∂κ_drift| (log scale)")
    ax.set_title("Issue #116 Task 4 — C1 (VQ-driven) vs C2 (relational-driven) κ signal")
    ax.set_xlabel("epoch")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    out_path = out_dir / "c1_vs_c2_kappa_grad.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] saved {out_path}")

    print(f"\n[OK] 6 plots saved to {out_dir}/")


if __name__ == "__main__":
    main()