"""
render_task10_15_plots.py — 给 task12/15 出 PNG 图，补完成指标

- task10_decay_histogram.png：每维 m3/m1 decay 直方图（v1 + v2 同图比较）
- task11_drift_curves.png：每层 ||r|| 在 step 100-1600 的漂移曲线（v2 17 ckpts）
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def render_task12():
    out_dir = Path("/home/wlia0047/ar57/wenyu/GeneRec/task10_v2_results")
    # v2 用 v2 的 decay，v1 用 v1 的 decay
    decay_v1 = np.load("/home/wlia0047/ar57/wenyu/GeneRec/task10_results/task10_decay.npy")
    decay_v2 = np.load(out_dir / "task10_v2_decay.npy")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # v1 (normalize=True)
    ax = axes[0]
    ax.hist(decay_v1, bins=80, color="#3b82f6", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(decay_v1), color="red", linestyle="--", label=f"mean = {np.mean(decay_v1):.3f}")
    ax.set_xlabel("decay = m_last / m_first (per-dim)")
    ax.set_ylabel("# dimensions (D=2048)")
    ax.set_title(f"task12 v1 (normalize=True)\nCV = {np.std(decay_v1)/np.mean(decay_v1):.3f}, "
                 f"min={decay_v1.min():.3f}, max={decay_v1.max():.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # v2 (normalize=False)
    ax = axes[1]
    ax.hist(decay_v2, bins=80, color="#22c55e", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(decay_v2), color="red", linestyle="--", label=f"mean = {np.mean(decay_v2):.3f}")
    ax.set_xlabel("decay = m_last / m_first (per-dim)")
    ax.set_ylabel("# dimensions (D=2048)")
    ax.set_title(f"task12 v2 (normalize=False)\nCV = {np.std(decay_v2)/np.mean(decay_v2):.3f}, "
                 f"min={decay_v2.min():.3f}, max={decay_v2.max():.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Task 14: BD-RFSQ dimension decay (Toys, 5-layer RKMeans) — "
                 "CV ≈ 0.05 either way → uniform, BD-RFSQ benefit limited",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "task10_decay_histogram.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")
    print(f"  v1 CV = {np.std(decay_v1)/np.mean(decay_v1):.4f}, "
          f"v2 CV = {np.std(decay_v2)/np.mean(decay_v2):.4f}")


def render_task13():
    out_dir = Path("/home/wlia0047/ar57/wenyu/GeneRec/task11_v2_results")
    with open(out_dir / "task1_v2_drift_report.json") as f:
        report = json.load(f)
    curves = report["new_layer_curves"]
    summary = report["summary"]

    # Plot: per-layer ||r|| curve over step
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: ||r|| mean vs step (layer 0-3)
    ax = axes[0]
    colors = {0: "#3b82f6", 1: "#22c55e", 2: "#f97316", 3: "#a855f7"}
    for layer_idx in sorted(curves.keys(), key=int):
        curve = curves[layer_idx]
        steps = [c["step"] for c in curve]
        norms = [c["norm_mean"] for c in curve]
        ax.plot(steps, norms, "o-", color=colors[int(layer_idx)],
                label=f"layer {layer_idx} (first_step={curve[0]['step_offset_from_first']})", linewidth=2)
    ax.set_xlabel("global training step")
    ax.set_ylabel("||r|| mean (over items)")
    ax.set_title("Task 15 v2: ||r|| mean per layer over training\n"
                 "(normalize=True → ||r|| → 1 for layers 1-3)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Subplot 2: per-step mean_norm_shift (relative drift)
    ax = axes[1]
    for layer_idx in sorted(curves.keys(), key=int):
        curve = curves[layer_idx]
        steps = [c["step"] for c in curve[1:]]  # shift 是相对前一个
        shifts = [c.get("mean_norm_shift", 0) for c in curve[1:]]
        ax.plot(steps, shifts, "o-", color=colors[int(layer_idx)],
                label=f"layer {layer_idx}", linewidth=2)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.6, label="5% threshold")
    ax.set_xlabel("global training step")
    ax.set_ylabel("mean_norm_shift (relative)")
    ax.set_yscale("log")
    ax.set_title("Task 15 v2: new-layer internal drift (relative shift)\n"
                 "Layer 1/2/3 stabilize within 100 steps of layer start")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Task 15: residual training non-stationarity (Toys, 4-layer RKMeans) — "
                 "new-layer drift < 100 steps", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "task11_drift_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")
    # 打印关键收敛信息
    for layer_idx, s in summary.items():
        print(f"  layer {layer_idx}: total_rel_shift={s['total_rel_shift']:.4f}, "
              f"convergence_steps={s.get('convergence_steps')}")


if __name__ == "__main__":
    print("=== render task12 PNG ===")
    render_task12()
    print()
    print("=== render task13 PNG ===")
    render_task13()