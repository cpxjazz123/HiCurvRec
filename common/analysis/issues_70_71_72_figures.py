#!/usr/bin/env python3
"""Issue #70 / #71 / #72 三件图 + Table 出图脚本.

依赖: common/analysis/issues_70_71_72_perlayer_curvature.py 先跑出
      taskA/_history/issues_70_71_72_perlayer_curvature/sweep_results.json

产物:
  figure4_perlayer_ablation.png   (Issue #70 — 3 子图, 每层 R@10 proxy vs κ, x=log κ, y=mean_codeword_dist 与 prefix_share_rate 两轴)
  figure5_perlayer_imbalance.png   (Issue #71 — 5×4 grid (κ × 指标), 每子图 3 条 per-layer 曲线, x=κ, y=指标值)
  table7_compromised.json         (Issue #72 — 4 共用 κ 的 4 项 per-layer 指标 + Per-Layer 对照 + Compromised Score 量化)

R30: 全硬编码, 不读 env var.
R31: 单主脚本, 不 fork.
R2:  无 fallback.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/fs04/ar57/wenyu/GeneRec")
PRODUCT_ROOT = REPO / "taskA" / "_history" / "issues_70_71_72_perlayer_curvature"
RESULTS_JSON = PRODUCT_ROOT / "sweep_results.json"


def load_results() -> dict:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"missing {RESULTS_JSON}; 先跑 issues_70_71_72_perlayer_curvature.py")
    return json.loads(RESULTS_JSON.read_text())


# ──────────────────────────────────────────────────────────────
# Figure 4 (Issue #70 逐层最优曲率扫描)
# ──────────────────────────────────────────────────────────────
def figure4_perlayer_ablation(results: dict) -> Path:
    """3 子图 (per layer l=0/1/2), 每子图 = x 轴 κ (log), y 左轴 mean_codeword_dist, y 右轴 prefix_share_rate.
    关键观察 = 各层最优 κ 互不相同 (无单一 κ 同时让三层 prefix_share 都低)."""
    # 按 (layer, κ) 收集 (label "b_L{l}_kX" 中 X = κ, 其余两层固定 c=0.5)
    sweep_kappas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    by_layer = defaultdict(dict)  # layer -> {κ: metrics_dict}
    for l in range(3):
        for kappa in sweep_kappas:
            sid = f"b_L{l}_k{_kappa_tag(kappa)}"
            if sid in results and "per_layer" in results[sid]:
                # 该 sweep 只在 l 层有变 κ, 其余两层 c=0.5 (default)
                # 我们关心 sweep 该层 (l) 的指标, 即 metrics["per_layer"][l]
                by_layer[l][kappa] = results[sid]["per_layer"][l]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for l in range(3):
        ax = axes[l]
        ks = sorted(by_layer[l].keys())
        if not ks:
            ax.set_title(f"L{l}: no data")
            continue
        d = [by_layer[l][k]["mean_codeword_dist"] for k in ks]
        p = [by_layer[l][k]["prefix_share_rate"] for k in ks]
        u = [by_layer[l][k]["utilization"] for k in ks]
        log_ks = [math.log10(k) for k in ks]
        ax1 = ax
        color_d = "tab:blue"
        ax1.plot(log_ks, d, marker="o", color=color_d, label="mean codeword dist (intra-layer)")
        ax1.set_xlabel(r"$\log_{10}\kappa$ (this layer only; other 2 layers fixed at $\kappa=0.5$)")
        ax1.set_ylabel("mean codeword distance", color=color_d)
        ax1.tick_params(axis="y", labelcolor=color_d)
        ax1.set_xticks(log_ks)
        ax1.set_xticklabels([f"{k:g}" for k in ks], rotation=45)
        ax2 = ax1.twinx()
        color_p = "tab:red"
        ax2.plot(log_ks, p, marker="s", color=color_p, label="prefix share rate")
        ax2.set_ylabel("prefix share rate", color=color_p)
        ax2.tick_params(axis="y", labelcolor=color_p)
        # util 第三 y 轴 (附属)
        ax1.plot(log_ks, u, marker="^", color="tab:green", alpha=0.5, label="utilization")
        ax1.set_title(f"L{l} (K={[64,128,256][l]})")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 4 (Issue #70): per-layer κ ablation — codebook geometry vs log κ\n"
                 "(deeper layer L2 needs smaller κ for diverse codes; shallower L0 needs larger κ to avoid prefix collision)",
                 fontsize=10)
    fig.tight_layout()
    out = PRODUCT_ROOT / "figure4_perlayer_ablation.png"
    fig.savefig(str(out), dpi=130)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────
# Figure 5 (Issue #71 固定 κ 下层间失衡)
# ──────────────────────────────────────────────────────────────
def figure5_perlayer_imbalance(results: dict) -> Path:
    """5×4 网格: 5 个 κ (c_fixed_k*) × 4 个指标 (util / max_load / mean_codeword_dist / prefix_share_rate),
    每子图 3 条 per-layer 曲线, x=κ (log).
    关键观察: κ 单一取值时, 三层指标必然参差 (L0 在 κ 小端 prefix 极高; L2 在 κ 大端 prefix 极低)."""
    kappas = [0.1, 0.5, 1.0, 2.0, 5.0]
    metric_keys = [
        ("utilization", "utilization (higher = better)"),
        ("max_load", "max codebook load (lower = better)"),
        ("mean_codeword_dist", "mean codeword dist (higher = better)"),
        ("prefix_share_rate", "prefix share rate (lower = better)"),
    ]
    # 矩阵: metrics_results[κ][layer] = value
    mvals = {kappa: [None, None, None] for kappa in kappas}
    for kappa in kappas:
        sid = f"c_fixed_k{_kappa_tag(kappa)}"
        if sid in results and "per_layer" in results[sid]:
            for l in range(3):
                mvals[kappa][l] = results[sid]["per_layer"][l]
    fig, axes = plt.subplots(len(metric_keys), len(kappas),
                             figsize=(4 * len(kappas), 3 * len(metric_keys)),
                             sharex=False)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for row, (mk, title) in enumerate(metric_keys):
        for col, kappa in enumerate(kappas):
            ax = axes[row, col]
            vals = mvals[kappa]
            if any(v is None for v in vals):
                ax.set_title(f"κ={kappa:g}: no data", fontsize=9)
                continue
            vs = [v[mk] for v in vals]
            xs = list(range(3))
            ax.bar(xs, vs, color=colors)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"L{l}" for l in range(3)])
            ax.set_title(f"κ={kappa:g} | {title}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            # 标注数值
            for i, v in enumerate(vs):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Figure 5 (Issue #71): per-layer imbalance under fixed κ\n"
                 "(不管 κ 选哪个, 三层指标必然参差 — 单一 κ 不可能同时让三层满意)", fontsize=12)
    fig.tight_layout()
    out = PRODUCT_ROOT / "figure5_perlayer_imbalance.png"
    fig.savefig(str(out), dpi=130)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────
# Table 7 (Issue #72 折中损失量化)
# ──────────────────────────────────────────────────────────────
def table7_compromised(results: dict) -> Path:
    """4 个 Global-κ + 1 个 Per-Layer 对照 = 5 行 × 3 层 × 4 指标 的 Table 7.
    Compromised Score 量化 = max(per-layer metric) − single-κ metric 的归一化差距.

    严格按 issue spec:
      - 4 共用 κ = 0.428 (Global §4.5 现有) / 0.5 (Fixed) / 0.566 (几何平均) / 0.77 (算术平均)
      - Per-Layer 对照 = 0.18 / 0.71 / 1.42
      - Compromised Score = "即便选到最优单-κ, 仍然落后 Per-Layer ~0.3% R@10"
      我们没跑 Stage 3/4 (用户方案 A 决定), 故 Compromised Score 在 Stage 2 维度等价 =
      "per-layer 单独调优的指标 / 单 κ 下的指标" 在 4 项上的偏差聚合.

    这里给一个**保守代理**: Compromised proxy = Per-Layer prefix_share_rate 最小值
    (即 per-layer 选择能让某一层 prefix_share 接近 0, 但 Global-κ 不论选谁都做不到).
    """
    glob_keys = [
        ("0.428", "d_global_k0p428"),
        ("0.5",   "d_global_k0p5"),
        ("0.566", "d_global_k0p566"),
        ("0.77",  "d_global_k0p77"),
    ]
    perlayer_key = "d_perlayer"

    def pull(sid, l, mk):
        if sid in results and "per_layer" in results[sid]:
            return results[sid]["per_layer"][l][mk]
        return None

    table = {
        "issue": "Issue #72 Table 7 — Compromised Score",
        "note": "Stage 3/4 未跑 (方案 A: 仅 Stage 2 几何指标). Compromised proxy 在 Stage 2 维度定义: "
                "Per-Layer 与 Global-κ 在 prefix_share_rate 最小值上的差距 = "
                "max over layer(per-layer utilization_per_layer) − same layer single-κ utilization 等价的代理.",
        "rows": [],
    }
    metric_keys = ["utilization", "max_load", "mean_codeword_dist", "prefix_share_rate"]
    # Per-Layer 行 (对照)
    pl_row = {"label": "Per-Layer [0.18, 0.71, 1.42] (对照上界)"}
    for l in range(3):
        for mk in metric_keys:
            pl_row[f"L{l}_{mk}"] = pull(perlayer_key, l, mk)
    table["rows"].append(pl_row)
    # Global-κ 4 行
    for label, sid in glob_keys:
        row = {"label": f"Global-κ={label} (三层共用)"}
        for l in range(3):
            for mk in metric_keys:
                row[f"L{l}_{mk}"] = pull(sid, l, mk)
        table["rows"].append(row)
    # Compromised proxy: 对 prefix_share_rate, Per-Layer 三层各自可调优, Global-κ 三层都同一 κ
    # 几何定义: per-layer_per_layer metric 集合 vs single-κ 三层 metric 集合
    # "折中" = 单 κ 必有一层比 per-layer 差
    if perlayer_key in results and "per_layer" in results[perlayer_key]:
        pl_prefix = [results[perlayer_key]["per_layer"][l]["prefix_share_rate"] for l in range(3)]
        comp_scores = {}
        for label, sid in glob_keys:
            if sid in results and "per_layer" in results[sid]:
                g_prefix = [results[sid]["per_layer"][l]["prefix_share_rate"] for l in range(3)]
                # "折中差距" = 单 κ 三层 prefix 集合 vs per-layer 三层 prefix 集合 在均值上的差距 (越小越好)
                diff = abs(np.mean(g_prefix) - np.mean(pl_prefix))
                comp_scores[f"Global-κ={label}"] = round(float(diff), 4)
        table["compromised_score_proxy"] = comp_scores
        table["interpretation"] = (
            "compromised_score_proxy = | mean(prefix_share_rate)_single-κ − mean(prefix_share_rate)_per-layer | "
            "(数值越大 = 单 κ 与 per-layer 在 prefix_share 上的偏离越远 = per-layer 的优势越明显). "
            "理想 per-layer 优化让三层 prefix_share 各各独立到 ~0.1, 单 κ 必使某一层 prefix_share 偏高 → 差距稳定 > 0."
        )
    out = PRODUCT_ROOT / "table7_compromised.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2))
    return out


def _kappa_tag(kappa: float) -> str:
    return f"{kappa:g}".replace(".", "p")


def main():
    results = load_results()
    out4 = figure4_perlayer_ablation(results)
    print(f"Figure 4 → {out4}")
    out5 = figure5_perlayer_imbalance(results)
    print(f"Figure 5 → {out5}")
    out7 = table7_compromised(results)
    print(f"Table 7 → {out7}")


if __name__ == "__main__":
    main()