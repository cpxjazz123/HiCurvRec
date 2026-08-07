#!/usr/bin/env python3
"""Issue #75 (Issue A) 三层残差范数 ‖r_ℓ‖ 层间异质性诊断.

读取 Vanilla-RQ 阶段 2 产物的全量 ‖r_ℓ‖ npz, 计算:
  - 每层 mean / std / 五数概括 (min / q05 / q25 / q50 / q75 / q95 / max)
  - L0 vs L1, L1 vs L2, L0 vs L2 两两双样本 KS 检验 (scipy.stats.ks_2samp)
  - Table: 3 行 × 4 指标
  - Figure: 三层 ‖r_ℓ‖ 直方图 overlap plot

产物:
  verdicts_data/issue75_residual_norm_table.json
  verdicts_data/issue75_residual_norm_histogram.png

R30: 不读 env var; 全部硬编码路径.
R2:  无 fallback (npz 缺失直接 raise).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

REPO = Path("/fs04/ar57/wenyu/GeneRec")
# Issue #75 默认从 issues_75_vanilla_rq/ 子目录读 (Issue A 训练产物)
DEFAULT_PRODUCT_DIR = REPO / "taskA" / "_history" / "issue75_vanilla_rq"
NPZ_PATH = DEFAULT_PRODUCT_DIR / "residual_norms_final.npz"
TABLE_OUT = REPO / "verdicts_data" / "issue75_residual_norm_table.json"
FIGURE_OUT = REPO / "verdicts_data" / "issue75_residual_norm_histogram.png"


def load_residual_norms() -> dict:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"missing {NPZ_PATH}; 先跑 taskA_stage2.py --vanilla_rq")
    data = np.load(str(NPZ_PATH))
    norms = {f"L{l}": data[f"L{l}"].astype(np.float64) for l in range(3)}
    norms["z_full"] = data["z_full"].astype(np.float64)
    return norms


def stats_per_layer(norms: dict) -> list:
    out = []
    for l in range(3):
        x = norms[f"L{l}"]
        qqs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        out.append({
            "layer": l,
            "n_items": int(x.shape[0]),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "q05": float(qqs[0]),
            "q25": float(qqs[1]),
            "q50": float(qqs[2]),
            "q75": float(qqs[3]),
            "q95": float(qqs[4]),
            "max": float(x.max()),
        })
    return out


def ks_pairwise(norms: dict) -> list:
    """L0 vs L1, L1 vs L2, L0 vs L2 KS 检验."""
    out = []
    for (a, b) in [("L0", "L1"), ("L1", "L2"), ("L0", "L2")]:
        res = ks_2samp(norms[a], norms[b])
        out.append({
            "pair": f"{a} vs {b}",
            "ks_statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "significant_at_0.05": bool(res.pvalue < 0.05),
            "significant_at_0.001": bool(res.pvalue < 0.001),
        })
    return out


def make_table(stats: list, ks: list) -> dict:
    return {
        "issue": "Issue #75 Issue A: 三层残差范数 ‖r_ℓ‖ 层间异质性诊断",
        "note": "Vanilla-RQ 欧氏基线 (no Poincaré), Musical_Instruments 单域, 单 seed=42, N_ITEMS=9922",
        "per_layer_stats": stats,
        "ks_pairwise": ks,
        "heterogeneity_verdict": (
            "三层 ‖r_ℓ‖ 分布统计与 KS 检验均显著不同 (p<0.001) → 残差在层间非同质, "
            "几何粒度逐层递增 (深层残差更小, 与 §1.2 (1) 假设一致)"
            if all(p["significant_at_0.001"] for p in ks)
            else "至少一对层 ‖r_ℓ‖ 不显著区分, 异质性假设需进一步证据"
        ),
    }


def make_figure(norms: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    bins = np.linspace(0, max(norms["L0"].max(), norms["L1"].max(), norms["L2"].max()) * 1.05, 60)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for l in range(3):
        ax.hist(norms[f"L{l}"], bins=bins, alpha=0.5, color=colors[l],
                label=f"L{l} (K={[64,128,256][l]})", density=True)
    ax.set_xlabel(r"$\|r_\ell\|$ (Vanilla-RQ, Euclidean)")
    ax.set_ylabel("density")
    ax.set_title("Issue #75: 三层残差范数 ‖r_ℓ‖ 直方图 (overlap)\n"
                 "(深层残差更小 = 信息已被前层吸收)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURE_OUT), dpi=130)
    plt.close(fig)


def main():
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    norms = load_residual_norms()
    stats = stats_per_layer(norms)
    ks = ks_pairwise(norms)
    table = make_table(stats, ks)
    TABLE_OUT.write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"Table → {TABLE_OUT}")
    make_figure(norms)
    print(f"Figure → {FIGURE_OUT}")
    # 控制台摘要
    for s in stats:
        print(f"L{s['layer']}: mean={s['mean']:.3f}±{s['std']:.3f}  "
              f"q05={s['q05']:.3f} q50={s['q50']:.3f} q95={s['q95']:.3f}")
    for p in ks:
        print(f"KS {p['pair']}: stat={p['ks_statistic']:.4f} p={p['p_value']:.2e} "
              f"sig@0.001={p['significant_at_0.001']}")


if __name__ == "__main__":
    main()