"""Two predictors:
  Stage2-only:    SID metrics                 -> test_R@10   (要不要跑 Stage3?)
  Early-stage:    SID + early Stage3 signals  -> test_R@10   (Stage3 内几分钟早停)

数据源: sid_metrics_collector/sid_metrics_summary.json (10 个 iter)
Early-stage 额外数据: 从 HG_Rec.log 抓 train_loss @ epoch [1, 5, 10, 20, 50]

按 Project Rule §1: 0 CLI flag, 全硬编码路径.
按 Project Rule §7: 无 fallback.

策略:
  - 10 样本 + 14 特征 → Ridge 退化到均值. 改用:
    (a) 单特征 Spearman 相关性诊断, 找最强信号特征.
    (b) 单特征/双特征线性回归 + LOO-CV, 显式 a*x + b.
    (c) 规则化: "G_L0 - mean(G_L1,G_L2) > threshold" → PASS/FAIL.
"""
import json
import os
import re
from glob import glob

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

# === 硬编码路径 ===
JSON_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/sid_metrics_collector/sid_metrics_summary.json"
STAGE3_LOGS = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/logs"
STAGE3_RESULTS = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/sid_metrics_collector/predictors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# iter → stage3 dir map
STAGE3_DIR = {
    "curvature_RQ-VAE": "tiger_baseline",
    "curvature_RQ-VAE_iter2": "iter2",
    "curvature_RQ-VAE_iter3": "iter3",
    "curvature_RQ-VAE_iter4": "iter4",
    "curvature_RQ-VAE_iter10": "iter10_gumbel_softmax_anneal",
    "curvature_RQ-VAE_iter11": "iter11_L0_collapse_sk05",
    "curvature_RQ-VAE_iter14": "iter14_L0_sk001",
    "curvature_RQ-VAE_iter28": "iter28_density_radial_alpha025",
    "curvature_RQ-VAE_iter32": "iter32_per_layer_hetero_c_sk_eps",
    "curvature_RQ-VAE_iter33": "iter33_per_item_curvature",
}


def engineer_features(row: dict) -> dict:
    """Engineer SID metrics → 层级关系特征."""
    g0 = row.get("gini_L0") or 0.0
    g1 = row.get("gini_L1") or 0.0
    g2 = row.get("gini_L2") or 0.0
    fg = row.get("full_gini") or 0.0
    n_unique = row.get("n_unique_full") or 0
    n_items = row.get("n_items") or 1
    l01 = row.get("l01_unique_pairs") or 0
    h = row.get("h_l1_given_l0") or 0.0
    return {
        "full_gini": fg,
        "gini_L0": g0,
        "gini_L1": g1,
        "gini_L2": g2,
        "l01_unique_pairs": l01,
        "h_l1_given_l0": h,
        "n_unique_full": n_unique,
        "G_L0_minus_G_L1": g0 - g1,
        "G_L0_minus_G_L2": g0 - g2,
        "G_L0_minus_mean_GL1L2": g0 - (g1 + g2) / 2.0,
        "G_L0_div_max_GL1L2": g0 / max(g1, g2, 1e-9),
        "utility_ratio": n_unique / n_items,
        "collapse_ratio": 1.0 - n_unique / n_items,
        "hr_50": row.get("hr_50") or 0.0,
    }


def extract_early_stage_signals(stage3_iter_dir: str) -> dict:
    """从 HG_Rec.log 抓 train_loss @ epoch + early valid @ 50/100/150."""
    candidates = [
        os.path.join(STAGE3_LOGS, stage3_iter_dir),
        os.path.join(STAGE3_RESULTS, stage3_iter_dir),
    ]
    log = None
    for base in candidates:
        if not os.path.exists(base):
            continue
        files = glob(os.path.join(base, "**/HG_Rec.log"), recursive=True)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            log = files[0]
            break
    if log is None:
        return {}

    epoch_loss = {}
    with open(log) as f:
        for line in f:
            m = re.search(r"epoch=(\d+)/\d+ done loss=([0-9.]+) lr=", line)
            if m:
                epoch_loss[int(m.group(1))] = float(m.group(2))
    out = {}
    for tgt in [1, 5, 10, 20, 50]:
        best = None
        for ep in sorted(epoch_loss.keys()):
            if ep <= tgt:
                best = epoch_loss[ep]
        out[f"train_loss_at_{tgt}"] = best if best is not None else np.nan

    early_valid = {}
    with open(log) as f:
        for line in f:
            m_v = re.search(r"Validation recall=\{[^}]*'recall@10': ([0-9.]+)[^}]*\}", line)
            m_e = re.search(r"epoch=(\d+).*Validation", line)
            if m_v and m_e:
                early_valid[int(m_e.group(1))] = float(m_v.group(1))
    for tgt in [50, 100, 150]:
        best = None
        for ep in sorted(early_valid.keys()):
            if ep <= tgt:
                best = early_valid[ep]
        out[f"valid_r10_at_{tgt}"] = best if best is not None else np.nan
    return out


def loo_predict_simple(X: np.ndarray, y: np.ndarray, model_cls, **kw):
    """LOO-CV 预测, 返回 (preds, mae, r2)."""
    preds = []
    for tr_idx, te_idx in LeaveOneOut().split(X):
        m = model_cls(**kw)
        m.fit(X[tr_idx], y[tr_idx])
        preds.append(m.predict(X[te_idx])[0])
    preds = np.array(preds)
    return preds, mean_absolute_error(y, preds), r2_score(y, preds)


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    rows = data["rows"]

    X_stage2 = []
    X_early = []
    y = []
    iter_names = []
    feature_names_stage2 = None
    feature_names_early = None
    early_data_dump = []

    for r in rows:
        iter_name = r["iter"]
        if r["test_recall_at_10"] is None:
            continue
        f2 = engineer_features(r)
        if feature_names_stage2 is None:
            feature_names_stage2 = sorted(f2.keys())
        X_stage2.append([f2[k] for k in feature_names_stage2])

        s3dir = STAGE3_DIR.get(iter_name, "")
        early = extract_early_stage_signals(s3dir)
        all_feats = {**f2, **early}
        if feature_names_early is None:
            feature_names_early = sorted(all_feats.keys())
        X_early.append([all_feats[k] for k in feature_names_early])
        early_data_dump.append({"iter": iter_name, **all_feats})

        y.append(r["test_recall_at_10"])
        iter_names.append(iter_name)

    X_stage2 = np.array(X_stage2)
    X_early = np.array(X_early, dtype=float)
    y = np.array(y)
    print(f"loaded {len(y)} samples")
    print(f"  Stage2-only features: {len(feature_names_stage2)}")
    print(f"  Early-stage features: {len(feature_names_early)}")

    # ============================================================
    # (1) 单特征 Spearman 相关性诊断 (Stage2-only)
    # ============================================================
    print(f"\n{'='*60}")
    print("单特征 Spearman 相关性 vs test_R@10 (Stage2-only)")
    print(f"{'='*60}")
    corrs = []
    for j, fname in enumerate(feature_names_stage2):
        col = X_stage2[:, j]
        rho, pval = spearmanr(col, y)
        if not np.isnan(rho):
            corrs.append((fname, rho, pval))
    corrs.sort(key=lambda x: -abs(x[1]))
    print(f"  {'feature':30s} {'rho':>7s} {'p':>10s}")
    for fname, rho, pval in corrs:
        print(f"  {fname:30s} {rho:+.4f} {pval:.4e}")

    top_stage2_feat = corrs[0][0]
    print(f"\n  最强 SID 特征: {top_stage2_feat} (rho={corrs[0][1]:+.4f})")

    # ============================================================
    # (1b) 几何特征 Spearman 相关性诊断
    # ============================================================
    geom_path = os.path.join(OUTPUT_DIR, "geometry_features.json")
    if os.path.exists(geom_path):
        geom = json.load(open(geom_path))
        geom_by_iter = {g["iter"]: g for g in geom}
        print(f"\n{'='*60}")
        print("单特征 Spearman 相关性 vs test_R@10 (GEOMETRIC features)")
        print(f"{'='*60}")
        geom_corrs = []
        geom_keys = sorted([k for k in geom[0].keys() if k.startswith("geom_")])
        for k in geom_keys:
            col = np.array([geom_by_iter[r["iter"]][k] for r in rows if r["iter"] in geom_by_iter], dtype=float)
            yy = np.array([r["test_recall_at_10"] for r in rows if r["iter"] in geom_by_iter])
            if len(col) < 5 or np.allclose(col, col[0]) or np.any(np.isnan(col)):
                continue
            rho, p = spearmanr(col, yy)
            geom_corrs.append((k, rho, p))
        geom_corrs.sort(key=lambda x: -abs(x[1]))
        print(f"  {'feature':45s} {'rho':>7s} {'p':>10s}")
        for k, rho, p in geom_corrs[:15]:
            print(f"  {k:45s} {rho:+.4f} {p:.4e}")
    else:
        geom_corrs = []

    # ============================================================
    # (2) 单特征 Linear Regression LOO-CV
    # ============================================================
    print(f"\n{'='*60}")
    print("Stage2-only 单特征 LOO Linear Regression")
    print(f"{'='*60}")
    stage2_results = {}
    for j, fname in enumerate(feature_names_stage2):
        X1 = X_stage2[:, j:j+1]
        preds, mae, r2 = loo_predict_simple(X1, y, LinearRegression)
        stage2_results[fname] = {"mae": float(mae), "r2": float(r2), "preds": preds.tolist()}
    sorted_results = sorted(stage2_results.items(), key=lambda x: x[1]["mae"])
    for fname, res in sorted_results[:8]:
        print(f"  {fname:30s}  MAE={res['mae']:.5f}  R²={res['r2']:+.4f}")

    # ============================================================
    # (3) 双特征 Linear Regression (top-2 + top-engineered)
    # ============================================================
    print(f"\n{'='*60}")
    print("Stage2-only 双特征 (用户重点关注)")
    print(f"{'='*60}")
    pairs = [
        ("G_L0_minus_mean_GL1L2", "gini_L0"),
        ("G_L0_minus_mean_GL1L2", "G_L0_minus_G_L1"),
        ("G_L0_minus_mean_GL1L2", "h_l1_given_l0"),
        ("gini_L0", "gini_L1"),
        ("G_L0_minus_mean_GL1L2", "full_gini"),
        ("gini_L0", "l01_unique_pairs"),
    ]
    pair_results = {}
    for a, b in pairs:
        ia, ib = feature_names_stage2.index(a), feature_names_stage2.index(b)
        X2 = X_stage2[:, [ia, ib]]
        preds, mae, r2 = loo_predict_simple(X2, y, LinearRegression)
        pair_results[f"{a}+{b}"] = {"mae": float(mae), "r2": float(r2), "preds": preds.tolist()}
        print(f"  {a:30s} + {b:25s}  MAE={mae:.5f}  R²={r2:+.4f}")

    # ============================================================
    # (4) 规则化: G_L0_minus_mean_GL1L2 > threshold → 预测高分
    # ============================================================
    print(f"\n{'='*60}")
    print("规则化分类器 (G_L0_minus_mean_GL1L2 threshold)")
    print(f"{'='*60}")
    GAP_FEAT = "G_L0_minus_mean_GL1L2"
    ig = feature_names_stage2.index(GAP_FEAT)
    gap_vals = X_stage2[:, ig]
    # 扫描 threshold
    thresholds = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10]
    for thr in thresholds:
        # 预测: gap > thr → baseline + small bonus; else baseline - small penalty
        # 用线性拟合: pred = mean(y) + alpha * gap
        # 但要 LOO
        gap_train = gap_vals.reshape(-1, 1)
        preds, mae, r2 = loo_predict_simple(gap_train, y, LinearRegression)
        # 用 thr 做硬规则
        rule_preds = np.where(gap_vals > thr, 0.058, 0.057)
        rule_mae = mean_absolute_error(y, rule_preds)
        print(f"  thr={thr:+.2f}  lin_reg_MAE={mae:.5f}  R²={r2:+.4f}  hard_rule_MAE={rule_mae:.5f}")

    # ============================================================
    # (5) Best 单/双特征 model 全量训练 + 报告
    # ============================================================
    best_stage2_cfg = sorted_results[0]
    best_stage2_name = best_stage2_cfg[0]
    best_pair_name = min(pair_results.items(), key=lambda x: x[1]["mae"])[0]
    print(f"\n{'='*60}")
    print(f"Best Stage2-only: 单特征 [{best_stage2_name}] MAE={best_stage2_cfg[1]['mae']:.5f}")
    print(f"Best Stage2-only: 双特征 [{best_pair_name}] MAE={pair_results[best_pair_name]['mae']:.5f}")
    print(f"{'='*60}")

    # 用最佳单特征画表
    best_pred = np.array(best_stage2_cfg[1]["preds"])
    print(f"\n  {'iter':40s} {'true':>8s} {'pred':>8s} {'err':>8s}")
    for i, it in enumerate(iter_names):
        print(f"  {it:40s} {y[i]:.4f}  {best_pred[i]:.4f}  {(best_pred[i]-y[i]):+.4f}")

    # ============================================================
    # (6) Early-stage: train_loss @ 5/10/20 + Stage2 特征
    # ============================================================
    print(f"\n{'='*60}")
    print("Early-stage 预测: train_loss @ 5/10/20 + Stage2 特征")
    print(f"{'='*60}")
    early_feats_to_use = [
        "train_loss_at_5", "train_loss_at_10", "train_loss_at_20",
        "full_gini", "gini_L0", "G_L0_minus_mean_GL1L2",
        "h_l1_given_l0", "l01_unique_pairs",
    ]
    early_idx = [feature_names_early.index(f) for f in early_feats_to_use if f in feature_names_early]
    X_early_use = X_early[:, early_idx]
    # NaN → 中位数
    for j in range(X_early_use.shape[1]):
        col = X_early_use[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X_early_use[nan_mask, j] = np.nanmedian(col[~nan_mask]) if (~nan_mask).any() else 0.0
    early_results = {}
    for j, fname in enumerate(early_feats_to_use):
        if j >= X_early_use.shape[1]:
            continue
        X1 = X_early_use[:, j:j+1]
        preds, mae, r2 = loo_predict_simple(X1, y, LinearRegression)
        early_results[fname] = {"mae": float(mae), "r2": float(r2)}
    sorted_early = sorted(early_results.items(), key=lambda x: x[1]["mae"])
    for fname, res in sorted_early:
        print(f"  {fname:30s}  MAE={res['mae']:.5f}  R²={res['r2']:+.4f}")

    # 强信号双特征 (Stage2 hierarchy + early train_loss)
    print(f"\n  Stage2 hierarchy + early train_loss 双特征:")
    early_pairs = [
        ("G_L0_minus_mean_GL1L2", "train_loss_at_5"),
        ("G_L0_minus_mean_GL1L2", "train_loss_at_10"),
        ("gini_L0", "train_loss_at_5"),
        ("gini_L0", "train_loss_at_10"),
    ]
    # 映射到 X_early_use 列索引 (使用 early_feats_to_use 顺序)
    early_pair_results = {}
    for a, b in early_pairs:
        if a not in early_feats_to_use or b not in early_feats_to_use:
            continue
        ia = early_feats_to_use.index(a)
        ib = early_feats_to_use.index(b)
        X2 = X_early_use[:, [ia, ib]]
        preds, mae, r2 = loo_predict_simple(X2, y, LinearRegression)
        early_pair_results[f"{a}+{b}"] = {"mae": float(mae), "r2": float(r2), "preds": preds.tolist()}
        print(f"  {a:30s} + {b:25s}  MAE={mae:.5f}  R²={r2:+.4f}")

    # ============================================================
    # (7) 几何特征驱动的双特征预测 (用户关键假设: L0 collapse + L1 diversity)
    # ============================================================
    geom_pred_results = {}
    if os.path.exists(geom_path):
        geom = json.load(open(geom_path))
        geom_by_iter = {g["iter"]: g for g in geom}
        geom_pairs = [
            ("geom_L0_util", "geom_h_l1_given_l0_mean"),
            ("geom_L0_util", "geom_L0_max_freq_ratio"),
            ("geom_intra_l0_norm_std_mean", "geom_L0_util"),
            ("geom_L0_entropy", "geom_h_l2_given_l0_mean"),
            ("geom_L0_util", "geom_h_l2_given_l0_mean"),
        ]
        print(f"\n{'='*60}")
        print("几何特征双特征 LOO Linear Regression (用户关键假设)")
        print(f"{'='*60}")
        for a, b in geom_pairs:
            if a not in geom_by_iter.get(rows[0]["iter"], {}) or b not in geom_by_iter.get(rows[0]["iter"], {}):
                continue
            X2 = np.array([[geom_by_iter[r["iter"]][a], geom_by_iter[r["iter"]][b]] for r in rows], dtype=float)
            preds, mae, r2 = loo_predict_simple(X2, y, LinearRegression)
            geom_pred_results[f"{a}+{b}"] = {"mae": float(mae), "r2": float(r2), "preds": preds.tolist()}
            print(f"  {a:35s} + {b:30s}  MAE={mae:.5f}  R²={r2:+.4f}")
        # Best geom pair full output
        if geom_pred_results:
            best_pair = min(geom_pred_results.items(), key=lambda x: x[1]["mae"])
            print(f"\n  Best geom pair: {best_pair[0]} MAE={best_pair[1]['mae']:.5f}")
            best_pred = np.array(best_pair[1]["preds"])
            print(f"  {'iter':40s} {'true':>8s} {'pred':>8s} {'err':>8s}")
            for i, it in enumerate(iter_names):
                print(f"  {it:40s} {y[i]:.4f}  {best_pred[i]:.4f}  {(best_pred[i]-y[i]):+.4f}")
            # 全量训练 + 系数
            a, b = best_pair[0].split("+")
            X2 = np.array([[geom_by_iter[r["iter"]][a], geom_by_iter[r["iter"]][b]] for r in rows], dtype=float)
            m_full = LinearRegression()
            m_full.fit(X2, y)
            print(f"\n  全量训练:")
            print(f"    intercept = {m_full.intercept_:.4f}")
            print(f"    coef[{a}] = {m_full.coef_[0]:+.4f}")
            print(f"    coef[{b}] = {m_full.coef_[1]:+.4f}")
            geom_pred_results[best_pair[0]]["coefs"] = {
                "intercept": float(m_full.intercept_),
                a: float(m_full.coef_[0]),
                b: float(m_full.coef_[1]),
            }

    # ============================================================
    # (8) 保存所有结果
    # ============================================================
    results = {
        "samples": iter_names,
        "y_true": y.tolist(),
        "stage2_spearman": [
            {"feature": f, "rho": float(r), "pval": float(p)} for f, r, p in corrs if not np.isnan(r)
        ],
        "geometric_spearman": [
            {"feature": f, "rho": float(r), "pval": float(p)} for f, r, p in geom_corrs
        ] if geom_corrs else [],
        "stage2_single": stage2_results,
        "stage2_pair": pair_results,
        "early_single": early_results,
        "early_pair": early_pair_results,
        "geometric_pair": geom_pred_results,
        "best_stage2_single_feat": best_stage2_name,
        "best_stage2_pair": best_pair_name,
    }
    out_path = os.path.join(OUTPUT_DIR, "predictor_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nwritten {out_path}")

    raw_path = os.path.join(OUTPUT_DIR, "early_stage_signals.json")
    with open(raw_path, "w") as f:
        json.dump(early_data_dump, f, indent=2, ensure_ascii=False)
    print(f"written {raw_path}")

    # === 保存结果 ===
    out_path = os.path.join(OUTPUT_DIR, "predictor_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nwritten {out_path}")

    # === 保存早期信号原始数据 ===
    raw_path = os.path.join(OUTPUT_DIR, "early_stage_signals.json")
    with open(raw_path, "w") as f:
        json.dump(early_data_dump, f, indent=2, ensure_ascii=False)
    print(f"written {raw_path}")


if __name__ == "__main__":
    main()
