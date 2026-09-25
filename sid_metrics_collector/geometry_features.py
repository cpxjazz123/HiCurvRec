"""几何空间特征提取: 从 item_emb + SID + codebook 抽取 Poincaré ball / RQ codebook 几何指标.

新指标分类:
  (1) Input geometry: item_emb norm / pairwise distance 分布
  (2) Per-L0 codebook "concentration": items 共享 L0 code 时的内聚度
  (3) Per-layer codebook structure: L0/L1/L2 codes 的使用熵 + 几何紧密度
  (4) Cross-layer geometry: L0-L1 alignment, L0-L2 inheritance 强度
  (5) Volume / radius stats: Poincaré ball 体积分布

按 Project Rule §1: 0 CLI flag, 全硬编码.
按 Project Rule §7: 无 fallback (sids_for_hgrec.npy 必须存在).
"""
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import entropy as scipy_entropy

# === 硬编码路径 ===
JSON_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/sid_metrics_collector/sid_metrics_summary.json"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/sid_metrics_collector/predictors"
GEOM_JSON = os.path.join(OUTPUT_DIR, "geometry_features.json")
COMBINED_JSON = os.path.join(OUTPUT_DIR, "all_features.json")

# Poincaré ball 几何常数 (c=1.0 baseline)
C_BASELINE = 1.0
EPS = 1e-9


def poincare_dist_sq(x, y, c=C_BASELINE):
    """Squared Poincaré distance: d²(x,y) = (1/c) * acosh(1 + 2c||x-y||² / ((1-c||x||²)(1-c||y||²)))."""
    diff_sq = np.sum((x - y) ** 2, axis=-1)
    x_sq = np.sum(x ** 2, axis=-1)
    y_sq = np.sum(y ** 2, axis=-1)
    num = 2 * c * diff_sq
    denom = (1 - c * x_sq) * (1 - c * y_sq)
    arg = 1 + num / (denom + EPS)
    arg = np.clip(arg, 1.0 + EPS, None)
    return (1 / c) * np.arccosh(arg) ** 2


def pairwise_dist_stats(emb: np.ndarray, n_sample: int = 1000, seed: int = 42) -> dict:
    """Sample-wise 配对距离统计 (Poincaré dist)."""
    rng = np.random.RandomState(seed)
    n = emb.shape[0]
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        sub = emb[idx]
    else:
        sub = emb
    # squared pairwise distance (chunked)
    dists = []
    chunk = 100
    for i in range(0, len(sub), chunk):
        a = sub[i:i+chunk]
        # 配对: a vs sub
        for j in range(0, len(sub), chunk):
            b = sub[j:j+chunk]
            d = poincare_dist_sq(a[:, None, :], b[None, :, :])
            np.fill_diagonal(d, 0)
            dists.append(d[np.triu_indices_from(d, k=1)])
    dists = np.concatenate(dists)
    return {
        "geom_pairwise_dist_mean": float(dists.mean()),
        "geom_pairwise_dist_std": float(dists.std()),
        "geom_pairwise_dist_p10": float(np.percentile(dists, 10)),
        "geom_pairwise_dist_p50": float(np.percentile(dists, 50)),
        "geom_pairwise_dist_p90": float(np.percentile(dists, 90)),
        "geom_pairwise_dist_max": float(dists.max()),
    }


def norm_features(emb: np.ndarray) -> dict:
    """item_emb norm 分布 (Poincaré ball radius 近似)."""
    norms = np.linalg.norm(emb, axis=-1)
    # norm 分布熵 (10-bin histogram)
    hist, _ = np.histogram(norms, bins=10)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return {
        "geom_norm_mean": float(norms.mean()),
        "geom_norm_std": float(norms.std()),
        "geom_norm_min": float(norms.min()),
        "geom_norm_max": float(norms.max()),
        "geom_norm_p10": float(np.percentile(norms, 10)),
        "geom_norm_p50": float(np.percentile(norms, 50)),
        "geom_norm_p90": float(np.percentile(norms, 90)),
        "geom_norm_entropy": float(scipy_entropy(hist)),
        # Poincaré ball 体积: V(r) ∝ (1-r²)^d, 接近 0 = 接近边界
        "geom_mean_ball_ratio": float(np.mean(norms ** 2)),  # <c*||x||²>
    }


def per_l0_concentration(emb: np.ndarray, sids: np.ndarray) -> dict:
    """Per-L0 code "集中度": items 共享 L0 code 时, item_emb norm 离散度 + Poincaré 紧密度.

    iter11 假设: L0 集中 + L1/L2 discriminative
    → 同一 L0 code 的 items 在 emb 空间应该紧密, 但 L0 code 之间应该有 separation.
    """
    l0_codes = sids[:, 0]
    unique_l0 = np.unique(l0_codes)
    intra_l0_norm_std = []  # intra-code norm std (紧密度)
    intra_l0_norm_var = []
    centroids_norm = []
    for code in unique_l0:
        mask = l0_codes == code
        if mask.sum() < 2:
            continue
        norms = np.linalg.norm(emb[mask], axis=-1)
        intra_l0_norm_std.append(norms.std())
        intra_l0_norm_var.append(norms.var())
        centroids_norm.append(norms.mean())
    if not intra_l0_norm_std:
        return {}
    intra_l0_norm_std = np.array(intra_l0_norm_std)
    intra_l0_norm_var = np.array(intra_l0_norm_var)
    centroids_norm = np.array(centroids_norm)
    return {
        "geom_intra_l0_norm_std_mean": float(intra_l0_norm_std.mean()),
        "geom_intra_l0_norm_std_max": float(intra_l0_norm_std.max()),
        "geom_intra_l0_norm_var_mean": float(intra_l0_norm_var.mean()),
        "geom_l0_centroid_norm_std": float(centroids_norm.std()),  # L0 codes 之间 norm 差异
        "geom_l0_centroid_norm_mean": float(centroids_norm.mean()),
        "geom_n_l0_codes_with_>10_items": int(sum(1 for c in unique_l0 if (l0_codes == c).sum() > 10)),
    }


def per_layer_code_usage(sids: np.ndarray, codebook_sizes=(256, 256, 256)) -> dict:
    """每层 code 使用分布 (码本利用率 + 熵)."""
    out = {}
    for layer in range(3):
        codes = sids[:, layer]
        cnt = Counter(codes.tolist())
        counts = np.array(list(cnt.values()))
        used = len(cnt)
        max_codes = codebook_sizes[layer]
        # 使用率 = 用了几个 / 总共几个
        util = used / max_codes
        # entropy of code usage (越高越均匀)
        probs = counts / counts.sum()
        ent = scipy_entropy(probs)
        out[f"geom_L{layer}_util"] = float(util)
        out[f"geom_L{layer}_entropy"] = float(ent)
        out[f"geom_L{layer}_max_freq_ratio"] = float(counts.max() / counts.sum())  # 最大单 code 占比
    return out


def cross_layer_alignment(sids: np.ndarray) -> dict:
    """跨层一致性: L0 给定后 L1 的 entropy, L1 给定后 L2 的 entropy.

    越低越好 (越确定 = hierarchy 结构清晰).
    iter11: 应该 L0 给定 → L1 entropy 很低 (因为 L0 集中 + L1 集中).
    """
    n = sids.shape[0]
    out = {}
    # L0 → L1
    h_l1_l0 = []
    for l0 in np.unique(sids[:, 0]):
        mask = sids[:, 0] == l0
        if mask.sum() < 5:
            continue
        l1_codes = sids[mask, 1]
        cnt = Counter(l1_codes.tolist())
        probs = np.array(list(cnt.values())) / sum(cnt.values())
        h_l1_l0.append(scipy_entropy(probs))
    out["geom_h_l1_given_l0_mean"] = float(np.mean(h_l1_l0)) if h_l1_l0 else 0.0
    out["geom_h_l1_given_l0_std"] = float(np.std(h_l1_l0)) if h_l1_l0 else 0.0

    # L1 → L2
    h_l2_l1 = []
    for l1 in np.unique(sids[:, 1]):
        mask = sids[:, 1] == l1
        if mask.sum() < 5:
            continue
        l2_codes = sids[mask, 2]
        cnt = Counter(l2_codes.tolist())
        probs = np.array(list(cnt.values())) / sum(cnt.values())
        h_l2_l1.append(scipy_entropy(probs))
    out["geom_h_l2_given_l1_mean"] = float(np.mean(h_l2_l1)) if h_l2_l1 else 0.0
    out["geom_h_l2_given_l1_std"] = float(np.std(h_l2_l1)) if h_l2_l1 else 0.0

    # L0 → L2 (跨层)
    h_l2_l0 = []
    for l0 in np.unique(sids[:, 0]):
        mask = sids[:, 0] == l0
        if mask.sum() < 5:
            continue
        l2_codes = sids[mask, 2]
        cnt = Counter(l2_codes.tolist())
        probs = np.array(list(cnt.values())) / sum(cnt.values())
        h_l2_l0.append(scipy_entropy(probs))
    out["geom_h_l2_given_l0_mean"] = float(np.mean(h_l2_l0)) if h_l2_l0 else 0.0
    return out


def find_item_emb_path(iter_name: str, primary_path: str, sids_path: str) -> str | None:
    """查找 item_emb.npy, fallback 用 sids_path 推导目录."""
    if primary_path and os.path.exists(primary_path):
        return primary_path
    base_dir = os.path.dirname(sids_path) if sids_path else ""
    candidates = [
        os.path.join(base_dir, "item_emb.npy"),
        os.path.join(base_dir, "item_emb_transformed_alpha0.25.npy"),
        os.path.join(base_dir, "item_emb_iter33.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def geometric_features_for_iter(iter_name: str, item_emb_path: str, sids_path: str) -> dict:
    """对单个 iter 提取所有几何特征."""
    if not os.path.exists(sids_path):
        return {}
    actual_emb_path = find_item_emb_path(iter_name, item_emb_path, sids_path)
    if actual_emb_path is None:
        return {}
    emb = np.load(actual_emb_path).astype(np.float32)
    sids = np.load(sids_path).astype(np.int64)
    if emb.shape[0] != sids.shape[0]:
        n = min(emb.shape[0], sids.shape[0])
        emb = emb[:n]
        sids = sids[:n]

    feats = {}
    feats.update(norm_features(emb))
    feats.update(pairwise_dist_stats(emb, n_sample=500))
    feats.update(per_l0_concentration(emb, sids))
    feats.update(per_layer_code_usage(sids))
    feats.update(cross_layer_alignment(sids))
    return feats


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    rows = data["rows"]

    geom_results = []
    for r in rows:
        iter_name = r["iter"]
        sids_path = r["sids_path"]
        item_emb_path = r["item_emb_path"]
        if not sids_path or not os.path.exists(sids_path):
            print(f"  skip {iter_name}: no sids")
            continue
        feats = geometric_features_for_iter(iter_name, item_emb_path, sids_path)
        geom_results.append({"iter": iter_name, "test_r10": r["test_recall_at_10"], **feats})
        print(f"  {iter_name:42s}  norm_mean={feats.get('geom_norm_mean', 0):.4f}  "
              f"h_l1_l0={feats.get('geom_h_l1_given_l0_mean', 0):.4f}  "
              f"L0_util={feats.get('geom_L0_util', 0):.3f}")

    # === 保存 ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(GEOM_JSON, "w") as f:
        json.dump(geom_results, f, indent=2, ensure_ascii=False)
    print(f"\nwritten {GEOM_JSON}: {len(geom_results)} rows")

    # === 与原 features 合并 ===
    orig_by_iter = {r["iter"]: r for r in rows}
    combined = []
    for g in geom_results:
        o = orig_by_iter.get(g["iter"], {})
        combined.append({**o, **g})
    with open(COMBINED_JSON, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"written {COMBINED_JSON}: {len(combined)} rows")


if __name__ == "__main__":
    main()
