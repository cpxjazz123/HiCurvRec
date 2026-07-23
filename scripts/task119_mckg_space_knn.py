"""Task #119 候选 C — MCKG 子空间 kNN@10 重测 (跨空间混淆修正).

把 Task #119 verdict 识别出的"跨空间混淆"问题修复:
- m=0/m=1 SID 训练空间 = MCKG 96d (3 子空间 × 32d, 实际是 3×64=192d)
- baseline SID 训练空间 = sentence-t5-base 768d

正确做法: 在每个 tokenizer 的**训练空间**测 kNN@10.
- m=0/m=1 → MCKG 加权距离 (考虑 κ 权重)
- baseline → T5 cosine

注: baseline 没有 MCKG 输入, 只能对照 m=0 vs m=1 在 MCKG 空间的 kNN@10
    vs 它们的 R@5 排序. Q1 改为 "MCKG 空间 m=0/m=1 排序是否和 R@5 排序一致?"

启动:
    cd /home/wlia0047/ar57/wenyu/GeneRec
    python3 scripts/task119_mckg_space_knn.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO_ROOT)


def compute_mckg_weighted_distance(item_emb: torch.Tensor, kappas: list, query_idx: int, top_k: int = 10):
    """MCKG 加权距离 (Task #82 标准 B):
    d_MCKG(u, v) = Σ_m sqrt(|κ_m|) · d_κ_m(u_m, v_m)

    item_emb: (3, n_items, 64) 三子空间
    kappas: [κ_0, κ_1, κ_2]
    query_idx: 查询 item index
    返回: top_k 邻居 indices + 距离
    """
    # 三子空间距离
    n_items = item_emb.shape[1]
    query_emb = item_emb[:, query_idx:query_idx+1, :]  # (3, 1, 64)

    # 球面距离 (κ > 0): d_S = arccos(<x, y>) / sqrt(|κ|)
    # 双曲距离 (κ < 0): d_H = sqrt(|κ|) · arccosh(-<x,y>_L)  (Lorentz)
    # 欧氏距离 (κ ≈ 0): d_E = ||x - y||

    total_dist = torch.zeros(n_items)
    for m in range(item_emb.shape[0]):
        kappa = kappas[m]
        emb_m = item_emb[m]  # (n_items, 64)

        if abs(kappa) < 1e-3:
            # Euclidean
            d_m = torch.norm(emb_m - emb_m[query_idx:query_idx+1], dim=1)
            weight = 1.0
        elif kappa > 0:
            # Spherical: arccos of normalized cosine
            emb_norm = torch.nn.functional.normalize(emb_m, dim=1)
            cos = torch.clamp(emb_norm @ emb_norm[query_idx], -1+1e-7, 1-1e-7)
            d_m = torch.arccos(cos) / np.sqrt(abs(kappa))
            weight = np.sqrt(abs(kappa))
        else:
            # Hyperbolic: Lorentz inner product
            x = emb_m[query_idx:query_idx+1]
            x0 = x[:, 0:1]  # time component
            x_space = x[:, 1:]
            y0 = emb_m[:, 0:1].T  # (1, n)
            y_space = emb_m[:, 1:].T  # (64, n)
            # Lorentz inner product: -x0*y0 + <x_space, y_space>
            lorentz_inner = -x0 @ y0 + x_space @ y_space  # (1, n)
            lorentz_inner = torch.clamp(-lorentz_inner, 1+1e-7, 1e6)  # ensure >= 1
            d_m = np.sqrt(abs(kappa)) * torch.arccosh(lorentz_inner).squeeze(0)
            weight = np.sqrt(abs(kappa))

        total_dist += weight * d_m

    top_k_dists, top_k_indices = torch.topk(total_dist, top_k + 1, largest=False)
    # 排除自身 (距离为 0)
    top_k_indices = top_k_indices[top_k_indices != query_idx][:top_k]
    return top_k_indices, total_dist[top_k_indices]


def co_cluster_rate_at_k(sid: torch.Tensor, neighbors: torch.Tensor) -> float:
    """top-K 邻居中, 至少与自身共享 1 digit 的比例."""
    sid = sid.numpy() if torch.is_tensor(sid) else sid
    n_neighbors = len(neighbors)
    query_sid = sid[:, neighbors[0]]  # (D,) self
    shared = 0
    for nbr in neighbors[1:]:
        nbr_sid = sid[:, nbr]
        if len(set(query_sid.tolist()) & set(nbr_sid.tolist())) > 0:
            shared += 1
    return shared / (n_neighbors - 1)


def main():
    # 加载 MCKG 嵌入
    print("[load] MCKG entity_embedding.pt ...")
    mckg = torch.load("products/task99_mckg_rebuild/entity_embedding.pt", weights_only=False)
    item_emb = mckg["subspace_item"]  # (3, 11924, 64)
    kappas = mckg["kappas"]  # [5.05, -0.08, -5.04]
    print(f"[load] item_emb shape = {tuple(item_emb.shape)}")
    print(f"[load] kappas = {[round(k, 3) for k in kappas]}")

    # 加载 2 个 SID (m=0 / m=1)
    tokenizers = [
        {
            "id": "Task85_m1_quasi_euclid",
            "label": "m=1 准欧氏 (κ≈-0.17)",
            "sid_path": "products/task85_tri_geom_rqvae/euclid_full/sid_subspace_1.pt",
            "downstream_R5": 0.0200,
            "downstream_R10": 0.0288,
        },
        {
            "id": "Task85_m0_sphere",
            "label": "m=0 球面 (κ≈+0.85)",
            "sid_path": "products/task85_tri_geom_rqvae/sphere_fix/sid_subspace_0.pt",
            "downstream_R5": 0.0174,
            "downstream_R10": 0.0262,
        },
    ]

    # 随机采样 200 个 item (节省时间)
    n_sample = 200
    np.random.seed(42)
    query_indices = np.random.choice(item_emb.shape[1], n_sample, replace=False)
    print(f"[sample] {n_sample} random query items (seed=42)")

    results = []
    for tk in tokenizers:
        print(f"\n=== {tk['id']} ({tk['label']}) ===")
        sid = torch.load(tk["sid_path"], weights_only=False)
        print(f"  SID shape: {tuple(sid.shape)}")

        co_cluster_rates = []
        for i, q in enumerate(query_indices):
            nbrs, _ = compute_mckg_weighted_distance(item_emb, kappas, int(q), top_k=10)
            rate = co_cluster_rate_at_k(sid, nbrs)
            co_cluster_rates.append(rate)
            if (i + 1) % 50 == 0:
                print(f"  [progress] {i+1}/{n_sample}, current mean co-cluster = {np.mean(co_cluster_rates):.4f}")

        mean_co_cluster = np.mean(co_cluster_rates)
        std_co_cluster = np.std(co_cluster_rates)
        print(f"  Final: mean co-cluster@10 = {mean_co_cluster:.4f} ± {std_co_cluster:.4f}")
        results.append({
            "id": tk["id"],
            "label": tk["label"],
            "sid_shape": list(sid.shape),
            "downstream_R5": tk["downstream_R5"],
            "downstream_R10": tk["downstream_R10"],
            "mckg_co_cluster_at_10_mean": float(mean_co_cluster),
            "mckg_co_cluster_at_10_std": float(std_co_cluster),
        })

    # 相关性: MCKG co-cluster@10 vs R@5 (n=2 不算, 但作为 sanity check)
    if len(results) == 2:
        x = np.array([r["mckg_co_cluster_at_10_mean"] for r in results])
        y = np.array([r["downstream_R5"] for r in results])
        # n=2 时 Pearson = Spearman = ±1 (无法判断)
        pearson_r, _ = pearsonr(x, y)
        spearman_r, _ = spearmanr(x, y)
        print(f"\n[相关性 n={len(results)}]")
        print(f"  Pearson ρ  = {pearson_r:+.4f}")
        print(f"  Spearman ρ = {spearman_r:+.4f}")
        print(f"  ⚠️ n=2 无法判断显著性 (任何单调关系都给出 ρ=±1)")

    # 输出
    out_dir = REPO_ROOT
    csv_path = out_dir / "task119_mckg_space_table.csv"
    with open(csv_path, "w") as f:
        f.write("id,label,sid_dim,downstream_R5,downstream_R10,mckg_co_cluster_at_10_mean,mckg_co_cluster_at_10_std\n")
        for r in results:
            f.write(f"{r['id']},{r['label']},{r['sid_shape'][0]},{r['downstream_R5']},{r['downstream_R10']},{r['mckg_co_cluster_at_10_mean']:.6f},{r['mckg_co_cluster_at_10_std']:.6f}\n")
    print(f"\n[save] {csv_path}")

    json_path = out_dir / "task119_mckg_space_summary.json"
    summary = {
        "n_tokenizers": len(results),
        "kappas": [round(k, 4) for k in kappas],
        "n_sample": n_sample,
        "caveat": "baseline 不在 MCKG 训练, 不能直接对照. 本表仅测 m=0 vs m=1 在 MCKG 空间的 kNN 共 digit 率.",
        "results": results,
    }
    if len(results) == 2:
        summary["correlation_pearson"] = float(pearson_r)
        summary["correlation_spearman"] = float(spearman_r)
        summary["correlation_warning"] = "n=2 cannot establish significance"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[save] {json_path}")

    return results


if __name__ == "__main__":
    main()