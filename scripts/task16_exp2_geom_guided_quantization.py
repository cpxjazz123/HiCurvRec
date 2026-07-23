"""
Task 64 — Exp2: 几何指导量化对比

验证几何标签能否改善量化 MSE。
方式A: 标准 (只用欧氏码书)
方式B: 硬选择 (argmax 选码书)
方式C: 软加权 (weighted distance)

用法: python scripts/task6_exp2_geom_guided_quantization.py
"""

import os, sys, argparse, json, time
import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quantize_euclidean(z, codebook):
    """Standard nearest-neighbor: way A."""
    dist = torch.cdist(z, codebook, p=2)
    indices = dist.argmin(dim=1)
    z_q = codebook[indices]
    return z_q, dist.min(dim=1).values


def quantize_hard_select(z, codebooks, geo_labels):
    """Way B: select codebook by argmax(geo_label)."""
    K = codebooks[0].shape[0]
    z_q_list = []
    min_dist_list = []
    for i in range(z.shape[0]):
        cb_idx = int(geo_labels[i].argmax().item())
        cb = codebooks[cb_idx]
        d = torch.norm(z[i] - cb, dim=1)
        best_k = d.argmin().item()
        z_q_list.append(cb[best_k])
        min_dist_list.append(d[best_k].item())
    return torch.stack(z_q_list), torch.tensor(min_dist_list)


def quantize_soft_weighted(z, codebooks, geo_probs):
    """Way C: weighted distance sum across all 3 codebooks."""
    z_q_list = []
    min_dist_list = []
    for i in range(z.shape[0]):
        w = geo_probs[i]  # (3,) softmax weights
        best_dist = float('inf')
        best_q = None
        # For each codeword index k (K total)
        K = codebooks[0].shape[0]
        for k in range(K):
            weighted_d = sum(w[j] * torch.norm(z[i] - codebooks[j][k]).item() for j in range(3))
            if weighted_d < best_dist:
                best_dist = weighted_d
                best_q = k
        # Need to choose one codebook for the final quantized vector
        # Use the codebook with highest weight
        dominant = int(w.argmax().item())
        z_q_list.append(codebooks[dominant][best_q])
        min_dist_list.append(best_dist)
    return torch.stack(z_q_list), torch.tensor(min_dist_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--codebooks", default="products/task16/from_task63/exp1_codebooks.pt")
    parser.add_argument("--geo_labels", default="products/task16/from_task64/exp1_geo_labels.npy")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 60)
    print("Task 64 — Exp2: 几何指导量化对比")
    print("=" * 60)

    # ── 加载 ──
    print(f"\n[1/4] Loading data...")
    embeddings = torch.load(args.embeddings, map_location="cpu").float()
    cbs = torch.load(args.codebooks, map_location="cpu")
    geo_probs = np.load(args.geo_labels)
    geo_probs_t = torch.from_numpy(geo_probs).float()

    print(f"  Embeddings: {list(embeddings.shape)}")
    print(f"  Codebooks: {[cbs[k].shape for k in cbs]}")
    for name in cbs:
        print(f"    {name}: {list(cbs[name].shape)}")

    # Use 3 codebooks: Euclidean, Poincare, Sphere (from Task #63)
    cb_names = ["Euclidean", "Poincare", "Sphere"]
    codebooks = [cbs[n].to(device) for n in cb_names]
    z = embeddings.to(device)

    # Train/test split (same as Exp1)
    N = z.shape[0]
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    z_train, z_test = z[train_idx], z[test_idx]
    geo_train, geo_test = geo_probs_t[train_idx], geo_probs_t[test_idx]

    print(f"  Train: {len(z_train)}, Test: {len(z_test)}")

    # ── Quantization ──
    print(f"\n[2/4] Running 3 quantization methods...")

    results = {}
    for mode_name, mode_fn in [
        ("A: Euclidean (baseline)", lambda data, labels: quantize_euclidean(data, codebooks[0])),
        ("B: Hard select", lambda data, labels: quantize_hard_select(data, codebooks, labels)),
        ("C: Soft weighted", lambda data, labels: quantize_soft_weighted(data, codebooks, labels)),
    ]:
        print(f"\n  {mode_name}...")
        t0 = time.time()

        z_q_train, dists_train = mode_fn(z_train, geo_train)
        z_q_test, dists_test = mode_fn(z_test, geo_test)

        mse_train = (z_train - z_q_train).pow(2).mean().item()
        mse_test = (z_test - z_q_test).pow(2).mean().item()
        avg_dist_train = dists_train.mean().item()
        avg_dist_test = dists_test.mean().item()

        elapsed = time.time() - t0
        print(f"    Train MSE: {mse_train:.8f}")
        print(f"    Test  MSE: {mse_test:.8f}")
        print(f"    Avg dist train: {avg_dist_train:.6f}")
        print(f"    Avg dist test:  {avg_dist_test:.6f}")
        print(f"    ({elapsed:.1f}s)")

        results[mode_name] = {
            "mse_train": mse_train,
            "mse_test": mse_test,
            "avg_dist_train": float(avg_dist_train),
            "avg_dist_test": float(avg_dist_test),
        }

    # ── 分析 ──
    print(f"\n[3/4] Comparison analysis...")
    baseline_mse = results["A: Euclidean (baseline)"]["mse_test"]
    for mode_name in ["B: Hard select", "C: Soft weighted"]:
        delta = ((results[mode_name]["mse_test"] - baseline_mse) / baseline_mse) * 100
        print(f"  {mode_name}: ΔMSE = {delta:+.4f}% vs baseline")
        results[mode_name]["mse_delta_pct"] = delta

    # ── 判定 ──
    print(f"\n[4/4] Decision:")
    mse_b = results.get("B: Hard select", {}).get("mse_test", 1)
    mse_c = results.get("C: Soft weighted", {}).get("mse_test", 1)
    mse_a = baseline_mse

    better_b = mse_b < mse_a * 0.995  # 0.5% improvement
    better_c = mse_c < mse_a * 0.995

    if better_b or better_c:
        winner = "B" if mse_b < mse_c else "C"
        print(f"  ✅ {'方式B' if better_b else ''}{'方式C' if better_c else ''} 改善 MSE")
        print(f"  最佳方式: {winner}")
    else:
        print(f"  ❌ 几何指导未改善 MSE")
        print(f"    方式A MSE={mse_a:.8f}")
        print(f"    方式B MSE={mse_b:.8f} ({((mse_b-mse_a)/mse_a*100):+.3f}%)")
        print(f"    方式C MSE={mse_c:.8f} ({((mse_c-mse_a)/mse_a*100):+.3f}%)")

    # Reason analysis
    cb_sim = []
    for i in range(3):
        for j in range(i + 1, 3):
            norms_i = codebooks[i] / codebooks[i].norm(dim=1, keepdim=True).clamp(min=1e-8)
            norms_j = codebooks[j] / codebooks[j].norm(dim=1, keepdim=True).clamp(min=1e-8)
            cos_sim = (norms_i @ norms_j.T).mean().item()
            cb_sim.append(cos_sim)
    mean_cb_sim = np.mean(cb_sim)
    print(f"\n  平均码书余弦相似度: {mean_cb_sim:.6f}")
    if mean_cb_sim > 0.99:
        print(f"  → 码书几乎相同，选择不同码书没有意义（符合 Task #63 结论）")

    # ── 保存 ──
    out_dir = "products/task16/from_task64"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp2_guidance_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── 整体 Exp2 判定 ──
    should_proceed_to_exp3 = better_b or better_c
    print(f"\n  → Exp3 继续条件: {'✅ YES' if should_proceed_to_exp3 else '❌ NO'}")


if __name__ == "__main__":
    main()
