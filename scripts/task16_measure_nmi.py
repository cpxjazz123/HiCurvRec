"""
Task 62 — Exp1: 残差 NMI + Probe 分析
用法: python scripts/task6_measure_nmi.py [--rqvae_ckpt products/task16/from_task62/rqvae_ckpt.pt]
"""

import os, sys, argparse, json
import torch
import numpy as np
from collections import Counter

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.stats import entropy


def load_metadata(path="products/task16/from_task62/item_metadata.pt"):
    """Load metadata list and return structured arrays."""
    meta_list = torch.load(path)
    N = len(meta_list)

    brands = [m["brand"] for m in meta_list]
    prices = [m["price"] if m["price"] is not None else 0.0 for m in meta_list]
    primary_cats = [m["categories"][-1] if m["categories"] else "Unknown" for m in meta_list]
    top_cats = [m["categories"][0] if m["categories"] else "Unknown" for m in meta_list]

    return {
        "brand": np.array(brands),
        "price": np.array(prices),
        "primary_category": np.array(primary_cats),
        "top_category": np.array(top_cats),
    }


def compute_nmi(labels_true, labels_pred):
    """计算归一化互信息"""
    return normalized_mutual_info_score(labels_true, labels_pred)


def discretize_price(prices, n_bins=5):
    """将价格分桶为离散标签"""
    bins = np.percentile(prices, np.linspace(0, 100, n_bins + 1))
    # 去重 bins
    bins = np.unique(bins)
    return np.digitize(prices, bins[:-1])


def extract_residuals(model, embeddings, device="cpu"):
    """提取 RQ-VAE 三层残差"""
    model.eval()
    model = model.to(device)
    embeddings = embeddings.to(device)

    residual = embeddings
    residuals_list = []
    indices_list = []

    with torch.no_grad():
        for q in model.quantizers:
            dist = torch.cdist(residual, q.codebook, p=2)
            ind = dist.argmin(dim=1)
            z_q = q.codebook[ind]
            r = residual - z_q
            residuals_list.append(r.cpu())
            indices_list.append(ind.cpu())
            residual = residual - z_q

    return residuals_list, indices_list


def run_nmi_analysis(residuals_list, metadata, K=64, n_seeds=3):
    """
    对每层残差做 KMeans 聚类 → 计算 NMI vs 各属性。
    residuals_list: list of 3 tensors (N, D)
    """
    layers = ["L1残差", "L2残差", "L3残差"]
    attributes = ["brand", "primary_category", "top_category", "price_discrete"]
    results = {attr: {} for attr in attributes}

    for attr_name, attr_values in metadata.items():
        if attr_name == "price":
            continue  # 用离散版
        # Prepare labels
        if attr_name == "price_discrete":
            labels = discretize_price(metadata["price"], n_bins=5)
        else:
            le = LabelEncoder()
            labels = le.fit_transform(attr_values)

        # 跳过属性值太少的
        if len(np.unique(labels)) < 2:
            print(f"  SKIP {attr_name}: only {len(np.unique(labels))} unique values")
            continue

        for li, res in enumerate(residuals_list):
            nmis = []
            for seed in range(n_seeds):
                km = KMeans(n_clusters=min(K, len(res)), random_state=seed, n_init="auto")
                cluster_labels = km.fit_predict(res.numpy())
                nmi_val = compute_nmi(labels, cluster_labels)
                nmis.append(nmi_val)

            results[attr_name][layers[li]] = {
                "mean": float(np.mean(nmis)),
                "std": float(np.std(nmis)),
                "values": [float(v) for v in nmis],
            }

    return results


def run_probe_analysis_light(residuals_list, metadata, test_split=0.2, random_state=42):
    """
    Lightweight probe: 只对低基数属性做 Logistic Regression
    """
    layers = ["L1残差", "L2残差", "L3残差"]
    results = {}

    np.random.seed(random_state)
    N = residuals_list[0].shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1 - test_split))
    train_idx, test_idx = idx[:split], idx[split:]

    # Only run probe on attributes with reasonable cardinality
    probe_attrs = ["primary_category"]  # 15 classes — manageable

    for attr_name in probe_attrs:
        attr_values = metadata[attr_name]
        le = LabelEncoder()
        labels = le.fit_transform(attr_values)

        if len(np.unique(labels)) < 2 or len(np.unique(labels)) > 100:
            continue

        accs = []
        for li, res in enumerate(residuals_list):
            X = res.numpy()
            clf = LogisticRegression(max_iter=200, multi_class="multinomial")
            clf.fit(X[train_idx], labels[train_idx])
            acc = accuracy_score(labels[test_idx], clf.predict(X[test_idx]))
            accs.append(float(acc))

        results[attr_name] = dict(zip(layers, accs))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqvae_ckpt", default="products/task16/from_task62/rqvae_ckpt.pt")
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--metadata", default="products/task16/from_task62/item_metadata.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=64, help="KMeans clusters")
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("Task 62 — Exp1: NMI + Probe 分析")
    print("=" * 60)

    # Check if checkpoint exists
    if not os.path.exists(args.rqvae_ckpt):
        print(f"ERROR: {args.rqvae_ckpt} not found. Train RQ-VAE first.")
        print("Run: python scripts/task6_train_rqvae.py")
        sys.exit(1)

    # Load
    print("\n[1/4] Loading model...")
    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu")
    from scripts.task62_train_rqvae import ResidualQuantization
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])

    print(f"[2/4] Loading embeddings ({args.embeddings})...")
    embeddings = torch.load(args.embeddings, map_location="cpu")

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[3/4] Extracting residuals (device={device})...")
    residuals_list, indices_list = extract_residuals(model, embeddings, device=device)
    for i, r in enumerate(residuals_list):
        print(f"  Layer {i+1}: residual shape={r.shape}")

    print(f"[4/4] Loading metadata...")
    metadata = load_metadata(args.metadata)

    # Price discretization
    metadata["price_discrete"] = metadata["price"]

    # Run NMI analysis
    print("\n--- NMI Analysis ---")
    nmi_results = run_nmi_analysis(
        residuals_list, metadata,
        K=args.K, n_seeds=args.n_seeds
    )
    print_nmi_table(nmi_results)

    # Save NMI results immediately (probe may be slow)
    out_dir = "products/task16/from_task62"
    os.makedirs(out_dir, exist_ok=True)
    out = {"nmi": nmi_results, "config": vars(args)}
    out_path = os.path.join(out_dir, "exp1_nmi_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"NMI results saved: {out_path}")

    # Run lightweight Probe analysis
    print("\n--- Probe Accuracy (light, primary_category only) ---")
    try:
        probe_results = run_probe_analysis_light(residuals_list, metadata)
        print_probe_table(probe_results)
        out["probe"] = probe_results
        out_path_full = os.path.join(out_dir, "exp1_results.json")
        with open(out_path_full, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Full results saved: {out_path_full}")
    except Exception as e:
        print(f"  Probe skipped (error: {e})")


def print_nmi_table(results):
    layers = ["L1残差", "L2残差", "L3残差"]
    print(f"\n{'':>20}", end="")
    for l in layers:
        print(f"  {l:>12}", end="")
    print()

    for attr, layer_data in results.items():
        print(f"{attr:>20}", end="")
        for l in layers:
            if l in layer_data:
                d = layer_data[l]
                print(f"  {d['mean']:.4f}±{d['std']:.4f}", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()


def print_probe_table(results):
    layers = ["L1残差", "L2残差", "L3残差"]
    print(f"\n{'':>20}", end="")
    for l in layers:
        print(f"  {l:>10}", end="")
    print()

    for attr, accs in results.items():
        print(f"{attr:>20}", end="")
        for l in layers:
            if l in accs:
                print(f"  {accs[l]:.4f}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()


if __name__ == "__main__":
    main()
