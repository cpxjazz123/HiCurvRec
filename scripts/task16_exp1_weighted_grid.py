"""
Task #66 Exp1 — 加权几何距离穷举搜索

目的：验证 d_mix = w_E·d_E + w_H·d_H + w_S·d_S 能否在 L2 归一化空间上
      打破 Task #62 Exp3 的"距离排序等价"。

设计：
  1. 加载 Task #62 RQ-VAE ckpt (768-dim) + 768-dim item embeddings
  2. 提取每层残差 r_1, r_2, r_3（已 L2 归一化）
  3. 对每层 ℓ，网格搜索 (w_E, w_H, w_S) 三元组
  4. 计算量化 MSE：MSE(w) = mean ||r - q||²，其中 q = argmin_k d_mix(r, c_k)
  5. 输出最优权重 + 与 baseline（w_E=1）的 ΔMSE

输入：
  - products/task16/from_task62/rqvae_ckpt.pt
  - products/task16/from_task62/item_embeddings.pt (11924, 768)
输出：
  - products/task16/from_task66/exp1_grid_results.json
  - products/task16/from_task66/exp1_best_weights.json
"""

import os, json, sys, time
import torch
import torch.nn.functional as F

# Paths
TASK62_DIR = "products/task16/from_task62"
TASK66_DIR = "products/task16/from_task66"
os.makedirs(TASK66_DIR, exist_ok=True)

# ───────── Load data ─────────
print("=" * 60)
print("Task #66 Exp1 — Weighted Distance Grid Search")
print("=" * 60)

print("\n[1/5] Loading Task #62 RQ-VAE ckpt...")
ckpt = torch.load(f"{TASK62_DIR}/rqvae_ckpt.pt", map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"]
n_layers = ckpt["n_layers"]
n_clusters = ckpt["n_clusters"]
n_features = ckpt["n_features"]
print(f"  n_layers={n_layers}, n_clusters={n_clusters}, n_features={n_features}")

# Extract codebooks: (n_layers, n_clusters, n_features)
codebooks = torch.stack(
    [state_dict[f"quantizers.{i}.codebook"] for i in range(n_layers)], dim=0
)
print(f"  codebooks shape: {codebooks.shape}")

print("\n[2/5] Loading item embeddings...")
embeddings = torch.load(f"{TASK62_DIR}/item_embeddings.pt", map_location="cpu", weights_only=False)
print(f"  embeddings shape: {embeddings.shape}")

# Normalize to unit sphere (matching RQ-VAE training assumption)
emb_norms = embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
emb_normalized = embeddings / emb_norms
print(f"  normalized to unit sphere (norm≈1.0)")

# ───────── Extract residuals via layer-by-layer quantization ─────────
print("\n[3/5] Extracting per-layer residuals...")


def quantize_with_euclidean(r, codebook):
    """Standard euclidean nearest-neighbor quantization.
    r: (N, D), codebook: (K, D)
    returns: q (N, D), indices (N,)"""
    dist = torch.cdist(r, codebook, p=2)
    indices = dist.argmin(dim=-1)
    q = codebook[indices]
    return q, indices


# Forward pass to extract residuals
residual = emb_normalized
residuals = []      # residuals[ℓ] = input to layer ℓ quantization; residuals[0]=z, residuals[n_layers]=r_n_layers
quantized = []      # quantized[ℓ] = q_ℓ
indices_per_layer = []

for layer_id in range(n_layers):
    residuals.append(residual.clone())  # input to this layer's quantizer
    q, idx = quantize_with_euclidean(residual, codebooks[layer_id])
    quantized.append(q)
    indices_per_layer.append(idx)
    residual = residual - q.detach()  # next layer's input

# After last layer, also store final residual (z - q_1 - ... - q_n_layers)
residuals.append(residual.clone())

# residuals[0] = z (input), residuals[ℓ] = r_ℓ = input to layer ℓ quantization
print(f"  residuals[0] (input z): mean norm = {residuals[0].norm(dim=-1).mean():.4f}")
for i in range(n_layers):
    r_label = f"r_{i+1}" if i < n_layers else f"final residual"
    print(f"  residuals[{i+1}] ({r_label}): mean norm = {residuals[i+1].norm(dim=-1).mean():.4f}")

# ───────── Distance functions ─────────
print("\n[4/5] Defining distance functions...")


def d_euclidean(x, c):
    """x: (N, D), c: (K, D) → (N, K) euclidean distance."""
    return torch.cdist(x, c, p=2)


def d_hyperbolic(x, c, eps=1e-5):
    """Poincaré ball distance after projection to ||x_p|| < 1.
    x: (N, D) on unit sphere, c: (K, D) on unit sphere.
    Project to Poincaré: x_p = tanh(||x||) · x/||x|| ≈ tanh(1)·x since ||x||≈1.
    """
    # Project to Poincaré ball with ||x_p|| < 1
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    norm_c = c.norm(dim=-1, keepdim=True).clamp_min(eps)
    # tanh(||x||)·x/||x|| — but x is already unit norm → tanh(1)·x
    x_p = torch.tanh(norm_x) * x / norm_x  # (N, D)
    c_p = torch.tanh(norm_c) * c / norm_c  # (K, D)

    sq_x = (1 - (x_p * x_p).sum(dim=-1, keepdim=True)).clamp_min(eps * eps)  # (N, 1)
    sq_c = (1 - (c_p * c_p).sum(dim=-1, keepdim=True)).clamp_min(eps * eps)  # (K, 1)
    diff_sq = ((x_p.unsqueeze(1) - c_p.unsqueeze(0)) ** 2).sum(dim=-1)  # (N, K)
    arg = 1 + 2 * diff_sq / (sq_x @ sq_c.T)  # (N, K)
    arg = arg.clamp_min(1 + eps)  # arcosh domain
    return torch.acosh(arg)


def d_spherical(x, c, eps=1e-7):
    """Spherical distance on unit sphere: arccos(<x, y>)."""
    cos_sim = x @ c.T  # (N, K)
    cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)
    return torch.arccos(cos_sim)


def mixed_distance(r, c, w_e, w_h, w_s):
    """Weighted combination of 3 distances."""
    return w_e * d_euclidean(r, c) + w_h * d_hyperbolic(r, c) + w_s * d_spherical(r, c)


def quantize_mixed(r, c, w_e, w_h, w_s):
    """Nearest-neighbor with mixed distance."""
    dist = mixed_distance(r, c, w_e, w_h, w_s)
    idx = dist.argmin(dim=-1)
    q = c[idx]
    return q, idx


def quant_mse(r, q):
    """||r - q||² per sample, averaged."""
    return ((r - q) ** 2).sum(dim=-1).mean().item()


# ───────── Grid search ─────────
print("\n[5/5] Grid search over (w_E, w_H, w_S)...")
print("  Constraint: w_E + w_H + w_S = 1, all ≥ 0")
print("  Step: 0.1")

step = 0.1
weights_list = []
w_e_values = [round(i * step, 2) for i in range(int(1.0 / step) + 1)]
for w_e in w_e_values:
    for w_h in [round(i * step, 2) for i in range(int((1.0 - w_e) / step) + 1)]:
        w_s = round(1.0 - w_e - w_h, 2)
        if w_s < -1e-6:
            continue
        w_s = max(0.0, w_s)
        weights_list.append((w_e, w_h, w_s))
print(f"  total combinations: {len(weights_list)}")

results = {"per_layer": {}, "metadata": {
    "n_layers": n_layers, "n_clusters": n_clusters, "n_features": n_features,
    "n_items": embeddings.shape[0], "step": step,
    "n_combinations": len(weights_list),
}}

start_time = time.time()
for layer_id in range(n_layers):
    r = residuals[layer_id + 1]  # the residual at this layer (input to quantize)
    c = codebooks[layer_id]
    # NOTE: residuals[0] = z (raw embedding); residuals[1] = r_1 (after q_1)
    # For layer ℓ (1-indexed), we quantize residuals[ℓ]
    # But ℓ=0 in array, ℓ=1 in physics. Let's index by physics layer.

    layer_results = []
    baseline_mse = None
    best = {"mse": float("inf"), "weights": None}

    for w_e, w_h, w_s in weights_list:
        with torch.no_grad():
            q, _ = quantize_mixed(r, c, w_e, w_h, w_s)
        mse = quant_mse(r, q)
        if abs(w_e - 1.0) < 1e-6 and abs(w_h) < 1e-6 and abs(w_s) < 1e-6:
            baseline_mse = mse
        if mse < best["mse"]:
            best = {"mse": mse, "weights": (w_e, w_h, w_s)}
        layer_results.append({
            "w_e": w_e, "w_h": w_h, "w_s": w_s,
            "mse": mse,
        })

    elapsed = time.time() - start_time
    print(f"\n  Layer ℓ={layer_id+1}:")
    print(f"    baseline (1,0,0): MSE = {baseline_mse:.6e}")
    print(f"    best     ({best['weights'][0]:.1f}, {best['weights'][1]:.1f}, {best['weights'][2]:.1f}): MSE = {best['mse']:.6e}")
    delta_pct = (best["mse"] - baseline_mse) / baseline_mse * 100 if baseline_mse else 0
    print(f"    ΔMSE = {delta_pct:+.3f}%")
    results["per_layer"][f"layer_{layer_id+1}"] = {
        "baseline_mse_euclidean": baseline_mse,
        "best_weights": list(best["weights"]),
        "best_mse": best["mse"],
        "delta_pct": delta_pct,
        "all_results": layer_results,
    }

results["metadata"]["total_elapsed_sec"] = round(time.time() - start_time, 2)

# ───────── Save results ─────────
out_full = f"{TASK66_DIR}/exp1_grid_results.json"
with open(out_full, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Full results saved: {out_full}")

# Save concise summary
summary = {
    "metadata": results["metadata"],
    "per_layer_summary": {},
}
for layer_name, lr in results["per_layer"].items():
    summary["per_layer_summary"][layer_name] = {
        "baseline_mse_euclidean": lr["baseline_mse_euclidean"],
        "best_weights": lr["best_weights"],
        "best_mse": lr["best_mse"],
        "delta_pct": lr["delta_pct"],
    }
out_summary = f"{TASK66_DIR}/exp1_best_weights.json"
with open(out_summary, "w") as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {out_summary}")

# ───────── Verdict ─────────
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
all_best_ee = all(
    abs(lr["best_weights"][0] - 1.0) < 1e-6
    for lr in results["per_layer"].values()
)
max_delta = max(
    abs(lr["delta_pct"])
    for lr in results["per_layer"].values()
)

if all_best_ee:
    print(f"❌ 所有层最优权重都是 (1.0, 0.0, 0.0) → 加权在 L2 归一化下完全等价于单欧氏距离")
    print(f"   几何类改进路线彻底关闭。")
    print(f"   max |ΔMSE| = {max_delta:.4f}%")
elif max_delta > 0.5:
    print(f"⚠️  加权距离对量化 MSE 有微小影响 (max |ΔMSE| = {max_delta:.3f}%)")
    print(f"   需要 Exp2 学习权重 + Exp3 端到端 R@10 验证")
else:
    print(f"✅ 加权距离有统计显著的微小效果 (max |ΔMSE| = {max_delta:.3f}%)")
    print(f"   继续 Exp2/3")

print("=" * 60)
