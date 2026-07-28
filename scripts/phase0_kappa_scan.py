"""Phase 0 (球面侧 κ 扫描, 30 min, CPU):
- 加载 E1 baseline ckpt (task226 e1_normcap epoch_199 collision 0.0953)
- 跑 E1 encoder 拿真实 latent
- 对每个 κ ∈ {-10, -3, -1, -0.3, 0, +0.3, +1, +3, +10}:
  - 计算 κ-Stereographic distance to codebook (per layer)
  - 测: agreement vs Euclidean argmin / dyn_ratio / usage / proj_needed / NaN / assign_entropy
- 三层各跑一遍, 输出 CSV + summary
"""
import sys, os, json, time
import numpy as np
import torch
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.hrqvae_free_curv import FreeCurvHRQVAE, geodesic_distance_sq

CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task226/e1_normcap/Jul-27-2026_00-09-44_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_199_collision_0.0953_model.pth"
DATA = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

KAPPA_LIST = [-10.0, -3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 10.0]
LAYERS = [(0, 64), (1, 128), (2, 256)]  # layer_idx, K
E_DIM = 32

print("=" * 70)
print("Phase 0: kappa 扫描 (球面侧 vs 双曲侧 vs 欧式)")
print("=" * 70)

# 加载 item emb
df = pd.read_parquet(DATA)
emb = np.stack(df['embedding'].values, axis=0)
print(f"item_emb shape = {emb.shape}")

# E1 config (跟 task226 launcher 一致): normcap + simvq + poincare loss + spread mode + 64/128/256
config = {
    "in_dim": 768, "num_emb_list": [64, 128, 256], "e_dim": 32, "M": 1,
    "kappa_max": 2.0, "layers": [512, 256, 128], "dropout_prob": 0.0, "bn": False,
    "loss_type": "poincare", "quant_loss_weight": 1.0, "beta": 0.5,
    "kmeans_init": False, "kmeans_iters": 100,
    "sk_eps": [0.003, 0.003, 0.003], "sk_iters": 3,
}

model = FreeCurvHRQVAE(**config)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
# E1 用的是 HRQVAE (非 free-curv) state_dict 命名不同, 试兼容 load
try:
    model.load_state_dict(ckpt, strict=False)
    print(f"loaded ckpt with strict=False")
except Exception as e:
    print(f"load_state_dict failed: {e}")
    raise
model.eval()

with torch.no_grad():
    x = torch.from_numpy(emb).float()  # (N, 768)
    z = model.encoder(x)  # (N, 32) latent
    print(f"latent z shape = {z.shape}, norm max = {z.norm(dim=-1).max().item():.4f}")
    # 取 codebook (三层)
    codebooks = []
    for li, vq in enumerate(model.hrq.vq_layers):
        cb = vq.embeddings.weight.detach()  # (K, 32)
        codebooks.append(cb)
        print(f"  L{li}: codebook shape {cb.shape}, norm max {cb.norm(dim=-1).max().item():.4f}")

# 对每层每个 κ, 跑一次完整评估
print()
print("=" * 70)
print(f"{'layer':>5} {'kappa':>6} {'side':>8} {'agreement':>10} {'dyn':>8} {'usage':>8} {'assign_H':>8} {'nan':>5} {'d_max':>8}")
print("-" * 70)

results = []
t0 = time.time()
for li, K in LAYERS:
    cb = codebooks[li]  # (K, 32)
    # 欧式 argmin (baseline)
    eucl_d = torch.cdist(z, cb)  # (N, K)
    eucl_idx = eucl_d.argmin(dim=-1)
    for kappa_val in KAPPA_LIST:
        kappa = torch.tensor(kappa_val)
        if kappa_val > 0:
            side = "sphere"
        elif kappa_val < 0:
            side = "hyper"
        else:
            side = "euclid"
        # κ-Stereographic distance (per-sample, per-code)
        d = geodesic_distance_sq(z.unsqueeze(1), cb.unsqueeze(0), kappa).squeeze(-1).sqrt()
        # argmin
        if kappa_val == 0:
            k_idx = eucl_idx  # 跟欧式相同
        else:
            k_idx = d.argmin(dim=-1)
        # metrics
        agreement = (k_idx == eucl_idx).float().mean().item()
        d_min = d.gather(1, k_idx.unsqueeze(1)).squeeze(1).mean().item()
        d_max_row = d.max(dim=-1).values.mean().item()
        dyn = (d.max(dim=-1).values / (d.min(dim=-1).values + 1e-8)).mean().item()
        usage_count = torch.bincount(k_idx, minlength=K)
        n_used = (usage_count > 0).sum().item()
        usage = n_used / K
        # 分配熵
        p = usage_count.float() / (usage_count.sum() + 1e-8)
        p_nz = p[p > 0]
        assign_H = -(p_nz * (p_nz + 1e-9).log()).sum().item()
        has_nan = bool(torch.isnan(d).any().item() or torch.isinf(d).any().item())
        # 双曲侧检查 proj
        if kappa_val < 0:
            hyp_bound = 1.0 / (abs(kappa_val) ** 0.5)
            needs_proj = (z.norm(dim=-1) > hyp_bound).any().item()
        else:
            needs_proj = False
        print(f"  L{li}  {kappa_val:>6.2f}  {side:>6} {agreement:>10.4f} {dyn:>8.4f} {usage:>8.4f} {assign_H:>8.4f} {str(has_nan):>5} {d_max_row:>8.4f}")
        results.append({
            "layer": li, "kappa": kappa_val, "side": side,
            "agreement": agreement, "dyn_ratio": dyn, "usage": usage,
            "assign_entropy": assign_H, "has_nan": has_nan,
            "needs_proj": needs_proj, "d_min": d_min, "d_max": d_max_row,
        })
print(f"\nPhase 0 完成, 耗时 {time.time()-t0:.1f}s")

# 保存 JSON
out = {"results": results, "config": config, "ckpt": CKPT}
out_path = "/home/wlia0047/ar57/wenyu/GeneRec/verdicts/_phase0_kappa_scan.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Results saved to {out_path}")

# Phase 0 判据总结
print()
print("=" * 70)
print("Phase 0 判据总结 (球面侧 usage ≥ 80% AND agreement 60-90%)")
print("=" * 70)
sphere_results = [r for r in results if r["side"] == "sphere"]
for r in sphere_results:
    ok = r["usage"] >= 0.80 and 0.6 <= r["agreement"] <= 0.9
    mark = "✅" if ok else "❌"
    print(f"  L{r['layer']} κ={r['kappa']:+.1f}  agreement={r['agreement']:.3f}  usage={r['usage']:.3f}  {mark}")
