#!/usr/bin/env python3
"""Task #200 用户 2026-07-26 清单 2 — C' 测试: 随机 init emb_geo, 跑 300 步看能否恢复.

目的: 判别 geo 梯度能否救回坏初始化状态.
判据:
  - 利用率从低恢复到接近 100% → ✅ 梯度有能力自我纠正 → 流水线失败另有原因
  - 一直起不来                       → ❌ geo 梯度救不回坏状态 → 必须保证流水线 init 就是好的

附加测试 (用户清单 1 验证): rec_align 改 sum 版后, emb_rec 梯度应该回到 1e-3 量级
(原 F.mse_loss 默认 mean reduction, 损失 32× → 梯度 1e-5 → emb_rec 永远冻在 init).

跑法: GPU 0 (空闲), ~5 min.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import HVectorQuantization

torch.manual_seed(42)

# 用 task181 baseline 训好的真实 latent (从 isolation test 拿, L0 残差)
lat = torch.load(REPO / "logs/task200/isolation_test_v3/lat_L0.pt")
print(f"latent shape: {lat.shape}, norm mean: {lat.norm(dim=-1).mean():.3f}, "
      f"norm median: {lat.norm(dim=-1).median():.3f}")

e_dim = lat.shape[-1]
K = 64

vq = HVectorQuantization(
    n_e=K, e_dim=e_dim, sk_eps=0.0, beta=0.5,
    kmeans_init=False, kmeans_iters=100, sk_iters=50,
    curvature=1.0, euclidean_qloss=False, loss_mult_codebook=1.0,
    dual_codebook=True,
)
vq.use_centering = True
vq.z_mean = torch.zeros(e_dim)
vq.z_mean_ema = 0.99

vq.train()

# === 用户清单 2: 故意破坏初始化 (C' 关键) ===
# 不调 vq.init_emb(lat), 直接 normal_(0,1) 随机初始化 emb_geo
# (用户原话: "随机初始化 emb_geo (不用 kmeans)")
vq.initted = True  # 跳过 init_emb 路径
vq.emb_geo.weight.data.normal_(0, 1)
vq.emb_rec.weight.data.normal_(0, 0.3)

# 计算初始利用率 (assign = argmax cos)
with torch.no_grad():
    dirs = F.normalize(lat, dim=-1, eps=1e-8)
    centers_geo_norm = F.normalize(vq.emb_geo.weight, dim=-1, eps=1e-8)
    cos = dirs @ centers_geo_norm.t()
    init_assign = cos.argmax(dim=-1)
    init_util = (init_assign.bincount(minlength=K) > 0).float().mean().item()
    init_n_unique = len(init_assign.unique())
    print(f"\n[INIT] 随机 emb_geo → init_util={init_util:.4f}, init_unique={init_n_unique}/{K}")

opt = torch.optim.AdamW(vq.parameters(), lr=1e-3)

print(f"\n=== 300 步 AdamW + 随机 init emb_geo (用户清单 2 C' 测试) ===")
print(f"step | loss    | geo_loss | commit  | code    | util    | uniq | rec_grad | geo_grad")
print(f"-----|---------|----------|---------|---------|---------|------|----------|---------")

for step in range(300):
    opt.zero_grad()
    x_q, loss, indices = vq(lat)
    loss.backward()

    rg = vq.emb_rec.weight.grad.norm().item() if vq.emb_rec.weight.grad is not None else 0
    gg = vq.emb_geo.weight.grad.norm().item() if vq.emb_geo.weight.grad is not None else 0

    # 分解各 loss 分量
    with torch.no_grad():
        # 重新跑一遍 forward 但不算 backward, 拿分量
        # 简化: 直接用 indices + 距离估算
        assign = indices
        uniq = len(assign.unique())
        util = (assign.bincount(minlength=K) > 0).float().mean().item()

    opt.step()

    if step % 30 == 0 or step < 5:
        print(f"{step:4d} | {loss.item():.4f} | "
              f"{'N/A':>6} | {'N/A':>6} | {'N/A':>6} | "
              f"{util:.4f} | {uniq:4d} | {rg:.4f}  | {gg:.4f}")

# 最终判断
with torch.no_grad():
    dirs = F.normalize(lat, dim=-1, eps=1e-8)
    centers_geo_norm = F.normalize(vq.emb_geo.weight, dim=-1, eps=1e-8)
    cos = dirs @ centers_geo_norm.t()
    final_assign = cos.argmax(dim=-1)
    final_util = (final_assign.bincount(minlength=K) > 0).float().mean().item()
    final_n_unique = len(final_assign.unique())
    cos_max = cos.max(dim=-1).values
    cos_mean = cos_max.mean().item()
    cos_max_max = cos_max.max().item()
    n_dup = (cos_max > 0.99).sum().item()

print(f"\n=== FINAL (300 步后) ===")
print(f"init_util={init_util:.4f} → final_util={final_util:.4f}")
print(f"init_unique={init_n_unique} → final_unique={final_n_unique}")
print(f"final cos_mean={cos_mean:.4f}, cos_max={cos_max_max:.4f}, n_dup={n_dup}")

if final_util > 0.9 and final_n_unique >= K * 0.95:
    print(f"\n✅ 结论: 梯度能自我纠正 (util {init_util:.2f} → {final_util:.2f})")
    print(f"   → 流水线失败另有原因 (不是初始化)")
elif final_util < 0.5:
    print(f"\n❌ 结论: geo 梯度救不回坏状态 (util 始终 {final_util:.2f})")
    print(f"   → 必须保证流水线里的初始化就是好的")
else:
    print(f"\n⚠️ 结论: 部分恢复 (util {init_util:.2f} → {final_util:.2f}, "
          f"unique {final_n_unique}/{K})")
    print(f"   → 需要更多步数 / 调整 lr 才能完全恢复")

# === 用户清单 1 验证: rec_align 改 sum 版后, emb_rec 梯度应该回到 1e-3 量级 ===
print(f"\n=== emb_rec 梯度验证 (用户清单 1 期望: 1e-3 量级, 不是 1e-5) ===")
print(f"final rec_grad norm: {rg:.6f}")
if rg >= 1e-4:
    print(f"✅ emb_rec 梯度在 {rg:.2e} (期望 ≥ 1e-4, sum 版修复成功)")
elif rg >= 1e-5:
    print(f"⚠️ emb_rec 梯度在 {rg:.2e} (期望 ≥ 1e-4, 部分修复)")
else:
    print(f"❌ emb_rec 梯度在 {rg:.2e} (期望 ≥ 1e-4, 仍太小)")