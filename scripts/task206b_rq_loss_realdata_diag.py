"""Task #206 Part B v2 — 真实数据 + 真实模型, 对比 poincare vs euclidean rq_loss.

数据流: item_emb.parquet (768-d) → MLP encoder (768→512→32) → quant loss 比较
"""
import sys, os
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from model.utils import HVectorQuantization, MLP

device = 'cuda:0'
torch.manual_seed(42)

# 真实数据 (item_emb.parquet 768-d)
df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
data768 = np.stack([np.asarray(e) for e in df['embedding'].values]).astype(np.float32)
data768_t = torch.tensor(data768).to(device)

# HG-Rec encoder head: 768 → 32 (跟 HG-Rec train_hrqvae.py MLP 结构对齐)
encoder_head = MLP(layers=[768, 512, 32], activation='relu').to(device)
encoder_head.eval()
with torch.no_grad():
    data32 = encoder_head(data768_t).float()
print(f'encoder 后 32-d norm: mean={data32.norm(dim=-1).mean():.4f}, max={data32.norm(dim=-1).max():.4f}')

# assert 在球内
max_norm = data32.norm(dim=-1).max().item()
if max_norm >= 0.9:
    print(f'⚠️  max norm {max_norm:.4f} >= 0.9, 不在 Poincaré ball 内.')
    print(f'⚠️  在该区间 poincare_distance 已被 proj_to_ball + artanh clamp 截断, 测不准双曲.')
    print(f'⚠️  把码字尺度缩到球内: data32 *= (0.85 / max_norm)')
    data32 = data32 * (0.85 / max_norm)
    print(f'缩放后 max norm = {data32.norm(dim=-1).max().item():.4f}')
else:
    print(f'✓ 数据范数检查通过 (max norm {max_norm:.4f} < 0.9 in Poincaré ball)')

e_dim = 32
K = 64
beta = 0.5

def make_hq(euclidean=False, seed=42):
    torch.manual_seed(seed)
    hq = HVectorQuantization(n_e=K, e_dim=e_dim, beta=beta, curvature=1.0, sk_eps=0.0,
                              euclidean_qloss=euclidean, kmeans_init=True, kmeans_iters=100).to(device)
    hq.train()
    return hq

def run_arm(name, hq, n_batches=8):
    indices_set = set()
    rq_losses = []
    batch_size = 256
    n_used = 0
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > len(data32):
            break
        batch = data32[start:end]
        try:
            x_q, rq_loss, indices = hq(batch, use_sk=False)
            rq_losses.append(rq_loss.item())
            n_used += batch_size
            for idx in indices.cpu().numpy():
                code = '-'.join([str(int(c)) for c in idx])
                indices_set.add(code)
        except Exception as e:
            print(f'  Batch {i} error: {e}')
            break
    if not rq_losses:
        return float('nan'), float('nan')
    mean_loss = float(np.mean(rq_losses))
    collision = 1 - len(indices_set) / max(n_used, 1)
    print(f'\n=== {name} ===')
    print(f'  {len(rq_losses)} batch rq_loss: {[f"{x:.6f}" for x in rq_losses]}')
    print(f'  mean rq_loss = {mean_loss:.6f}')
    print(f'  collision_rate = {collision:.4f}')
    return mean_loss, collision

# 臂 A: 双曲 (Poincaré)
hq_hyp = make_hq(euclidean=False)
mean_hyp, coll_hyp = run_arm('臂 A (双曲 poincare)', hq_hyp)

# 臂 B: 欧式 (--euclidean_qloss)
hq_euc = make_hq(euclidean=True)
mean_euc, coll_euc = run_arm('臂 B (欧式 F.mse_loss)', hq_euc)

print('\n===== 真实数据量级对比 =====')
print(f'臂 A 双曲 rq_loss mean = {mean_hyp:.6f}')
print(f'臂 B 欧式 rq_loss mean = {mean_euc:.6f}')
if mean_euc > 1e-9:
    ratio = mean_hyp / mean_euc
    print(f'ratio 双曲/欧式 = {ratio:.1f}x')
    print(f'  期望: 用户预测 ~128x (双曲公式 d² ≈ 4·‖u−v‖²)')
    if ratio > 50:
        print(f'  ✅ 量级差距确认 (ratio {ratio:.0f}x 远大于 1)')
    elif ratio > 10:
        print(f'  ⚠️  量级部分不匹配 (ratio {ratio:.0f}x, 用户预测 128)')
    else:
        print(f'  ❌ 量级匹配 (ratio {ratio:.1f}x, 58.66% 另有原因)')
else:
    print('!! mean_euc ≈ 0')
