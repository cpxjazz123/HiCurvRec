"""Task #206-L: 码字半径扫描 — 在不同 c‖e‖² 跨距下测 Poincaré vs Euclidean argmin 一致率.

用户预测:
    c‖e‖² = 0.03 (现状) → ~99.9% 一致
    c‖e‖² = 0.30         → ~95%
    c‖e‖² = 0.60         → ~85%
    c‖e‖² = 0.90         → ~60%

如果实测趋势吻合 → 验证 "铺开半径 + 推高 c‖e‖²" 能让几何主导 → 投入训练工程.
如果推 0.9 一致率还有 95% → 这条路也不通, 收.

策略:
  1. 加载双曲 ep19 ckpt encoder + 3 个码本
  2. 真实数据 → encoder → 32-dim latent (跟 ep19 训练时一致)
  3. 对每个 level:
    a. 模拟 RQ 残差链 → 该 level 的输入 latent
    b. 对每个 target_norm²(=c·‖e‖²), 把码本线性放大到该跨度
       (也同步放大 latent, 因为缩放只是改变"球内位置")
    c. Poincaré argmin vs Euclidean argmin 一致率
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from model.utils import poincare_distance, MLP

ckpt_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task206/stage1_baseline_retrain/Jul-26-2026_14-47-13_beta_1.000_codebook_[64,128,256]_sk_0.000/epoch_19_collision_0.4281_model.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# 加载 ckpt
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 重建 encoder
encoder = MLP(layers=[768, 512, 256, 128, 64, 32], activation='relu').to(device)
enc_sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
encoder.load_state_dict(enc_sd, strict=False)

# 加载 3 个码本
codebooks = {
    0: sd['hrq.vq_layers.0.embeddings.weight'].to(device).float(),
    1: sd['hrq.vq_layers.1.embeddings.weight'].to(device).float(),
    2: sd['hrq.vq_layers.2.embeddings.weight'].to(device).float(),
}

# 真实数据 → encoder → z0
df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
data768 = np.stack([np.asarray(e) for e in df['embedding'].values]).astype(np.float32)
data768_t = torch.tensor(data768).to(device)

encoder.eval()
with torch.no_grad():
    z0 = encoder(data768_t).float()

# argmin 函数
def argmin_dist(latent, codebook, mode='poincare', c=1.0):
    B = latent.shape[0]
    K = codebook.shape[0]
    x = latent.unsqueeze(1).expand(B, K, -1)
    cb = codebook.unsqueeze(0).expand(B, K, -1)
    if mode == 'poincare':
        diff = poincare_distance(x.reshape(-1, latent.shape[-1]),
                                  cb.reshape(-1, latent.shape[-1]), c=c)
        diff = diff.reshape(B, K)
    else:
        diff = (x - cb).reshape(-1, latent.shape[-1]).norm(dim=-1).reshape(B, K)
    return diff.argmin(dim=-1)

def scale_to_c_norm_sqr(tensor, target_c_norm_sqr, c=1.0):
    """线性缩放 tensor 让 c·‖e‖² = target.
       target_c_norm_sqr = c·E[‖e‖²] (per-element mean; 用户用此口径)

       实际操作: 算当前 c·mean(‖e‖²), 求倍数, 整体缩放.
    """
    cur_sqr = (tensor ** 2).sum(dim=-1).mean().item()
    cur_c_norm_sqr = c * cur_sqr
    if cur_c_norm_sqr < 1e-12:
        return tensor.clone()
    factor = np.sqrt(target_c_norm_sqr / cur_c_norm_sqr)
    return tensor * factor

# 目标 c·‖e‖² 跨度
TARGETS = [0.03, 0.10, 0.30, 0.60, 0.85, 0.95]

print('=== Task #206-L 半径扫描 argmin 一致率 ===\n')
print(f'{"level":8s} {"target c‖e‖²":>14s} {"cb_norm_max":>12s} {"lat_norm_max":>13s} '
      f'{"agree":>8s} {"disagree":>10s}')
print('-' * 70)

# 模拟 RQ 残差链, 算各 level 的输入 latent
def get_level_latent(level, codebooks, z0):
    z = z0
    for i in range(level):
        idx = argmin_dist(z, codebooks[i], mode='poincare')
        z = z - codebooks[i][idx]
    return z

for level in [0, 1, 2]:
    lat = get_level_latent(level, codebooks, z0)
    cb_orig = codebooks[level]

    # 记录原始 norm
    cb_orig_max = cb_orig.norm(dim=-1).max().item()
    cb_orig_c_norm_sqr = (cb_orig ** 2).sum(dim=-1).mean().item()
    lat_orig_max = lat.norm(dim=-1).max().item()
    print(f'\n[Level {level}] 原始码本: c·mean(‖e‖²) = {cb_orig_c_norm_sqr:.4f}, max_norm = {cb_orig_max:.4f}')
    print(f'[Level {level}] 原始latent: max_norm = {lat_orig_max:.4f}')

    for target in TARGETS:
        # 同时缩放 cb 和 lat (跟 argmin 只差相对关系)
        cb_scaled = scale_to_c_norm_sqr(cb_orig, target)
        lat_scaled = scale_to_c_norm_sqr(lat, target)

        # 球内 guard: c·max(‖e‖²) < 0.99 才安全
        c_max_norm_sqr = ((cb_scaled ** 2).sum(dim=-1)).max().item()
        lat_max_norm_sqr = ((lat_scaled ** 2).sum(dim=-1)).max().item()

        # 如果超球 (max > 0.999² → boundary 风险), 提示
        cb_at_risk = c_max_norm_sqr > 0.95
        lat_at_risk = lat_max_norm_sqr > 0.95

        # 仍算 argmin (Poincaré 用 proj 截断)
        idx_p = argmin_dist(lat_scaled, cb_scaled, mode='poincare')
        idx_e = argmin_dist(lat_scaled, cb_scaled, mode='euclidean')
        agree = (idx_p == idx_e).float().mean().item()

        cb_max_norm = np.sqrt(c_max_norm_sqr)
        lat_max_norm = np.sqrt(lat_max_norm_sqr)
        flag = '⚠️过球' if (cb_at_risk or lat_at_risk) else ''
        print(f'  {level:6d} {target:14.3f} {cb_max_norm:12.4f} {lat_max_norm:13.4f} '
              f'{agree*100:7.2f}%  {(1-agree)*100:9.2f}%  {flag}')

print('\n=== 结论判据 ===')
print('如果 c‖e‖²=0.30 → 一致率 ≈ 95%, 表明这条路(铺开半径)能引入几何效应, 值得训练工程')
print('如果 c‖e‖²=0.90 → 一致率 ≈ 60%, 也证明能引入, 但需要球边界风险评估')
print('如果 c‖e‖²=0.90 → 一致率 ≈ 95%, 表明这条路也不通, 几何在该数据下完全不可测')
