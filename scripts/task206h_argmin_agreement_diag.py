"""Task #206-H: 5-min argmin 一致率诊断.

加载已训好的双曲 ep19 ckpt (real encoder + real codebooks),
对每个 level: 模拟 RQ 链 (z → residual → next), 测 Poincaré vs Euclidean argmin 一致率.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter

from model.utils import poincare_distance, MLP

ckpt_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task206/stage1_baseline_retrain/Jul-26-2026_14-47-13_beta_1.000_codebook_[64,128,256]_sk_0.000/epoch_19_collision_0.4281_model.pth'
device = 'cuda:0'
torch.manual_seed(42)

# 读 ckpt
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
elif isinstance(ckpt, dict) and 'model' in ckpt:
    sd = ckpt['model']
else:
    sd = ckpt

# 重建 encoder: ckpt keys 是 encoder.mlp.{1,4,7,10,13}.*
# MLP layers = [768, 512, 256, 128, 64, 32] (跟 train_hrqvae.py --layers 512 256 128 64 + e_dim=32)
# MLP with activation='relu' 在 mlp.{1,4,7,10,13}.weight (Linear 6 层 + 5 ReLU 间隔)
encoder = MLP(layers=[768, 512, 256, 128, 64, 32], activation='relu').to(device)
enc_sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
print(f'encoder load: missing={len(missing)} unexpected={len(unexpected)}')

# 加载 3 个码本
codebooks = {
    0: sd['hrq.vq_layers.0.embeddings.weight'].to(device).float(),
    1: sd['hrq.vq_layers.1.embeddings.weight'].to(device).float(),
    2: sd['hrq.vq_layers.2.embeddings.weight'].to(device).float(),
}
for k, cb in codebooks.items():
    norms = cb.norm(dim=-1)
    print(f'codebook {k}: shape={cb.shape}, norm mean={norms.mean():.4f}, max={norms.max():.4f}, min={norms.min():.4f}')

# 加载数据
df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
data768 = np.stack([np.asarray(e) for e in df['embedding'].values]).astype(np.float32)
data768_t = torch.tensor(data768).to(device)

# 编码 (data → 32-dim latent)
encoder.eval()
with torch.no_grad():
    z0 = encoder(data768_t).float()
lat_norm = z0.norm(dim=-1)
print(f'\nlatent z0: shape={z0.shape}, norm mean={lat_norm.mean():.4f}, max={lat_norm.max():.4f}, min={lat_norm.min():.4f}')

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

print('\n=== 5-min Argmin 一致率诊断 (level 0, z0 only) ===\n')
for lvl in [0, 1, 2]:
    cb = codebooks[lvl]

    # 拿该 level 的 latent 输入: 模拟 RQ 残差链
    z = z0
    for i in range(lvl):
        # 找最近的码字, 减去作为残差
        idx_tmp = argmin_dist(z, codebooks[i], mode='poincare')
        z_q = codebooks[i][idx_tmp]
        z = z - z_q
    lat_lvl = z  # level lvl 的输入 latent

    # 测
    print(f'\n--- Level {lvl} (K={cb.shape[0]}) ---')
    print(f'  input latent norm: mean={lat_lvl.norm(dim=-1).mean():.4f}, max={lat_lvl.norm(dim=-1).max():.4f}, min={lat_lvl.norm(dim=-1).min():.4f}')
    print(f'  codebook norm:     mean={cb.norm(dim=-1).mean():.4f}, max={cb.norm(dim=-1).max():.4f}')

    idx_p = argmin_dist(lat_lvl, cb, mode='poincare')
    idx_e = argmin_dist(lat_lvl, cb, mode='euclidean')
    agree = (idx_p == idx_e).float().mean().item()

    # 分布
    p_dist = Counter(idx_p.cpu().numpy().tolist())
    e_dist = Counter(idx_e.cpu().numpy().tolist())
    n_unique_p = len(p_dist)
    n_unique_e = len(e_dist)

    print(f'  agreement = {agree:.4f} ({agree*100:.2f}%)')
    print(f'  disagreement = {(1-agree)*100:.2f}% ({int((1-agree)*len(lat_lvl))} / {len(lat_lvl)} items)')
    print(f'  unique codes: poincare={n_unique_p}, euclidean={n_unique_e}')

    # 冲突样本数 (二者分配到不同码字)
    n_diff = (idx_p != idx_e).sum().item()
    print(f'  items where p != e: {n_diff}')

print('\n=== 结论 ===')
print('  > 99% agreement → 几何在分配阶段不改变, 欧式 vs 双曲 对比无意义 (在该尺度)')
print('  < 95% agreement → 几何显著改变分配, 需要先调 ×4 + 找 epoch-aligned 对齐点')
print('  中间区间 → 谨慎, 检查冲突是否集中在某些物品/某些码字')
