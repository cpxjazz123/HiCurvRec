"""Task #206m-v4: 每层独立 c 强度对齐扫描 (内存优化版).

用户纠正: 三层码字长度不同, 同 c ≠ 同几何强度.
用 per-layer c 让三层 c·max‖x‖² 对齐到 {0.60, 0.85}.
然后看 util/熵 — layer-dependent 是否真实.
"""
import sys, gc
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import numpy as np
import pandas as pd
from model.utils import poincare_distance, expmap0, MLP

torch.manual_seed(42)
device = 'cpu'

ckpt_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task206/stage1_baseline_retrain/Jul-26-2026_14-47-13_beta_1.000_codebook_[64,128,256]_sk_0.000/epoch_19_collision_0.4281_model.pth'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

encoder = MLP(layers=[768, 512, 256, 128, 64, 32], activation='relu').to(device)
enc_sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
encoder.load_state_dict(enc_sd, strict=False)

codebooks_tan = {
    i: sd[f'hrq.vq_layers.{i}.embeddings.weight'].to(device).float()
    for i in range(3)}

df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
data768 = np.stack([np.asarray(e) for e in df['embedding'].values]).astype(np.float32)
data768_t = torch.tensor(data768).to(device)
del df

encoder.eval()
with torch.no_grad():
    z0 = encoder(data768_t).float()
del encoder, data768_t

def argmin_dist_mem_eff(latent_tan, codebook_tan, mode='poincare', c=1.0, chunk_size=1024):
    """内存节约版 argmin: 分 chunk 处理."""
    B = latent_tan.shape[0]
    K = codebook_tan.shape[0]
    indices = torch.zeros(B, dtype=torch.long)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        batch = latent_tan[start:end]
        cb_h = expmap0(codebook_tan, c)  # (K, d)
        if mode == 'poincare':
            lh = expmap0(batch, c)
            d = poincare_distance(lh.unsqueeze(1).expand(end-start, K, -1),
                                   cb_h.unsqueeze(0).expand(end-start, K, -1), c).squeeze(-1)
        else:
            lh = expmap0(batch, c)
            d = (lh.unsqueeze(1).expand(end-start, K, -1) - cb_h.unsqueeze(0).expand(end-start, K, -1)).norm(dim=-1)
        indices[start:end] = d.argmin(dim=-1)
    return indices

def util_entropy(indices, K):
    usage = torch.bincount(indices, minlength=K).float()
    util = (usage > 0).float().mean().item()
    p = usage / usage.sum().clamp_min(1e-12)
    ent = -(p * (p + 1e-12).log()).sum().item()
    return util, ent, ent / np.log(K)

def get_level_latent(level, cbs, z):
    # 残差链: argmin (欧式, c=1) → 减
    zc = z.clone()
    for i in range(level):
        idx = argmin_dist_mem_eff(zc, cbs[i], mode='euclidean', c=1.0)
        zc = zc - cbs[i][idx]
    return zc

# 测每层 r_max
r_max = {}
for L in range(3):
    r = codebooks_tan[L].norm(dim=-1)
    r_max[L] = r.max().item()
    print(f"[L{L}] r: mean={r.mean().item():.4f}, max={r_max[L]:.4f}")
    gc.collect()

# c = (artanh(√s) / r_max)²
def calc_c(target_strength, r_max):
    s = min(target_strength, 0.999)
    return (np.arctanh(np.sqrt(s)) / r_max) ** 2

STRENGTHS = [0.60, 0.85]
print(f'\n=== v4 强度对齐扫描 ===\n')
for s in STRENGTHS:
    cs = [calc_c(s, r_max[L]) for L in range(3)]
    print(f'  强度 {s:.2f}: c = [{cs[0]:.0f}  {cs[1]:.0f}  {cs[2]:.0f}]')

print(f'\n{"L":3s} {"str":>4s} {"c":>5s} {"agree%":>8s} {"util_p":>7s} {"util_e":>7s} {"Δutil":>8s} {"ent%":>6s} {"c·max":>7s}')
print('-' * 70)

z_lat = {L: get_level_latent(L, codebooks_tan, z0) for L in range(3)}
gc.collect()

for L in range(3):
    cb = codebooks_tan[L]
    K = cb.shape[0]
    for s in STRENGTHS:
        c_val = calc_c(s, r_max[L])
        idx_p = argmin_dist_mem_eff(z_lat[L], cb, mode='poincare', c=c_val)
        idx_e = argmin_dist_mem_eff(z_lat[L], cb, mode='euclidean', c=c_val)
        agree = (idx_p == idx_e).float().mean().item()
        up, ep, _ = util_entropy(idx_p, K)
        ue, _, _ = util_entropy(idx_e, K)
        cb_h = expmap0(cb, c_val)
        c_max = (c_val * (cb_h**2).sum(dim=-1)).max().item()
        print(f'L{L}  {s:.2f} {c_val:4.0f}  {agree*100:6.2f}% {up*100:6.2f}% {ue*100:6.2f}% {up-ue:+7.2f}% {ep/np.log(K)*100:5.1f}% {c_max:6.4f}')
        gc.collect()

# baseline c=1
print(f'\n--- baseline c=1 ---')
for L in range(3):
    cb = codebooks_tan[L]
    K = cb.shape[0]
    idx_p = argmin_dist_mem_eff(z_lat[L], cb, mode='poincare', c=1.0)
    idx_e = argmin_dist_mem_eff(z_lat[L], cb, mode='euclidean', c=1.0)
    up, ep, _ = util_entropy(idx_p, K)
    ue, _, _ = util_entropy(idx_e, K)
    print(f'L{L}  1.00    1  {up*100:6.2f}% {ue*100:6.2f}% {up-ue:+7.2f}% {ep/np.log(K)*100:5.1f}%')

print('\n=== 判断 ===')
print('如果强度对齐后 util/Δutil 三层一致 → layer-dependent 是伪影 (支持用户预测)')
print('如果强度对齐后 util/Δutil 仍不一致 → layer-dependent 真实存在')
