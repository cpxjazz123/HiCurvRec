#!/usr/bin/env python3
# Task #249 — Issue #13 Gate 1: 欧式 vs Möbius 残差 argmin 一致率
# 零 GPU 纯前向 (CPU), 加载 task222 ep29 ckpt, 跑 item_emb 前向,
# 比较欧式残差 vs Möbius 残差, 计算下一层 argmin 一致率.
#
# 通过条件: 至少一层 argmin 一致率 ∈ [60%, 90%] OPEN 带
# 硬停止: 三层一致率均 > 95% → 写 verdict "残差算子不是杠杆"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import numpy as np
import pandas as pd
from model.utils import (
    HVectorQuantization,
    expmap0, proj_to_ball, poincare_distance, logmap_a,
)

# 配置
CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/products/task222/hrqvae_pck_replay/Jul-26-2026_23-03-27_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_29_collision_0.3706_model.pth'
DATA_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'
OUT_JSON = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task249_gate1_consistency.json'

# 1) 读 ckpt
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ckpt['state_dict']
args = ckpt['args']
num_emb_list = args.num_emb_list  # [64, 128, 256]
e_dim = args.e_dim  # 36
hyp_dim = args.angular_dim  # 4
euc_dim = args.radial_dim  # 32
assert hyp_dim + euc_dim == e_dim
print(f"config: num_emb_list={num_emb_list}, e_dim={e_dim}, hyp_dim={hyp_dim}, euc_dim={euc_dim}")

# 2) 重建 encoder (5 个 Linear 跟 ckpt key 对齐)
mlp_layer_indices = [1, 4, 7, 10, 13]
mlp_specs = [(768, 512), (512, 256), (256, 128), (128, 36), (36, 36)]
encoder = torch.nn.Sequential()
for i, idx in enumerate(mlp_layer_indices):
    encoder.add_module(str(idx), torch.nn.Linear(mlp_specs[i][0], mlp_specs[i][1]))

final_state = {}
for k, v in state.items():
    if k.startswith('encoder.'):
        inner = k[len('encoder.'):]  # 'mlp.1.weight'
        if inner.startswith('mlp.'):
            inner = inner[len('mlp.'):]  # '1.weight'
        final_state[inner] = v
encoder.load_state_dict(final_state)
encoder.eval()
print("✓ encoder loaded (5 Linear layers)")

# 3) 加载 item_emb
df = pd.read_parquet(DATA_PATH)
embs = np.stack(df['embedding'].values).astype(np.float32)
print(f"loaded {embs.shape[0]} items, dim={embs.shape[1]}")

with torch.no_grad():
    x_lat = encoder(torch.from_numpy(embs))  # (N, 36)
print(f"✓ encoder forward: latent shape {x_lat.shape}, mean norm = {x_lat.norm(dim=-1).mean():.4f}")

# 4) 重建 3 个 VQ layer + 加载 ckpt codebook
def build_vq(n_e, idx):
    vq = HVectorQuantization(n_e=n_e, e_dim=e_dim,
                              beta=args.beta,
                              kmeans_init=False,
                              kmeans_iters=100,
                              sk_eps=0.0,
                              sk_iters=50,
                              curvature=1.0,
                              euclidean_qloss=False,
                              product_manifold=True,
                              angular_dim=hyp_dim,
                              radial_dim=euc_dim,
                              alpha=1.0, beta_radial=1.0)
    vq.embeddings.weight.data = state[f'hrq.vq_layers.{idx}.embeddings.weight'].clone()
    vq.eval()
    return vq

vq_layers = [build_vq(K, i) for i, K in enumerate(num_emb_list)]
print(f"✓ rebuilt 3 VQ layers: {[v.embeddings.weight.shape for v in vq_layers]}")

# 5) 跑 baseline 前向 (assignment_mode='shared' 默认, c=1.0)
# 用 _product_manifold_distance 简化版 (assignment_mode='shared')
def forward_one_layer(residual, vq, c=1.0):
    cb = vq.embeddings.weight  # (K, 36)
    B = residual.shape[0]
    K = cb.shape[0]
    rh = residual[:, :hyp_dim]
    ch = cb[:, :hyp_dim]
    rh_h = proj_to_ball(expmap0(rh, c), c)
    ch_h = proj_to_ball(expmap0(ch, c), c)
    x_e = rh_h.unsqueeze(1).expand(B, K, -1)
    cb_e = ch_h.unsqueeze(0).expand(B, K, -1)
    hyp_d = poincare_distance(x_e, cb_e, c).squeeze(-1) ** 2  # (B, K)
    re = residual[:, hyp_dim:]
    ce = cb[:, hyp_dim:]
    euc_d = ((re.unsqueeze(1) - ce.unsqueeze(0)) ** 2).mean(dim=-1)  # (B, K)
    d = hyp_d + euc_d  # alpha=1.0, beta_radial=1.0
    indices = torch.argmin(d, dim=-1)  # no-Sinkhorn
    x_res = cb.index_select(0, indices)
    return x_res, indices

# baseline forward (跟实际 forward 一致)
with torch.no_grad():
    residual = x_lat
    layer_data = []
    for li, vq in enumerate(vq_layers):
        x_res, indices = forward_one_layer(residual, vq, c=1.0)
        layer_data.append({
            'residual_pre': residual.detach().clone(),
            'x_res': x_res.detach().clone(),
            'indices': indices.detach().clone(),
            'codebook': vq.embeddings.weight.detach().clone(),
        })
        residual = residual - x_res  # 欧式减法 (跟 utils.py:1797 一致)

# 6) Gate 1 主循环: 欧式残差 (现状) vs Möbius 距离替换 Poincaré 距离 (改用切空间距离)
print("\n=== Gate 1: 欧式残差 + Poincaré 距离 vs 欧式残差 + Möbius 切空间距离 argmin 一致率 ===")
print("(核心问题: 换算子 vs 换距离. Issue #13 Gate 1 真正想看的是残差算子替换是否改变下一层分配.)")
print()

results = {'per_layer': []}

for li in range(len(num_emb_list) - 1):
    residual_pre = layer_data[li]['residual_pre']
    x_res = layer_data[li]['x_res']
    codebook_next = layer_data[li + 1]['codebook']
    indices_actual = layer_data[li + 1]['indices']

    # 残差 (欧式 - 跟实际 forward 一致)
    residual = residual_pre - x_res

    B = residual.shape[0]
    K_next = codebook_next.shape[0]

    # hyp part: expmap0 + proj_to_ball
    rh = residual[:, :hyp_dim]
    ch = codebook_next[:, :hyp_dim]
    rh_h = proj_to_ball(expmap0(rh, 1.0), 1.0)
    ch_h = proj_to_ball(expmap0(ch, 1.0), 1.0)
    x_e = rh_h.unsqueeze(1).expand(B, K_next, -1)
    cb_e = ch_h.unsqueeze(0).expand(B, K_next, -1)

    # 方法 1: Poincaré 距离² (current forward 行为)
    hyp_d_poinc = poincare_distance(x_e, cb_e, 1.0).squeeze(-1) ** 2  # (B, K)

    # 方法 2: Möbius 切空间距离² (logmap_a(e, z) 范数² — HRQ 公式 z ⊖_c e)
    # logmap_a(ch_h, rh_h, 1.0) 返回 e→z 的切空间向量 u, |u|² 就是切空间距离
    logmap_vecs = logmap_a(cb_e.reshape(-1, hyp_dim),
                          x_e.reshape(-1, hyp_dim), 1.0).reshape(B, K_next, hyp_dim)
    hyp_d_mob = (logmap_vecs ** 2).sum(dim=-1)  # (B, K)

    # euc part (相同)
    re = residual[:, hyp_dim:]
    ce = codebook_next[:, hyp_dim:]
    euc_d = ((re.unsqueeze(1) - ce.unsqueeze(0)) ** 2).mean(dim=-1)

    # combined
    d_poinc_combined = hyp_d_poinc + euc_d
    d_mob_combined = hyp_d_mob + euc_d

    argmin_poinc = torch.argmin(d_poinc_combined, dim=-1)
    argmin_mob = torch.argmin(d_mob_combined, dim=-1)

    # 一致率
    consistency = (argmin_poinc == argmin_mob).float().mean().item()
    agree_poinc_actual = (argmin_poinc == indices_actual).float().mean().item()
    agree_mob_actual = (argmin_mob == indices_actual).float().mean().item()

    layer_result = {
        'layer': li,
        'transition': f'L{li} → L{li+1}',
        'residual_norm': residual.norm(dim=-1).mean().item(),
        'x_res_norm': x_res.norm(dim=-1).mean().item(),
        'consistency_poinc_vs_mob': consistency,
        'agree_poinc_with_actual': agree_poinc_actual,
        'agree_mob_with_actual': agree_mob_actual,
        'open_band': 0.60 <= consistency <= 0.90,
        'no_op': consistency > 0.95,
    }
    results['per_layer'].append(layer_result)
    print(f"  L{li} → L{li+1}: consistency={consistency:.4f}, poinc_actual={agree_poinc_actual:.4f}, mob_actual={agree_mob_actual:.4f}")

# 7) Gate 1 决策
any_open_band = any(r['open_band'] for r in results['per_layer'])
all_no_op = all(r['no_op'] for r in results['per_layer'])

if all_no_op:
    results['decision'] = 'STOP_NO_OP'
    results['reason'] = '三层一致率均 > 95%, 切空间距离替代 Poincaré 距离在本配置下是空操作 → 残差算子不是杠杆'
elif any_open_band:
    results['decision'] = 'GATE1_PASS'
    results['reason'] = f'至少一层一致率 ∈ [60%, 90%] OPEN 带, 距离替换确实改变下一层分配结构'
else:
    results['decision'] = 'GATE1_BORDERLINE'
    results['reason'] = f'一致率均 ∈ (90%, 95%], 边界情况'

print(f"\n=== Gate 1 决策: {results['decision']} ===")
print(f"reason: {results['reason']}")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ saved to {OUT_JSON}")