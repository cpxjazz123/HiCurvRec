#!/usr/bin/env python3
# Task #254 — Issue #13 Gate 3: Logmap distance (query→codeword) vs Euclidean distance argmin
# 零 GPU 纯前向 (CPU), 加载 task222 ep29 ckpt, 跑 item_emb 前向,
# 比较欧式 query-codeword 距离 vs Logmap query-codeword 距离 (切空间距离 of hyperbolic part),
# 计算 query→codeword argmin 一致率.
#
# 通过条件: 至少一层 argmin 一致率 ∈ [60%, 90%] OPEN 带
# 硬停止: 三层一致率均 > 95% → 写 verdict "Logmap 距离不是 argmin 杠杆"
#
# 跟 Gate 1 区别: Gate 1 比较 hyp part 不同距离 (Poincaré vs Logmap), Hall 3 直接比较
# 欧式 vs Mobius-(hyp-L2 + euc-L2) 总距离. 也就是: Gate 3 替换"是不是 Logmap 距离整体替换欧式距离".
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
OUT_JSON = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task254_gate3_logmap_argmin.json'

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

# 2) 重建 encoder
mlp_layer_indices = [1, 4, 7, 10, 13]
mlp_specs = [(768, 512), (512, 256), (256, 128), (128, 36), (36, 36)]
encoder = torch.nn.Sequential()
for i, idx in enumerate(mlp_layer_indices):
    encoder.add_module(str(idx), torch.nn.Linear(mlp_specs[i][0], mlp_specs[i][1]))

final_state = {}
for k, v in state.items():
    if k.startswith('encoder.'):
        inner = k[len('encoder.'):]
        if inner.startswith('mlp.'):
            inner = inner[len('mlp.'):]
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

# 4) 重建 3 个 VQ layer
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

# 5) 跑 baseline forward (欧式 query-codeword 距离, 跟实际 forward 一致)
# 注意: 量化时 query 还没 proj_to_ball, codebook emb 已经 expmap0 到 ball;
# 实际 utils.py 用 expmap0(query) - expmap0(codebook) 不完全 — 但 product_manifold 实现的
# poincare_distance 已 expected 两个输入都在 ball 内. 这里 follow Gate 1 模式.
def forward_euc_distance(residual, vq, c=1.0):
    """欧式 query-codeword 总距离 (query 和 codebook 都先 expmap0 到 ball)"""
    cb = vq.embeddings.weight  # (K, 36)
    B = residual.shape[0]
    K = cb.shape[0]
    rh = residual[:, :hyp_dim]
    ch = cb[:, :hyp_dim]
    rh_h = proj_to_ball(expmap0(rh, c), c)
    ch_h = proj_to_ball(expmap0(ch, c), c)
    # 欧式 hyp part 距离
    hyp_d = ((rh_h.unsqueeze(1) - ch_h.unsqueeze(0)) ** 2).mean(dim=-1)  # (B, K)
    re = residual[:, hyp_dim:]
    ce = cb[:, hyp_dim:]
    euc_d = ((re.unsqueeze(1) - ce.unsqueeze(0)) ** 2).mean(dim=-1)  # (B, K)
    d = hyp_d + euc_d
    indices = torch.argmin(d, dim=-1)
    x_res = cb.index_select(0, indices)
    return x_res, indices, d

def forward_logmap_distance(residual, vq, c=1.0):
    """Logmap 距离: hyp part 用 logmap distance (切空间距离²), euc 保留欧式"""
    cb = vq.embeddings.weight  # (K, 36)
    B = residual.shape[0]
    K = cb.shape[0]
    rh = residual[:, :hyp_dim]
    ch = cb[:, :hyp_dim]
    rh_h = proj_to_ball(expmap0(rh, c), c)
    ch_h = proj_to_ball(expmap0(ch, c), c)
    # logmap distance: logmap at query (rh_h) to codeword (ch_h)
    logmap_vecs = logmap_a(ch_h.unsqueeze(0).expand(B, K, -1).reshape(-1, hyp_dim),
                          rh_h.unsqueeze(1).expand(B, K, -1).reshape(-1, hyp_dim), c).reshape(B, K, hyp_dim)
    hyp_d = (logmap_vecs ** 2).sum(dim=-1)  # (B, K)
    re = residual[:, hyp_dim:]
    ce = cb[:, hyp_dim:]
    euc_d = ((re.unsqueeze(1) - ce.unsqueeze(0)) ** 2).mean(dim=-1)
    d = hyp_d + euc_d
    indices = torch.argmin(d, dim=-1)
    x_res = cb.index_select(0, indices)
    return x_res, indices, d

# baseline forward (欧式 - 跟实际 forward 一致)
with torch.no_grad():
    residual = x_lat
    layer_data = []
    for li, vq in enumerate(vq_layers):
        x_res, indices, d = forward_euc_distance(residual, vq, c=1.0)
        layer_data.append({
            'residual_pre': residual.detach().clone(),
            'x_res': x_res.detach().clone(),
            'indices': indices.detach().clone(),
            'codebook': vq.embeddings.weight.detach().clone(),
            'd_euc': d.detach().clone(),
        })
        residual = residual - x_res  # 欧式减法 (跟 utils.py 实际一致)

# 6) Gate 3 主循环: 欧式 query-codeword 距离 vs Logmap 距离 argmin 一致率
print("\n=== Gate 3: 欧式 query-codeword vs Logmap query-codeword argmin 一致率 ===")
print("(核心问题: hyp part 欧式 hyp-d 替换为 logmap 距离, 是否改变 argmin 决策)")
print()

results = {'per_layer': []}

for li in range(len(num_emb_list)):
    residual_pre = layer_data[li]['residual_pre']
    codebook = layer_data[li]['codebook']
    indices_actual = layer_data[li]['indices']
    vq = vq_layers[li]

    B = residual_pre.shape[0]
    K = codebook.shape[0]

    # 距离 1 (欧式) - 复用 layer_data
    d_euc = layer_data[li]['d_euc']
    argmin_euc = torch.argmin(d_euc, dim=-1)

    # 距离 2 (Logmap)
    _, _, d_logmap = forward_logmap_distance(residual_pre, vq, c=1.0)
    argmin_logmap = torch.argmin(d_logmap, dim=-1)

    # 一致率
    consistency = (argmin_euc == argmin_logmap).float().mean().item()
    agree_euc_with_actual = (argmin_euc == indices_actual).float().mean().item()
    agree_logmap_with_actual = (argmin_logmap == indices_actual).float().mean().item()

    layer_result = {
        'layer': li,
        'transition': f'L{li} argmin',
        'residual_norm': residual_pre.norm(dim=-1).mean().item(),
        'consistency_euc_vs_logmap': consistency,
        'agree_euc_with_actual': agree_euc_with_actual,
        'agree_logmap_with_actual': agree_logmap_with_actual,
        'open_band': 0.60 <= consistency <= 0.90,
        'no_op': consistency > 0.95,
    }
    results['per_layer'].append(layer_result)
    print(f"  L{li}: consistency={consistency:.4f}, euc_actual={agree_euc_with_actual:.4f}, logmap_actual={agree_logmap_with_actual:.4f}")

# 7) Gate 3 决策
any_open_band = any(r['open_band'] for r in results['per_layer'])
all_no_op = all(r['no_op'] for r in results['per_layer'])

if all_no_op:
    results['decision'] = 'STOP_NO_OP'
    results['reason'] = '三层 argmin 一致率均 > 95%, Logmap 距离替代欧式距离在该配置下是空操作 → 距离公式不是 argmin 杠杆'
elif any_open_band:
    results['decision'] = 'GATE3_PASS'
    results['reason'] = f'至少一层 argmin 一致率 ∈ [60%, 90%] OPEN 带, Logmap 距离确实改变 argmin 决策, 可推 Phase 2 Stage 3 50 epoch'
else:
    results['decision'] = 'GATE3_BORDERLINE'
    results['reason'] = f'一致率均 ∈ (90%, 95%], 边界, 需更多 evidence'

print(f"\n=== Gate 3 决策: {results['decision']} ===")
print(f"reason: {results['reason']}")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ saved to {OUT_JSON}")
