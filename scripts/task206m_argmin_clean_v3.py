"""Task #206m-v3: 切空间参数化 (expmap0) 扫描 — 数据全有效.

用户 2026-07-26 修正方案:
    v2 用球内坐标缩放 (cb_in_ball 被 proj_to_ball clamp 出球 → 无效数据).
    v3 改用切空间参数化: 把 cb 当切空间向量, expmap0(cb, c) 映射进球.
    这样 c‖x‖² = tanh²(√c·r) 永远 < 1, 不可能 clamp.

L0 实测 (r_mean=0.095, r_max=0.166) 代入:
    c=1    → mean c‖x‖²=0.010, max=0.028  (✅)
    c=42   → mean c‖x‖²=0.300, max=0.63   (✅ 不出球)
    c=118  → mean c‖x‖²=0.600, max=0.90   (✅)
    c=282  → mean c‖x‖²=0.850, max=0.985  (✅)

预期:
    c=1   → 一致率 ~99%, util 不变
    c=42  → 一致率 ~90%, util 略降
    c=118 → 一致率 ~75%, util 明显降
    c=282 → 一致率 ~60%, util 显著降

§五 隐忧 (用户预测: c↑ → util↓) 这次在三层都应该成立.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from model.utils import poincare_distance, expmap0, proj_to_ball, MLP

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

# 加载 3 个码本 (注意: ckpt 里存的是切空间 embeddings.weight, 不是球内坐标!)
codebooks_tan = {  # tangent space codebook (per layer)
    0: sd['hrq.vq_layers.0.embeddings.weight'].to(device).float(),
    1: sd['hrq.vq_layers.1.embeddings.weight'].to(device).float(),
    2: sd['hrq.vq_layers.2.embeddings.weight'].to(device).float(),
}

# 真实数据 → encoder → z0 (切空间)
df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
data768 = np.stack([np.asarray(e) for e in df['embedding'].values]).astype(np.float32)
data768_t = torch.tensor(data768).to(device)

encoder.eval()
with torch.no_grad():
    z0 = encoder(data768_t).float()

# argmin 函数 (切空间参数化版本)
def argmin_dist(latent_tan, codebook_tan, mode='poincare', c=1.0):
    """切空间参数化: latent_tan 和 codebook_tan 都是切空间向量.
       expmap0 映射进球 — 自动满足 ‖x‖_E < 1/√c, c‖x‖² < 1."""
    B = latent_tan.shape[0]
    K = codebook_tan.shape[0]

    # 切空间 → 球内 (tanh 压缩, 永不出球)
    latent_h = expmap0(latent_tan, c)
    codebook_h = expmap0(codebook_tan, c)

    if mode == 'poincare':
        x = latent_h.unsqueeze(1).expand(B, K, -1)
        cb = codebook_h.unsqueeze(0).expand(B, K, -1)
        diff = poincare_distance(x.reshape(-1, latent_h.shape[-1]),
                                  cb.reshape(-1, latent_h.shape[-1]), c=c)
        diff = diff.reshape(B, K)
    else:
        x = latent_h.unsqueeze(1).expand(B, K, -1)
        cb = codebook_h.unsqueeze(0).expand(B, K, -1)
        diff = (x - cb).reshape(-1, latent_h.shape[-1]).norm(dim=-1).reshape(B, K)
    return diff.argmin(dim=-1)

# 利用率 + 分配熵
def util_entropy(indices, K):
    usage = torch.bincount(indices, minlength=K).float()
    util = (usage > 0).float().mean().item()
    p = usage / usage.sum().clamp_min(1e-12)
    entropy = -(p * (p + 1e-12).log()).sum().item()
    entropy_norm = entropy / np.log(K)
    return util, entropy, entropy_norm

# c 列表 (用户 §一 给的)
C_LIST = [1, 42, 118, 282]

print('=== Task #206m-v3 切空间参数化扫描 ===\n')
print('L0 切空间实测 r: mean=0.095, max=0.166')
print('切空间 → 球内 (expmap0) → c‖x‖² = tanh²(√c·r), 永远 < 1, 不出球\n')
print('预期 (用户预测):')
print('   c=1   → 一致率 ~99%, Δutil=0')
print('   c=42  → 一致率 ~90%, Δutil 略负')
print('   c=118 → 一致率 ~75%, Δutil 明显负')
print('   c=282 → 一致率 ~60%, Δutil 显著负\n')

# 模拟 RQ 残差链 (切空间版本, 也用 expmap0)
def get_level_latent(level, codebooks_tan, z0_tan):
    z = z0_tan
    for i in range(level):
        # 切空间残差链: 量化 z → expmap0 进球 → argmin → 减 emb_rec
        # 但这里只是为了拿到 layer i 的输入 latent, 简化为: argmin (切空间 → 球内), 然后减切空间 cb.
        z_h = expmap0(z, 1.0)  # c=1 简单
        cb_h = expmap0(codebooks_tan[i], 1.0)
        idx = (z_h.unsqueeze(1) - cb_h.unsqueeze(0)).norm(dim=-1).argmin(dim=-1)
        z = z - codebooks_tan[i][idx]  # 残差链在切空间
    return z

# 表头
print(f'{"level":6s} {"c":>4s} {"agree%":>8s} {"util_p":>8s} {"util_e":>8s} '
      f'{"Δutil":>8s} {"ent_p":>7s} {"ent_e":>7s} {"Δent":>7s} '
      f'{"c·mean":>8s} {"c·max":>8s}')
print('-' * 110)

for level in [0, 1, 2]:
    lat_tan = get_level_latent(level, codebooks_tan, z0)
    cb_tan = codebooks_tan[level]
    K = cb_tan.shape[0]

    # 切空间参数 r = ‖cb_tan‖
    cb_r_mean = cb_tan.norm(dim=-1).mean().item()
    cb_r_max = cb_tan.norm(dim=-1).max().item()

    for c in C_LIST:
        idx_p = argmin_dist(lat_tan, cb_tan, mode='poincare', c=c)
        idx_e = argmin_dist(lat_tan, cb_tan, mode='euclidean', c=c)
        agree = (idx_p == idx_e).float().mean().item()

        util_p, ent_p, ent_p_norm = util_entropy(idx_p, K)
        util_e, ent_e, ent_e_norm = util_entropy(idx_e, K)

        # 计算实际 c·mean(‖x‖²) 和 c·max(‖x‖²) (在球内, expmap0 后)
        cb_h = expmap0(cb_tan, c)
        cb_norm_sqr = (cb_h ** 2).sum(dim=-1)
        c_norm_sqr_mean = (c * cb_norm_sqr).mean().item()
        c_norm_sqr_max = (c * cb_norm_sqr).max().item()

        # 验证 c·max < 1 (永不出球)
        safe = '✅' if c_norm_sqr_max < 0.99 else '❌'

        print(f'  {level:4d} {c:4d} {agree*100:7.2f}% {util_p*100:7.2f}% {util_e*100:7.2f}% '
              f'{(util_p-util_e)*100:+7.2f}% {ent_p_norm*100:6.1f}% {ent_e_norm*100:6.1f}% '
              f'{(ent_p_norm-ent_e_norm)*100:+6.1f}% {c_norm_sqr_mean:7.4f} {c_norm_sqr_max:7.4f} {safe}')

print('\n=== §五 隐忧验证 (v3 数据全有效) ===')
print('若 c↑ → util_p↓ (Δutil 持续 -5% 以上): 公式预测成立')
print('若 c↑ → util_p 不变或上升: 半径惩罚被别的机制补偿')
print('若 ent_p 也下降: 分配偏向少数码字 (分布更不均匀)')
print('对照 v2 (球内缩放): L0 c=42 因出球被 proj 截断 → util 反向 +12.5% (无效数据)')