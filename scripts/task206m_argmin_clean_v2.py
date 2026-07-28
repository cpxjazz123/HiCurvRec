"""Task #206m: 提 c 扫描 argmin 一致率 (不出球) + 利用率/分配熵诊断.

用户 2026-07-26 新方案 (§一):
    - 不放大坐标 (老方法会过球)
    - 保持切空间参数不动, 只提高 c.
    - 因为 c‖x‖² = tanh²(√c·r), 球内范数自动满足 < 1/√c, 永远在球内.
    - 验证 §五隐忧: 双曲 argmin 是否真的让利用率↓ (公式预测) 跟论文
      "更均匀利用" 主张 (HG-Rec paper) 矛盾.

策略:
  1. 加载双曲 ep19 ckpt (跟 #206-L 一样)
  2. 真实数据 → encoder → z0
  3. 模拟 RQ 残差链 → 各 level 的输入 latent
  4. 对每个 c ∈ {1, 12, 42, 118}:
    a. Poincaré argmin (不同 c, 同一 cb) → idx_p
    b. Euclidean argmin (无 c 影响) → idx_e
    c. 一致率 = (idx_p == idx_e).mean()
    d. 利用率_p = unique(idx_p) / K
    e. 利用率_e = unique(idx_e) / K (对照)
    f. 分配熵_p = H(idx_p)
    g. 分配熵_e = H(idx_e) (对照)
    h. c·mean(‖e‖²) 范围 (在原 cb 上, 提示实际 c‖e‖² 跨距)
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

# 利用率 + 分配熵
def util_entropy(indices, K):
    usage = torch.bincount(indices, minlength=K).float()
    util = (usage > 0).float().mean().item()
    p = usage / usage.sum().clamp_min(1e-12)
    entropy = -(p * (p + 1e-12).log()).sum().item()
    # 最大可能熵 (均匀分布) = log(K), 归一化看利用率强度
    entropy_norm = entropy / np.log(K)
    return util, entropy, entropy_norm

# c 列表 (用户 §一 给的: 1/12/42/118 对应 c‖x‖² ∈ {0.01, 0.10, 0.30, 0.60})
C_LIST = [1, 12, 42, 118]

print('=== Task #206m 提 c 扫描 argmin 一致率 + 利用率/熵 ===\n')
print('用户 §一 表格 (L0 实测 r: mean=0.095, min=0.03, max=0.166):')
print('   c=1    → c‖x‖² mean≈0.010, range 0.001-0.028, w-range 1.00-1.03')
print('   c=12   → c‖x‖² mean≈0.100, range 0.010-0.27,  w-range 1.01-1.37')
print('   c=42   → c‖x‖² mean≈0.300, range 0.037-0.63,  w-range 1.04-2.68')
print('   c=118  → c‖x‖² mean≈0.600, range 0.10-0.90,   w-range 1.11-10.3')
print('§五验证: c↑ → util_p↓? (公式预测) ↔ HG-Rec paper "更均匀" 主张 (矛盾?)\n')

# 模拟 RQ 残差链, 算各 level 的输入 latent
def get_level_latent(level, codebooks, z0):
    z = z0
    for i in range(level):
        idx = argmin_dist(z, codebooks[i], mode='poincare')
        z = z - codebooks[i][idx]
    return z

# 表头: c | 一致率 | util_p | util_e | util_diff | ent_p | ent_e | ent_diff | c·mean‖e‖² | c·max‖e‖²
print(f'{"level":6s} {"c":>4s} {"agree%":>8s} {"util_p":>8s} {"util_e":>8s} '
      f'{"Δutil":>8s} {"ent_p":>7s} {"ent_e":>7s} {"Δent":>7s} '
      f'{"c·mean":>8s} {"c·max":>8s} {"w_max":>7s}')
print('-' * 110)

for level in [0, 1, 2]:
    lat = get_level_latent(level, codebooks, z0)
    cb = codebooks[level]
    K = cb.shape[0]

    # 计算 c·mean(‖e‖²) 和 c·max(‖e‖²) 范围 (在原 cb 上, 不动坐标)
    cb_norm_sqr = (cb ** 2).sum(dim=-1)  # (K,)
    cb_norm_sqr_mean = cb_norm_sqr.mean().item()
    cb_norm_sqr_max = cb_norm_sqr.max().item()

    for c in C_LIST:
        idx_p = argmin_dist(lat, cb, mode='poincare', c=c)
        idx_e = argmin_dist(lat, cb, mode='euclidean', c=c)
        agree = (idx_p == idx_e).float().mean().item()

        util_p, ent_p, ent_p_norm = util_entropy(idx_p, K)
        util_e, ent_e, ent_e_norm = util_entropy(idx_e, K)

        # 用户 §一 公式: w_max = 1/(1 - c·max(‖e‖²))
        # 这里 ‖e‖² 是球内坐标 (expmap0 后), 跟 ckpt 里 embeddings.weight 直接对应
        # 因为 ckpt 里存的本来就是切空间, 实际量化用 expmap0 后.
        # 但这里我们直接拿 ckpt 的 embeddings 当球内坐标用, 因为 ckpt 训练时
        # 已经走 expmap0(proj_to_ball(...)). 所以 ‖e‖_E ≈ 球内范数.
        c_norm_sqr_max = c * cb_norm_sqr_max
        c_norm_sqr_mean = c * cb_norm_sqr_mean
        # w_max = 1 / (1 - c·max(‖e‖²)) (近似; 实际公式更复杂)
        if c_norm_sqr_max < 0.99:
            w_max = 1.0 / (1.0 - c_norm_sqr_max)
        else:
            w_max = float('inf')

        print(f'  {level:4d} {c:4d} {agree*100:7.2f}% {util_p*100:7.2f}% {util_e*100:7.2f}% '
              f'{(util_p-util_e)*100:+7.2f}% {ent_p_norm*100:6.1f}% {ent_e_norm*100:6.1f}% '
              f'{(ent_p_norm-ent_e_norm)*100:+6.1f}% {c_norm_sqr_mean:7.4f} {c_norm_sqr_max:7.4f} '
              f'{w_max:7.2f}')

print('\n=== §五 隐忧验证判据 ===')
print('若 c↑ → util_p↓ (Δutil 持续 -5% 以上): 公式预测成立, 论文"更均匀"主张与数学矛盾')
print('若 c↑ → util_p 不变或上升: 半径惩罚被别的机制补偿, 需要解释')
print('若 ent_p 也下降: 分配偏向少数码字 (分布更不均匀)')