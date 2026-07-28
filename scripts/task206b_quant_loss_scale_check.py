"""Task #206 Part B — 5-min quant_loss 量级对比."""
import sys, os
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from model.utils import HVectorQuantization, poincare_distance

device = 'cuda:0'
torch.manual_seed(42)

# Stage 1 输入 32 dim; 真实 embedding 来自 item_emb.parquet (768 dim)
# 但 RQ-VAE 自己有 encoder 把 768→32; HG-Rec 训练时 encoder + quant 一起学.
# 这里直接用 32 dim 随机向量 (因 quant_loss 量级比较跟 dim 无关)
e_dim = 32
K = 64
# 真实数据尺度: 用 stage1 latent mean norm ~0.5
data_t = (torch.randn(256, e_dim, device=device) * 0.5)
data_t.requires_grad_(False)

# 双曲 (c=1.0, 默认)
hq_hyp = HVectorQuantization(n_e=K, e_dim=e_dim, beta=0.5, curvature=1.0, sk_eps=0.0,
                             kmeans_init=True, kmeans_iters=10).to(device)
hq_hyp.train()
out = hq_hyp(data_t, use_sk=False)
print(f'forward returns {len(out)} values')
x_q_hyp = out[0]
commit_hyp = out[2]
code_hyp = out[3]
print(f'臂 A 双曲: commit_loss = {commit_hyp.item():.6f}, code_loss = {code_hyp.item():.6f}')

# 欧式 (euclidean_qloss) — 现行实现
hq_euc = HVectorQuantization(n_e=K, e_dim=e_dim, beta=0.5, curvature=1.0, sk_eps=0.0,
                             euclidean_qloss=True, kmeans_init=True, kmeans_iters=10).to(device)
hq_euc.train()
out2 = hq_euc(data_t, use_sk=False)
x_q_euc = out2[0]
commit_euc = out2[2]
code_euc = out2[3]
print(f'臂 B 欧式: commit_loss = {commit_euc.item():.6f}, code_loss = {code_euc.item():.6f}')

print('\n=== 量级比 (双曲 / 欧式) ===')
print(f'  commit: 双曲 = 欧式 × {commit_hyp.item()/max(commit_euc.item(),1e-9):.1f}')
print(f'  code:   双曲 = 欧式 × {code_hyp.item()/max(code_euc.item(),1e-9):.1f}')
print(f'  期望 ≈ 32 (因为 F.mse_loss mean over 32 dim vs sum over 32 dim)')

# 修复: 欧式用 sum(-1) 跟双曲同 dim-reduction
commit_sum = ((x_q_euc.detach() - data_t) ** 2).sum(-1).mean()
code_sum   = ((x_q_euc - data_t.detach()) ** 2).sum(-1).mean()
print(f'\n=== 修复 (sum over dim) ===')
print(f'  修复 commit = {commit_sum.item():.6f}, code = {code_sum.item():.6f}')
print(f'  修复/原始 = ×{commit_sum.item()/max(commit_euc.item(),1e-9):.1f}')
print(f'  修复/双曲 = ×{commit_sum.item()/max(commit_hyp.item(),1e-9):.3f}  (期望 ~1 = 等量级)')

commit_4x = 4.0 * commit_sum
print(f'\n=== 再 ×4 (Berman factor) ===')
print(f'  修复×4 = {commit_4x.item():.6f}, 双曲/修复×4 = {commit_hyp.item()/max(commit_4x.item(),1e-9):.3f}')
