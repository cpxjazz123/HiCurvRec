# Issue #64 v6b verdict — Low-rank Learnable B_geo (U·V^T rank=16) NO-GO

**Issue**: #64 Hyperbolic Attention Bias (HAB) for HG-Rec T5
**Variant**: v6b — learnable B_geo via low-rank factorized embeddings U_l (K_l, r), V_l (K_l, r), r=16
**Date**: 2026-08-07
**Author**: MiniMax
**Status**: NO-GO (历史第二, 但不超 v4)

---

## 4 Gate verdict

### Gate 1 (实施完整性): PASS
- 修改 `common/hyperbolic_attention_bias.py`: `self.bias = ParameterList([nn.Parameter(Dbar_list[l])])` → `self.U, self.V = nn.ModuleList([nn.Embedding(K_l, 16)])`
- SVD 初始化: `U = U_svd[:, :r] * sqrt(S[:r])`, `V = V_svd[:, :r] * sqrt(S[:r])` (Eckart-Young 最优 rank-r 近似)
- forward: `Dbar_ij = einsum('bir,bjr->bij', U_l[k_i], V_l[k_j])` (无 /sqrt(r), init scale 已吸收)
- 参数总数: 86016 → 14336 (砍 83%)
- stage3_train_pure_t5.py: param group 用 `id()` 排除 U/V embedding weight (避免 DDP prefix 误判)
- precheck: 4 Gate 全 PASS, SVD 重建相对误差 <3%

### Gate 2 (训练健康度): PASS
- 训练 95 epoch (用户终止, EARLY_STOP=30 loss-based 未触发)
- loss: 4.81 → 1.78 (持续下降)
- 12s/epoch (vs v6 满秩 51s/epoch, 4.2× 加速 — 根因诊断: fancy index scatter backward 优化为 embedding backward)
- lambda_raw + U/V 梯度流通正常, λ_eff 学到 ±0.20 边界
- DDP 4-card 稳定, 无 NCCL 死锁

### Gate 3 (valid R@K): PARTIAL-GO
- v6b valid R@10 best = **0.1291** @ ep45 (超 v4 best 0.1285 +0.0006, 超 baseline 0.1267 +0.0024)
- ep60 R@10=0.1283, ep70=0.1272, ep75=0.1262, ep80=0.1267, ep95=~0.127
- valid 在 ep45 见顶后震荡下行, 但 loss 持续下降 — 过拟合信号

### Gate 4 (端到端 test R@K): NO-GO (历史第二)
| 方案 | valid R@10 | test R@10 | vs baseline |
|---|---|---|---|
| Issue #61 baseline (无 HAB) | 0.1267 | 0.1024 | — |
| v4 (frozen Dbar + 100× λ_raw lr + detach fix) | **0.1285** | **0.1038** | **+1.4% PASS** |
| v6 (满秩 86k learnable bias) | 仅 ep5 跑通, 51s/epoch 太慢未继续 | — | — |
| **v6b (低秩 14k learnable bias) ep45 (valid-best)** | 0.1291 | 0.1001 | **-2.3% FAIL** |
| **v6b ep60 (loss-best@1.94)** | 0.1283 | **0.1028** | **+0.04% PARTIAL** (历史第二) |
| **v6b ep85 (loss-best@1.81)** | ~0.127 | 0.0991 | -3.2% FAIL |
| **v6b ep95 (loss-best@1.78, final)** | ~0.127 | **0.1011** | -1.3% FAIL |

- v6b 全部 ckpt test R@10 都低于 v4 (0.1038), **不超 v4**
- v6b 唯一超 baseline 的是 ep60 ckpt (+0.04%), 但比 v4 低 -0.001
- NDCG@20 v6b ep60 = 0.0837 vs baseline 0.0821 (+1.9%, 历史第二)

---

## 关键发现

### 1. learnable bias 路径 NO-GO
**原因**: v6b (low-rank learnable U·V^T) 端到端 test R@10 (best 0.1028) < v4 (frozen Dbar + λ_raw 0.1038).
- 学到的 U·V^T bias 偏离 stage2 预计算 Dbar 几何先验, 但没有找到更优的解
- T5 的生成路径已经充分挖掘 stage2 几何, learnable bias 微调空间有限

### 2. 速度瓶颈诊断 (PyTorch 内部优化经验)
- v6 (满秩 86k) 51s/epoch → 根因不是 DDP sync / bucket / static_graph, 而是 **fancy index `bias_l[k_i, k_j]` 的 backward scatter**
- scatter 在 PyTorch 中没有 fused kernel, 每个 (i,j) pair 走 ATen → CUDA 单 op 路径, ~1μs/op × 38M ops/epoch = 38s/epoch ✓ 与测量吻合
- v6b 改用 `nn.Embedding` lookup + `einsum`: backward 是标准 Embedding backward (PyTorch 优化), 速度回 12s/epoch

### 3. SVD 初始化数学陷阱
- 第一版 SVD init 在 `sqrt(S/r)` 和 forward `/sqrt(r)` 都加了 `/sqrt(r)` 因子, 双因子抵消成 `1/(r·sqrt(r))`, 重建误差 98%
- 修复: 单一 `sqrt(S)` 因子到 init, forward 不带 `/sqrt(r)`, 重建误差 <3% (Eckart-Young 最优)

---

## Issue #64 终极方案

**v4 (commit 历史) 关闭 Issue #64**:
- frozen Dbar matrix + λ_raw init=0.5*λ_max (非零, fast path 不触发) + 100× lr param group + 修复 `lambda_eff.detach()` 切断梯度的 bug
- test R@10=**0.1038** (+1.4% vs baseline 0.1024), **历史最佳**
- ckpt 路径: `/tmp/v64_hab/HG_Rec_best.pth` (v4 final)

**v6b 探索 NO-GO**:
- learnable bias 不能突破 frozen bias 的天花板
- 速度优化经验 (fancy index → embedding) 已沉淀, 可供未来设计参考

---

## 文件变更

- `common/hyperbolic_attention_bias.py`: `HyperbolicAttentionBias.__init__` 用 SVD 初始化 U/V embeddings; `get_B_geo` 用 einsum 替换 fancy index
- `common/stage3/stage3_train_pure_t5.py`: param group 用 `id()` 排除 U/V embedding; loss-based EARLY_STOP; bias_l2_norm trace → U/V_l2_norm
- `common/stage4/stage4_eval_pure_t5.py`: 无需修改, 默认 `bias_rank=16` 已匹配

## 产物

- `/tmp/v64_hab_v6b/HG_Rec_best.pth` (v6b ep95 loss-best, test 0.1011)
- `/tmp/v64_hab_v6b/eval_test.json` (final test verdict)
- `/tmp/v64_hab_v6b/train_pure_t5.log` (95 epoch 训练日志)