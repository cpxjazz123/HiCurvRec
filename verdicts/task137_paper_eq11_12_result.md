# Task #137 Verdict — paper Eq11/12 ρ 单调假设 REFUTED

**日期**: 2026-07-29
**决定**: ⛔ **paper Eq12 单调假设 REFUTED**, 因果链 (a)→(b) 断开
**Stage**: 无需 GPU (zero-GPU 验证), 直接落 verdict

## 1. 验证对象

paper §3.2.3 Theorem 3.4 → Eq11/12:
- Eq 11: K_ℓ = K_1 · e^{(n-1)√c·ρ}  (码字数随 ρ 指数增长)
- Eq 12: K_ℓ = K_1 · γ^ℓ, γ = e^{(n-1)√c·Δρ} (等 Δρ ⇒ K 几何级数)
- paper §3.2.3 末尾: "our experiments use K_1=64, L=3, γ=2, giving codebook sizes [64,128,256]"

**关键假设 R1**: 相邻 layer codeword 半径增量 Δρ = const (正向)
**因果链 R2**: 等 Δρ → γ 稳定 → K 几何级数 → 容量分配合理

## 2. 验证数据

3 个 Stage 1 ckpt (200 epoch 训练, musical_instruments 9922 items, e_dim=32, n-1=31):

| Variant | c | L0 norm | L0 ρ | L1 norm | L1 ρ | L2 norm | L2 ρ |
|---------|---|---------|------|---------|------|---------|------|
| B control | [10,10,10] | 0.1286 | 0.2729 | 0.0825 | 0.1688 | 0.0637 | 0.1291 |
| A1 | [10,1,1] | 0.1302 | 0.2769 | 0.0863 | 0.1730 | 0.0638 | 0.1277 |
| A2 | [30,3,3] | 0.0790 | 0.1691 | 0.0642 | 0.1289 | 0.0473 | 0.0947 |

公式: ρ = (2/√c)·artanh(√c·‖e‖_E)

## 3. 关键发现

### 3.1 ρ 序列反向 (REFUTED)

| Variant | Δρ(L0→L1) | Δρ(L1→L2) | γ_pred(L0→L1) | γ_pred(L1→L2) | γ_target |
|---------|-----------|-----------|---------------|---------------|----------|
| B | **-0.104** | **-0.040** | 2.1e-5 | 0.0205 | 2.0 |
| A1 | -0.104 | -0.045 | 2.1e-5 | 0.2455 | 2.0 |
| A2 | -0.040 | -0.034 | 0.0011 | 0.1597 | 2.0 |

**所有 3 个变体 ρ 严格下降 (L0 > L1 > L2)**, paper Eq12 Δρ > 0 假设 REFUTED.

### 3.2 γ_pred 全部 < 1

按 Eq12 公式 γ = e^{(n-1)·√c·Δρ}, 因 Δρ < 0, γ_pred < 1 (代码 K 应指数衰减, 但实际 K 翻倍). 训练自然解 (Euclidean-shrinkage) 跟 paper 假设 (Hyperbolic-growth) 反向.

### 3.3 实际训练解: Euclidean-shrinkage (反 paper Eq12)

K 翻倍 (64→128→256), ρ 减半 (0.27→0.13). 这是 RQ-VAE **残差 hierarchical compression** 的自然结果:
- L0 拟合粗粒度 (大码字)
- L1/L2 拟合残差 (细粒度, 码字小)
- K 翻倍因为细粒度需要更多码字覆盖精度
- 跟 paper Eq12 假设的 "K 翻倍 = ρ 增加" 方向相反

**paper "等 Δρ ⇒ K 几何级数" 因果链断开**:
- (a) 等 Δρ 假设: REFUTED ❌
- (b) K 几何级数: 维持 ✓ (作为 codebook_size 参数, 不是 paper 公式推导结果)
- (c) 因果方向: 实际是 "K 参数预设 → ρ 训练结果 (反向)", 不是 paper "等 Δρ → K 几何级数"

## 4. 决策 (per R11.3 自主决策)

| Gate | 决策 | 原因 |
|------|------|------|
| paper Eq12 单调假设 R1 | ❌ REFUTED | Δρ < 0 三次确认 |
| paper 因果链 (a)→(b) | ❌ 断开 | 训练方向跟 paper 假设相反 |
| paper Eq11 公式 | ✅ 公式正确 | K 关于 ρ 单调关系数学正确, 但 ρ 不是预设 |
| 实验接受 paper "γ=2" 设定 | ✅ 维持 | codebook_size=[64,128,256] 是 recipe, 跟 paper 表述一致, 跟 paper 推导无关 |

## 5. 后续启示

1. **paper Theorem 3.4 公式层正确**: K_ρ = e^{(n-1)√c·ρ} 在给定 ρ 时是对的, 但 ρ 本身是训练结果不是预设.
2. **paper §3.2.3 "实验使用 γ=2" 是 recipe, 不是 paper 公式的预测**: 实验员硬编码 K 几何级数, 没验证训练是否真产生等 Δρ.
3. **未来方向**: ρ 几何不是 R@10 杠杆 (Task #229 c-scan 6+3 arm 已证). 真正杠杆候选:
   - Sinkhorn during training (Issue #10 redesign D/E/F)
   - 训练时长 (task200 dual_v5 验证)
   - 跨架构 R@10 baseline (LETTER 0.0509, FDSA 0.0594, Caser 0.0378 都没到 0.1020)
   - SID Sinkhorn 平衡 vs 训练时长 = c-scan 上限

## 6. 与其他 task 的关系

| Task | 关系 |
|------|------|
| #218 (Per-Codeword κ) | 攻 Eq11 "固定 c" 假设 → 成功 (c_k~U[0.5,20] 打破) |
| #219 (Gromov) | 攻 Eq11 "距离公式" 假设 → 成功 (Gromov product 打破) |
| #229 c-scan | 攻 Eq11 "c 是单个超参" 假设 → FAIL (c-scan 9 arm collision 0.0876 是上限) |
| **#137 (本任务)** | **攻 Eq12 "等 Δρ" 假设 → REFUTED, ρ 训练结果是反向 Euclidean-shrinkage** |

**12 方向合并 NO-GO**: paper §3.2.3 "codebook 几何 = R@10 杠杆" 的整套理论 prescription 都不实现. 7 内部 + 3 NO-HOPE + 1 钉半径 + **1 公式假设 (本任务) 11→12 全 NO-GO**.

## 7. 产物

- descriptions/task137_paper_eq11_12_rho_monotonicity.md
- verdicts/task137_paper_eq11_12_result.md (本文件)
- 复用 ckpt: products/task239/gate2_*/best_loss_model.pth (3 个)

## 8. 验证脚本 (zero-GPU)

```python
import torch, math, os
def poincare_radius(e_norm, c):
    sqrt_c = math.sqrt(c)
    x = sqrt_c * e_norm
    if x >= 1.0: return float('inf')
    return (2.0 / sqrt_c) * math.atanh(x)

for d, c_list in [('gate2_c10_10_10', [10,10,10]), 
                  ('gate2_c10_1_1', [10,1,1]),
                  ('gate2_c30_3_3', [30,3,3])]:
    run = sorted([x for x in os.listdir(f'products/task239/{d}') if os.path.isdir(f'products/task239/{d}/{x}')])[0]
    sd = torch.load(f'products/task239/{d}/{run}/best_loss_model.pth', map_location='cpu', weights_only=False)['state_dict']
    for l in range(3):
        e = sd[f'hrq.vq_layers.{l}.embeddings.weight']
        print(f'L{l} c={c_list[l]} norm={e.norm(dim=-1).mean():.4f} rho={poincare_radius(e.norm(dim=-1).mean().item(), c_list[l]):.4f}')
```

## 9. Status

- ✅ paper Eq12 Δρ>0 假设 REFUTED (3/3 ckpt 验证)
- ❌ paper "等 Δρ → K 几何级数" 因果链断开
- ⏭️ 下一个主动推进候选: 跨架构 R@10 重新基线 (S3Rec paper-aligned fix 后 Stage 3+4) / Issue #10 redesign 4-arm (等用户)

result: Task #137 — paper Eq11/12 ρ 单调假设 REFUTED
