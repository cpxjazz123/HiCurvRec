# Poincaré Attention Pooling Precheck Verdict

**日期**: 2026-08-07
**方案**: 在 Stage3 T5 encoder 输出后注入 Poincaré attention pooling (DECOR PromptFormer 的双曲版)
**状态**: Precheck PASS / 实施待决策

---

## 实施物

- `common/poincare_attention_pool.py`: PoincareAttentionPool 模块
  - `poincare_distance(x, y, c=1.0)`: 完整双曲距离公式 (Ganea 2018)
  - `expmap0(v, c=1.0)`: 切空间 → 流形
  - `logmap0(y, c=1.0)`: 流形 → 切空间
  - `mobius_matvec(M, x, c=1.0)`: Ganea 2018 Eq.(7) 矩阵乘法
  - `PoincareAttentionPool(d_model, num_queries=64, alpha_init=0.1)`: 主模块
    - bos_queries: nn.Parameter (64, 128), xavier uniform init + tanh 0.5
    - alpha_raw: nn.Parameter scalar, sigmoid → (0, 1) gate

- `verdicts/poincare_attn_pool_precheck.py`: 4 Gate 验证

---

## 4 Gate verdict

### Gate 1 (数值稳定性): PASS
- norm~0.001 点: d=0.0045 (与预期一致)
- norm~0.99 点: d=0.01 (球面附近距离放大但稳定)
- 距离对称性: d(x,y) == d(y,x) (误差 <1e-6)
- logmap0 + expmap0 round-trip: 误差 = 0.0 (精确)
- 32×16 随机点: 全 finite, 无 NaN/Inf

### Gate 2 (梯度流通): PASS
- bos_queries.grad.norm() = 147 (强梯度信号)
- alpha_raw.grad = -0.77 (明确负方向 → alpha 应该上升)
- 无 NaN 梯度

### Gate 3 (与 Euclidean 差异): **意外发现 ⚠**
**预期**: Poincaré attention 比 Euclidean 对 norm 翻转更敏感 (因双曲距离的非线性)
**实测**: 
- 反转 norm 后 Euclidean 差异 = 1.40, Poincaré 差异 = 0.036
- 比率 Poincaré/Euclidean = **0.03** (远小于 1, 即 Poincaré 反而**不敏感**)

**根因**:
- `e_ctx = expmap0(mean(logmap0(fused_p)))` 是双曲 Frechet mean, 对 norm 翻转有平均化效应
- attention 后的 weighted sum (mobius 切空间) 同样抹平 norm 差异
- softmax(-d) 的软加权本身就有平均效果
- **单纯替换 Euclidean → Poincaré attention pooling 不一定有效**

### Gate 4 (集成钩子): PASS
- 模块文件就位: `common/poincare_attention_pool.py`
- 待集成位置: `common/stage3/stage3_train_pure_t5.py` 中 `geo_forward` / `geo_generate` (HG_Rec 模型 forward hook), 在 `model.shared(input_ids)` 之后注入

---

## 关键发现

### ⚠ 单纯 Euclidean → Poincaré attention 替换不能保证增益

数学根因：
- 双曲距离 attention 权重 = softmax(-d(x, q))
- 但 attention output 是 weighted sum, 几何意义被 softmax 平滑掉
- norm 翻转的影响被 expmap0(mean(.)) 平均化

### 三条候选改进路径

**A. Norm-weighted attention** (推荐优先尝试)
不用双曲距离做 attention weight, 而是**显式 norm 加权**:
```python
# attn_weight ∝ 1/norm (norm 小的 token 权重高 → 反映抽象概念)
norms = torch.linalg.vector_norm(fused_p, dim=-1, keepdim=True)  # (B, L, 1)
attn = F.softmax(1.0 / norms.squeeze(-1).clamp(min=_EPS), dim=-1)  # (B, L)
bos_vec = (attn.unsqueeze(-1) * fused_p).sum(dim=1)  # 加权和
```
→ 强制 norm 小的 token (抽象概念) 主导

**B. Poincaré distance attention (当前版) 但加 norm mask**
保留 distance-based attention, 但显式 norm mask:
```python
attn = F.softmax(-d, dim=-1)
attn = attn * norms.squeeze(-1).pow(-1)  # norm 加权
attn = attn / attn.sum(dim=-1, keepdim=True)  # 重归一化
```

**C. 保留 Euclidean attention, 加 Poincaré norm-aware gate**
```python
euclid_attn = standard_dot_product_attention(...)  # 欧氏版
poincare_norm_weight = exp(-beta * norm_fused_p)  # 双曲 norm 加权
final_attn = euclid_attn * poincare_norm_weight
```

---

## 端到端预训练验证

**未做** — 当前 precheck 阶段 Gate 3 暴露关键问题, 需要先选定路径 (A/B/C) 再做 micro-training 验证。

**建议下一步** (按 ROI):
1. **实施路径 A (norm-weighted)**: 改 PoincareAttentionPool.forward, 用 norm 加权替代 distance attention
2. 重跑 Gate 3 验证敏感性提升
3. micro-training (5000 样本, 5 epoch, taskA stage3 issue61 ckpt 起步) 验证 loss 下降
4. 若 loss 正常 → 开 issue 跑完整 95 epoch 训练

---

## 当前产物

- 模块: `common/poincare_attention_pool.py` (90 行)
- precheck 脚本: `verdicts/poincare_attn_pool_precheck.py`
- verdict: 本文件

## Why
DECOR 的 alpha-gated context-aware embedding 是高 ROI 改造点 (预期 +3~6%), 但其 attention pooling 用 Euclidean 点积
不能直接搬到 Poincaré ball. Gate 3 实测证明 distance-based Poincaré attention pooling 被 softmax 平均化效应抹平.
需要 norm-weighted variant 才能真正反映层次结构.

## How to apply
新 issue 应当:
- 优先实施路径 A (norm-weighted Poincaré attention), 而非路径 B/C
- precheck 必须包含 Gate 3 (Euclidean vs Poincaré 敏感性对比), 避免凭直觉盲训
- micro-training 必须用现有 ckpt (taskA_stage3_issue61/HG_Rec_best.pth) 起步, 不要从零训练
- 集成位置: stage3 geo_forward / geo_generate hook, 在 model.shared(input_ids) * d_model_sqrt 之后注入
- 保留现有 B_geo / HAB / CodewordGeoResidual 不动, 仅在 fused_embeds 层加 Poincaré pooling