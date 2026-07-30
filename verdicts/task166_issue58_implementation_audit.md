# Issue #58 (实现审计 #55/#56 mode collapse) — Bug 确认, 重新开放 (2026-07-31)

## 用户问题

Issue #55 / #56 以 NO-GO 告终, 症状: α_l/scale_l 静止, SID mode collapse (3-digit unique 0.01%). **是否实现 bug 而非方向问题**?

## Step 1 + Step 2 审计结果

### Bug #1 (致命): α_l_raw 是 dead parameter

**根因**: 在 `FreeCurvVectorQuantization.forward` 中:

```python
d = self._per_component_dist_sq(latent, codebook)   # ← 用了 alpha
indices = torch.argmin(d, dim=-1)                    # ← argmin 截断梯度
x_q_hard = codebook.index_select(0, indices)         # ← index_select 无梯度
x_q_st = x + (x_q - x).detach()                       # ← straight-through
recon_loss = ‖x - x_q_st‖²                            # ← 不依赖 alpha

kappa = self.kappa_m()  # 在 MixedCurv 子类返回 fixed kappa
commitment_loss = d_κ(x_q, x)²  +  beta * d_κ(x, x_q)²  # ← 用 fixed kappa, 不用 alpha
```

**gradient flow 路径**:
- `_per_component_dist_sq` 用 alpha → 结果 d 只用于 argmin (无梯度)
- `commitment_loss` 用 `kappa_m()` = fixed kappa → 不用 alpha
- `x_q_st` 用 detach → recon_loss 不依赖 d

→ **α_l_raw 对 total_loss 的梯度严格为 0**

**反向验证 (T4)**: 直接用 `min(d)` 作为 loss, α_l_raw.grad = **0.8908** (非零!). 证明 α_l 本身可微, 只是 forward 路径设计让它 dead.

### Bug #2 (致命): scale_l 是 dead parameter

`scale_l` 在 forward 中**完全未使用**:
- `_per_component_dist_sq` 没引用 scale_l
- 其他 forward 路径也没引用

代码 docstring 写 "codebook = embeddings.weight * scale_l.unsqueeze(-1)", 但这个操作没实现. **docstring 撒谎**.

### Bug #3 (中等): RiemannianAdamW 公式错误

代码用 `(1+‖x‖²)²`, 标准 Poincaré 球公式是 `(1-κ‖x‖²)²/4`:
- 1+ 应改为 1- (跟 #47 同款)
- 漏了 κ 因子
- 漏了 /4 因子

边界行为对比 (κ=0.74):
| ‖x‖ | Code (1+‖x‖²)² | Std (1-κ·r²)²/4 | Ratio |
|------|------------------|------------------|-------|
| 0.0 | 1.0 | 0.25 | 4× |
| 0.5 | 1.56 | 0.17 | 9× |
| 0.866 | 3.06 | 0.05 | 62× |
| 1.0 | 4.0 | 0.017 | **237×** |

代码公式在 ‖x‖→1 时把梯度放大约 237×, 跟正确 Riemannian 流形方向相反.

### Bug #4 (次要): Issue #55 verdict 数学描述错误

verdict 写 "在 ‖x‖→1 时接近 0", 但 `1/(1+1)² = 0.25` ≠ 0. 实际公式 `1/(1+‖x‖²)²` 在 ‖x‖=1 时是 0.25, 不会趋于 0. 用户 Issue #58 §疑点 1 已准确指出.

## 历史类比 (跟 #47 同模式)

| Issue | 误判 | 真实原因 |
|-------|------|---------|
| #47 | "统一 κ-stereographic 公式设计有问题" | 2 个实现 bug: Möbius 加法符号错误 + κ=0 NaN |
| #55 | "Riemannian 优化器/曲率感知优化方向有问题" | 3 个实现 bug: α_l/scale_l dead param, Riemannian 公式 1+ vs 1- |
| #56 | "混合曲率乘积空间方向有问题" | 2 个实现 bug: α_l dead param, commitment loss 路径用 fixed kappa 不用 alpha |

## 验证标准 (per Issue #58 §验证标准)

- ✅ Step 1/2 发现实现 bug (公式错误、梯度断连、无关改动副作用): **确认**
- ⏸️ 修复后重跑 Stage 1-3, 看 mode collapse 和"参数停滞"是否消失 → **待执行**
- ❓ 若不消失, 维持 #55/#56 NO-GO

## 后续行动

### 必须执行 (per Issue #58 §验证标准)
1. **重新开放 Issue #55 + #56** (本次审计已确认 NO-GO 是 buggy 实现)
2. 修复 Bug #1 (α_l): commitment/codebook loss 也用 mixed_curv_dist
3. 修复 Bug #2 (scale_l): 在 _per_component_dist_sq 中应用 scale_l
4. 修复 Bug #3 (Riemannian): 替换为 (1-κ‖x‖²)²/4
5. 重跑 Stage 1-3, 跟 HG-Rec baseline R@10=0.1020 对比
6. 若修复后仍 NO-GO, 再维持 closed

### 优先级
- 等 GPU 0/3 空闲 (Task #158/#159 完成后, ~01:52 AEST)
- 先修复 bug + Stage 1 训练 (~3h)
- 若 Stage 1 仍 mode collapse → 维持 NO-GO
- 若 Stage 1 改善 → 继续 Stage 2 + Stage 3

## 关键 takeaway (供未来 reference)

Issue #58 复刻了 Issue #47 的 "公式 bug 误判为方向问题" 模式. 教训:
- 静态代码审计必须先于"方向 NO-GO"结论
- α_l/scale_l 类参数"梯度 ≈ 0"必须区分 (a) 没注册 (b) detach 截断 (c) saturation
- verdict 数学描述必须跟实际代码公式一致

---
result: Issue #58 Step 1+2 完成. 确认 4 个 bug (Bug #1+#2+#3 实现 bug + Bug #4 verdict 描述错). 重新开放 Issue #55 + #56, 修复 bug 后重测. 等 #158/#159 完成释放 GPU.
