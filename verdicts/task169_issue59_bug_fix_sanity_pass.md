# Issue #59 (修复+重跑 #55/#56 实现 bug) — Bug 修复完成, Sanity 5/5 PASS

## 任务摘要

基于 Issue #58 审计发现的 3 个实现 bug, 修复 `hrqvae_issue55_56.py` 中的死参数路径, 5/5 sanity test 验证修复有效. 等 GPU 0/3 空闲后启动 Stage 1 重训练.

## 修复内容 (代码位置: `HG-Rec/model/hrqvae_issue55_56_fixed.py`)

### Bug #1 修复: α_l_raw 现在有非零梯度

**根因**: 父类 `FreeCurvVectorQuantization.forward` 的 commitment loss 路径用 `kappa = self.kappa_m()` 拿曲率, 而 MixedCurv 子类 `kappa_m()` 返回 fixed kappa tensor (跟 α_l 无关).

**修复**: 在 `FreeCurvVectorQuantizationMixedCurvFixed` 中完全 override `forward`, commitment/codebook loss 改用 `mixed_curv_dist_scalar(xq_m, xl_m, α_m, κ)` —— α_l 通过 sigmoid(α_l_raw) 进入 loss.

**验证**: T1 — `alpha_l_raw.grad = tensor([8.3407])` (非零, Bug #1 fixed).
**反向验证**: T4 — 同结构下普通 forward 也得到 `alpha_l_raw.grad = -0.0618`, 说明整条 forward 路径都对 α_l 可微.

### Bug #2 修复: scale_l 现在有非零梯度

**根因**: 原代码 `_per_component_dist_sq` 应用 scale_l (在子类的 override 中), 但这个 d 只用于 argmin 截断梯度; 而 commitment loss 用的是 `codebook.index_select(0, indices)` 拿到的 **unscaled** codeword.

**修复**: `_mixed_curv_commitment_loss` 中也应用 scale_l:
```python
xq_m_scaled = xq_m * s_m  # scale_l 应用
d_c = mixed_curv_dist_scalar(xq_m_scaled, xl_m.detach(), a_m, κ)
```

**验证**: T2 — `scale_l.grad = tensor([-0.4499])` (非零, Bug #2 fixed).

### Bug #3 修复: RiemannianAdamW 公式 (1+ vs 1-) 数学正确

**根因**: 原代码 `(1 + ‖x‖²)²`, 标准 Poincaré 球公式是 `(1 - κ‖x‖²)²/4`:
- 1+ → 1-
- 漏 κ 因子
- 漏 /4 因子

**修复**: `RiemannianAdamWFixed` 用正确公式 `(1 - κ*‖x‖²)² / 4`, 通过 param_group 的 kappa 参数传入 (默认 1.0).

**验证**: T3 — param [0.5, 0.5, 0.5] (norm 0.866) 单步后变 [0.395, 0.395, 0.395] (norm 0.684), 验证 Riemannian retraction 正确投影回 Poincaré ball (‖x‖ < 1/√κ = 1).

## Sanity Test 5/5 PASS

| Test | 检查 | 期望 | 实测 | 状态 |
|------|------|------|------|------|
| T1 | α_l_raw.grad | 非零 | 8.3407 | ✅ |
| T2 | scale_l.grad | 非零 | -0.4499 | ✅ |
| T3 | Riemannian 公式 + retraction | norm 下降 + 留在球内 | 0.866→0.684 (< 1) | ✅ |
| T4 | Forward + backward 全流程 | finite + grad 非零 | OK | ✅ |
| T5 | RVQ (3 层) 完整 build + forward | shape + loss finite | OK | ✅ |

## 跟 Issue #47 同模式确认

Issue #47 是 Möbius 加法符号错误 + κ=0 NaN, 当时也误判为方向问题. Issue #58 复刻这一模式, 确认 α_l/scale_l "梯度 ≈ 0" 是 (b) detach 截断路径, 不是 (a) 没注册 或 (c) saturation. 修复路径明确, 不是神秘现象.

## 后续执行计划 (per Issue #59 §验证标准)

| 阶段 | 内容 | 状态 | 预计耗时 |
|------|------|------|----------|
| **Gate 0** | 5/5 sanity (已 PASS) | ✅ | done |
| **Gate 1** | Stage 1 重训练 #56 fixed (α_l active) | ⏸️ 等 GPU 0/3 | ~3h |
| **Gate 2** | Stage 1 重训练 #55 fixed (α_l + scale_l active) | ⏸️ 等 GPU 0/3 | ~3h |
| **Gate 3** | Stage 2 SID 推断 (需 #156/#162 update) | ⏸️ 依赖 Gate 1 | ~10min |
| **Gate 4** | Stage 3 T5 训练 + R@10 对比 baseline 0.1020 | ⏸️ 依赖 Gate 3 | ~3h |

### Stage 1 重训练 Recipe (跟原 #55/#56 一致, 仅换模型类)

```bash
# Issue #56 fixed
python3 scripts/task156_issue56_stage1_train.py \
  --vq_class FreeCurvVectorQuantizationMixedCurvFixed \
  --kappa_fixed 0.74 --lr 1e-3 --num_epochs 1000

# Issue #55 fixed  
python3 scripts/task157_issue55_stage1_train.py \
  --vq_class FreeCurvVectorQuantizationMixedCurvWithScaleFixed \
  --kappa_fixed 1.0 --lr 1e-3 --num_epochs 1000
```

### 验证指标

- α_l_raw 训练中是否仍全程停在 0.5 (i.e., alpha_l=0.5)? → 修复后应该漂移
- scale_l 训练中是否仍停在 1.0? → 修复后应该漂移
- Stage 1 ckpt loss 是否仍 0.0002 (collapse)? → 修复后应该 > 0.01
- Stage 2 SID 3-digit unique 是否仍 0.01%? → 修复后应该 > 50%

### 决策阈值 (跟 Issue #58 §验证标准 一致)

| 修复后结果 | 结论 |
|-----------|------|
| α_l/scale_l 漂移 + Stage 1 loss > 0.01 + SID 3-digit unique > 50% | Issue #55/#56 维持 GO, 进入 Stage 3 |
| α_l/scale_l 仍静止 / Stage 1 仍 collapse | Issue #55/#56 维持 NO-GO, 方向本身失败 |

## 关键 takeaway (供未来 reference)

Issue #58/59 完整闭环复刻 Issue #47 模式:
1. **静态代码审计 (Step 1)** → 发现 4 个 bug
2. **梯度单元测试 (Step 2)** → 反向验证 bug 是实现错不是方向错
3. **修复 + 验证 (Step 3)** → 5/5 sanity pass
4. **下一步 Stage 1 重跑** → 验证方向本身是否可行

教训:
- α_l/scale_l 类参数"梯度 ≈ 0"必须先审计: (a) 注册了吗 (b) detach 截断吗 (c) saturation? — (b) 是大头
- verdict 数学描述必须跟实际代码公式一致 (Issue #58 Bug #4)
- 修复 + 重测是完整闭环, 不能停在"NO-GO"

## 关联产物

- 修复文件: `HG-Rec/model/hrqvae_issue55_56_fixed.py` (380 行, 完整 5/5 sanity)
- Bug 审计: `verdicts/task166_issue58_implementation_audit.md`
- 梯度测试: `scripts/task167_issue58_step2_gradient_unit_test.py`
- 历史 NO-GO (待重测): `verdicts/task156_issue56_mode_collapse_no_go.md`, `verdicts/task157_issue55_mode_collapse_no_go.md`

---
result: Issue #59 Bug #1+#2+#3 修复完成, 5/5 sanity PASS (α_l_raw.grad=8.34, scale_l.grad=-0.45, Riemannian 公式正确). 等 Task #158/#159 释放 GPU 0/3 后启动 Stage 1 重训练.
