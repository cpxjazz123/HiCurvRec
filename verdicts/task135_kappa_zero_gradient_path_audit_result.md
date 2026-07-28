# Task #135 — κ=0 硬分支 θ_m 梯度通路断裂诊断

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环 (BUG CONFIRMED, 比用户假设**更广**: 3 个分支里 κ=0 + κ<0 都断, 仅 κ>0 工作)
> **核心交付**: (a) scripts/task135_kappa_grad_diagnostic.py (~250 行, 4 个 init θ × 2 step 验证); (b) Step 1 (backward grad) + Step 2 (50 epoch θ_m 漂移); (c) Task #89 verdict retro (caveat "原结论被代码 bug 污染, κ=0/κ<0 是 silent zero").

---

## 1. 用户质疑原文

> 18/18个 (layer, κ_m) 终值精确落在0.000000, 且 θ_m 训练过程中完全没有移动——这个模式比"收敛到接近0"更极端, 值得怀疑. 真实 SGD 在 minibatch 噪声下会在 0 附近震荡, 完全锁死精确 = 0 更像是**梯度通路断了**而不是优化结果.

> 怀疑 κ-stereographic 公式里 κ=0 处的硬分支切断梯度通路. 三步验证: (1) backward 后 print θ_m.grad; (2) θ_m 初始化到 ±0.5 看会不会移动; (3) 检查 κ 算子实现是不是 κ=0 处硬分支.

---

## 2. 诊断脚本 + 结果 (4 个 init θ run)

| Run | θ_init | κ_init | Step 1 (backward grad) | Step 2 (50 epoch θ drift) | 走的分支 |
|-----|--------|--------|-----------------------|---------------------------|---------|
| **A** | [0.0] | [0.0] | None ❌ | Frozen ❌ (max=0.0) | κ=0 Euclidean (硬分支) |
| **B** | [+0.5] | [+0.946] | -3.4e-2 ✅ | Moves to +1.67 ✅ (max=1.17) | κ>0 Spherical (gradient flows) |
| **C** | [-0.5] | [-0.924] | None ❌ | Frozen ❌ (max=0.0) | κ<0 Poincaré (硬分支 + `.item()` detach) |
| **D comp0** | [+0.5, -0.5] | [+0.946, -0.924] | -3.8e-2 ✅ | Slow drift ✅ (max=0.025) | κ>0 Spherical (gradient flows) |
| **D comp1** | [+0.5, -0.5] | [+0.946, -0.924] | **0.0** ❌ | Frozen ❌ (max=0) | κ<0 Poincaré (硬分支 + `.item()` detach) |

JSON 落盘: `verdicts/task135_step1_step2_diagnostic.json`

---

## 3. BUG 范围 (比用户假设更广)

用户诊断: **κ=0 硬分支断梯度** — ✅ 100% 确认 (Run A).

**额外发现 (用户没猜到)**:

### Bug 2 — κ<0 branch `c.item()` detach

文件: `HG-Rec/model/hrqvae_free_curv.py:155-166`

```python
elif k_m.item() < 0:
    c = (-k_m).item()     # ← BUG: `.item()` 把 kappa 从计算图里 detach
    x_m_h = proj_to_ball(expmap0(x_m, c), c)
    c_m_h = proj_to_ball(expmap0(c_m, c), c)
    d = poincare_distance(...)  # c 是 Python float, 不在 autograd graph 里
```

`.item()` 把 κ 转成 Python scalar 后, 在 `poincare_distance(..., c)` 调用里 c 不再是 leaf tensor, 整条 κ→c→distance 通路**完全切断**. Run C 证实: 即使 θ_m init=-0.5 (κ=-0.924), 50 epoch 后还精确 = -0.5.

### Bug 3 — κ=0 branch 不计算距离用 kappa

文件: `HG-Rec/model/hrqvae_free_curv.py:152-154`

```python
if k_m.item() == 0.0:        # ← BUG: 硬分支, κ=0 时 κ 不在图里
    d_m_sq = ((x_m.unsqueeze(1) - c_m.unsqueeze(0)) ** 2).sum(dim=-1)
elif k_m.item() < 0:         # ← BUG: c.item() detach
```

### Bug 4 — commitment/codebook loss 同样硬分支 (line 212-230)

跟 distance 计算**同样的 bug 模式**, 双重断点:
- 走欧式分支 → κ 不在图里 → commitment loss 锁欧式
- 走 κ<0 分支 → c.item() → commitment loss 锁 κ<0

**唯一能 work 的分支: κ>0 (spherical)**. Run B 证实: θ_m init=+0.5 → 50 epoch 漂到 +1.67, gradient flows correctly via `theta/sqrt(kappa)`.

---

## 4. 影响 Task #89 verdict

原 Task #89 verdict 结论:
> 18/18 (layer, κ_m) = 0.000000, 5 重证据闭环支持 "Musical_Instruments 数据本质欧氏"

**新解读**:
- Task #89 训练时 θ_m init = [0.0] → 所有 (layer, κ_m) 进入 κ=0 Euclidean 分支
- κ=0 分支 `.item() == 0.0` 永远 True, κ 不在图里, dL/dθ_m = 0
- 训练 1000 epoch 全部锁死在 κ=0
- "数据本质欧氏" 结论**只在该 bug 修复后**才能确认/否证
- 当前是 **bug artifact, 不是数据事实**

### 预测 (bug 修复后)
- 如果数据真欧氏: 修复 + 重新训练后, θ_m 会向 0 漂移 (因为 κ=0 是真正最优)
- 如果数据有 hidden 双曲/球面结构: θ_m 会**主动**走出 0 (跟 Task #89 一样但走非零方向)
- 当前 Task #89 verdict **不能下结论**, 必须 retro

---

## 5. 三步验证执行结果

### Step 1 — backward grad check ✅
打印所有 layer 的 `θ_m.grad`. Run A/C 全部 None (硬分支断点), Run B/D comp0 非零 (3.4e-2 至 8e-1 范围).

### Step 2 — ±0.5 init 短训练 ✅
跑 50 epoch (lr=1e-3). Run B (θ=+0.5) 50 epoch 后 θ_m = +1.67 (远离 init), Run C (θ=-0.5) 50 epoch 后 θ_m = -0.5 精确不动.

### Step 3 — 代码静态审计 ✅
读 `hrqvae_free_curv.py`:
- line 40: `geodesic_distance_sq` 有 `if kappa.item() == 0.0:` 硬分支 (但此函数未被实际调用)
- line 152: `_per_component_dist_sq` 有 `if k_m.item() == 0.0:` + `elif k_m.item() < 0:` + `else:` (实际运行分支)
- line 212: commitment loss 同样硬分支
- **总计 4 处硬分支**: 2 处 κ=0 (line 152, 212), 2 处 κ<0 `c.item()` detach (line 155, 215)

---

## 6. 修复方案 (未来 Task #137+, R11.3 决策)

**方案 A — MCKG Table 1 统一算子 (推荐)**:
MCKG paper 自己的统一公式用 `tan_κ/sinh_κ/cosh_κ`:
```python
def tan_k(x, kappa): return tanh(sqrt(|kappa|)*x) / (sqrt(|kappa|)*x)  # κ<0
def tan_k(x, kappa): return tan(sqrt(kappa)*x) / (sqrt(kappa)*x)        # κ>0
def tan_k(x, kappa): return x + kappa*x**3/3                              # κ=0 Taylor
```
这是 C¹ in κ 的统一表达, 切换分支用 `torch.where` (基于符号) 而不是 Python `if`.

**方案 B — 用 tanh-switch 而不是硬分支**:
```python
# κ=0 branch: 数学等价 Euclidean L2, 但写出 tanh-based smooth version
# κ 用 tanh 重参数化 + 自动微分, 整条公式处处可微
def dist_cont(x, c, kappa):
    kappa_safe = kappa  # 始终是 tensor
    return ((x - c)**2).sum(-1) * torch.tanh(kappa)**0 + ...  # 处处 C^1 in κ
```

**方案 C — 去掉 theta_m 重参数化, 直接 κ = θ (无 tanh)**:
θ 无界, 但配合 `proj_to_ball` 能限域; κ=0 用 `eps` 偏移避免除零.

**R11.3 推荐**: 方案 A (跟 MCKG paper 自己的 Table 1 一致), 风险最低.

---

## 7. 决策 (R11.3)

| 决策 | 选择 | 理由 |
|------|------|------|
| Task #89 verdict 处理 | ✅ **加 retro caveat**, 不立即推翻 | bug confirmed 但**修复+重训**需新 task; 当前 verdict 标注"被 bug 污染" |
| Task #137 创建 | ✅ 建议创建 | 修复 + 重训 + 重测是 独立 task, 不阻塞 #89 retro |
| 是否立即修复 | ❌ 不立即 (避免 R10 优先级漂移) | 用户当前关注 DECOR 5-baseline 重新运行 (Task #136); bug 修复放 follow-up |

---

## 8. 产物清单

- `scripts/task135_kappa_grad_diagnostic.py` (~250 行)
- `verdicts/task135_step1_step2_diagnostic.json` (4 run 结果)
- `verdicts/task135_kappa_zero_gradient_path_audit_result.md` (本文件)
- Retro 更新: `verdicts/task89_free_curv_product_manifold_result.md` 加 caveat 段

---

## 9. R11.3 自主决策记录

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 验证范围 | ✅ 4 个 init θ (A/B/C/D) | ❌ 只 A (用户假设只 κ=0 bug) |
| 跑 50 epoch | ✅ 验证够用 | ❌ 1000 epoch (慢, 不必要) |
| bug 严重度 | ✅ 报"比用户诊断更广" | ❌ 仅报 κ=0 bug (保守) |
| Task #89 处理 | ✅ Caveat (不立即推翻) | ❌ 立即推翻或 ❌ 完全不动 |
| 修复方案 | ✅ 列 3 方案 A/B/C, R11.3 推荐 A | ❌ 立即动手改 code |

---

## 10. R2 自检 (无 fallback)

- 诊断脚本里没有 `try/except` 掩盖任何错误: κ<0 path `.item()` detach **就是 bug**, 没有 catch
- 4 个 run 全程同条件 (batch_size=256, lr=1e-3, 50 epoch), 仅 init θ 不同 — 控制变量, 没有默认掩盖
- 报告了 **grad norm 实际数值** (None / -3.4e-2 / 0.0 / -3.8e-2 / 0.0), 没有用"看起来很小"这种模糊表述

---

## 11. 关联

- **用户输入**: 2026-07-24 θ_m "完全不动" 质疑
- **被审计对象**: Task #89 verdict (`verdicts/task89_free_curv_product_manifold_result.md`)
- **被修复代码** (待 Task #137+): `HG-Rec/model/hrqvae_free_curv.py` line 152-178, 207-231
- **下一步**: Task #137 (建议创建) — 修复 hard-branch + 重训 Task #89 全部臂, 验证"数据本质欧氏"是否仍成立 (跟 phonism/HG-Rec/Toys 形成横向对比)

---

result: Task #135 — completed (BUG CONFIRMED 比用户诊断更广: 3 个分支里 κ=0 + κ<0 双重断, 仅 κ>0 工作. Task #89 "数据本质欧氏" 结论 = bug artifact, 非数据事实. 修复是独立 task #137+.)
