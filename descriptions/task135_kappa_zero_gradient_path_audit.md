# Task #135 — κ=0 硬分支导致 κ_m 梯度通路断裂诊断 + 修复

> **任务目的**: 用户 2026-07-24 反馈: Task #89 结论"18/18 (layer, κ_m) 精确 = 0.000000 + θ_m 训练完全不动"比"收敛到接近 0"更可疑, 可能不是数据本质, 而是代码梯度通路断了. 三步验证 + 修复 + 重测.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 Task #89 verdict (被质疑)

Task #89 verdict `verdicts/task89_free_curv_product_manifold_result.md` 第 1 段:

> 18 个 (layer, κ_m) 终值 = 精确 0.000000, **θ_m 训练 1000 epoch 后仍精确 0**

5 重证据闭环支持 "Musical_Instruments 数据本质欧氏". 但用户指出: "Adam 在 minibatch 噪声下通常在 0 附近震荡 (±1e-3), 完全锁死精确 = 0 更像是**梯度通路断**而不是优化结果".

### 1.2 用户提出的具体根因 (按概率)

**最可能**: `_per_component_dist_sq` 和 commitment loss 的**硬分支**:

```python
# HG-Rec/model/hrqvae_free_curv.py:152
if k_m.item() == 0.0:        # ← Python 硬分支
    d_m_sq = ((x_m.unsqueeze(1) - c_m.unsqueeze(0)) ** 2).sum(dim=-1)
elif k_m.item() < 0:
    c = (-k_m).item()
    x_m_h = proj_to_ball(expmap0(x_m, c), c)
    ...
```

当 `theta_m=0` 时, `kappa_m = kappa_max * tanh(0) = 0.0` **精确**等于 0.0, 永远走第一个分支. 该分支内 `kappa` 张量**从未出现**在计算图里 → `dL/d_kappa = 0` 精确 → `dL/d_theta_m = 0` 精确 → θ_m 一动不动.

### 1.3 用户提议的三步零成本验证

| 步 | 验证 | 期望结果 (bug 成立) |
|----|------|---------------------|
| 1 | backward 后 print `θ_m.grad` | 精确 0.0 或 None |
| 2 | θ_m 初始化到 ±0.5, 跑短调试 | 仍不动 (精确 0) |
| 3 | 检查 κ 算子实现 | 找到硬分支 |

---

## 2. 实验设计

### 2.1 Step 1 — backward grad check (~10 min CPU)

**目的**: 加载 Task #89 Arm A 已训 ckpt (或重训 5 epoch), 在第一次 backward 后打印 `θ_m.grad`.

**判据**:
- 若 `θ_m.grad = 0.0` (精确) 或 tensor 为 None → **bug 确认** (梯度通路断)
- 若 `θ_m.grad ≠ 0.0` (例如 ≈ 1e-5 量级) → 阈值偏低但不是 bug, 检查学习率

### 2.2 Step 2 — 远点初始化 (±0.5) 短训练 (~30 min GPU)

**做法**: A 臂 (M=1), `θ_m=0` → `θ_m=±0.5`, 跑 100 epoch, 每 10 epoch 记录 `θ_m` 终值.

**判据**:
- 若 `θ_m` 从 ±0.5 仍不动 → **bug 确认** (梯度传不到)
- 若 `θ_m` 缓慢漂回 0 → 这是 "0 是 attractor" 的干净信号, 即使 bug 修了也大概率落 0

### 2.3 Step 3 — 代码审计 (静态) (~5 min)

读 `hrqvae_free_curv.py` line 34-55 (geodesic_distance_sq) + 133-178 (_per_component_dist_sq) + 207-231 (commitment loss). 列出所有 `if k.item() == 0.0` 硬分支位置.

### 2.4 Step 4 — 修复并重训小窗口 (~6 h GPU)

**修复方案**: 把硬分支换成**处处光滑的连续公式**, 用 tanh-based soft-sign:

```python
# 替换: if k_m.item() == 0.0:
#       elif k_m.item() < 0:
# 改成: 用 κ_m 绝对值 + sign 连续切换, 或用直接 tanh 序列展开
#
# MCKG Table 1 统一写法: tan_κ / sin_κ / cosh_κ 等,
# 这些在 κ→0 时所有 κ-dependent 项都用泰勒展开保证 C¹ 连续.
```

具体方案**等 Step 3 审计完再定** (R11.3 自主决策).

**重训**: A 臂 (M=1), seed=42, **200 epoch** (短窗口, 看 κ_m 是否会动) + **每 10 epoch 打印 θ_m.grad 范数**.

**判据**:
- 修复后 `θ_m.grad ≠ 0` (量级 >1e-6) → 梯度通路恢复, bug 确认
- 修复后 θ_m 仍稳定 0 → "数据本质欧氏" 结论仍然成立 (但**这个证据才干净**)

### 2.5 Step 5 — 更新 Task #89 verdict (RETRO)

无论 Step 4 结果如何, Task #89 verdict 都需要 RETRO / 加 caveat:
- **若 bug 修复后 θ_m 仍锁 0**: 维持 "数据本质欧氏" 结论 (但证据更干净, 注明"已排除 κ=0 梯度阻断")
- **若 bug 修复后 θ_m 学到非 0**: **强烈推翻**原结论, Task #89 verdict 必须 RETRO, 重新跑 Stage 1-4

---

## 3. 决策触发 (vs Task #89 现有结论)

| 验证结果 | 决策 |
|----------|------|
| Step 1 grad 精确 0 + Step 2 远点不动 | **bug 确认**, Task #89 verdict 必须 RETRO |
| Step 1 grad 精确 0 + Step 2 远点**动了** | **矛盾信号** — bug 局部但不全断, 进一步分析 |
| Step 1 grad 非 0 | **非 bug**, 原 Task #89 结论保留 |
| Step 4 修复后 θ_m 学到非 0 | **推翻** "数据本质欧氏", Task #89 重跑 |
| Step 4 修复后 θ_m 仍锁 0 | **维持** "数据本质欧氏", 但证据更干净 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Step 1 backward grad check | ~10 min (CPU) |
| Step 2 ±0.5 初始化短训练 | ~30 min GPU |
| Step 3 代码静态审计 | ~5 min |
| Step 4 修复 + 200 epoch 重训 | ~1.5 h GPU (A 臂 1 卡) |
| Step 5 verdict retro + §16 更新 | ~30 min |
| **总计** | **~2.5 h GPU + ~30 min CPU** |

---

## 5. 风险与缓解

**风险 1**: 修复后 κ_m 跑到极端值 (-2 或 +2), 数值爆
→ 缓解: κ = κ_max · tanh(θ) 已有界重参数化, κ_max=2 已限死; 200 epoch 短训练不会跑到极端

**风险 2**: 三步验证都通过 (原代码没问题), 但 Task #89 结论仍可疑
→ 缓解: 即使 grad 非零, 也跑 Step 4 看 θ_m 训练曲线是否真在 0 附近震荡; 写"非 bug 但信号弱" verdict

**风险 3**: 修改 `HG-Rec/model/hrqvae_free_curv.py` 属上游代码改动 (R11.4 关键决策)
→ 缓解: 已 dry-run 报告修改位置 (Step 4 待审), 不静默改

---

## 6. 完成度跟踪

- [ ] Step 1: backward grad check 脚本
- [ ] Step 1 跑完, 确认 grad 状态
- [ ] Step 2: ±0.5 初始化短训练脚本
- [ ] Step 2 跑完, 看 θ_m 是否移动
- [ ] Step 3: 代码静态审计 (列出所有硬分支位置)
- [ ] Step 4: 修复方案确定 + 代码改
- [ ] Step 4: 200 epoch 重训跑完
- [ ] Step 5: Task #89 verdict retro (R11.3 决策: 维持 / 推翻 / 加 caveat)
- [ ] Step 5: verdicts/task135_*.md 写出
- [ ] Step 5: loop.md §16 更新 (Task #89 状态从 ✅ → ⚠️, 加 #135 活跃任务)

---

## 7. 关联

- **用户输入**: 2026-07-24 "θ_m 完全不动"质疑
- **被审计对象**: Task #89 verdict (`verdicts/task89_free_curv_product_manifold_result.md`)
- **被修改代码**: `HG-Rec/model/hrqvae_free_curv.py` (上游 HG-Rec 衍生, 不是 snap-research/GRID 原版, 允许改)
- **后续**: Task #89 verdict retro → 后续 paper §6 写法更新

---

## 8. 关键决策点 (R11.3 自决)

1. **Step 4 修复方案**: 待 Step 3 审计完定. 候选 (a) tan_κ/sin_κ 统一算子表 (MCKG Table 1), (b) tanh-based soft switch, (c) 用 `|k| + ε` 把硬分支换成 torch.where. (R11.3: 优先 a, 因为这是 MCKG paper 自己用的方案)
2. **RETRO vs 加 caveat**: 取决于 Step 4 结果. 若修复后 θ_m 仍锁 0 → 加 caveat (证据更干净); 若非 0 → 强制 RETRO 整份 verdict (R2 不允许 silent fall-back)
3. **修改上游代码**: 属 R11.4 critical decision, 必须 dry-run 报告修改位置 + 用户授权后才改 (`Bash sed` 或 Edit 工具)
4. **新训练是否需要 1000 epoch**: 200 epoch 短窗口仅做 κ_m 信号侦测, 不下结论; 若信号强再补长训 (~6h)

---

## 9. 用户疑虑原文 (引用)

> 18/18个(layer, κ_m)终值精确落在0.000000,且θ_m训练过程中完全没有移动——这个模式比"收敛到接近0"更极端,值得怀疑. 真实的随机梯度下降,哪怕最优点确实在0附近,Adam在有限学习率+minibatch噪声下通常会在0附近震荡(比如±1e-3量级),而不是从头到尾锁死在一个精确值上不动.
>
> 这背后最常见的原因是κ-stereographic公式里的分段实现问题...如果代码用的是硬分支(if kappa == 0: ... elif kappa > 0: ...这种Python if/else,而不是一个处处光滑、能连续求导过κ=0这个点的写法),PyTorch的autograd只会追踪实际走过的那条分支,κ这个变量在"走欧式分支"时可能根本不出现在计算图里.
>
> 建议在写进最终结论前,先做三个几乎零成本的验证(不需要重新训练完整模型):
> (1) backward()之后、optimizer.step()之前,打印一个θ_m的.grad
> (2) θ_m初始化到远离0的地方(+0.5和-0.5),跑一个几十步的短调试
> (3) 检查κ-stereographic算子的具体实现是不是在κ=0处用了硬分支
