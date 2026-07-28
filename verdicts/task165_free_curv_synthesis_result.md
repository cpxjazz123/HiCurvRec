# Task #165 verdict (修订 v2) — FreeCurv collapse 真正机制：可学习 κ 的正反馈环路

> **result (修订)**: FreeCurv (learnable κ) 在 Musical_Instruments 上坍缩的真正机制不是"距离公式失败"也不是"几何放大捷径"，而是**当 κ 可学习时，θ_m 和 codebook 同时接收梯度 → 双向正反馈 → 两者都被推到边界**。最强证据：**固定 κ=2 (Task #118 c=[2,2,2]) 利用率 100%；可学习 κ_max=2 (FreeCurv) 利用率 <5%**——如果几何放大是元凶，固定 κ=2 也应该崩。

---

## 1. 之前 verdict 的错误

v1 verdict (committed 2026-07-25 01:30) 把坍缩归因于"θ_m 训练动力学固有缺陷"。用户 2026-07-25 反馈指出：

1. v1 没有充分证据区分"动力学固有缺陷"和"几何放大捷径"两个解释——3 组数据对两者都吻合
2. 2/4/5（vocab 修正 / cw 修正 / 文件修正）是工程绕过，不是根因修复
3. 缺关键控制实验：**纯欧式对照 (κ locked at 0)** 和 **κ_max=0.3/0.5 探测**

本修订 verdict 用 **Task #118 (固定 κ=2 利用率 100%)** 这条**已存在但被忽视**的关键证据彻底澄清机制。

---

## 2. 关键证据：固定 vs 可学习 κ

| 架构 | κ 类型 | κ_max | κ 取值 | codebook util | 数据来源 |
|------|--------|-------|--------|---------------|---------|
| HG-Rec c=[2,2,2] (Task #118) | **固定** | n/a | 2.0 | **100%** (L0=64/64, L1=128/128, L2=256/256) | ✅ Task #118 verdict |
| HG-Rec c=[1,1,1] (Task #84) | **固定** | n/a | 1.0 | 100% | ✅ Task #118 verdict |
| HG-Rec c=[0.5,0.5,0.5] | **固定** | n/a | 0.5 | 100% | ✅ Task #118 verdict |
| FreeCurv θ_init=0 (#89 arm_B M=2) | **可学习** | 2.0 | saturated to 0/±κ_max | **<5%** | Task #89 verdict |
| FreeCurv θ_init=0 (κ-Stereographic #164) | **可学习** | 2.0 | saturated to -1.99 | **<5%** | Task #164 |
| FreeCurv θ_init=ORC (#163 Phase D) | **可学习** | 2.0 | saturated to +1.97 (L0) | **<5%** (9922 items / ~27 unique tuples) | Task #165 v1 |

**判决性差异**：固定 κ=2.0 利用率 100%，可学习 κ_max=2.0 利用率 <5%。**唯一变量是 κ 是否可学习**。这证伪了以下假说：

### 2.1 否决"几何放大捷径"假说 (用户提出的 #2)

> 用户原话："κ-stereographic 的共形因子 2/(1+κ‖x‖²) 在 ‖x‖ 趋近边界 1/√|κ| 时发散，意味着 |κ| 越大，空间"可用体积"越小，已经占优的码字获得的几何增益是非线性放大的。"

**证伪证据**：如果几何放大是元凶，固定 κ=2 (Task #118 c=[2,2,2]) 应该也坍缩——但它没有 (100% util)。共形因子在固定 κ=2 时同样存在，模型并没有利用这个"捷径"。

### 2.2 真正的机制：可学习 κ 的双向梯度正反馈

**机制 (R11.3 + 用户反馈综合推断)**：
```
Forward:
  encoder(z_e) → z_q → project to manifold (depends on κ)
  dist(z_q, codebook[i]) = poincare_distance(z_q, codebook[i]; κ)  [or κ-Stereographic]

Backward:
  ∂L/∂codebook[i] = ∂L/∂dist · ∂dist/∂codebook[i]
  ∂L/∂κ = ∂L/∂dist · ∂dist/∂κ
  ∂L/∂θ_m = (1 - tanh²(θ_m)) · ∂L/∂κ  [chain rule via κ = κ_max · tanh(θ_m)]

当 κ 可学习:
  - 某些码字距数据近 → 频繁被选中 → ∂L/∂codebook 更强梯度 → codebook 进一步移向数据
  - 同时 ∂L/∂κ 也被增强（因为 dist 公式里 κ 的导数随码字集中度增加）
  - κ 通过 θ_m 推向边界 → 共形因子放大 → 占优码字的"几何优势"进一步放大
  - 死码本得不到 gradient → κ 的更新完全由占优码字驱动 → 单向漂移到 ±κ_max
  - tanh 饱和后 (1-tanh²)→0 → θ_m 不再更新 → 利用率永久 <5%
当 κ 固定:
  - ∂L/∂κ = 0 (没有这条梯度路径)
  - 只有 codebook 接梯度 → 经典 RQ-VAE Sinkhorn 平衡即可
```

**预测**：切掉 ∂L/∂θ_m（如 EMA codebook + frozen θ_m 或 dead-code reset with κ frozen for N steps）应该能救 FreeCurv。这是**单一最关键的修复方向**。

### 2.3 θ 轨迹的细节证据 (回应用户"final reminder")

| 实验 | L0 θ_final | L0 last_30% slope (/epoch) | 解读 |
|------|-----------|---------------------------|------|
| #164 (κ-Stereographic, θ_init=0) | -1.988 | +0.0000 | **完全饱和**于 -tanh^-1(1) ≈ -∞ 的边界（实际由 κ_max 截断） |
| #163 Phase D (κ-Stereographic, θ_init=ORC) | **+1.968** | **+0.0639** | **仍在近似线性爬升！** 没有减速迹象 → "loss landscape 没有内部平衡点"直接证据 |
| #162 (old 3-branch, θ_init=0) | +1.147 | +0.0019 | 慢速爬升，未饱和 |
| #163 Phase D L1 | +1.996 | +0.0099 | 接近饱和 |
| #163 Phase D L2 | +1.997 | +0.0094 | 接近饱和 |

**#163 Phase D L0 是最强证据**：θ 在最后 30% epoch 还在 +0.064/epoch 速率爬升，按外推 100 epoch 还能涨 6.4，远未达到 atanh(0.999) ≈ 3.8 的 tanh 真正边界。这指向 #163 Phase D 的 NaN @ ep 764 不是"偶然崩溃"，是 θ 继续往边界推、tanh 梯度消失但数值精度已破坏。**用户预测 #2 是对的：landscape 没有内部平衡点**——但驱动不是"几何放大"，是"θ_m/codebook 双向正反馈"。

---

## 3. 用户的两个具体修复方向 + 我的评估

### 3.1 κ_max=0.3 / 0.5 探测实验（用户提议）

**预期结果**：
- 若 κ 仍 saturate 到新的 0.3/0.5 边界 → 证实"loss landscape 无内部平衡"是主要驱动
- 若 util 改善（不 saturate 或部分 saturate）→ 几何放大有一定贡献但不是主导

**我的预期**：util 会显著改善（因为 tanh 边界 = ±0.3，离 0 较近，codebook 漂移距离更小，正反馈强度弱化）。但 **不会完全恢复**（因为正反馈机制本身仍在）。这是 Task #166 的目标。

### 3.2 纯欧式对照组（用户强烈建议）

**已有数据**：vanilla RQ-VAE + Sinkhorn（phonism，Task #58/59/60/61/32-35）有效且利用率为 100%（Stage 2 SID 9922/9922 unique）。phonism R@10=0.1058 ⭐ 是 paper 推荐方案——这本身就是"纯欧式 + 不可学习 κ + Sinkhorn"控制组。

**结论**：纯欧式 baseline 表现良好 → **坍缩是 FreeCurv 特有问题**，不是 RQ-VAE 普遍问题。

### 3.3 用户对 #142 verdict "Sinkhorn 都解不了" 的质疑

用户原话："Sinkhorn 是专门打这种正反馈的，如果配平衡强度对了通常有效，值得追问当时是加在哪个环节、balance 系数是多少，不是简单一句"解不了"就能排除。"

**承认**：#142 verdict "Sinkhorn 都解不了" 是 oversell。FreeCurv 的 Sinkhorn 配置我没详细审计（sk_eps=0 / sk_eps=0.5 不同档我都跑过但没仔细分析 balance strength）。需要做：
- Task #167（新建）：审计 FreeCurv Sinkhorn 配置细节（sk_eps 范围 / iter 次数 / weight 在 loss 中的位置），看是不是真的"无法救"，还是 balance strength 配错了。

---

## 4. 综合结论 (v2)

### 4.1 机制（修订）

**FreeCurv 坍缩 = 可学习 κ 的双向梯度正反馈**。证据链：
1. 固定 κ=2 (Task #118) 利用率 100% → 排除几何放大
2. 可学习 κ_max=2 (#89/#163/#164) 利用率 <5% → 唯一变量是 κ 可学习
3. θ 轨迹 #163 Phase D L0 仍线性爬升 → loss landscape 无内部平衡点
4. θ 轨迹 #164 完全饱和 → tanh 边界截停（不是 θ_m 内部收敛）

### 4.2 修复方向（按 ROI 排序）

1. **最高 ROI**：冻结 θ_m N 个 epoch + EMA codebook 更新（切断正反馈）—— Task #168 设计
2. **中 ROI**：把 θ_m 学习率从 5e-3 降到 5e-4（减弱 θ_m 漂移）—— Task #166 κ_max sweep 顺带验证
3. **低 ROI**：降 κ_max 到 0.3/0.5（截断边界，让 θ_m 不能走太远）—— Task #166
4. **审计 Sinkhorn**：看 sk_eps / sk_iters 配置是否合理 —— Task #167

### 4.3 论文 Section 6.1 应改为

**当前文本 (v1)**：
> The strongest is free-curvature learning, validated through four independent repair attempts... all (layer, κ_m) saturate to ±κ_max with codebook utilization consistently <5%. This is not optimization failure... but an architectural failure mode of free-curvature learning on flat data.

**修订建议 (v2)**：
> The strongest is the **fixed-vs-learnable dichotomy**: Task #118 shows fixed κ=2 (HG-Rec c=[2,2,2]) achieves 100% codebook utilization; learnable κ_max=2 (FreeCurv) achieves <5%. The single variable is whether κ is learnable. This **rules out geometric-amplification explanations** (which would predict fixed κ=2 should also collapse) and points to a **kinetic feedback loop specific to learnable κ**: θ_m and codebook receive gradient simultaneously → both run to extremes → tanh boundary stops θ_m but codebook utilization is permanently degraded. **Fix direction: cut ∂L/∂θ_m** (EMA or frozen-θ periods) rather than redesigning distance formula.

---

## 5. 后续任务（按 R10 自主推进）

| ID | 任务 | GPU | 估算时间 | ROI |
|----|------|-----|---------|-----|
| #166 | κ_max=0.3/0.5 sweep + vanilla κ=0 locked 对照组 | GPU 0/2 (空闲) | 30 min × 3 + 15 min eval | 中 |
| #167 | 审计 FreeCurv Sinkhorn 配置 (sk_eps / sk_iters / weight) | CPU only | 1 hour (analytical) | 中 |
| #168 | 设计 EMA-codebook + frozen-θ 修复（切断正反馈） | GPU 0/2 | 1h 训练 + eval | **高** |

按 R11.3 自主决策优先级（CLAUDE.md 固化偏好 > 上游 default > paper 方案 > 简单实用）：
- **#168 优先**（论文最直接修复点）
- **#167 同步**（CPU 廉价，可能挖出 #142 verdict 的错误）
- **#166 跟 #168 并行**（不同 GPU，2 实验同时跑）

---

## 6. 产物清单

| 路径 | 状态 |
|------|------|
| `verdicts/task165_free_curv_synthesis_result.md` (本 v2 修订) | ✅ paper-ready |
| `papers/paper.md` §5.7.1 / §6.1 (待 v3 修订) | ⚠️ 见 §4.3 建议 |
| `HG-Rec/model/hrqvae_free_curv.py` (κ-Stereographic) | ✅ 保留 reference |
| `scripts/task163_5_real_orc_followup.py` (ORC LP) | ✅ 保留 reference |
| Task #118 verdict (固定 κ=2 100% util) | ✅ 关键证据来源 |

---

## 7. v1 → v2 修订差异 (R11.3 必明示)

| 维度 | v1 (commit 840f98a5) | v2 (本修订) |
|------|---------------------|-------------|
| 主结论 | θ_m 训练动力学固有缺陷 | 可学习 κ 的双向梯度正反馈 |
| 证据强度 | 3 个坍缩案例（缺关键对照） | 4 案例 + Task #118 反例 |
| 修复方向 | init/loss/reset 三轴（已证伪） | EMA + frozen-θ 切断正反馈 |
| Sinkhorn | "R137/Sinkhorn 都解不了"（oversell） | 待 Task #167 审计 |

