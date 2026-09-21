# iter6 Direction Decision (Agent B: direction-judge)

**任务**: 评判 Agent A 提出的 P1/P2/P3 候选机制的方向合理性,产出唯一推荐。

**iter5 dominant bottleneck** (来自 Agent A 报告): attention temperature_scale 经 softplus + cyclic_factor.detach() + +0.25 bias 三联把 cyclic c(t) 联动淹没,attention logits 量级远小于 distances 量级,机制未产生预期直接效应 DE-1。

**iter5 forbidden next directions**:
1. 再加 attention / codebook attention
2. softplus 长漂路径
3. cyclic_factor.detach() 截断反向
4. +0.25 bias 隐藏 cyclic 振幅
5. 任何隐藏 cyclic c(t) 振幅的常数偏置

**baseline 对照**:
- iter11 (sk_L0=0.50): test_R@10=0.0602 (+1.86% vs TIGER 0.0591), 几何指纹 L0_util=0.22 (强 collapse) + H(L1|L0)=4.39 (强 diversity)
- v318 cyclic-c |sin| baseline (Instruments domain): test_R@10=0.1179, Gini 0.038, collision 3.9%
- 17 iter LOO R² = +0.024 (geom_L0_util ρ=-0.69)
- 0.065 hard target, 当前 best 0.0602 (差 ~8%)

---

## 评分表

| 维度 (满分 10) | P1 Sinkhorn linear ε-anneal | P2 Lorentz manifold 替换 | P3 Per-item π(c\|item) 软路由 |
|----------------|----------------------------|------------------------|---------------|
| (a) 上轮失败根因明确修复路径 | **8** | **8** | **8** |
| (b) 2023+ 论文实证 | **9** (Liu 2026 EPH-ASC, Geneva 2022) | **9** (Nickel 2018, HG-Rec 2026, HRQ 2025) | **8** (Weighted-PM 2023, CurvGCL 2024) |
| (c) 与 Poincare + Sinkhorn + M2/M3 + cyclic c(t) 兼容 | **9** (Sinkhorn 已是 v318 栈内组件,改 schedule 而非引入新组件) | **5** (manifold 完整替换,需要重写 KMeans init + encoder/decoder + commit,工程量大,违反"Rotate don't stack") | **7** (per-item 路由是新增路由器,独立模块,可挂载在 c(t) 上) |
| (d) Stage 1 端纯曲率变更 ceiling 风险 | **6** (Sinkhorn ε 调优边际效应 L1 sweep 已证弱 ~2.3%, 17 iter L1 sweep 信号弱暗示该方向接近 ceiling) | **5** (manifold 替换是 Stage 1 端大幅变更, ceiling 风险中,但与 P3 类似属"per-layer 异质"路线族 — 见 iter31 NO-GO) | **4** (per-item 路由维度新但样本量 n=10, 难以脱离 iter31 per-layer 异质 NO-GO 的拖累) |
| (e) **Gap-closing relevance (强制)** | **9** (完全绕过 attention logits 死路, ε(t) = ε_min + (ε_max-ε_min)·\|sin\| 振幅完整 1.0 峰谷差, 与 c(t) 显式乘法耦合可微, 无 detach 无 softplus 无 +0.25 bias, 路径上 DE-1 直接可打印) | **8** (根本不需要 attention logits 路径, assignment 直接 arccosh(-⟨x,y⟩_L/c), c(t) 直接进 arccosh, 无任何截断/偏置, 路径上 DE-1 直接可打印) | **8** (完全不用 attention, π(c\|item) 软路由无 detach 无 softplus 无 bias, c_eff = E_π[c(t)] 完整 sin 振幅放大, 路径上 DE-1 直接可打印) |
| (f) 创新与曲率相关性合规 (6 类 ≥1 + 关键词 ≥1) | **10** (曲率 c 在训练中动态变化 + 几何变换 Sinkhorn ε(t), 关键词: hyperbolic / manifold / curvature 全命中) | **10** (manifold 几何替换, 关键词: Lorentz / hyperbolic / manifold / Riemannian / curvature / distance metric 全命中) | **9** (per-item 异质曲率 + 几何变换, 关键词: curvature / manifold / mixed-curvature / routing 命中) |
| **三态硬资格** | | | |
| Forbidden directions 检查 | ✓✓✓✓✓ (无 attention / 无 softplus / 无 detach / 无 bias / 无 hidden auxiliary) | ✓✓✓✓✓ | ✓✓✓✓✓ |
| Step ≤ 5000 数值证据 | ✓ (step 1000/2500/5000 打印 ε_actual 与 c(t) 相关系数 ≥ 0.95) | ✓ (step 500 Lorentz norm residual < 1e-5; step 1000 距离双峰; step 5000 ⟨x,x⟩_L = -1/c 相对误差 < 1e-4) | ✓ (step 1000 π entropy; step 2500 per-item c_eff 多峰; step 5000 ∂L/∂π 梯度范数 > 0) |
| **总加权分** | **51/60** | **45/60** | **44/60** |

---

## 唯一推荐: **P1 — Sinkhorn linear ε-anneal + Lipschitz 检测** (替代 iter5 指数衰减 + 三联 bias)

### 三行理由

1. **Gap-closing 路径最干净**: P1 完全绕过 iter5 失败的 attention logits 量级失衡死路,改在 Sinkhorn assignment 层做 ε(t) = ε_min + (ε_max-ε_min)·|sin(π t/T)|,振幅完整 1.0 峰谷差且与 c(t) 显式乘法耦合可微,无 detach / 无 softplus / 无 +0.25 bias,路径上无任何截断或常数偏置;ε(t) 与 c(t) 在 step 5000 的相关系数 ≥ 0.95 可作为 DE-1 的硬验证点。

2. **Sinkhorn 已在 v318 栈内**: P1 不引入新组件,只改 Sinkhorn 的 ε schedule (从 iter5 失败的指数衰减 + 三联 bias 改成 Liu 2026 EPH-ASC 提出的线性 stability law + 完整 sin 振幅),与 Poincare + Sinkhorn + M2/M3 + cyclic c(t) 完全兼容,工程量最小,符合"Rotate don't stack"反模式约束。

3. **风险相对可控**: P2 manifold 替换需要重写 KMeans init + encoder/decoder + commit,DDP 4 卡稳定性未知 (v334 KMeans init 卡死教训),且属"per-layer 异质"路线族 (与 iter31 NO-GO 同源);P3 per-item 软路由样本量证明力 n=10 极弱,且 iter31 per-layer 异质 ceiling lock 已锁死 Stage 1 端纯曲率变更天花板;P1 风险最低,Liu 2026 已给出指数衰减→线性 stability 的理论依据。

---

## BACKUP: 无

iter6 任务限定 single iteration 单机制选择,按"Rotate don't stack"反模式严禁 P1+P2 / P1+P3 叠加,且三个候选互斥关系已明示,故不推荐 backup。

---

## 评判签字

- 评判依据: Agent A lit_search_iter6.md (P1/P2/P3 候选) + memory iter11 fingerprint + memory v318 baseline
- 评分硬资格: P1/P2/P3 三态硬资格 (forbidden / step ≤ 5000) 全部 PASS
- 推荐: **P1** (51/60 分最高)
- 完成时间: 2026-09-21
- 评判者: Agent B (direction-judge)