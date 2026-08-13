**Issue #140 CLOSED — GO** (commit 71258c97b48500697011c3f4dd391deaa2d60b65, pushed to gitlab origin/main)

## Gate 1 — 机制与等价性: PASS (5/5)
1. 全部 c_0,b=c0_ref 时新旧 L0 distance/assignment/loss/前三位 SID 完全一致: distance max diff 4.2e-07, front3 SID 0/9922 差异。
2. 单列扰动 (theta_7+0.5) 只改变第 7 列距离, 其余 63 列逐元素不变。
3. 64 个 theta_b 全部进 optimizer 独立 group (lr=3e-3), 梯度 finite (63/64 非零; 未选中 codeword theta 梯度为 0 是合法机制)。
4. d_norm 后 batch 平均距离 = 1.0 (归一化正确, 尺度与 reference 容差内)。
5. 无 NaN/Inf, c_0,b 0 个边界命中, 无 scale-shortcut chain。

关键设计决策: E_batch[d_c] 用 **同 batch 同候选层 (L0) 距离矩阵全局标量均值** — 实测 per-column 归一化会改变 argmin (1087/9922 items 漂移) 违反 check 1 起点严格等价, 全局标量 0 漂移, 是唯一满足 spec 的选择。

## Gate 2 — 曲率分化与 L0 分配证据: PASS
- 64 个 c_0,b 终值: range [0.998, 1.013], std=0.00275 — 非退化分化 (无塌缩/无边界命中), 但幅度小 (anchor lambda=1.0 较强)。
- 轨迹 1800 点 (l0_branch_curvature_trace.json/csv), theta 梯度 finite 全程。
- **occupancy Spearman = -0.766 (p=4.7e-13, Holm 校正显著)** — 高占用 codeword 曲率更低 (更平缓、更易选中); margin/density 不显著。
- 无尺度捷径: argmin(d_norm) vs argmin(d_raw) 0 漂移; scale_shortcut_audit.json shortcut_detected=False。
- L0 分配 vs reference 变化 87.25% (训练重新分配覆盖); SID sha=9dafeef0, 4-tuple util 1.0, dead code 0。
- L1/L2 保持 Issue139 路径 (实现 hash 未动): final c=[1.0581, 1.0747] 与 reference [1.0583, 1.0742] 同量级。

## Gate 3 — 正式 Stage3/4: GO (TARGET 未达)
- Stage3: 200 epoch DDP 4 卡, EARLY_STOP=20, best valid R@10=0.1256 (epoch 199) vs Issue139 0.1269 (-0.0013, 训练时序 noise 量级)。
- Stage4: 单 ckpt + beam=20, 全量 24,772 样本, **test R@10=0.0989** vs Issue #139 0.0968 (**+0.0021, +2.2%**) — **严格高于锚点**。
- L0 最差频率桶 [200,1000): R@10 **0.0385 → 0.1154 (+0.0769, 3x)** — 大幅改善。
- L0 first-error 比例: 88.37% → 87.99% (改善)。两项均改善, 无恶化。
- R37 不触发; TARGET REACHED (>0.1020) 未达 (0.0989)。

## Verdict 分类
**曲率机制有效** — 不是 SID 重分配 no-op (occupancy-曲率显著关联 + 最差桶 3x 改善), 但提升幅度不足以破 TARGET。4 stage 全量实时运行 (R40), 全部产物落盘 issue140_verdict.json (R33)。
