# Task #231 — Issue #6 调参版 (Per-Codeword κ c_k range sweep) 结果

## 结论
**🟡 PARTIAL SUCCESS — L0 OPEN 了, 但 L1/L2 退化到接近 FAIL 阈值 (>92%).**

| Layer | Uniform(0.5,20) baseline | Uniform(0.5,5) | Uniform(1,5) |
|-------|--------------------------|----------------|--------------|
| L0 (K=64)  | 28.90% ± 2.15% TOO_STRONG | **81.12% ± 0.48% OPEN** ✅ | **82.68% ± 0.29% OPEN** ✅ |
| L1 (K=128) | 67.15% ± 0.22% OPEN      | 92.37% ± 0.30% OPEN      | 93.18% ± 0.36% OPEN      |
| L2 (K=256) | 75.69% ± 0.79% OPEN      | 94.04% ± 0.17% OPEN      | 94.60% ± 0.21% OPEN      |

**Issue #6 提议部分成立**: 缩窄 c_k range 让 L0 从 28.90% 跃迁到 81-83% (跨进 OPEN band), **但** L1/L2 也跟着涨到 92-95% (逼近 FAIL 阈值 95%). Uniform(1,5) 比 Uniform(0.5,5) 略高 (L0 +1.56pp, L1 +0.81pp, L2 +0.56pp), 极接近临界点.

## 关键发现
1. **Issue #6 假设成立 (L0 方向)**: c_k spread [0.5, 20] 太宽, 少量大 c_k codewords (5/64 = 7.8% of L0) 几何主导整个 L0 分配. 缩窄到 [0.5, 5] 或 [1, 5] 移除这些 outlier, L0 一致率立即从 28.90% 进 81-83%.
2. **Joint constraint 陷阱 (Issue #6 自己预测)**: L1/L2 一致率随 c_k range 缩窄而单调上涨 (L1: 67%→92%→93%, L2: 76%→94%→95%). 同一参数同时影响三层, 没有"只动 L0 不动 L1/L2"的 c_k range sweep.
3. **Pareto front**: 三个 range 都在 Pareto front 上. Uniform(1,5) L0 OPEN 但 L2 = 94.60% (距 FAIL 0.40pp). 进一步缩窄 (e.g., Uniform(1,3) 或 Uniform(2,4)) L2 会 FAIL — **缩窄 c_k range 不是完整修复**, 只把 L0 从 TOO_STRONG 拉进 OPEN 的同时让 L1/L2 离 FAIL 更近.
4. **机械解释**: c_k range 缩窄 → 1/sqrt(c_k) weight 缩窄 → per-codeword distance formula 退化为更接近 baseline 欧式 argmin → 一致率 ↑. L0 受益最大因为 K=64 (少量 outlier 主导最强). L1/L2 K=128/256, outlier 影响被稀释, 一致率 baseline 已经 OPEN, 缩窄只是把它们推到临界.

## 决策
- **🟡 PARTIAL GO**: Issue #6 提议**部分成功** — L0 OPEN band 验证成功 (Issue #6 的核心假设成立).
- ❌ **不推荐**继续缩窄 c_k range (L1/L2 已经接近 FAIL, 边际 ROI 负).
- ✅ **推荐 hybrid**: 跟 Issue #7 (Task #232 Gromov weight=0.5 三层全 OPEN) 组合 — L0 用 Per-Codeword κ (Uniform(1,5)) 保留 L0 OPEN, L1/L2 用 Gromov weight=0.5 替代 Per-Codeword κ (因为 Gromov 在 L1/L2 比 PC κ 更稳定地 OPEN).

## 产物
- scripts/task231_pck_phase0_ckrange_sweep.py (3 ranges × 3 layers × 3 seeds = 27 measurements)
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task231_pck_phase0_results.json
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task231_pck_phase0.log

## 跟 Task #218 / #219 / #220 / #221 的关系
- Task #218 (Stage 1 PC κ 训练): PARTIAL SUCCESS (epoch 29 healthy, epoch 35+ 坍缩) — 训练动力学不支持
- Task #220 (Stage 1 PC κ 训练, task218 verdict 跟踪): PARTIAL SUCCESS (跟 task218 是同一实验, 不同 task number)
- Task #219 (Gromov Phase 0 forward-pass, weight=1.0): L0 OPEN, L1/L2 TOO_STRONG
- **Task #231 (本次)**: PC κ c_k Uniform(1,5) 让 L0 OPEN, 但 L1/L2 涨到 93-95%
- **Task #232 (本次)**: Gromov weight=0.5 让三层全 OPEN (74-90%) — **Pareto 最优**

**整体进展**: 8 个测试方向 (Issue #1-3, 5 + Task #218-232) 中, **Task #232 (Gromov weight=0.5) 是首个三层全 OPEN 方案**. Issue #6 单独不是完整修复, 跟 Issue #7 组合才有完整 SID escape.

result: Task #231 PARTIAL SUCCESS — Issue #6 假设成立 (L0 OPEN via c_k range narrowing), 但 L1/L2 退化到接近 FAIL. 推荐 hybrid: Issue #6 (PC κ Uniform(1,5)) + Issue #7 (Gromov weight=0.5).
