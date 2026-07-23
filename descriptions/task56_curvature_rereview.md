# Task #56 — 曲率重审 / Task #85 三几何清算

> **任务目的**: 在 dist_kappa κ→0 L'Hôpital 边界修复 (threshold 1e-2) 后, 重审 Task #85 三几何独立 SID (m=0 球面 / m=1 准欧氏 / m=2 双曲) 的曲率机制, 看是否改变原 verdict
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #85 终局 verdict (m=0 球 R@5=0.0174 / m=1 准欧氏 R@5=0.0200 / m=2 双曲 trivial bias) + 战线一 Campaign 修复:
1. Task #85 三几何训练时, dist_kappa κ→0 处仍会得 2× 欧氏距离 (L'Hôpital 边界 bug)
2. 修复后 (threshold 1e-2), 可学习 κ 完全可能漂过 0 时仍给欧氏距离, 几何决策更稳定
3. Task #85 训出的 κ 值需要重审 (看是否曾漂过 0)

## 2. 实验设计

**变量**: dist_kappa κ→0 行为 (旧: 2× 欧氏 / 新: 1× 欧氏)
**保持不变**:
- Task #85 三子空间 SID tensor (sid_subspace_0/1/2.pt)
- TIGER 训练配置
- seed=42

**启动命令**:
```bash
# Stage A: 复现 Task #85 Stage 2.1/2.2 用新 dist_kappa (rerun RQ-VAE, ~2 h)
# Stage B: 用 Task #85 已训好的 SID tensor + 新 TIGER 推断 + Recall@5 (~30 min)
# Stage C: 对比原 R@5 (m=0: 0.0174, m=1: 0.0200) vs 新 R@5
```

## 3. 决策触发

| 新 R@5 vs 原 R@5 | 判定 |
|------------------|------|
| 新 R@5 ≥ 原 + 10% | ✅ L'Hôpital 修复显著影响曲率机制, 需重写 Task #85 verdict |
| 新 R@5 ≈ 原 (±5%) | ❌ L'Hôpital 修复对端到端 Recall 无影响, Task #85 verdict 维持 |
| 新 R@5 < 原 -10% | ⚠️ 修复使曲率机制恶化, 撤销修复或调整 threshold |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage A: 重训三子空间 RQ-VAE (用新 dist_kappa) | ~2 h |
| Stage B: TIGER 推断 + Recall@5 评估 | ~30 min |
| Stage C: 对比分析 + verdict | ~30 min |
| **总计** | **~3 h** |

## 5. 风险与缓解

**风险 1**: 重新训练三子空间 RQ-VAE 与 Task #85 历史 baseline (m=1 0.0200) 不完全可比 (新代码 vs 旧代码) → 缓解: 复用 Task #85 训练数据 + 固定 seed, 仅改 dist_kappa
**风险 2**: 修复后新 R@5 仍低于 Task #87 fused baseline (0.0194) → 缓解: 这是预期结果 (单 κ 信息量 < fused), 修复 verdict 仅在数字层面
**风险 3**: Task #85 ckpt 已不存在 → 缓解: 仅需重训 Stage 2 RQ-VAE (无需重训 TIGER)

## 6. 完成度跟踪

- [ ] 验证 Task #85 历史 κ 值 (查 log) 是否曾漂过 0
- [ ] Stage A: 重训 m=0/1/2 三子空间 RQ-VAE (用新 dist_kappa)
- [ ] Stage B: TIGER 推断 + Recall@5
- [ ] Stage C: 对比 verdict (新 R@5 vs 原 R@5)
- [ ] 写 verdict (Task #85 verdict v3 / 曲率重审结果)
