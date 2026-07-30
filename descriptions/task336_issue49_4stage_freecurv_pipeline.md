# Task #336 — Issue #49 — FreeCurvHRQVAE 完整 4 阶段流水线

**状态**: 进行中 (2026-07-30)

## 目的

基于 Issue #47 修复后的统一 κ-stereographic 公式，首次在"公式干净"前提下测试 per-layer 可变曲率。
包含测地初始化 + κ-codebook 解耦调度 + 对称初始化 (3θ 臂) + Stage 1→2→3→4 全流程执行。

## 关键设计

1. **统一公式**: Issue #47 修复（Möbius addition 符号修复 + tan_κ⁻¹ NaN 修复）, 直接硬编码在 `hrqvae_free_curv.py` 中
2. **测地 kmeans**: `init_emb_geodesic()` 为每个码字层做双曲测地 kmeans 初始化
3. **κ-Codebook 解耦调度**:
   - Phase A (epoch 1-200): θ 冻结于初始值, codebook 正常训练 (Sinkhorn enabled, sk_eps=0.01)
   - Phase B (epoch 201-400): θ 解冻 + 小 lr (1e-5) 学习, codebook 持续训练
4. **3 初始化臂**: θ_init = {+0.02, -0.02, 0.0} (正/负曲率/零曲率对称)

## Stage 1 Gate 1 标准

- (a) L0/L1/L2 utilization ≥ 90%
- (b) three-digit collision_rate ≤ 0.20
- (c) 至少一组 κ_m 最终值非饱和 (|κ| ≪ κ_max=2.0) 非退化 (|κ| > 0.01)

## Stage 2-4

- Stage 2: Sinkhorn 推断 (5 iter) + 4-digit dedup → (9922, 4) SID
- Stage 3: T5-mini 200 epoch 训练 (复用 `task84_hgrec_stage3_train.py` 协议)
- Stage 4: 评估 R@10 对比 baseline 0.1020 (决策阈值: > 0.1022 GO, ≤ 0.1022 NO-GO)

## 产物

| 阶段 | 路径 |
|------|------|
| Stage 1 训练 | `products/task336/{arm_plus,arm_minus,arm_zero}/` |
| Stage 2 SID | `products/task336/{arm}/sid_*.npy` |
| Stage 3 T5 | `products/task336/{arm}/t5_*/` |
| Stage 4 评估 | `products/task336/{arm}/eval_*.json` |
| Verdict | `verdicts/task336_issue49_result.md` |
