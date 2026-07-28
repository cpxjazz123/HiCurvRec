# Task #209 A3 Stage 4 — Test Eval Verdict

> **完成日期**: 2026-07-26
> **状态**: ❌ **NO-GO** (test R@10 = 0.0863, vs HG-Rec baseline 0.1020, **-15.4%**)

---

## §1 测试结果 (test split, beam_size=20)

| 指标 | A3 (本任务) | HG-Rec baseline (#84) | Δ vs baseline | 决策阈值 |
|------|------------|----------------------|----------------|----------|
| Recall@5 | 0.0735 | 0.0816 | -9.9% | - |
| **Recall@10** | **0.0863** | **0.1020** | **-15.4%** | > 0.1020 GO, ≤ NO-GO |
| Recall@20 | 0.1053 | 0.1279 | -17.7% | - |
| NDCG@5 | 0.0657 | 0.0690 | -4.8% | - |
| NDCG@10 | 0.0698 | 0.0755 | -7.5% | - |
| NDCG@20 | 0.0746 | 0.0821 | -9.1% | - |

**结论**: ❌ **NO-GO** — A3 在所有 6 个 test 指标上**全面低于** HG-Rec baseline, 最大差距 -17.7% (Recall@20).

---

## §2 Val vs Test 过拟合分析

| Split | Recall@10 | NDCG@20 |
|-------|-----------|---------|
| Val (best epoch 58) | **0.1050** | **0.0882** |
| Test | 0.0863 | 0.0746 |
| **Gap** | **-17.8%** | **-15.4%** |

**关键观察**:
- val R@10=0.1050 看起来**高于** HG-Rec baseline (0.1020), 但 test R@10=0.0863 **低于** baseline.
- val/test gap 接近 -18%, 提示**严重过拟合** — path_reg hyp + ρ_targets + soft norm 组合让 RQ-VAE 在 train/val 上拟合到了某种**虚假几何模式**, 但该模式不能泛化到 test.

**根因假说** (per Task #209 Phase 1 verdict):
- 软约束 + path_reg 让码字沿测地线聚集, 但产品 manifold 的 hyp subspace 仍被推到 boundary (‖x‖_E ≈ 1).
- T5-mini 学到了这种"边界附近聚集的 SID 模式", 但 test items 跟 train SID 几何分布有差异, 边界聚集反而限制了泛化.
- 跟 Task #205 "c=100 mode collapse" + Task #203 "scale_norm REFUTED" 一致 — boundary saturation 不带来泛化收益.

---

## §3 A3 训练配置 (本任务参数)

| 字段 | 值 |
|------|-----|
| RQ-VAE 配方 | --norm_target 1.0 1.35 1.70 + --gamma_norm 0.1 + --scale_norm poincare + --w_path 1.0 --path_geometry hyp --rho_targets_path 2.0 2.7 3.4 |
| Stage 1 ckpt | task209/phase1_arm_A3/best_collision_model.pth (val collision 0.9999, **mode collapse**) |
| Stage 2 SID | Instruments_t5_rqvae_task209_A3.npy (99.97% 唯一性, **Sinkhorn-balanced identity-like**) |
| T5 架构 | T5-mini 9.18M: 6 enc + 4 dec, d_model=128, num_heads=6, vocab=11000 |
| Best epoch | 58 (early stop @ 95, counter 20) |

---

## §4 历史教训 (与 Phase A/B 联动)

A3 NO-GO 闭环了 **Task #209 软约束 + path_reg hyp** 完整 path. 累积证据链:

| Task | 阶段 | 结论 | 关键指标 |
|------|------|------|----------|
| #209 Phase 1 (硬投影) | Stage 1 | ❌ collision 0.9999 | 4 臂全 mode collapse |
| #209 Phase 1 (软约束) | Stage 1 | ⚠️ collision 健康 | 但 val/test gap -18% |
| #209 A3 (本任务) | Stage 4 | ❌ test R@10 -15.4% | 全面低于 baseline |
| #210 Phase A scan | - | 几何探索 | 找到可行区 (λ≥4.7, util≥0.9) |
| #210 Phase B B1 | (进行中) | TBD | product_manifold only |

**核心结论**: **path_reg hyp + 软 norm + product_manifold 三者叠加** 即使在 val 上有 0.1050 (vs 0.1020), 在 test 上仍然 NO-GO. **路径正则化对 generalization 没有帮助**.

---

## §5 关键决策 (R11.3)

**主决策**: A3 作为 Task #209 Phase 1 最终评估完成, 写入 NO-GO verdict.
**后续**: 不再花 GPU 重跑 A3 变体. 把 GPU 留给 Phase B B1 (product_manifold only) 看**去掉 path_reg 后是否更好**.

**下一步** (per R10 主动推进):
1. 等 B1 Stage 3 训练完成 (~80 min)
2. B1 Stage 4 test eval
3. 写 Phase B verdict (B1 real R@10 + B2/B3 NO-GO 根因)

---

## §6 当前产物

- `products/task209/t5small_A3/Instruments/Jul-26-2026_18-40-36/HG_Rec_best.pth` (27 MB, best epoch 58)
- `verdicts/task209_A3_test_metrics.json` (机器可读)
- `verdicts/task209_A3_test_result.md` (本 verdict)
- `logs/task209/A3_stage4_eval_jul-26-2026_19-51-49.log` (eval 日志)

**result:** ❌ Task #209 A3 test R@10=0.0863 (-15.4% vs HG-Rec baseline 0.1020). val/test gap -18% 提示 path_reg hyp 软约束版有过拟合. 路径正则化对 generalization 无帮助, NO-GO 闭环 Phase 1 软约束路径.
