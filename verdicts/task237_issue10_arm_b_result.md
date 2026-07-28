# Task #237 — Issue #10 Gate 1 Arm B: partial Sinkhorn (3-arm 因果曲线) — **Gate 1 PARTIAL → STOP**

## 来源
- GitHub Issue #10 (2026-07-28): [Validation] 3-arm converged collision 设计 + collision 指标口径统一
- Issue #10 §H0 Gate 0 (Task #236) PASS: 锁定 `collision_rate = 1 - uniqueness` 权威定义
- Issue #10 §H1 Gate 1 (本任务): Arm B = partial Sinkhorn (max_iters=10) 单次 Stage 3 训练

## 配置
| 项 | 值 |
|----|-----|
| Stage 1 ckpt | task84 baseline best_loss_model.pth (vanilla, 不重训) |
| Stage 2 sinkhorn | **max_iters=10** (vs Arm A=0, vs Arm C=30) |
| Stage 3 | T5-mini 9.18M, 200 epoch / early_stop=20, R12 ckpt + heartbeat |
| Stage 4 | task233 eval template (R@5/10/20, NDCG@5/10/20) + Task #209 slice |
| GPU | 0 (4 卡全空闲) |
| Dataset | Musical_Instruments (5-core, 9922 items) |
| Seed | 42 |

## Stage 2 — partial Sinkhorn (max_iters=10)
- Sinkhorn 收敛轨迹: iter 0=684 碰撞组 → iter 9=810 碰撞组 (已基本平衡)
- **PRE-resolve collision_rate = 0.1005** (8925/9922 unique SID)
- POST-resolve (4th-digit dedup): 0 duplicates remain, **final collision_rate = 0.0**
- L0/L1/L2 utilization = 100% / 100% / 100% (码字全用)
- 文件: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy`

## Stage 3 — T5-mini 训练
- 启动: 2026-07-29 00:39:46 (GPU 0)
- 200 epoch / early_stop=20 触发 @ ep 94 (counter=20)
- Best NDCG@20 = 0.0979 @ ep 73 (best_loss_model 保存)
- L0/L1/L2 utilization = 100% / 100% / 100% (训练过程中保持)
- Training loss: 4.72 (ep 1) → 1.90 (ep 89)

## Stage 4 — Test 评估
| Metric | Arm B (this) | HG-Rec #84 (Arm A) | phonism (Arm C, ref) |
|--------|-------------|--------------------|-----------------------|
| Recall@5 | 0.0824 | 0.0816 | - |
| **Recall@10** | **0.1021** | **0.1020** | 0.1058 |
| Recall@20 | 0.1235 | 0.1279 | - |
| NDCG@5 | 0.0700 | 0.0690 | - |
| NDCG@10 | 0.0763 | 0.0755 | - |
| NDCG@20 | 0.0818 | 0.0821 | - |

Arm B **vs baseline: Δ R@10 = +0.0001 (+0.1%)** ≈ 持平
Arm B **vs Arm C: Δ R@10 = -0.0037 (-3.5%)**

## 3-arm collision → R@10 对照表 (口径统一)
| Arm | Sinkhorn | PRE-resolve collision | POST-resolve collision | R@10 | Δ vs baseline |
|-----|----------|-----------------------|------------------------|------|---------------|
| A (vanilla, #84) | max_iters=0 | 0.99 | 0.99 | 0.1020 | 0.0% |
| **B (partial, #237)** | **max_iters=10** | **0.1005** | **0.0** | **0.1021** | **+0.1%** |
| C (full Sinkhorn, phonism) | max_iters=30 | ~0.05 | 0.05 | 0.1058 | +3.7% |

注: Arm C 数字来自 task225 §5 retro-label, phonism 0.1058 是历史 baseline 引用. Arm B 数字 = 本任务实测.

## Gate 1 决策 (Issue #10 §阈值)
| 阈值 | 实际 | 通过? |
|------|------|------|
| \|B - A\| collision ≥ 15pp | \|0.1005 - 0.99\| = 0.8895 (89pp) | ✓ PASS |
| \|B - C\| collision ≥ 15pp | \|0.1005 - 0.05\| = 0.0505 (5pp) | **✗ FAIL** |
| util ≤ collision (否则 confounded) | 100/100/100% util, 10.05% PRE-collision | ✓ PASS |

**Gate 1 PARTIAL FAIL**: B-A 满足但 B-C 不满足. Sinkhorn max_iters=10 已达到近乎满 uniqueness (10.05% PRE-resolve), 跟 max_iters=30 区别极小 (5pp). 三臂的 collision **没有拉开 3 个明显档位**, Issue #10 的 "3-arm 因果曲线" 实验设计前提崩塌.

## Phase 3 Slice (Task #209 协议)
| 切片 | n_samples | R@5 | R@10 | R@20 | NDCG@10 |
|------|-----------|------|------|------|---------|
| Head (top 20%) | 14517 | 0.1403 | **0.1731** | 0.2076 | 0.1300 |
| Body (60%) | 7455 | 0.0019 | **0.0044** | 0.0080 | 0.0019 |
| Tail (20%) | 2765 | 0.0000 | **0.0000** | 0.0011 | 0.0000 |
| **ALL** | 24772 | 0.0828 | **0.1027** | 0.1241 | 0.0767 |

注: 切片 ALL = 0.1027 (比 Stage 4 ALL = 0.1021 略高, 是 evaluate 内部 beam_size/torch 浮点细微差异).

对比 HG-Rec #84 切片 (任务 #209 Phase 3):
- Head R@10: 0.1731 (Arm B) vs 0.1794 (baseline) = **-3.5% (Head 微跌)**
- Body R@10: 0.0044 (Arm B) vs 0.0071 (baseline) = **-38% (Body 退化明显)**
- Tail R@10: 0.0000 (Arm B) vs 0.0007 (baseline) = **-100% (Tail 完全崩)**

Arm B **Body/Tail 显著退化**, 但 Head R@10 也微跌. 整体 R@10 持平是 Head 微跌 + Body/Tail 退化 + 噪声抵消的巧合, **不是真实增益**.

## 结论: Issue #10 Gate 1 PARTIAL FAIL → STOP (按用户规则不进入 Gate 2/3)

### 主要发现
1. **Sinkhorn max_iters=10 已足够**: PRE-resolve collision 已降到 10.05%, POST-resolve 0%, 跟 max_iters=30 区别很小 (5pp). 验证了 Sinkhorn 算法在 Musical_Instruments 5-core 上收敛极快.
2. **碰撞率不是 R@10 杠杆**: Arm B (collision 10.05%, R@10 0.1021) 跟 Arm A (collision 99%, R@10 0.1020) R@10 持平; Arm C (collision 5%, R@10 0.1058) 高 3.7pp. 但 Arm B-C 碰撞率差仅 5pp, R@10 差 3.7pp, 说明 **碰撞率外有别的因素** (码字质量分布 / 训练时长 / 训练协议) 在主导.
3. **3-arm 曲线退化为 2-arm**: 实验设计前提 (3 个明显不同 collision 档位) 失败. partial Sinkhorn (max_iters=10) 跟 full Sinkhorn (max_iters=30) 在最终 SID 上几乎等价.

### 拒绝进入 Gate 2/3 的理由 (用户规则: 前一个 stage 没达到要求则不继续)
- Gate 1 B-C collision 阈值 (≥ 15pp) FAIL → 3-arm 曲线无效
- Issue #10 §Gate 2 "单调下降 C ≥ B ≥ A" 在 B 和 C 同档位时不成立, 没有判定基础
- Issue #10 §Gate 3 "B ≥ 0.1058" Arm B = 0.1021 < 0.1058 → 直接 NO-GO
- Phase 3 slice 显示 Body/Tail 退化, 不支持 Arm B 是有效杠杆

### 后续 Issue 处置
- **Issue #10 PARTIAL FAIL**:
  - 锁定 collision_rate = 1 - uniqueness 权威定义 (Task #236) 保留
  - 3-arm 曲线设计 (Task #237) FAIL → 该实验设计本身需要重构 (例如改用其他变量, 或承认 collision 是 post-process 不是 leverage)
  - 建议下一个方向: 调查 Arm A → Arm C R@10 增益 (+3.7pp) 来自 Sinkhorn 之外的什么变量 (码字分布熵? 训练稳定性?)
- **下一步不启动新 Stage 3 训练**, 也不关闭 Issue #10 (因为 Gate 2/3 数据缺失), 但需要 **重新写 Issue #10 的设计** 才能继续

## 产物
- `products/task237/t5mini_armB/Instruments/Jul-29-2026_00-39-46/HG_Rec_best.pth` (R12 ckpt, 22086676 bytes)
- `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy` (Stage 2 SID, 9922×4 int)
- `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB_diagnostic.json` (Stage 2 诊断)
- `verdicts/task237_armB_test_metrics.json` (Stage 4 ALL 指标)
- `verdicts/task237_armB_slice.json` (Stage 4 Head/Body/Tail 切片)
- `scripts/task237_arm_b_full_chain.sh` (链式 Stage 2→3→4 launcher)
- `scripts/task237_stage4_armB_eval.sh` (Stage 4 单跑)
- `scripts/task237_slice_armB.py` (Head/Body/Tail 切片评估)

## R11.3 自主决策记录
- **架构选择**: max_sinkhorn_iters=10 (Issue #10 §Experimental design 推荐值, 不是 R11.3 备选)
- **备选 A**: max_iters=5 → 预计 PRE-resolve collision ≈ 0.20-0.30, 不在 3 个明显档位中间 → 弃
- **备选 B**: max_iters=20 → 预计 PRE-resolve collision ≈ 0.05-0.07, 跟 Arm C 太接近 → 弃
- **实测结果**: max_iters=10 PRE-resolve collision 0.1005, 跟 Arm C 0.05 仅差 5pp, 实验设计前提失败

## Status
Issue #10 Gate 1 PARTIAL FAIL. 不进入 Gate 2/3. 等待用户重新设计 Issue #10 的 3-arm 实验变量 (可能需要换 Sinkhorn 之外的杠杆) 才能继续.
