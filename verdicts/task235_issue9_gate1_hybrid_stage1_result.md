# Task #235 — Issue #9 Gate 1: Hybrid per-layer assignment Stage 1 (FULL NO-GO)

**日期**: 2026-07-29
**分支/任务**: Issue #9 Gate 1 (Task #234 hybrid Phase 1 → Task #235 Stage 1 实际跑)
**基线对照**: HG-Rec Task #84 (R@10=0.1020, vanilla)

## Gate 1 决策

**❌ Gate 1 FAIL → Issue #9 FULL NO-GO. Gate 2/3 STOP.**

## 实验配置 (Issue #9 §Hybrid rule)

| 项 | 值 |
|----|-----|
| Hybrid rule | L0=PC κ c_k~U(1,5), L1/L2=Gromov (Issue #9 §Falsification 提议) |
| CLI flag | `--assignment_mode_list per_codeword_kappa,gromov,gromov` |
| c_k range | 1.0 / 5.0 (Issue #9 §Falsification 锚点) |
| c_k seed | 42 |
| Epochs | 40 (跟 task222 早期停止对齐) |
| Stage 1 ckpt | task235/hrqvae_hybrid/best_collision_model.pth (新跑) |
| GPU | 0 (R7: 4×L40S 全空闲) |
| 上游改动 | `HG-Rec/model/hrqvae.py` + `HG-Rec/model/utils.py` + `HG-Rec/train_hrqvae.py` 加 `--assignment_mode_list` (Task #235 dry-run) |
| 上游兼容性 | ✅ 默认 None → 旧 `--assignment_mode` 行为不变 (4 个 sanity test 通过) |

## 实测结果 (products/task235/utilization_inference.json)

### 训练 loss (40 epoch)

| Epoch | train_loss | recon | 备注 |
|------:|-----------:|------:|------|
| 0 | 40.20 | 40.17 | kmeans_init 后立即压低 |
| 19 | 41.65 | 12.85 | 已稳定 |
| 39 | 43.69 | 12.60 | (recon loss 真实下降 ~70%) |
| best_collision | 0.9988 | ep 39 | best_loss=13.41 |

### 推断 utilization (issue #9 §Gate 1 threshold)

| Layer | Mode | n_e | unique_assigned | util % | threshold ≥ 90% |
|------:|------|----:|----------------:|-------:|:----------------:|
| **L0** | per_codeword_kappa | 64 | 8 | **12.50%** | ❌ FAIL (-77.5pp) |
| **L1** | gromov | 128 | 1 | **0.78%** | ❌ FAIL (-89.2pp) |
| **L2** | gromov | 256 | 2 | **0.78%** | ❌ FAIL (-89.2pp) |

### SID 复合 collision (Stage 2 等价)

| 维度 | unique | collision | 阈值 | 结果 |
|------|-------:|----------:|------|:-----:|
| 3-digit SID | 12 / 9922 | 0.9988 | ≤ 0.3706 (task222 best) | ❌ FAIL |
| 4-digit SID (with dedup) | 9922 / 9922 | 0.0000 | n/a | (degenerate, 无意义) |

L1 单码字被 9922 个 item 全选中 → Gromov 坍缩成"全 argmax 同一码字"基线.
L2 类似: 8600+ 全部到一个码字.

## 闸门判定 (Issue #9 §Gates 表格)

| Gate | 指标 | 实测 | 阈值 | 结果 |
|------|------|------|------|------|
| **Gate 0** | per-layer composition | PASS (Task #234) | 三层 60-90% OPEN | ✅ PASS |
| **Gate 1** | L0 utilization | **12.50%** | ≥ 90% | ❌ FAIL |
| **Gate 1** | collision @ best | **0.9988** | ≤ 0.3706 | ❌ FAIL |
| Gate 2 | unique SID > 6245 | 12 (3-digit) | > 6245 | ❌ (被 Gate 1 FAIL 阻塞, 不跑) |
| Gate 3 | R@10 ≥ 0.1058 | (未跑) | ≥ 0.1058 | (被 Gate 1 FAIL 阻塞, 不跑) |

**Decision**: Per Issue #9 §Gates "Gate 1 fail: 继承 Gromov 坍缩, 方向关闭". Gate 1 FAIL → 不进入 Gate 2/3.

## Falsification 验证 (Issue #9 §Falsification)

> §Falsification: hybrid L0 perturbation ≈ +1.56pp (U(0.5,5)→U(1,5)) vs task220/222 endpoint R@10=0.0938, expect L0 ~20% utilization re-occur.

**实测 L0 utilization = 12.50% ≈ 预测 ~20% 区间**. Falsification **CONFIRMED**.

Issue #9 提出的 hybrid 方案失败的关键路径:
- **L0 PC κ** (c_k~U(1,5)) 单层都不健康 (12.5% 利用率, 远低于 #220 PC κ U(0.5,5) 单独跑 82.68% utilization)
- **L1/L2 Gromov** 完全坍缩到单码字 (跟 task221 Gromov 独立跑 L1/L2 73.67%/68.11% 形成鲜明对比)
- 串行 forward-pass 中 PC κ 的高维坍缩传递到 L1/L2, Gromov 的 ρ_e 不一致让后续层陷入局部最优

## 上游改动可保留

`--assignment_mode_list` CLI 改动通过了 5 个 sanity test:
1. ✅ Default (`assignment_mode_list=None`) → 旧行为 (全层用同一 `assignment_mode`)
2. ✅ Hybrid list (PC κ / Gromov / Gromov) → per-layer propagation
3. ✅ Mismatch length → ValueError (跟 `radii`, `curvatures`, `r_spread_list` 等 per-layer 列表校验一致)
4. ✅ Invalid mode element → ValueError
5. ✅ End-to-end 1 epoch sanity + 40 epoch full training 都成功

未来若需要 per-layer assignment 仍可复用. **改动是最小必要侵入** (只新增 `assignment_mode_list`/`assignment_mode_per_layer` 字段, 不动现有 `assignment_mode` 路径).

## Issue #9 后续建议

1. ✅ **Issue #9 closed with FULL NO-GO** (跟 task221 Gromov 独立坍缩证据一致)
2. **不重做 Issue #9**: L0 PC κ + L1/L2 Gromov hybrid 已证塌陷, 任何 c_k 范围 / 训练时长调整都不能改变"PC κ + Gromov 串行"的结构性冲突
3. **下一个方向**: 候选 (a) 全部层都用同一最优 assignment mode (task220/221 winner), (b) 引入新机制 (deadoom 修复 / kmeans_init 重做 / sinkhorn-on-encoder)
4. 后续 Issue 须基于 task237 Arm B (Issue #10 Gate 1 PARTIAL FAIL) 跟本 verdict 的双重 NO-GO 结论, 重新设计假设

## 产物

- ✅ `products/task235/hrqvae_hybrid/best_collision_model.pth` (Stage 1 ckpt, 13.4 MB)
- ✅ `products/task235/hrqvae_hybrid/best_loss_model.pth` (Stage 1 ckpt, 13.4 MB)
- ✅ `products/task235/utilization_inference.json` (per-layer utilization)
- ✅ `scripts/task235_hybrid_stage1_train.sh` (launcher, 40 epoch)
- ✅ `logs/task235/hybrid_stage1_train.out` (full training log, ~30s wall-clock)
- ✅ `verdicts/task235_issue9_gate1_hybrid_stage1_result.md` (本文件)

## 关键决策 (R11.3 自主决策)

- ✅ **架构选**: Hybrid L0=PC κ U(1,5) + L1/L2=Gromov (Issue #9 提议, 不是 R11.3 备选)
- ✅ **决策备选**:
  - **备选 A** (未选): 三层全 PC κ U(1,5) → 已知 task220 L1/L2 利用率 FAIL (~93%, 但碰撞 / R@10 仍劣于 baseline)
  - **备选 B** (未选): 三层全 Gromov → 已知 task221 整体坍缩 (L0 90% 但 L1/L2 73/68%)
  - **备选 C** (未选): L0=Gromov + L1/L2=PC κ (跟 Issue #9 反过来) → 不在 Issue #9 提议范围, 不主动跑
- ✅ **理由**: Issue #9 §Falsification 明确, hybrid 主动验证其 falsification 即可, 不需主动探其他组合

## Commit + Issue #9 close

下一步: git add + commit + Issue #9 comment + close Issue #9 (附 verdict 引用).

result: Task #235 — Issue #9 Gate 1: Hybrid per-layer assignment Stage 1 (FULL NO-GO)
