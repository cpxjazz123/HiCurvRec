# Task #242 / Issue #11 Gate 1b (Arm A+) — FULL NO-GO 关闭方向

**日期**: 2026-07-29
**状态**: Gate 1b FAIL → Issue #11 FULL NO-GO, 关闭.

## 配置

| 项 | 值 |
|----|-----|
| 配置 | Task #242 Arm A 配置 + `--anti_collapse dead_revive` |
| L0 c_k | U(1.0, 5.0) |
| L1 c_k | U(0.5, 20.0) |
| L2 c_k | U(0.5, 20.0) |
| 死码字复活 | 每个 eval step (epoch 9/14/19/24/29/34/39) |
| Wall clock | 15 s (40 epoch × 0.32 s) |

## Gate 1b 数值

| ckpt | L0 util | L1 util | L2 util | collision_rate | L0 ≥ 90%? | collision ≤ 0.3706? |
|------|---------|---------|---------|----------------|-----------|----------------------|
| **best_collision** (epoch 34, 0.9958) | **3.12%** (2/64) | 3.12% (4/128) | 7.03% (18/256) | 0.9945 | ❌ FAIL | ❌ FAIL |
| best_loss (epoch ?) | 14.06% (9/64) | 28.12% (36/128) | 19.92% (51/256) | 0.9749 | ❌ FAIL | ❌ FAIL |

**Gate 1b 两条通过条件全部不满足, 且比 Arm A 更差.**

## 对照表 (Issue #11 全部 Stage 1 尝试)

| Run | L0 c_k range | L0 util | L1 util | L2 util | collision | 备注 |
|-----|--------------|---------|---------|---------|-----------|------|
| task220 (200 ep, ep29) | U(0.5, 5) 全层 | 20.31% | 96.09% | 93.75% | 0.3835 | 全层 PC κ |
| task222 (40 ep, ep29) | U(0.5, 5) 全层 | 20.31% | 98.44% | 91.02% | 0.3706 | task220 healthy 复现 |
| **task242 Arm A (40 ep, ep14)** | **U(1,5) L0 / U(0.5,20) L1/L2** | **23.44%** | **46.09%** | **38.28%** | **0.9385** | **Issue #11 推荐配置** |
| **task242 Arm A+ (40 ep, ep34)** | **同上 + dead_revive** | **3.12%** | **3.12%** | **7.03%** | **0.9945** | **Gate 1b 补救尝试** |

## 决策

按 Issue #11 §阶段闸门:
> Gate 1b 仍未过 → 该方向 FULL NO-GO, 关闭, 不进入 Stage 2/3/4. 此时 H2 与 H3 同时被证伪, 应当记录"逐层曲率参数化无法在 Stage 1 存活", 并停止对 PC κ 参数空间的进一步调参.

### 关键发现

1. **H2 反传机制不存在**: Issue #11 推荐 Arm A 押注"L1/L2 用宽区间 (U(0.5, 20)) 释放几何参与 → 反向梯度结构变化 → L0 利用率回升". 实测 L1/L2 利用率反而从 98%/91% 暴跌到 46%/38%, L0 只回升 3.13pp (20.31→23.44), 跟 H2 预测相反.
2. **H3 死码字复活无效**: dead_revive 不仅没改善, 反而把 L0 利用率从 23.44% 砸到 3.12% (降幅 20pp). 死码字被随机重 init 后立即被反向传播再次杀掉, 形成"复活 → 再死"循环, 实际有效码字数越来越少.
3. **PC κ 参数空间耗尽**: 历次扫描 (task218 / task220 / task222 / task231 / task242 Arm A / task242 Arm A+) 已经覆盖:
   - 全局 c_k range: U(0.5,5), U(0.5,20), U(1,5)
   - per-layer c_k range: L0 U(1,5) + L1/L2 U(0.5,20)
   - 死码字复活: on/off
   
   没有任何组合让 L0 utilization 跨过 90% 阈值. PC κ 机制本身在 Stage 1 训练动力学下无法让 L0 码本展开.

### 上游改动保留

`HG-Rec/model/utils.py` + `HG-Rec/model/hrqvae.py` + `HG-Rec/train_hrqvae.py` 的 `--c_k_range_list` plumbing 改动保留 (HG-Rec/ 在 .gitignore). 后续 per-layer κ 实验可复用.

### Issue #11 状态

**Issue #11 FULL NO-GO 关闭.**

按 Issue #11 §后续方向:
- 后续方向不应再尝试 "逐层 c_k range 调参" (PC κ 参数空间耗尽)
- 候选方向:
  1. 全部层用同一最优 mode (task220 PC κ U(0.5,5) / task221 Gromov weight=0.5 — 但 Gromov Stage 1 全坍缩)
  2. 引入新机制 (kmeans_init 重做 / sinkhorn-on-encoder) — 跟 Issue #12 沙漏效应路径可能交叉
  3. 跨架构 (LETTER / S3Rec / FDSA) 而非在 HRQVAE 内部调参

## 产物

- `products/task242/hrqvae_perlayer_ck_deadrevive/Jul-29-2026_02-35-49_*/best_collision_model.pth` (L0 util 3.12%)
- `products/task242/hrqvae_perlayer_ck_deadrevive/Jul-29-2026_02-35-49_*/best_loss_model.pth` (L0 util 14.06%)
- `products/task242/hrqvae_perlayer_ck_deadrevive/Jul-29-2026_02-35-49_*/epoch_*_collision_*_model.pth`
- `logs/task242/perlayer_ck_deadrevive_stage1_train.out`
- `scripts/task242_issue11_gate1b_perlayer_deadrevive_stage1.sh`

## 历史 issue 引用

- verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md (Gate 0 PASS — Phase 0 OPEN 三层)
- verdicts/task242_issue11_gate1_perlayer_stage1_result.md (Gate 1 NO-GO)
- verdicts/task242_issue11_gate1b_perlayer_deadrevive_result.md (本文件, FULL NO-GO)
