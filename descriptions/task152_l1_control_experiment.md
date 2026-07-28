# Task #152 — L1 对照实验 (用户 2026-07-24 提议)

> **任务目的**: 验证 Task #149 L1 κ=0 是 "数据真实偏好" 还是 "θ=0 梯度死区 bug"
> **完成日期**: 2026-07-24
> **状态**: 🟢 **闭环** — verdict 写于 `verdicts/task152_l1_control_result.md`

---

## 1. 背景

承接 Task #149 三层 κ 学习结论:
- L0 κ = -0.127 (双曲, 学习到了)
- L1 κ = **0.0000 全程** (欧式, 没动)
- L2 κ = +0.163 (球面, 学习到了)

L1=0 的解读有 2 种:
- (a) L1 真实就是欧式, 数据偏好 κ=0
- (b) θ=0 是梯度死区, L1 init=0 没法跳出来

需要判别性实验区分.

---

## 2. 实验设计

**变量**: L1 θ_init: 0 → +0.15
**保持不变**:
- L0 θ_init=-0.3, L2 θ_init=+0.3 (跟 Task #149 一致)
- Phase A 100 epoch frozen + Phase B 100 epoch unfrozen (lr_theta=1e-5)
- Stage 1 RQ-VAE 训练 (task89_stage1_train_rqvae.py)

**启动命令**: `scripts/task152_l1_control.sh`

**判据**:
- L1 漂回 ~0 → "0 是 L1 真实偏好"
- L1 停在 +0.15 附近 → "θ=0 是梯度死区"

---

## 3. 决策触发

| 观察 | 解读 |
|------|------|
| L1 漂回 ~0 | 假设 (a) 成立, Task #149 结论保留 |
| L1 停在 +0.07 ~ +0.15 | 假设 (b) 成立, θ=0 死区是 bug, 改 init 即可修复 |

---

## 4. 预算

| 阶段 | 估算 |
|------|------|
| Stage 1 训练 | ~5 min (200 epoch) |
| κ_history 分析 + verdict | ~30 min |
| **总计** | **~35 min** |

---

## 5. 风险

- **Phase B lr 太小**: lr_theta=1e-5 可能让 L1 漂移太慢, 看不出明显方向 → 已用 100 epoch Phase B 给足时间
- **codebook 干扰**: L1 codebook utilization 100% 全程, 无坍缩