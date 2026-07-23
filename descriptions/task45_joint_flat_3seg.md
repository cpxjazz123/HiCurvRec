# Task #45 — 战线三 3.1 Joint-Flat-3seg

> **任务目的**: 验证 Joint-Flat-3seg (3 段独立 codebook + 联合训练) 是否能追上 E4 concat-192d 表现, 分离"训练耦合 vs 分段产品结构"两个变量
> **完成日期**: 2026-07-20
> **状态**: ✅ 已完成

---

## 1. 背景

承接 Task #44 V4 矩阵结论: E4 concat-192d L2 (0.482) 远低于 E3 Flat-3seg-L2 (1.160). 是否因为联合训练耦合 / 拼接 representation power / 还是其他?

## 2. 实验设计

3 段 unfused subspace (NORM_CAP=5.0, 与 V4 E2 对齐), 无曲率机制, 联合训练 (单一全局 192d norm² loss + 共享 commitment).

## 3. 决策触发

| 指标 | 范围 | 决策 |
|------|------|------|
| recon_seg < 1.5 | 接近 E4 (1.75x) | 架构罪在训练解耦 |
| recon_seg ≈ 1.16 | 与 E2/E3 同 | 分段产品结构本质劣势 |

## 4. 预算

~10 min (2000 steps, K=64, 3-layer)

## 5. 结果 (回填)

- raw clip recon_seg = 1.390 (与 E2/E3 同水平, joint 训练无帮助)
- +log1p recon_seg = 0.389 (与 Flat-3seg-log1p 0.364 同水平)
- **结论**: 训练耦合不带来额外增益, log1p 是真正的修复.

## 6. 产物

`scripts/task153_joint_flat_3seg.py` + `products/task153_joint_flat3seg/task153_summary.json`
