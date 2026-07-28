# Task #255 — Issue #10 方向 A 预准备: Sinkhorn 强度扫描 + Arm B 候选配置

## 现有数据点 (从已 commit verdict 汇总)

| max_iters | collision_rate | uniqueness_rate | R@10 | 来源 |
|-----------|---------------|-----------------|------|------|
| 0 (无 Sinkhorn) | 0.9907 (task225 口径) / 0.9100 (task223 口径) | 0.0093 / 0.09 | 0.1020 | task84 baseline |
| 10 | 0.1005 | 0.8995 | 0.1021 | task237 Arm B |
| 30 | 0.05 (task223 rough) | 0.95 | 0.1058 | task225 §5 vanilla+Sinkhorn |

3-arm 因果曲线已存在 3 点 (max_iters ∈ {0, 10, 30}). 关键问题: **这 3 点是不是 collision 单调 → R@10 单调的反向?**

| max_iters | collision | R@10 | ΔR@10 vs baseline |
|-----------|-----------|------|-------------------|
| 0 | 0.99 | 0.1020 | baseline |
| 10 | 0.10 | 0.1021 | +0.0001 |
| 30 | 0.05 | 0.1058 | +0.0038 |

观察:
- max_iters 0 → 10: collision 暴跌 0.99 → 0.10 (-0.89), R@10 几乎不动 (+0.0001)
- max_iters 10 → 30: collision 暴跌 0.10 → 0.05 (-0.05), R@10 涨 0.0038 (**这是 "collision 杠杆" 的 evidence**)
- **curve 是 U 形 / S 形, 不是简单单调**: 大头 collision 跌 (0.99 → 0.10) **没换来 R@10**, 小头 collision 跌 (0.10 → 0.05) 换 R@10 +0.0038

## 建议 Arm B 候选

**目标 collision**: 0.05 (跟 Arm C 同档, fine-tune if necessary) **OR 0.10 (跟 Arm B 同档, 进一步 Sinkhorn)**?

A 方向推荐 max_iters = 20 (落在 10/30 中点, 预期 collision 0.06-0.08, 探索 Sinkhorn 强度曲线):

| 候选 | max_iters | 预期 collision | 预期 R@10 |
|------|-----------|---------------|-----------|
| A1 | 15 | 0.07-0.08 | 0.1030-0.1050 |
| A2 | 20 | 0.05-0.07 | 0.1040-0.1055 |
| A3 | 25 | 0.05-0.06 | 0.1050-0.1060 |

**R11.3 决策**: 候选 A2 (max_iters=20) 是首选 — collision 中点, R@10 应在 0.1040-0.1055 区间. 若 A2 R@10 < 0.1020 → "collision 不是 R@10 杠杆" 成立, Issue #10 关闭. 若 A2 R@10 ∈ 0.1020-0.1058 → gap 缩小, 多跑 1 个 A1/A3 即可.

**注**: 不在本任务启动 Stage 3. 上述只是 dry-run 计划, 等用户决策 Issue #10 方向 A/B/C.

## 关键决策点 (R11.3)

- **任务范围**: 零 GPU 纯分析, 只汇总已有 verdict 数据点 + 写 Arm B 候选. **不启动 Stage 3**.
- **等 Issue #10 用户决策**: 方向 A/B/C 任一, 这个候选表都直接可用.
- **不写代码**: 没有新代码需求, 纯 verdict 文件即可.
- **R11.4 不可逆**: 启动 Stage 3 训练 50 epoch 是不可逆 (GPU 时间), 必须用户决策.

## 物理产物

```
verdicts/task255_issue10_arm_b_sinkhorn_strength_dryrun.md
```

(无 scripts/, 无 descriptions/, 无新增数据)

## 未来 (等用户决策)

- 方向 A 启动: `bash scripts/task223_stage2_codebook.sh --max_sinkhorn_iters 20 --output_path ...` → 1 次 Stage 4 eval
- 方向 B 关闭: 本 task 已是总结, 直接 close Issue #10
- 方向 C: 用户提新方向, 本 task 可作为参考

result: Task #255 — Issue #10 方向 A 候选 Arm B 配置 dry-run 完成. 现 3-arm Sinkhorn 强度曲线已收集 (max_iters ∈ {0, 10, 30}). Arm B 候选 = max_iters=20 (推荐, collision-target 0.06-0.08). 等用户决策 Issue #10 方向 A/B/C 后可立即启动.
