# Task #275 — A2 + A3 Stage 1 curriculum 并行验证 (L0 ≥ 90%)

## 背景

Task #270 设计 3 配方 (A1/A2/A3) + Task #271 验证 A1 β=0.0 FAIL (L0 max 29.7% / ep30 26.6% < 90%).

剩下 2 个待验证配方:
- **A2**: Curriculum β — 先 30 ep β=0.0 拿到散开码字 → 热启动 → 30 ep β=0.5 双曲精修
- **A3**: β=0.5 + encoder freeze @ ep 20 (Task #193 已有实现)

## 任务范围 (本 cron tick)

1. **并行启动 A2 (GPU 1) + A3 (GPU 2)** Stage 1 训练, ~30 min/配方
2. **Stage 1 only** — 不进 Stage 2/3/4. 通过条件: L0 ≥ 90% at ep30+
3. **如 A2 或 A3 任一 PASS** → 后续 cron tick 写 verdict 闭环 + 评估是否进入 Stage 2 推断
4. **如全部 FAIL** → 闭环 NO-GO, 写 verdict + 候选 5 (L0 ≥ 90% 阈值重审) 升级用户拍板

## 通过条件

| 层 | 通过阈值 |
|---|---|
| L0 (K=64) | ≥ 90% (≥58 unique codes) |
| L1 (K=128) | ≥ 80% (≥103 unique) |
| L2 (K=256) | ≥ 80% (≥205 unique) |
| collision_rate | ≤ 0.5% (≤ 50 unique SIDs lost to collisions) |

## 关键决策点 (R11.5 + R7 + 用户 override)

- **并行而非串行**: 4×L40S 全空闲, 串行浪费 GPU (R7 禁止闲置)
- **A2 + A3 在不同 GPU**: 互不干扰, GPU 1 / GPU 2
- **Stage 1 only**: Issue #17 §H3 已明示本方向"不申请任何 Stage 3/4 预算" — Stage 1 验证 PASS 才进下一阶段
- **A2 需要 A1 ckpt**: A1 best_collision_model.pth 已在 products/task270/A1_euclidean/, launcher 自动检测

## 物理产物

```
descriptions/task275_a2_a3_curriculum_parallel.md  (本文件)
verdicts/task275_a2_a3_curriculum_parallel_result.md
products/task270/A2_curriculum/<run_id>/  (Stage 1 ckpt)
products/task270/A3_freeze_enc/<run_id>/  (Stage 1 ckpt)
logs/task270/stage1_A2_*.log
logs/task270/stage1_A3_*.log
```

result: Task #275 — A2 + A3 Stage 1 curriculum 并行验证, 通过条件 L0 ≥ 90% at ep30+. R7 并行 GPU 1+2. 通过则后续 Stage 2/3/4, 全 FAIL 则 NO-GO 闭环 + 候选 5 升级用户拍板.