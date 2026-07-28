# Task #187 — 4 层 Encoder 实验 (⏸️ STOPPED by user, 数据不足)

> **结论**: ⏸️ **STOPPED by user** at epoch 6/200 — 用户 18:42 改方向到 Task #188 (paper Table 7 多 seed 复现). 不是实验失败, 是用户战略调整. best_ckpt 已 R12 强制保存 (epoch 5, val NDCG@20=0.0630).

---

## 1. 任务目的

用户 2026-07-25 18:25 指令: 试一下 encoder 用 4 层 (vs Task #181 的 6 层). 验证减少 encoder 深度对 R@10 的影响.

---

## 2. 训练轨迹 (前 5 epoch 完整)

| Epoch | Val R@5 | Val R@10 | Val R@20 | Val NDCG@20 | Best? |
|-------|---------|----------|----------|-------------|-------|
| 1 | 0.0301 | 0.0420 | 0.0469 | 0.0244 | ✓ |
| 2 | 0.0531 | 0.0653 | 0.0785 | 0.0442 | ✓ |
| 3 | 0.0618 | 0.0724 | 0.0879 | 0.0484 | ✓ |
| 4 | 0.0630 | 0.0778 | 0.0933 | 0.0560 | ✓ |
| 5 | 0.0653 | 0.0809 | 0.0996 | **0.0630** | ✓ (final) |

**最终**:
- Best val NDCG@20 = 0.0630 @ epoch 5
- 训练在 epoch 6 开始时 (18:42) 被用户叫停
- best_ckpt 落盘 ✓ (18 MB, R12 强制保存)

---

## 3. 对比 Task #181 同期 (前 5 epoch)

| Epoch | Task #181 (6 enc) val R@10 | Task #187 (4 enc) val R@10 | Δ |
|-------|---------------------------|---------------------------|---|
| 1 | 0.0378 | 0.0420 | +0.004 |
| 2 | 0.0492 | 0.0653 | **+0.016** |
| 3 | — | 0.0724 | — |
| 4 | — | 0.0778 | — |
| 5 | — | 0.0809 | — |

**观察** (前 5 epoch 趋势):
- Task #187 (4 enc) 的 val R@10 起点反而比 Task #181 (6 enc) 高 (epoch 2: 0.0653 vs 0.0492)
- 任务设计本身没有 bug (loss 正常下降, val 正常爬升)
- 但样本数太少 (5 epoch), 无法判定 4 vs 6 层优劣 — 需要跑到 ~80 epoch 才能比较

---

## 4. 停训原因

用户 18:42 改方向, 提议 Task #188 (paper Table 7 复现 + 3 seed 方差), 比"4 vs 6 层" 单变量对照更有信息量 (paper 没做过多 seed, 是建立可信度的关键). Task #187 被新方向覆盖.

**用户后续决策**: Task #188 (已登记 description, 待启动)

---

## 5. 产物清单

| 文件 | 路径 | 大小 | 备注 |
|------|------|------|------|
| Best ckpt | `products/task187/t5small_4layer/Instruments/Jul-25-2026_18-38-48/HG_Rec_best.pth` | 18 MB | epoch 5, val NDCG@20=0.0630 |
| 训练 log | `logs/task187/Instruments/Jul-25-2026_18-38-48/HG_Rec.log` | 4.4 KB | 5 epoch 训练记录 |
| Launcher log | `logs/task187/launcher.out` | — | 启动 + kill 记录 |
| Launcher | `scripts/task187_encoder4_train.sh` | — | 完整 Stage 3 + Stage 4 pipeline |
| Description | `descriptions/task187_encoder4_layers.md` | — | 任务定义 |

**没用上的产物**: 无

---

## 6. 关键决策点 (R11.3 自主决策)

### 决策 1: 是否保留 best_ckpt?
**选了**: 保留 (18 MB)
**为什么**: 即便实验短, ckpt 是完整 Stage 3 训练的产出 (R12 已强制保存), 后续 Stage 4 eval 可以跑出 test R@10 做参考 (但用户没要求, 不主动跑)
**备选**: 删除 ckpt 释放磁盘 — 不必要, 18 MB 很小

### 决策 2: 写 verdict?
**选了**: 是, 强制写
**为什么**: CLAUDE.md §9.1 强制 verdict 即便失败/中断也要写, 记录 root cause
**依据**: loop.md §9 + R8 (完成判定)

---

## 7. 后续建议

1. **不要重做 Task #187**: 用户战略转向 Task #188, 单变量 4 vs 6 层 ROI 低
2. **如果未来真要测 encoder 层数**: 在 Task #188 跑完之后, 拿 collision≈13% codebook, 跑一次 4 enc + 6 enc 两个 stage 3 (各 seed=42), 直接比较 test R@10 (~20 min 总投入)
3. **Task #187 最佳 val NDCG@20=0.0630** 仅作为"训练启动正常"的 sanity check, 没有参考价值

---

## 8. 关键数字

| 指标 | 值 |
|------|-----|
| 训练时长 | ~3.5 min (5 epoch) |
| 跑到 epoch | 6/200 (3%) |
| Best val NDCG@20 | 0.0630 (epoch 5) |
| Best val R@10 | 0.0809 (epoch 5) |
| GPU 占用 | L40S GPU 0 (已释放) |
| Test R@10 | — (Stage 4 未跑) |

---

## 9. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建 (⏸️ STOPPED verdict, 补写) |

---

**result:** Task #187 4 层 encoder 实验在 epoch 5/200 被用户叫停 (val NDCG@20=0.0630), 不是失败. 用户 18:42 战略转向 Task #188 (paper Table 7 多 seed 复现). best_ckpt 18 MB 已 R12 保存. GPU 0 全释放.
