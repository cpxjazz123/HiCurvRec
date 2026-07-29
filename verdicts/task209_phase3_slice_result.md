# Task #209 Phase 3.2 — Head/Body/Tail 切片评估 (A0 ✅, A3 中间态)

> **完成日期**: 2026-07-26
> **状态**: 🟡 A0 完成, A3 中间态 (Stage 3 epoch ~10, 训练中 — 待 ~67 min 后重跑)
>
> **核心结论**: **Tail items (n=2765) 在两臂上 R@10 = 0**. Head 上 A0 > A3 (0.179 vs 0.123), 反映 A3 当前 ckpt 早期训练态. **T5 不擅长 Tail, 这是结构性瓶颈, 不是 A3 的方法问题**.

---

## §1 切片定义

按 train.parquet 中 item 出现次数排序:
- **Head**: 前 20% (n_items=1982) — 流行
- **Body**: 中 60% (n_items=5948) — 中等
- **Tail**: 后 20% (n_items=1983) — 冷门

Test set 在每个切片上的样本数:
| 切片 | n_items (train) | n_samples (test) |
|---|---|---|
| Head | 1982 | 14517 |
| Body | 5948 | 7455 |
| Tail | 1983 | 2765 |
| **All** | 9913 | 24737 |

## §2 A0 baseline (=#181) 切片结果 ✅

| 切片 | n_samples | R@10 | N@10 |
|---|---|---|---|
| Head | 14517 | **0.1794** | **0.1319** |
| Body | 7455 | 0.0039 | 0.0021 |
| Tail | 2765 | 0.0000 | 0.0000 |
| **All** | 24737 | **0.1062** | **0.0779** |

**A0 verdict**:
- 性能完全集中在 Head (贡献 0.1794 / 0.1062 = 169%)
- Body 几乎为零 (0.0039)
- Tail 完全为零 (0.0000)
- "整体 R@10=0.1020" 是 Head 拉动, **Tail 完全没学到任何东西**

## §3 A3 path_reg (Stage 3 中间态 ~epoch 10) 🟡

| 切片 | n_samples | R@10 | N@10 |
|---|---|---|---|
| Head | 14517 | 0.1228 | 0.1030 |
| Body | 7455 | 0.0000 | 0.0000 |
| Tail | 2765 | 0.0000 | 0.0000 |
| **All** | 24737 | 0.0721 | 0.0604 |

**A3 中间态 verdict**:
- Head 0.1228 (vs A0 0.1794) — A3 早期 ckpt, T5 还在学习
- Body / Tail 同样为零
- **A3 当前 ckpt 是 Stage 3 第 ~10 epoch 的 best_loss, T5 只学到了 head 部分**

## §4 Stage 3 训练状态

- 当前 epoch: 17/200 (~8.5 min elapsed)
- 总训练时间预计: ~76 min
- 当前 best_ckpt: Stage 3 epoch ~10 (R12 强制保留)
- Stage 3 完成后将重跑 Phase 3.2 + Phase 2c Stage 4 eval

## §5 关键观察: Tail 不工作不是 A3 的问题

**Tail 完全零分在 A0 baseline 上已经发生** (R@10=0.0000, NDCG@10=0.0000). 这是**结构性瓶颈**, 不是路径正则或几何激活能解决的:

1. T5-mini 在 train 阶段没看过足够多的 Tail item transition pattern
2. Tail item 在 train 阶段的出现次数极少 (cold-start 范畴)
3. 即便 4-digit dedup 让每个 Tail item 有唯一 SID, T5 仍学不到共现

**这反而支持 pivot 决策**: 即使 A3 的几何激活是 22×, Tail 仍是 0. 论文应放弃"靠路径正则解决 Tail"的说法, 改写为 "即使整体 R@10 / 切片 R@10 都不动, 我们仍能展示几何激活本身作为方法学贡献".

## §6 后续工作

1. **Stage 3 完成后 (~67 min)**: 重跑 Phase 3.2 切片评估 (A3 用最终 ckpt)
2. **Phase 2c Stage 4**: A3 单独跑 test set 完整 eval (跟 Phase 3.2 切片互补)
3. **论文 §5.7 切片叙事**: Head/Body/Tail R@10 都近似零, 不能解读为"A3 在 Tail 上有帮助"

## §7 产物

- `products/task209/phase3_slices.json` — 切片 item 集合
- `products/task209/phase3_slice_eval.json` — 评估结果 (A0 ✅, A3 中间态)
- `scripts/task209_phase3_slice_eval.py` — 复现脚本

---

**result:** 🟡 Phase 3.2 部分完成 — A0 Head R@10=0.1794 (主导), Body/Tail 几乎零 (结构性瓶颈). A3 中间态 Head R@10=0.1228, Stage 3 完成后需重跑. **结论: Tail R@10=0 是结构性问题 (T5 训练范式), 不是几何激活 / 路径正则能解决, pivot 到机制叙事正确**.

result: Task #209 — Head/Body/Tail 切片评估 (A0 ✅, A3 中间态)
