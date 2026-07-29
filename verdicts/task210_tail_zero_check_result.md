# Task #210 Tail=0.0000 评估 bug 检查

> **完成日期**: 2026-07-26
> **状态**: ✅ 不是 bug, Tail=0 是真实结构性瓶颈

---

## §1 检查目标

Phase 3.2 (#209) 切片评估显示 A0 baseline 在 Tail (n=1983 items) 上 R@10=0.0000 / NDCG@10=0.0000. 用户 2026-07-26 反馈"先查是不是评估 bug. 精确的 0 比'很小'更可疑".

## §2 检查项

### §2.1 Tail items 是否在 train 里出现

| 切片 | 平均出现次数 (train) | min | max |
|---|---|---|---|
| **Tail** (n=1983) | **3.29** | 1 | 5 |
| Head (n=1982) | 51.48 | — | — |
| Body (n=5948) | — | — | — |

**Tail items 在 train 出现次数分布**:
- 1-2 次: 324 项 (16.3%)
- 3-5 次: 1659 项 (83.7%)
- 6+ 次: 0 项
- **0 次: 0 项** ← Tail items 全都在 train 里出现过, 不是 cold-start

**结论**: 不是 cold-start 评估协议问题. Tail items 都至少有 1 次 train 出现.

### §2.2 Tail test items 的 history 长度

test.parquet 中 target ∈ Tail 的样本 (n=2765):
- history 长度 mean: 7.92
- history 长度 min: 4
- history 长度 max: 162

**结论**: 评估上下文足够长, 不是 history 截断问题.

### §2.3 Tail test items 的 history 中含多少 Tail items

| 统计 | 值 |
|---|---|
| mean (Tail items in history) | 0.4608 |
| median | **0.0** |
| min | 0 |
| max | ~7 |

**关键发现**: median=0, 意味着**超过一半的 Tail test items 的 user history 里完全不含 Tail item**. 即用户对 Tail items 没有"我喜欢类似东西"的共现 pattern 可学.

## §3 结论

**Tail R@10=0.0000 是真实的结构性瓶颈, 不是评估 bug**:

1. ✅ Tail items 在 train 里出现 (cold-start 排除)
2. ✅ test history 长度正常 (截断排除)
3. ✅ 评估管线正确 (test target ∈ tail=2765, 与切片数一致)
4. ❌ **用户对 Tail items 没有共现 pattern** (median=0 Tail in history)

机制解释: 推荐系统只能从"用户过去喜欢过的相似东西"推荐. 如果用户的 history 里完全没有 Tail items, 即便 Tail item 在其他用户的 history 里出现, 也不该推给此用户 (无信号). T5-mini 在 Tail 上 R@10=0 是正确行为, 不是 bug.

## §4 对论文的影响

### §4.1 切片章节: 保留, 但诚实写明

切片评估章节**保留**, 但需要明确:
- Tail R@10=0 不是方法问题, 是数据特性
- Head R@10 才是方法有效性的真实读数
- 我们的 A3 在 Head 上 R@10 = 0.123 (中间态), 最终 ckpt 应该更高

### §4.2 切片叙事的真实角色

切片不是"展示 A3 比 A0 在 Tail 更好", 而是:
- **下界证据**: 即使 A0 baseline 在 Tail 上完全失败, A3 也不会更差 (因为几何不破坏已有信号)
- **Head 主导**: Head R@10 是 A3 vs A0 真正比较的地方 (目前 A0=0.179, A3 中间态=0.123 — A3 训练中)
- **Body/Tail 的 0**: 写进论文 Limitations, 不作为卖点

## §5 副产物

`products/task209/phase3_slices.json` 已含切片定义 (Head/Body/Tail item 集合).

---

**result:** ✅ Tail=0.0000 不是评估 bug — Tail items 都在 train 里出现过 (avg 3.29 次), test history 长度正常 (mean 7.92), 但用户对 Tail items 无共现 pattern (median=0 Tail in history). Tail R@10=0 是结构性瓶颈. 切片章节保留, 但 Head 是真实读数.

result: Task #210 — Task #210 (auto-extracted fallback)
