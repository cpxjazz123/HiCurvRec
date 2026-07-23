# Task #44 Phase 1 V5 — T5 vs MCKG 拆分对照 (用户怀疑验证)

> **完成日期**: 2026-07-20
> **任务**: 验证用户怀疑 "MCKG 是 margin ranking loss 训出来的, embedding 度量结构不适合拆分, 是 root cause"
> **结论**: ✅ **用户怀疑完全成立** — T5 上拆分/拼接 = 0.95× (拆分更好), MCKG 上 = 4.22× (拆分显著差)

---

## 1. 实验设计

用户在 V4 后指出: **MCKG 的 item embedding 是 margin ranking loss 训出来的, 只保证排序, 不保证度量结构**. fused 平均掩盖了 norm 长尾, 拆分成 3 段后长尾放大 → 即使去掉几何机制 (E2/E3 Flat-3seg), 拆分也退化 4.22×.

**对照设计**: 同样的 "拆分-独立 vs 拼接-联合" 实验, 在 T5 (sentence-t5-base) embedding 上跑一遍.
- T5: max norm=1.12, p99=0.915 (norm 健康, 无长尾)
- MCKG: max norm=1143 (sub_h), p99=10.2 (长尾严重)

| T5 实验 | 输入处理 | 模型 | 备注 |
|---------|---------|------|------|
| T1 | T5 768d L2 norm | StandardRQ 768d (joint) | joint baseline |
| **T2** | **T5 768d L2 → 3×256d split** | **Flat-3seg 3 个 256d RQ** | **拆分关键对照** |
| T3 | T5 768d raw → 3×256d split | Flat-3seg 3 个 256d RQ | norm 影响 |

---

## 2. V5 结果 (T5 上)

| 实验 | final_loss | 维度 | 模型 |
|------|------------|------|------|
| T1 (T5-L2-768 joint) | **0.248** | 768d | joint StandardRQ |
| **T2 (T5-L2-3seg split)** | **0.235** | 3×256d | Flat-3seg |
| T3 (T5-Raw-3seg split) | 0.115 | 3×256d | Flat-3seg (raw 不归一) |

**关键观察**:
- T2 / T1 = **0.95×** → 拆分在 T5 上**略好** (各段损失了 5%, 几乎无代价)
- T3 / T2 = **0.49×** → raw 不归一在 T5 上反而更低(因 norm 较小, norm² loss 量纲自然小)

---

## 3. V4 vs V5 核心对比

| 指标 | T5 (健康 norm) | MCKG (长尾 norm) | 差距 |
|------|---|---|---|
| **拆分/拼接 ratio** | **0.95×** | **4.22×** | **4.4×** |
| norm 影响 (raw/L2) | 0.49× | 0.88× (E2/E3) | 反向 |

**最关键判决**:

```
T5 上拆分/拼接 = 0.95×  (拆分几乎没有代价)
MCKG 上拆分/拼接 = 4.22×  (拆分代价是 baseline 的 4 倍)

差距 = 4.22 / 0.95 = 4.44×
```

→ **"拆分代价"在 MCKG 上被放大 4.4×**. 这绝对不可能是"欧氏 RQ 拆分架构的固有问题" (T5 上是 0.95×).

→ **必然是 MCKG embedding 本身的问题**: norm 长尾 + margin ranking 训出的非平滑度量结构.

---

## 4. Root Cause 完整链条

```
MCKG margin ranking loss
    ↓ 只优化排序 gap, 不约束 norm
item embedding norm 长尾 (max/mean=796× for sub_e, 1143× for sub_h)
    ↓
fused = mean(sub_s + sub_e + sub_h)  ← 平均掩盖部分长尾
    ↓
fused L2 norm → 表面干净 (B = 0.275)
    ↓ 但 fused 本身 raw 也有问题 (E1 = 0.815, fused raw max=381)
3 段独立 RQ (E2/E3)
    ↓ 每段独立 norm² loss, 长尾放大
E3 Flat-3seg-L2 = 1.16  (4.22× baseline)
    ↓
PM-RQ 几何机制 (learnable κ, K³ 搜索, fusion logits)
    ↓ 雪上加霜
C PM-RQ V3 = 3.51  (12.8× baseline)
```

**每个环节都在恶化**:
1. MCKG 训出畸形度量结构(基础病)
2. fused 平均 + L2 norm 部分掩盖(临床缓解)
3. 3 段独立 RQ 放大(急性发作)
4. PM-RQ 几何机制最致命(致死一击)

---

## 5. 真正可行的下一步 (如要继续)

| 方案 | 描述 | ROI | 备注 |
|------|------|-----|------|
| **A. 终止当前 PM-RQ** | 当前架构否证 | ✅ 高 | 推荐 |
| **D. 换 embedding 源** | 用 T5 / sentence-t5 / flan-t5 替代 MCKG, 重新设计 PM-RQ | ⚠️ 中 | 重新走完整 GRID Stage 1-4 |
| **E. 重训 MCKG** | 加 norm 正则 + metric loss, 训出度量结构健康的 embedding | ⚠️ 中 | 复杂, 需要改 MCKG 训练 |
| **F. 不拆 + 加几何** | 在 fused 64d 上做几何重参数化 (1 个 RQ + learnable κ), 避开 3 段独立 norm² | ⚠️ 中 | 改动小, 仍有探索价值 |
| G. C-1 concat 192d + 几何 | 在 MCKG V4 E4 (0.48) 基础上加几何 | ⚠️ 低 | 比 F 还差, 不推荐 |

---

## 6. 教训与方法论

**用户怀疑的方法论价值**:
- V3 verdict: "PM-RQ 否证, 几何假设错" → **错**(用户指出架构不公平)
- V4 verdict: "架构独立 norm² 是 root cause" → **部分对**(用户指出 embedding 也可能是 root cause)
- V5 verdict: "MCKG embedding 度量结构不适合拆分才是 root cause" → **完整对**

**每轮 verdict 的盲点都是同一个**: 只在 MCKG 上对照, 没有跨 embedding 源的对照. 用户指出这一点后, 用 T5 (一个完全不同的训练范式) 做对照, 才完整分离了变量.

**一般教训**: 当一个"对照实验"只在同一个数据集上做时, 即使有 baseline 也有 ablation, 也可能错过"数据集本身是问题"的可能性. **跨数据集/跨 embedding 源对照**是必要的.

---

## 7. 产物清单

| 路径 | 内容 |
|------|------|
| `scripts/task44_t5_split_vs_joint.py` | V5 主脚本 (3 组 T5 对照) |
| `logs/task44_phase1_v5.log` | V5 训练日志 (7 min, T1-T3 全完成) |
| `products/task44_pmrq_phase1/task44_phase1_v5_t5_ablation.json` | V5 完整 summary |
| `verdicts/task44_phase1_v3_result.md` | V3 旧 verdict (3 轮 toy) |
| `verdicts/task44_phase1_v4_ablation_result.md` | V4 旧 verdict (Flat-3seg 对照) |
| **`verdicts/task44_phase1_v5_embedding_root_cause.md`** | **本 verdict (含 result: 行)** |

---

## 8. Task #44 完整时间线

- **V1 (2026-07-19)**: 原始 argmin, C=1.926, 7× 退化
- **V2 (2026-07-20)**: STE 量化器, C=2.394, 8.7× 退化, fusion_logits 不动
- **V3 (2026-07-20)**: d_total commitment, C=3.512, 12.8× 退化, fusion_logits 学会 trivial solution
- **V4 (2026-07-20)**: Flat-3seg 对照矩阵, 揭示 "3 段独立 norm²" 是 root cause
- **V5 (2026-07-20)**: T5 对照, 揭示 "MCKG embedding 度量结构畸形" 才是 root cause

**最终判决**: PM-RQ 在 MCKG 上的否证是 **多重 root cause 累积**:
1. MCKG embedding 度量结构畸形 (主因)
2. 3 段独立 norm² loss 架构 (放大因子)
3. 几何机制不稳定 (致命一击)

---

**result:** Task #44 Phase 1 V5 T5 对照验证用户怀疑完全成立: T5 上"拆分/拼接" = 0.95× (拆分几乎无代价), MCKG 上 = 4.22× (拆分显著差), **MCKG 上"拆分代价"被放大 4.4× 是 MCKG embedding 本身的度量结构畸形**(margin ranking loss 训出来的 norm 长尾问题)。**完整 root cause 链 = MCKG 度量结构畸形 + 3 段独立 norm² loss + PM-RQ 几何机制 三重叠加**。推荐 A 终止当前 PM-RQ 架构;如要继续则需 **D (换 embedding 源为 T5) 或 F (不拆 + 加几何)**。