# Task #93 — Paper Section 6 Draft (Discussion + Limitations + Future Work)

> **任务目的**: 起草论文 Section 6 (Discussion + Limitations + Future Work), 基于 Task #92 Section 5 闭环结论. 涵盖: (a) "数据本质欧氏"发现的理论含义, (b) 何时双曲几何真正有效, (c) 单 seed 评估的局限, (d) 未来工作方向.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #92 Section 5 草稿已完成, Section 6 需要更高层次的 Discussion + Limitations + Future Work. 这是 paper deliverable 的下一块.

## 2. Section 6 结构

### 6.1 Theoretical Implications: Is Hyperbolic Geometry Necessary?

基于 5 重证据 (Task #82/88/89/117 + #90), 讨论:
- 数据集本质欧氏 → 是否所有 RQ-VAE 应用都不需要 hyperbolic
- 双曲几何真正有效的场景是什么 (层级结构 / 幂律分布)
- Musical_Instruments 数据集不具备层级结构的原因

### 6.2 When Does Hyperbolic Geometry Help?

- 对比 Musical_Instruments vs 有层级结构的数据 (e.g., taxonomy-based, knowledge graph)
- 提出"geometric prior transferability"概念: 几何先验需与数据真实结构匹配
- HG-Rec 在论文 Toys 数据集上 geometric prior 匹配, 我们的 Musical_Instruments 不匹配

### 6.3 Mechanism vs Geometry Trade-off

- Sinkhorn vs Differential-Length Codebook 的对比
- Task #90 显示 vanilla + Sinkhorn 是更优的"机制组合"
- 启示: 简单的机制 + 有效的 post-processing > 复杂的几何先验

### 6.4 Limitations

- 单 seed 评估 (seed=42): 无法估计统计不确定性
- 数据集单一 (Musical_Instruments): 结论的 transferability 需更多数据集验证
- sentence-t5-base embedding: 其他 embedding (flan-t5, BGE) 的影响未探索
- 单一 codebook size [64,128,256]: 其他 size 的影响未探索

### 6.5 Future Work

- Multi-seed 验证 (待用户决策解除 multi-seed 禁令后)
- 多数据集验证 (Beauty/Sports 当前已删, 需重新获取)
- 代码本 size sweep ([32,64,128], [128,256,512])
- Embedding 选择影响 (flan-t5-base vs sentence-t5-base vs hybrid)
- 真正层级数据集 (DBpedia, Yelp taxonomy) 上的 hyperbolic 实验

## 3. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 整合 Section 5 闭环 | ~20 min | 无 |
| Discussion 段落写稿 | ~30 min | 无 |
| Limitations + Future Work | ~20 min | 无 |
| **总计** | **~70 min** | **0 GPU** |

## 4. 产物清单

- `verdicts/task93_paper_section6_draft.md` — Section 6 完整草稿

## 5. 关联

- 前置: Task #92 (Section 5 草稿), Task #84/87/88/89/90/91/94/95
- 后续: 论文最终写作, Section 7 Conclusion

---

**核心交付**: 论文 Section 6 完整 markdown 草稿, Discussion + Limitations + Future Work.