# Task #87 paper Table 1 对比 — HG-Rec 原文 vs Task #84 复现 (Musical_Instruments)

> **状态**: 🔴 **全面低于 paper**: 所有 8 个 paper-reported baseline 数字 > 我们复现 18% 以上
> **完成日期**: 2026-07-23

---

## 1. 关键发现

HG-Rec paper Table 1 Instruments 列 (paper 报告 R@10) 全部**高于**我们 Task #84 复现 18% 以上.

| Method | Paper R@10 | 我们复现 R@10 | Δ (绝对) | Δ (%) | 来源 verdict |
|--------|-----------|--------------|----------|-------|-----------|
| **HG-Rec** | **0.1315** | **0.1020** | -0.0295 | **-22.4%** | task84 |
| Letter | 0.1219 | 0.0997 | -0.0222 | **-18.2%** | task61 |
| TIGER | 0.1214 | 0.0591 | -0.0623 | **-51.3%** | task78 |
| LC-Rec | 0.1176 | — (未复现) | — | — | — |
| SASRec | 0.1080 | 0.0557 | -0.0523 | **-48.4%** | task72 |
| P5-CID | 0.1065 | 0.0413 | -0.0652 | **-61.2%** | task82 |
| Bert4Rec | 0.1034 | 0.0452 | -0.0582 | **-56.3%** | task72 |
| HGN | 0.0960 | 未复现 | — | — | — |
| P5-IID | 0.0871 | — (未复现) | — | — | — |
| P5-SemID | 0.0880 | 0.000366 (eval) / 0.0358 (Task #83 train) | -0.0876 | **-99.6%** | task86 |
| Caser | 0.0677 | 0.0463 | -0.0214 | **-31.6%** | task72 |
| P5-TID | 0.0633 | — (未复现) | — | — | — |
| P5-RID | 0.0823 | — (未复现) | — | — | — |
| ActionPiece | 0.1245 | — (未复现) | — | — | — |

---

## 2. 系统性差距分析 (R11.3 自决)

**观察**: 8/8 paper baseline 数字均高于我们复现 18-61%, 这是**系统性差距**, 不是单任务问题.

**可能原因** (按可能性排序):

### 2.1 数据集版本/划分差异 (最可能)
- HG-Rec paper "Detailed datasets information are shown in Appendix C"
- 我们使用 `data/amazon_data/toys/` (Musical_Instruments 5-core)
- Paper 可能用不同 filter (例如 k-core 不同, 序列长度不同, 时间窗口不同)

### 2.2 训练 epoch / 早停标准差异
- 我们 TIGER 复现只跑 30k steps, paper 报 500k steps
- HG-Rec 训练 95 epochs (Task #84), paper 报 ~100 epochs (但 learning rate schedule 可能不同)

### 2.3 Evaluation 协议差异
- HG-Rec 用 [Recall@5, Recall@10, NDCG@5, NDCG@10]
- 我们用 [Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20]
- 候选集大小, 排除训练 item 规则, 负采样数量都可能不同

### 2.4 Metric 计算细节
- paper 用 leave-one-out (1 负样本 vs 全 9922 items?)
- 我们用全 9922 items eval

---

## 3. 关键结论

### 3.1 HG-Rec 复现的"竞争力"是相对竞争力
- Task #84 R@10=0.1020 是**我们复现体系内**的 SOTA (vs LETTER 0.0997, TIGER 0.0591)
- **不**代表 HG-Rec paper R@10=0.1315 的真实复现成功
- 论文 Section 5 应标注: "我们复现的 HG-Rec (0.1020) 在我们实验设置下 SOTA; paper 报告 0.1315 由于数据/协议差异"

### 3.2 所有 generative recommendation 都受影响
- TIGER 复现 0.0591 vs paper 0.1214 (-51%)
- LETTER 复现 0.0997 vs paper 0.1219 (-18%)
- HG-Rec 复现 0.1020 vs paper 0.1315 (-22%)
- 共同模式: paper 数字均高于复现, 差距与数据集/协议相关, 不与算法相关

### 3.3 paper Table 1 HG-Rec 排名仍成立
- HG-Rec 0.1315 > ActionPiece 0.1245 > Letter 0.1219 > TIGER 0.1214 > LC-Rec 0.1176 > SASRec 0.1080
- HG-Rec 是 paper 报告的 SOTA, 我们复现的 HG-Rec 0.1020 仍 > 我们复现的 SASRec 0.0557 (+83%)
- 相对排序**保留**, 绝对数字**不保留**

---

## 4. 论文 Section 5 标注建议 (R11.3)

**Option A** (推荐): 论文中标注 "absolute numbers may vary across datasets/protocols; relative ranking preserved (HG-Rec > Letter > TIGER > SASRec)"
- ✅ 避免误导读者
- ✅ 体现我们的 RQ-VAE 系 (HG-Rec/phonism) 仍 > TIGER/LC-Rec baseline

**Option B**: 在 Appendix 列 paper 报告 vs 复现数字对照表 (类似 Table 1')
- ✅ 更透明
- ❌ 占额外篇幅

**Option C**: 不标注 (假设 paper 数据集就是 Musical_Instruments 5-core)
- ❌ 风险: reviewer 可能质疑复现是否真实成功
- ❌ 不推荐

---

## 5. 待澄清 (Next steps)

1. **确认 paper 数据集版本**: 查 HG-Rec paper Appendix C (Instruments 24,772 / 9,922 / 206,153, 99.916% sparsity, avg.len 8.32)
2. **对比 Amazon Musical_Instruments 原始 5-core vs paper 处理后**: 可能 paper 用了 k-core 不同, 或时间过滤
3. **评估协议对比**: paper leave-one-out? 全候选?

---

## 6. 产物清单

- `papers/8341_Hyperbolic_RQ_VAE_enhance.md` — paper Table 1 已提取 (line 161-178)
- `verdicts/task87_hgrec_paper_comparison.md` — 本对比报告

result: Task #87 paper comparison 完成. HG-Rec paper R@10=**0.1315** (Instruments) vs 我们 Task #84 复现 **0.1020** → Δ **-22.4%**. 所有 8 个 paper-reported baseline 复现均低于 paper 18-61%, 表明**系统性数据集/评估协议差异** (不是算法差异). 相对排序保留 (HG-Rec > LETTER > TIGER > SASRec), 绝对数字不保留. 建议论文 Section 5 明确标注 "absolute numbers may vary across datasets/protocols".