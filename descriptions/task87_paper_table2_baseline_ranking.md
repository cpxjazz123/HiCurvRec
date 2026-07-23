# Task #87 — Paper Table 2 baseline 综合排名 (Stage 5 summary)

> **任务目的**: 汇总 paper Table 2 全部 baseline 复现结果, 输出 Musical_Instruments dataset 上的 final ranking, 作为 paper Table 2 复现终判.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #78/#79/#80/#81/#82/#83/#89/#90/#91 共 8 个 baseline 复现 + Task #84 HG-Rec 新 baseline. 需要把所有 test metrics 汇总到 paper Table 2 格式, 输出 R@5/10/20 + NDCG@5/10/20 的 ranking, 与 paper 报的数字对比, 给出最终 paper Table 2 复现判定.

---

## 2. 实验设计

**输出**: 
- `products/task87/paper_table2_ranking.csv` — 全部 baseline + 我们的复现, 按 R@10 排序
- `verdicts/task87_paper_table2_baseline_ranking_result.md` — Markdown 报告, 含 (a) ranking 表格 (b) vs paper 偏差 (c) HG-Rec 在 table 中位置 (d) 决策结论

**输入数据**:
- Task #78 TIGER: verdict `task84_tiger_t5_inference_result.md` (hit@10=0.0591)
- Task #79 CoST: `task79_cost_no_official_impl_result.md`
- Task #80 FDSA: `task85_fdsa_test_eval_result.md` (R@10=0.0594)
- Task #81 S³Rec: 训练中, ~2h 后产出 eval
- Task #82 P5-CID: `task82_p5_cid_instruments_result.md` (hit@10=0.0413)
- Task #83 P5-SID: Task #86 eval 进行中
- Task #89 LightGCN: `task89_*_result.md` (R@10=0.0455, +0.2% vs paper)
- Task #90 FMLP-Rec: NO-GO
- Task #91 DuoRec: `task91_duorec_repro_result.md` (R@10=0.0672, +48%)
- Task #84 HG-Rec: `task84_hgrec_main_repro_instruments_result.md` (R@10=0.1020)

---

## 3. 决策触发

| 总 baseline 完成度 | 决策 |
|------------------|------|
| ≥ 7 个 baseline 复现 | ⭐ 出 paper Table 2 综合 ranking 报告 |
| 5-7 个 baseline | 🟡 部分 ranking + 标注缺哪些 |
| < 5 个 | ❌ 推迟到所有 baseline 完成 |

---

## 4. 完成度

- [ ] 收集所有 baseline verdict 文件
- [ ] 解析 paper Table 2 数字 (Yelp/Beauty/Musical_Instruments)
- [ ] 生成 paper_table2_ranking.csv
- [ ] 写 verdict + result: 行

result: Task #87 — Paper Table 2 baseline 综合排名 (Stage 5 summary) 待启动.