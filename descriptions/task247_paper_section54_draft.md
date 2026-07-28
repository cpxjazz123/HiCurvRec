# Task #247 — paper Section 5.4 推荐措辞草案 (零 GPU)

## 来源
- Task #246 v3 ranking (2026-07-29): paper-aligned fixes 增量更新综合 ranking
- Task #94 (2026-07-23): HG-Rec paper Table 1 对比 (paper 0.1315 vs 复现 0.1020, Δ -22.4%)
- Task #87 v2 (2026-07-24): 综合 ranking 22 行 baseline
- Task #88 (2026-07-24): per-layer curvature 6 网格 (c555 R@10=0.1051 微弱最优)
- Task #89 (2026-07-24): free-curvature product manifold (κ=0 NO-GO)
- Task #141 (2026-07-24): Caser paper-aligned fix (R@10=0.0378)
- Task #143 (2026-07-24): FDSA paper-aligned fix NO-GO (撤回, 沿用 task85)
- Task #150 (2026-07-24): LETTER paper-aligned fix (R@10=0.0509)
- Task #152 (2026-07-24): L1 control experiment (κ=0 数据真实偏好, 非 bug)

## 任务目的

为 paper Section 5.4 提供推荐措辞草案:
1. **Table 2 推荐版**: 用 Task #246 v3 ranking (27 baseline), 标注 "复现 R@10" + "vs Paper" + 来源 verdict
2. **Table 3 推荐版**: HG-Rec variants (c555 / c222 / c215 / c111 / c1055 / c512 + A-arm) vs 其他 generative 最佳 (phonism 0.1058 / LETTER paper-aligned 0.0509 / TIGER 0.0591 / FDSA 0.0594)
3. **paper-aligned systematic bias 段措辞**: 解释 paper absolute numbers 系统性偏低 (-22% ~ -57%) 不是 bug, 是 paper-reported 跟 RecBole/HG-Rec 真实复现协议差异
4. **retro-label 段**: 解释 Task #87 报告的 LETTER 0.0997 / Caser 0.0463 是 over-trained "+ve outlier", 已被 paper-aligned fix 替换
5. **HG-Rec vs other generative 比较措辞**: 强化 "RQ-VAE + T5 生成 vs 其他 generative" 领先幅度 (0.10+ vs 0.05-0.06)
6. **draft LaTeX table**: Table 2 + Table 3 完整 LaTeX 代码片段, 可直接复制粘贴

## 决策 (R11.3 自主决策)
- 不写完整 paper, 只写 Section 5.4 推荐措辞 + draft LaTeX
- 不重写 Task #94 paper Table 1, 引用即可
- 保留 v3 ranking 数据 (Task #246), 不再修改
- LaTeX 格式: 用标准 IEEE/ACM 两栏会议格式 + \begin{table} ... \end{table} 结构

## 产物
- `verdicts/task247_paper_section54_draft.md` (推荐措辞 + LaTeX 片段)
- `verdicts/task247_table2_v3.tex` (Table 2 LaTeX)
- `verdicts/task247_table3_v3.tex` (Table 3 LaTeX)

## 状态
零 GPU 数据整理. 复用 task246 v3 ranking 数据 + task94 paper comparison. 不写脚本.