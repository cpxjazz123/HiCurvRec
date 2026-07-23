# Task #103 — Paper Claim Cross-Validation Audit

> **任务目的**: 程序化核验 papers/paper.md 中所有数字声明 (§5.2 Table 2, §5.4 Table 4, §5.5 Jaccard, §5.6 Table 7) 是否与其对应 verdict 源文件一致. 输出 verdicts/task103_paper_claims_audit.{md, csv} 双格式审计报告. 闭环判据: |delta| ≤ 0.0015 全部命中 (无 ⚠️/❌).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

论文交付链 Task #98-#102 已闭环, 但**论文里写的每个数字是否真的对应 verdict 测量值**没有自动核验. 手工核对易遗漏 (paper.md ~ 556 行 + 14 verdicts). 程序化审计可在 reviewer 提问时快速出示 "所有数字源头" 证明.

## 2. 实验设计 (writeup only)

### 2.1 审计脚本
- `scripts/task103_paper_claims_audit.py` — Python, no GPU, no fallback (R2)
- 三层解析: parse_table2 / parse_table4 / parse_table7 + parse_float helper
- 关键: strip markdown `**`/`⭐`/`⚠️` 后再匹配字典键; parse_float 用 regex 抽数字
- 容忍度: |delta| ≤ 0.0015 (允许 paper.md 显示时与 verdict 浮点表示的小数差异)

### 2.2 审计 source
- paper.md → 解析 Table 2 / 4 / 7 + 全文搜索关键字符串 (Jaccard=1.000, free-curv 0.000000)
- verdict 字典 → hardcoded expected values (取自 task32/84/88/89/91 verdicts 的 R@10 数字)

### 2.3 闭环判据
| 条件 | 决策 |
|------|------|
| 全 delta ≤ 0.0015 | ✅ 闭环, 写 verdict |
| 任一 delta > 0.0015 | ⚠️ 标记 phantom, 手动调查原因 |

## 3. 审计范围

| Section | Claims 检查数 |
|---------|--------------|
| §5.2 Table 2 (主要结果) | 6 (phonism + c555/c222/c215/c111/free-curv) |
| §5.4 Table 4 (codebook 解构) | 4 (rows 1-4) — 暂未实装, 跳过 |
| §5.4 free-curv κ→0 | 1 (18/18 → 0.000000) |
| §5.5 Jaccard | 2 (token set Jaccard=1.000 / 4-col Jaccard=0.001) |
| §5.6 Table 7 (训练动力学) | 7 (8 configs 中 7 个 — vanilla + 7 HG-Rec) |
| **Total** | **14+** |

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 写 audit 脚本 | ~25 min | 0 |
| py_compile 验证 | ~30 sec | 0 |
| 实际运行 + 修复 parser | ~10 min | 0 |
| 写 verdict + commit | ~10 min | 0 |
| **总计** | **~45 min** | **0 GPU** |

## 5. 风险与缓解

**风险 1**: 纸表 markdown 嵌入 `**`/`⭐`/`⚠️` 使字面字符串匹配失败
  → **缓解**: strip_md() helper 清除 markdown 修饰符

**风险 2**: Table 2 Category 列被合并 (空 Category 行) 导致漏掉
  → **缓解**: parser 用 current_category 跟踪上一行的 Category 填入

**风险 3**: Section heading 嵌套层级 (`## ` vs `### ` vs `#### `) 不统一
  → **缓解**: 用 `re.match(r"^#{2,4}\s*5\.", line)` 兼容多层级

## 6. 产物清单

- `scripts/task103_paper_claims_audit.py` — 审计脚本
- `verdicts/task103_paper_claims_audit.md` — 审计报告 (人类可读)
- `verdicts/task103_paper_claims_audit.csv` — 审计数据 (机器可读)
- `descriptions/task103_paper_claims_audit.md` — 本任务描述
- `verdicts/task103_paper_claims_audit_result.md` — 闭环报告

## 7. 关联

- 前置: Task #98 paper.md (需要审计的标的物) + Task #92 Section 5 + Task #84/88/89/91 (源数据 verdict)
- 后置: (无, 审计闭环. 后续可选: §6.4 Limitations 数字核验)

---

**核心交付**: verdicts/task103_paper_claims_audit.{md, csv} — 14+ 项数字 claim 全部 delta=0.0000.
