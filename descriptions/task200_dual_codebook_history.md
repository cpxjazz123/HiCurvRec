# Task #200 — 双码本几何解耦 (历史 placeholder)

> **任务目的**: 历史 placeholder — 引用 Task #200 系列 verdicts (Phase 0/1/Stage 3+4)
> **完成日期**: 2026-07-26
> **状态**: ✅ 全部闭环 (placeholder for 历史 TaskList 条目)

---

## 历史说明

Task #200 在 taskList 内部有 9 个条目 (Phase 0 重跑 / Phase 1 v1-v5 / Stage 3 dual_v5 等), 但因 descriptions/ 文件缺失, R9 next-task 编号会冲突. 此 placeholder 用于填补 descriptions/ 空洞, 引用现有 verdicts/ 文件.

| 实际 verdict 文件 | 内容 |
|------------------|------|
| `verdicts/task200_phase0_result.md` | Phase 0 双码本初始化验证 (v3 PASS) |
| `verdicts/task200_phase1_result.md` / `task200_phase1_v2_result.md` / `task200_phase1_v3_result.md` | Phase 1 50 epoch 冒烟 (v1/v2/v3, 最后 v5 收敛) |
| `verdicts/task200_v4_5diagnoses_result.md` | 用户 5 点诊断清单 |
| `verdicts/task200_v5_user_4step_result.md` | 用户根因诊断 4 步 |
| `verdicts/task200_stage3_dual_v5_template.md` | Stage 3 launcher 模板 |
| `verdicts/task200_dual_v5_stage3_4_result.md` | Stage 3+4 闭环 (R@10=0.0915, -10.3% vs HG-Rec) |
| `verdicts/task200_dual_v5_test_metrics.json` | Stage 4 test R/N 完整 metrics |

**核心结论**: 用户原"双码本几何解耦"路径 (Phase 1 v5 EMA batch-mean + 3 层 centering + α_geo=1.0) 在 Stage 1 训练期正常, 但 Stage 3+4 端到端验证**未救** baseline (-10.3% R@10). 见 `verdicts/task200_dual_v5_stage3_4_result.md`.

---

## R9 备注

- descriptions/ 历史 max=199, 此 placeholder 填补 #200 空洞
- 后续 R9 next-task = 201 (按 max+1)
- Task #200 系列实验代码与脚本保留在 `scripts/task200_*.py|sh`