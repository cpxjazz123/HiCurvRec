# Task #78 — TIGER T5 训练 (paper Table 2 baseline #9)

> **任务目的**: 训练 LETTER/TIGER T5 baseline (paper Table 2 #9), 在 Musical_Instruments 上 reproduce paper R@10=0.0574
>
> **完成日期**: 2026-07-23 (训练中)
> **状态**: 🟢 在跑 (R12 checkpoint-45835 已落盘, step 45835/50000, ~91.67% 进度, ~20min 剩余)

---

## 1. 背景

承接 Task #77 (LETTER t5-base backbone 升级), 用 t5-base backbone 训练 TIGER T5 模型, 验证 LETTER 升级后下游推荐性能。

## 2. 实验设计

**变量**: T5 backbone (t5-base vs t5-small)
**保持不变**:
- TIGER SID recipe: 复用 Task #50 LETTER RQ-VAE tokenizer (692 tokens)
- 数据集: Musical_Instruments (5-core RecBole format)
- 训练配置: 50000 steps

**启动命令**: (已在 §16 R12 checkpointing 监控)

## 3. 决策触发

| 指标条件 | R@10 区间 | 决策 |
|----------|-----------|------|
| R@10 ≥ 0.0574 (paper target) | ≥ 0.057 | ✅ paper reproduced |
| R@10 ∈ [0.045, 0.057) | partial | ⚠️ backbone 升级未明显提升 |
| R@10 < 0.045 | < 0.045 | ❌ 退化 |

## 4. 风险与缓解

**风险 1**: LETTER/finetune.py 需 JSON 格式 sid, 而 Task #50 输出 .pt → blocking 启动
→ 缓解: Task #78 启动前已完成 .pt → .json 转换 (sid_tiger_text.json)

**风险 2**: checkpoint OOM 在 t5-base 上
→ 缓解: R12 checkpoint-45835 已落盘, 每 save_total_limit=1 自动覆盖

## 5. 完成度跟踪

- [x] sid_tiger_text.json 转换
- [x] T5 训练启动 (R12 PID 监控)
- [x] R12 ckpt-45835 已落盘
- [ ] step 50000 完成 + 评估
- [ ] 写 verdict

(此 placeholder 文档由 Task #79 R9 gap 填补生成, 实际训练配置在 §16 表格中跟踪)