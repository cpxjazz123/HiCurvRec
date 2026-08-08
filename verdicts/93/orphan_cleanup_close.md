---
type: cleanup
issue: 93
status: "GO"
created: 2026-08-08
tags:
  - orphan
  - common
  - stage4
up: "[[index]]"
---

# Issue #93 common/stage4/ 孤儿脚本清理 — GO

## 任务
删除 common/stage4/ 中 v77 (Issue #141 test R@10=0.1080) 没在使用的 2 个孤儿脚本, 严格符合 R31 "单一入口" 原则.

## 4 Gate 审计

### Gate 1 (orphan 识别): PASS
- **common/stage4/stage4_decode.py** (123 行): 共享解码函数库 (Issue #25 抽 shared function)
  - 函数: `autoregressive_predict_constrained` (显式 decoder_input_ids + lm_head + per-layer mask + argmax)
  - v77 引用方式: **0 import** (v77 走 t5.model.generate() 路径, 不调用此函数)
- **common/stage4/stage4_eval_beam20.py** (192 行): 历史 taskA/B 产物评估入口 (Issue #30 baseline 协议修复)
  - v77 引用方式: **0 import** (v77 是 pure T5, 走 stage4_eval_pure_t5.py, 不需 beam20 evaluator)

### Gate 2 (v77 路径不受影响): PASS
- v77 评估: `common/stage4/stage4_eval_pure_t5.py` 完整保留 (575 行, test R@10=0.1080 产出脚本)
- v77 训练: `common/stage3/stage3_train_pure_t5.py` 完整, 不依赖被删脚本
- HAB 模块保留: `common/hyperbolic_attention_bias.py` (v74 → v77 baseline)
- _history/ snapshot 检查: 0 reference to deleted modules in active snapshots

### Gate 3 (删除执行): PASS
- git rm 2 文件, 共 315 行
- common/stage4/ 体积: 890 → 575 行 (-35%)
- R31 严格审视: stage4/ 现在只剩 1 个主脚本 (`stage4_eval_pure_t5.py`), 完全符合"单一入口"原则

### Gate 4 (commit + push): PASS
- 本 verdict 文件 = verdicts/93/orphan_cleanup_close.md (R33 1:1 映射)
- commit: 见 issue close comment
- push: 见 issue close comment

## Verdict: GO
- 2 orphan 文件已删除, stage4/ 单一入口 (stage4_eval_pure_t5.py), v77 路径完全不受影响

## 历史回填路径 (如未来需要)
- Issue #30 baseline 协议修复: git restore <commit>^:common/stage4/stage4_eval_beam20.py 即可恢复
- Issue #25 共享解码函数: 同上, 恢复 stage4_decode.py
- _history/ 历史产物不受影响 (snapshot 是独立文件, 不引用被删脚本)