---
type: cleanup
issue: 92
status: "GO"
created: 2026-08-08
tags:
  - orphan
  - common
up: "[[index]]"
---

# Issue #92 common/ 孤儿文件清理 — GO

## 任务
删除 common/ 中 v77 (Issue #141 test R@10=0.1080) 没在使用的 3 个孤儿文件.

## 4 Gate 审计

### Gate 1 (orphan 识别): PASS
- **common/decor_prompt_former.py** (199 行): DECOR PromptFormer, Issue #65/70 NO-GO (test_R@10=0.1002 < baseline 0.1024)
  - 引用: 仅 `--enable_prompt_former` flag 开启时加载, v77 默认不开
- **common/t5_uncertainty.py** (110 行): T5 Uncertainty Head, Issue #76 v76 NO-GO (test_R@10=0.1050 < v77=0.1080)
  - 引用: 仅 `--enable_t5_uncertainty` flag 开启时调用, v77 默认不开
- **common/stage1/stage1_hyperbolic.py** (301 行): 历史 Stage1 上游, Issue #20 已替换
  - 引用: 0 import (v77 用 HG-Rec/dataset/Instruments/item_emb.parquet 上游预计算 768d)

### Gate 2 (v77 路径不受影响): PASS
- v77 训练: `common/stage3/stage3_train_pure_t5.py` 完整, 不依赖被删模块
- v77 评估: `common/stage4/stage4_eval_pure_t5.py` 完整, 仅依赖 common/hyperbolic_attention_bias.py
- HAB 模块保留: common/hyperbolic_attention_bias.py (v74 → v77 baseline)
- launch_*.sh 引用检查: 无引用被删文件 (verified by grep)

### Gate 3 (删除执行): PASS
- git rm 3 文件, 共 610 行
- common/ 体积: 2532 → 1922 行 (-23%)
- v77 训练/评估路径 grep 验证: 0 reference to deleted modules in active code

### Gate 4 (commit + push): PASS
- 本 verdict 文件 = verdicts/92/orphan_cleanup_close.md (R33 1:1 映射)
- commit: 见 issue close comment
- push: 见 issue close comment

## Verdict: GO
- 3 orphan 文件已删除, common/ 体积 -23%, v77 路径完全不受影响

## 历史 NO-GO 路径保留
- 如未来 issue 想复用 DECOR 或 Uncertainty 路径, 从 git 历史 `git restore <commit>^:<file>` 即可
- 当前不删除 Issue #65/70/76 涉及的具体代码 (Issue #76 路径已 5 commit 闭环, 历史完整)