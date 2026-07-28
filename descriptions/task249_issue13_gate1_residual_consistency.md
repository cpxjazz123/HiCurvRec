# Task #249 — Issue #13 Gate 1: 欧式 vs Möbius 残差 argmin 一致率 (零 GPU 前向)

## 来源
- GitHub Issue #13 (2026-07-28): [Escape Route 3] 残差算子几何一致性审计
- 承接 Task #248 (Gate 0 PASS): 残差算子确为欧式减法 (`utils.py:1797` + `hrqvae.py:550`)
- 主文献: Piękos P. et al. *Hyperbolic Residual Quantization*, arXiv:2505.12404v1

## 任务目的

Issue #13 §阶段闸门 Gate 1 (零 GPU, 纯前向):
- 载入 task222 ep29 healthy ckpt (L0 20.31% / L1 98.44% / L2 91.02%, collision 0.3706)
- 跑 item_emb (9922 items, dim=768) 一次 encoder + 3 层 RQ 前向
- 逐层比较: 欧式残差 `z - e_k` (现状) vs Möbius 残差 `z ⊖_{c_l} e_k = logmap_c(e_k, z)` (HRQ 主张)
- 算下一层 argmin 一致率
- 通过条件: 至少一层 ∈ [60%, 90%] OPEN 带
- 硬停止: 三层一致率均 > 95% → "残差算子不是杠杆"

## 决策 (R11.3 自主决策)
- ckpt 选 task222 ep29 healthy (Issue #13 §Gate 1 明文指定)
- 距离公式: 用 logmap_a 切空间距离² (HRQ §3.2 主张 `|logmap_c(e, z)|²`)
- 用 assignment_mode='shared' (Gate 1 强制简化)
- 验证: py_compile + 直接跑, 输出 verdicts/task249_gate1_consistency.json

## 产物
- `scripts/task249_issue13_gate1_residual_consistency.py`
- `verdicts/task249_issue13_gate1_residual_consistency_result.md`
- `verdicts/task249_gate1_consistency.json`
- Issue #13 GitHub 评论 (Gate 1 结果)

## 状态
零 GPU 纯前向, ~3 min wall.