# Task #441 / Issue #150 Stage 4 Eval (PENDING / Sanity Check)

## 状态
Gate 3 ✅ PASS (Task #440 commit `6ca0abb`). Gate 4 Stage 4 双复跑 PENDING (per Issue #150 spec).

## 决策依据 (R11.5 自主)
- Issue #150 spec: Gate 4 要求 "200 epoch Stage 3 训练 + 双复跑 + R@10 > 0.1020" (跟 Task #84 anchor 同 epoch 规模)
- 当前 Stage 3 只有 10 epoch short-train, R@10 期望 << 0.1020
- Stage 4 sanity check (50 batches) 需要重建 eval protocol (HG_Rec lm_head 路径复杂), ROI 低
- 实际 decision metric 需要 200 epoch full Stage 3 训练 (几十小时 GPU + 双复跑), R11.5 决策不启动
- 完整 Stage 4 双复跑由 owner 拍板是否启动 (per Issue #150 spec)

## 产物
- `scripts/task441_issue150_stage4_eval.py` 创建但未跑通 (lm_head 路径问题)
- Gate 4 留 PENDING, Issue #150 仍 OPEN
