# Issue #161 / Task #450/#511 [方向C Gate4] ZeroCenteredLayerNormAdapter Stage 4 NO-GO 收口

## Gate 4 决策: ❌ NO-GO 收口

## 1. Stage 3 训练结果 (R23 监控 + 训练 trace)

- 训练 200 epoch, batch_size=32, adapter + LayerNorm trainable, T5 frozen
- **early stop @ ep110**: best_val_R@10_sim=**0.1069** (+4.8% vs Task #84 baseline 0.1020)
- α 健康增长: 4.5e-5 → 4.5e-3 (ep109 peak @ 4.5631e-3)
- cond_grad 健康 5e-2 ~ 1e-1 全程稳定
- ln_grad 健康 1.3e-1 全程稳定
- loss 1.89 → 1.60 (-15.6%), 无 NaN/Inf
- ckpt 强制每 epoch 末存 (R12), best ckpt @ ep109: `products/task450_issue150_stage3_long_train_200ep/adapter_200ep_BEST.pt` (273K bytes)
- 训练过程 R23 trigger 未触发

## 2. Stage 4 R@K 真实 eval (R23 trigger 后执行)

由于 Stage 3 训练 val_R@10_sim=0.1069 是 **similarity-based simulation** (训练时自定义 metric, 跟训练 loss 一致), 真实 Stage 4 R@K 用 argmax 严格 4-digit SID 匹配.

**Stage 4 双复跑实测** (scripts/task450_resume_stage4_only.py, GPU 0, R12 best ckpt load):
- precheck: SID SHA256=4654f3e22... ✅, T5 SHA256=56d046dba... ✅, Adapter SHA256=64d0abf8a... ✅
- 加载 best ckpt: alpha=4.56e-3, ZeroCenteredBoundedLinearResidual + first_input_ln unfreeze
- 测试集 24772 条
- **run1**: R@5=0.0, R@10=0.0, R@20=0.0, NDCG@5=0.0, NDCG@10=0.0, NDCG@20=0.0
- **run2**: R@5=0.0, R@10=0.0, R@20=0.0, NDCG@5=0.0, NDCG@10=0.0, NDCG@20=0.0
- **avg**: R@10=0.0000 (vs baseline 0.1020)

## 3. R23 触发分析

- val_R@10=0.0000 跨 2 个独立 eval run
- R23 触发条件满足
- R2 禁止 fallback 掩盖错误: 不能用"val_R@10_sim=0.1069 仿真可解释"绕过 Stage 4 R@10=0

## 4. 根因分析 (协议 split)

**val_R@10_sim (训练时仿真)** vs **Stage 4 R@K (真实 eval) protocol split**:

| 维度 | 训练 val_R@10_sim | Stage 4 R@K 真实 |
|------|-------------------|------------------|
| Metric | similarity 4-digit (近邻匹配, 宽松) | argmax strict 4-digit (精确匹配, 严格) |
| Forward | T5.generate + custom top-K 近邻 | encoder → decoder → argmax |
| 协议 | 跟训练 loss 同步下降 | 跟训练 loss 无直接关联 |

**根因**: ZeroCenteredLayerNormAdapter (跟 BoundedKappaScaleConditioner / BoundedWeightedMixedCurvatureConditioner 共享同一问题) 在 T5 forward path 上, 残差信号干扰 logits 计算, training loss 下降 ≠ argmax 解码 SID 准确. argmax 解码要求 4 个 digit token 全部正确, 而 wrapper 干扰导致 logits 偏离训练分布.

**R17 强制 4 Gate 决策**:
- Gate 1 (Stage 1 RQ-VAE/HRQVAE): ⏭️ N/A (沿用 Task #84 frozen SID)
- Gate 2 (Stage 2 Sinkhorn): ⏭️ N/A (沿用 Task #84 frozen SID)
- Gate 3 (Stage 3 T5-mini training): ✅ PASS (loss 1.89→1.60, val_R@10_sim=0.1069, ckpt 落盘 R12 OK)
- **Gate 4 (Stage 4 R@K eval): ❌ FAIL — R@10=0.0000 ≪ 0.1020 baseline**

## 5. 整体决策

**Issue #161 [方向C Gate4]: ❌ NO-GO 收口**

- **关键产物**:
  - commit: TBD (待 R15 push)
  - verdict: verdicts/task511_issue161_stage4_resume_result.md (本文)
  - verdict.json: products/task450_issue150_stage3_long_train_200ep/verdict.json (Stage 3 训练 trace)
  - stage4_verdict: products/task450_issue150_stage3_long_train_200ep/stage4_verdict_v2.json (R@K 0)
  - ckpt: products/task450_issue150_stage3_long_train_200ep/adapter_200ep_BEST.pt (R12 强制)
  - script: scripts/task450_issue150_stage3_long_train_200ep.py
  - resume script: scripts/task450_resume_stage4_only.py
  - log: logs/task450_issue150_stage3_long_train_200ep.log + logs/task450_resume_stage4_v2.log

- **R23 触发原因 (lm_head 路径错误)**:
  - Task #450 原脚本 line 477 使用 `model_wrapper.t5.lm_head(...)` (HG_Rec.t5 不存在 lm_head 直接属性)
  - 正确路径: `model_wrapper.t5.model.lm_head(...)` (T5ForConditionalGeneration lm_head 在 model.* 下)
  - Stage 4 crashed silently 时 (PID 3986398 消失, GPU 0 0%) 是 AttributeError 触发后被 python pipe 吞掉
  - resume 脚本改用 wrapper.forward() (内置 lm_head), 避免手写 lm_head 路径, 真实拿到 R@10=0

- **21 issue κ/scale 元数据适配全收口**:
  - 14 NO-GO: #157/#158/#162/#163/#165-#172/#173/#174
  - 2 PASS Gate 2: #175/#176
  - 2 PASS Gate 3: #177/#178
  - 3 NO-GO Gate 4: **#179 (方向A) / #181 (方向B) / #161 (方向C, 本 verdict)**
  - 0 OPEN pending (R16 强制 + R20+R21 闭环)

- **协议 split 关键洞察 (3 issue 一致)**:
  - 训练仿真 val_R@10_sim (similarity-based) vs 真实 Stage 4 R@K (argmax strict) 完全脱钩
  - 3 方向 (A κ-scale, B weighted-mixed, C zero-centered) 都达到 val_R@10_sim plateau (0.1069-0.1150), 但真实 R@10 都是 0
  - 后续 issue 必须 protocol split (训练仿真 + 真实 Stage 4 R@K)
