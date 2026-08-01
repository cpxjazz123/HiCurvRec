# Issue #179 / Task #472 [方向A Gate4] κ-scale 有界残差 conditioner Gate 4 NO-GO 收口

## Gate 4 决策: ❌ NO-GO 收口

## 1. Stage 3 训练结果 (R23 监控 + 训练 trace)

- 训练 200 epoch, batch_size=32, 132737 trainable + 5508736 frozen
- **early stop @ ep35**: best_val_R@10_sim=**0.1150** (+12.7% vs Task #84 baseline 0.1020)
- α 健康增长轨迹: 4.54e-5 (ep0) → 1.27e-2 (ep1) → 4.58e-1 (ep2) → **5.00e-1 bound=True (ep3-onwards)**
- cond_grad 健康 ep1=4.34e-2 → ep9=5.72e-3 (transient dip) → ep10-12 1e-4 → ep13 1.03e-6 → ep14 8.36e-8 → ep15+ 恢复 3-9e-2
- ln_grad 健康 5.5e-1 ~ 7.2e-1 全程
- loss 9.13 → 1.04 (-89%), 无 NaN/Inf
- ckpt 强制每 epoch 末存 (R12), best ckpt @ ep35: `products/task472_issue179_direction_a_gate4_200ep/adapter_200ep.pt` (532K bytes)
- 训练过程 R23 trigger 未触发 (loss 健康 + cond_grad 短暂塌缩但 ep15+ 恢复)

## 2. Stage 4 R@K 真实 eval (R23 trigger 后执行)

由于 Stage 3 训练 val_R@10_sim=0.1150 是 **similarity-based simulation** (训练时自定义 metric, 跟训练 loss 一致), 真实 Stage 4 R@K 用 argmax 严格 4-digit SID 匹配. R18 强制 Gate 4 必须实测 R@10 > 0.1020 baseline.

**Stage 4 双复跑实测** (scripts/task472_resume_stage4_only.py, GPU 1, R12 best ckpt load):
- precheck: SID SHA256=2dab2922... ✅, T5 SHA256=56d046db... ✅
- 加载 best ckpt: alpha=5.00e-1, BoundedKappaScaleConditioner
- 测试集 24772 条
- **run1**: R@5=0.0, R@10=0.0, R@20=0.0, NDCG@5=0.0, NDCG@10=0.0, NDCG@20=0.0
- **run2**: R@5=0.0, R@10=0.0, R@20=0.0, NDCG@5=0.0, NDCG@10=0.0, NDCG@20=0.0
- **avg**: R@10=0.0000 (vs baseline 0.1020)

## 3. R23 触发分析

- val_R@10=0.0000 跨 2 个独立 eval run (run1 + run2 都是 0)
- R23 触发条件满足: val_R@10=0 跨 ≥2 评估
- R2 禁止 fallback 掩盖错误: 不能用"val_R@10_sim=0.1150 仿真可解释"绕过 Stage 4 R@10=0

## 4. 根因分析

**val_R@10_sim (训练时仿真)** vs **Stage 4 R@K (真实 eval) protocol split**:

| 维度 | 训练 val_R@10_sim | Stage 4 R@K 真实 |
|------|-------------------|------------------|
| Metric | similarity 4-digit (近邻匹配, 宽松) | argmax strict 4-digit (精确匹配, 严格) |
| Forward | T5.generate + custom top-K 近邻 | encoder → decoder → argmax |
| 协议 | 跟训练 loss 同步下降 | 跟训练 loss 无直接关联 |

**根因**: BoundedKappaScaleConditioner 在 α=0.5 (饱和 bound) 时, 残差信号幅度太大干扰 T5 正常 forward path. training loss 下降 ≠ generate/R@K 提升. argmax 解码要求 4 个 digit token 全部正确, 而残差干扰导致 logits 偏离训练分布.

**R17 强制 4 Gate 决策**:
- Gate 1 (Stage 3 训练机制): ✅ PASS (L0/L1/L2 usage 100%, ckpt 落盘 R12 OK)
- Gate 2 (Stage 2 Sinkhorn): ✅ PASS (reused from #157 SID NPY)
- Gate 3 (Stage 3 T5-mini training): ✅ PASS (loss 9.13→1.04, ckpt 落盘)
- **Gate 4 (Stage 4 R@K eval): ❌ FAIL — R@10=0.0000 ≪ 0.1020 baseline**

## 5. 整体决策

**Issue #179 [方向A Gate4]: ❌ NO-GO 收口**

- **关键产物**:
  - commit: TBD (待 R15 push)
  - verdict: verdicts/task472_issue179_direction_a_gate4_200ep_result.md (本文)
  - verdict.json: products/task472_issue179_direction_a_gate4_200ep/verdict.json (Stage 3 训练 trace)
  - stage4_verdict: products/task472_issue179_direction_a_gate4_200ep/stage4_verdict.json (R@K 0)
  - ckpt: products/task472_issue179_direction_a_gate4_200ep/adapter_200ep.pt (R12 强制)
  - script: scripts/task472_issue179_direction_a_gate4_200ep_stage3_stage4.py
  - resume script: scripts/task472_resume_stage4_only.py
  - log: logs/task472_issue179_direction_a_gate4_200ep.log + logs/task472_resume_stage4.log

- **实施差异 vs Task #450** (方向C ZeroCenteredLayerNormAdapter):
  - BoundedKappaScaleConditioner 强制 α ≤ 0.5 bound + κ/scale 元数据
  - 训练 val_R@10_sim: 0.1150 (Task #472) vs 0.1069 (Task #450, +7.6% simulation)
  - **Stage 4 R@10 真实**: 0.0000 (Task #472) vs Task #450 (待 Stage 4 实测, R23 monitor 等中)
  - 协议 split: 训练仿真 ≠ 真实 argmax R@K

- **后续建议**: protocol split = 训练仿真 vs 真实 Stage 4 必须同时跑才能判断 wrapper 真实效果. val_R@10_sim 是必要不充分条件.