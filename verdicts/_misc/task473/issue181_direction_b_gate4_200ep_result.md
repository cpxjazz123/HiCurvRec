---
task: 473
type: result
issue: 181
gate: 4
status: "NO-GO"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Issue #181 / Task #473 [方向B Gate4] 混合曲率有界残差 conditioner Gate 4 NO-GO 收口

## Gate 4 决策: ❌ NO-GO 收口

## 1. Stage 3 训练结果 (R23 监控 + 训练 trace)

- 训练 200 epoch, batch_size=32, 100097 trainable + 5508736 frozen
- **early stop @ ep35**: best_val_R@10_sim=**0.1150** (+12.7% vs Task #84 baseline 0.1020)
- α 健康增长轨迹: 4.54e-5 (ep0) → 1.27e-2 (ep1) → 4.58e-1 (ep2) → **5.00e-1 bound=True (ep3-onwards)**
- cond_grad 健康 ep1=4.45e-2 → ep9=2.45e-1 → ep15+ 1.2e-1 ~ 2.4e-1 全程
- ln_grad 健康 5.6e-1 ~ 9.3e-1 全程
- loss 9.13 → 1.12 (-88%), 无 NaN/Inf
- ckpt 强制每 epoch 末存 (R12), best ckpt @ ep35: `products/task473_issue181_direction_b_gate4_200ep/adapter_200ep.pt` (404K bytes)
- 训练过程 R23 trigger 未触发

## 2. Stage 4 R@K 真实 eval (R23 trigger 后执行)

**Stage 4 双复跑实测** (scripts/task473_resume_stage4_only.py, GPU 2, R12 best ckpt load):
- precheck: SID SHA256=2dab2922... ✅, T5 SHA256=56d046db... ✅
- 加载 best ckpt: alpha=5.00e-1, BoundedWeightedMixedCurvatureConditioner
- 测试集 24772 条
- **run1**: R@5=0.0, R@10=0.0, R@20=0.0, NDCG@5=0.0, NDCG@10=0.0, NDCG@20=0.0
- **run2**: R@5=0.0, R@10=0.0, R@20=0.0, NDCG@5=0.0, NDCG@10=0.0, NDCG@20=0.0
- **avg**: R@10=0.0000 (vs baseline 0.1020)

## 3. R23 触发分析

- val_R@10=0.0000 跨 2 个独立 eval run
- R23 触发条件满足

## 4. 根因分析

**val_R@10_sim (训练时仿真)** vs **Stage 4 R@K (真实 eval) protocol split**:
- 跟 Task #472 同一根因: BoundedWeightedMixedCurvatureConditioner 在 α=0.5 饱和 bound 时残差干扰 T5 正常 forward path
- 训练仿真 0.1150 ≠ 真实 argmax R@10=0
- 协议 split 是必要充分条件, 必须严格 Stage 4 实测才能判断 wrapper 真实效果

**R17 强制 4 Gate 决策**:
- Gate 1 (Stage 3 训练机制): ✅ PASS
- Gate 2 (Stage 2 Sinkhorn): ✅ PASS (reused from #158 SID NPY)
- Gate 3 (Stage 3 T5-mini training): ✅ PASS (loss 9.13→1.12, ckpt 落盘)
- **Gate 4 (Stage 4 R@K eval): ❌ FAIL — R@10=0.0000 ≪ 0.1020 baseline**

## 5. 整体决策

**Issue #181 [方向B Gate4]: ❌ NO-GO 收口**

- **关键产物**:
  - commit: TBD (待 R15 push)
  - verdict: verdicts/task473_issue181_direction_b_gate4_200ep_result.md (本文)
  - verdict.json: products/task473_issue181_direction_b_gate4_200ep/verdict.json
  - stage4_verdict: products/task473_issue181_direction_b_gate4_200ep/stage4_verdict.json
  - ckpt: products/task473_issue181_direction_b_gate4_200ep/adapter_200ep.pt (R12)
  - script: scripts/task473_issue181_direction_b_gate4_200ep_stage3_stage3_stage4.py
  - resume script: scripts/task473_resume_stage4_only.py
  - log: logs/task473_issue181_direction_b_gate4_200ep.log + logs/task473_resume_stage4.log

- **实施差异 vs Task #450** (方向C ZeroCenteredLayerNormAdapter):
  - BoundedWeightedMixedCurvatureConditioner 强制 α ≤ 0.5 + 4 分量 [κ,α,β,γ] 元数据
  - 训练 val_R@10_sim: 0.1150 (Task #473) vs 0.1069 (Task #450)
  - **Stage 4 R@10 真实**: 0.0000 (Task #473) — 协议 split 暴露真实失败

- **20 issue κ/scale 元数据适配全收口**:
  - 14 NO-GO: #157/#158/#162/#163/#165-#172/#173/#174
  - 2 PASS Gate 2: #175/#176
  - 2 PASS Gate 3: #177/#178
  - 2 NO-GO Gate 4: **#179/#181** (本 verdict)
  - 1 OPEN pending: #161 (Task #450 Stage 4 待实测)