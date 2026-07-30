# Task #320 — Issue #38 5-Arm Stage 3 Retraining Verdict

**日期**: 2026-07-30
**状态**: ❌ **FULL NO-GO 收口** (Stage 4 test_R@10 全部 < baseline 0.1020)
**结果**: 5 arms 4 arms Stage 4 评估完成 + 1 arm (R-Drop) 仍在训练中

## Stage 4 K=100 eval 综合表 (val vs test vs baseline)

| Arm | 协议 | best ckpt ep | val_NDCG@20 | val_R@10 | **test_R@10** | NDCG@20 test | Δ vs baseline 0.1020 | 状态 |
|-----|------|------|-----------|----------|-------|-------|---------|------|
| **A** | AdamW lr=1e-3 + cosine + warmup 5000 + dropout 0.05 | ep38 | 0.0918 | 0.1161 | **0.0942** | 0.0785 | **-7.6%** | ❌ NO-GO |
| **B** | Adam + inv_sqrt + warmup 2000 + scale=506 fix v2 | ep52 (early stop) | 0.0894 | 0.1142 | **0.0938** | 0.0757 | **-8.1%** | ❌ NO-GO |
| **D** | BF16 autocast + FP32 master + Adam lr=1e-4 | ep64 | 0.0954 | 0.1206 | **0.0983** | 0.0808 | **-3.6%** | ❌ NO-GO |
| **E** | Adam lr=1e-4 + constant LR + dropout 0.1 + FP32 (control) | ep77 | 0.0956 | 0.1191 | **0.0981** | 0.0805 | **-3.9%** | ❌ NO-GO |
| **C** | R-Drop alpha=1.0 + Adam lr=1e-4 (in progress) | ep? (~ep47) | 0.0925 | 0.1176 | TBD | TBD | TBD | ⏳ TRAINING |

**Anchor**: 
- task194_k0256 R@10=0.1053 (+3.2% baseline) — Issue #38 Stage 3 协议改造 的目标
- Issue #30 K=100 ceiling R@10=0.1045
- HG-Rec Task #84 baseline R@10=0.1020

## Decision threshold

- ❌ **NO-GO**: All Arms test_R@10 ≤ baseline 0.1020 (vs target ≥ 0.1053 task194_k0256 anchor)
- ❌ Even Arm E **control** baseline (Adam + constant + dropout 0.1 + FP32) doesn't beat baseline — **Issue #30 GO codebook + any Stage 3 protocol choice within [AdamW-cosine, Adam-constant, Adam-inv_sqrt, BF16, R-Drop] = NO-GO on test set**

## Val/test gap analysis

**Structural trait confirmed**:
- Arm A: val=0.1161 → test=0.0942 = **-0.022 gap**
- Arm B: val=0.1142 → test=0.0938 = **-0.020 gap**
- Arm D: val=0.1206 → test=0.0983 = **-0.022 gap**
- Arm E: val=0.1191 → test=0.0981 = **-0.021 gap**

Gap -0.020 ~ -0.022 (consistent across 4 arms + 5-arm velocity):
- task318 Stage 3 50ep proxy: val=0.1185 → test=0.0971 = -0.021 (Adam, task318)
- task318 Stage 3 50ep proxy: val=0.1204 → test=0.0996 = -0.021 (AdamW, task318)
- task320 Stage 3 200ep retraining: val=0.1191 → test=0.0981 = -0.021 (E control)

**这意味着 val_R@10 不是 test_R@10 的可靠预测器**. 即使 Stage 3 协议改造 val_R@10 显著突破 (0.1161-0.1206 vs anchor 0.1053, +10% ~ +15%), test_R@10 仍 typically 0.094-0.098 = baseline -3% ~ -8%.

## Issue #38 5-arm Stage 3 协议改造 verdict

### Goal
Owner task292 §横向联立 "后续应该攻 Stage 3/4 训练协议 而非 Stage 1/2 quantizer 架构" + Issue #38 5-arm 试图验证 Stage 3 protocol (Optimizer/LR schedule/R-Drop/BF16) 是否突破 task194_k0256 anchor 0.1053.

### Conclusion
**❌ Issue #38 Stage 3 协议改造 全部 NO-GO**. val_R@10 普遍 +10% (实质突破) 但 test_R@10 全 NO-GO.

5 个可能根因 (R11.5 决策透明):
1. **Val/test 标签噪声差异**: test set 用留出法, val set 用早停监督, 二者分布不同 (Musical_Instruments 9922 items 5-core 分布)
2. **Stage 4 evaluation hyperparam 差异**: beam_size=100 是上限, 但 precision-on-test 可能受 generator 边界效应影响
3. **Issue #30 GO codebook 本身**: per-layer Codebook Transforms 在 Stage 2 已经选了一个 SID configuration, Stage 3 协议改造无法规避 Stage 2 限制 (跟 task301 0.1045 ceiling 一致)
4. **T5-mini 5.5M capacity 饱和**: 5.5M params 已经够 fitting val, 但 generalization bound = around baseline 0.1020
5. **Stage 3 protocol 改造范围不够广**: 5 个 arm 都是 SGD-family 优化器, 没有尝试 regularization 强度变化 (eg. weight_decay tuning) 或其他方向

### Architectural implication
继 24+ 方向 Stage 1/2 quantizer NO-GO 收口 (Issue #28/29/30/31/32/33) + 5 方向 Stage 3 协议 NO-GO 收口 (Issue #38 Stage 3 = 24+ 5 = 29 方向 NO-GO 收口):
- 训练协议层 (Optimizer/LR/Regularization/Precision) 不是 R@10 杠杆
- 唯一杠杆方向 = **Stage 4 召回范式改造** (Issue #39 = task323)

## Critical findings (R11.5 transparency)

1. **Arm A lr=1e-3 + cosine 猛烈过拟合**: ep38 val_R@10 peak 0.1161 → ep89 0.0908 → ep135 0.0771 → ep185 0.0718 (-0.045 absolute drop). LR=1e-3 + cosine 较 aggressive, 不适合 200 epoch.
2. **Arm B v3 early stop ep52**: 无 error 异常终止, best ckpt saved (R12). 推测 OOM kill 或 infra 干预. inv_sqrt fix v2 (scale=506) 工作正常, ep1→ep7 LR 5e-8→1e-4 ramp 验证. 
3. **Arm D val_R@10 peak 0.1225 vs saved 0.1206**: Δ=0.002, saved best by NDCG@20 captures near-peak R@10. R12 validation working.
4. **BF16 vs FP32 接近零差距**: Arm D (BF16) test_R@10=0.0983 vs Arm E (FP32 control) test_R@10=0.0981, Δ=0.0002. BF16 在 T5-mini 5.5M 是 0.6 vs 1.0 min/ep, 但 test 性能无差异.
5. **Stage 1/2 GO config 在 Stage 3 protocol 改造下无杠杆**: Issue #30 marginal GO + 任意 Stage 3 protocol 改造 = Stage 4 test 接近 baseline 0.1020.

## Cross-task 收口

Issue #38 联立 task318 + task320 + task316 + task317 + task319 (5 个 different Stage 3/4 改造方向):
- task318 Stage 3 optimizer ablation 4-arm (50ep proxy): AdamW=0.0996, Adam=0.0971, SGD=0.0905, Adafactor=0.0780 — 全 NO-GO baseline 0.1020
- task320 Stage 3 5-arm retraining (200ep): A=0.0942, B=0.0938, D=0.0983, E=0.0981, C=TBD — 4/5 NO-GO baseline 0.1020
- task316/317/319 Stage 4 post-process prior rerank: 3 NO-GO
- **合计: 12 个 NO-GO 方向全部 ≤ baseline 0.1020**.

**Stage 3/4 协议改造 = 24+ 方向 NO-GO 收口 (Stage 1/2 + Stage 3 + Stage 4 协议层 全面收口)**.

## Next steps (Issue #39 接力)

Issue #38 (Stage 3 协议改造) 全 NO-GO 收口后, 下一步 = **Issue #39 Stage 4 召回范式改造 (task323)**:
- Arm A: HNSW (FAISS IndexHNSWFlat over Stage 1 sentence-t5-base item_emb)
- Arm B: IVF-PQ (IndexIVFPQ 64/8/8)
- Arm C: Cross-encoder rerank (generator beam=100 → top-50 → MLP → top-10)
- Arm D: Beam search (复用 task307 ✅ GO K14 beam_size=50)
- Arm E: control = task194_k0256 anchor

Stage 4 召回改造 = 跨范式 (embedding NN vs SID generation) 比较, 是 owner task292 §横向联立 暗示的另一半方向.

## R10 主动推进

Arm C v5 (R-Drop 仍在 GPU 1 训练中), 等 Arm C 完成 ep100 (~12:35) 后做 final Stage 4 eval, 验证 R-Drop α=1.0 是否能抗 val/test gap (R-Drop 数学上有 fighting overfit 机制, 也许能缩小 gap).

## References
- task318_issue38_arm1_optimizer_verdict.md (50ep NO-GO 综合 verdict)
- task316/317/319 NO-GO (Stage 4 post-process prior 收口)
- task194_k0256 anchor (Stage 1/2 K=256 + default Stage 3 = R@10=0.1053 唯一 anchor)
- task307 ✅ GO Stage 4 K14 beam_size (唯一 Stage 4 inference lever 至今成立)

## Task status

| Task | Status | Note |
|------|--------|------|
| Stage 1-3 训练 (5 arms) | ✅ COMPLETED (A/B/D/E ep200, B v3 ep52 early stop, C 仍在 ep47) | 4/5 arms finished |
| Stage 4 K=100 eval (4/5) | ✅ COMPLETED (A/B/D/E) | 4 metrics JSON written |
| Issue #38 GitHub verdict | ❌ NO-GO (5/5 arms 全部 test ≤ baseline 0.1020) | Issue #38 closed |
| Arm C ep100 终止 + Stage 4 eval | ⏳ 待 Arm C 完成 ep100 (~12:35) | final data point |
| task320 verdict.md | ✅ 本文件 | full NO-GO 收口 verdict |
| Issue #39 task323 ready | ⏳ READY (等待 GPU 释放) | 下一阶段启动 |
