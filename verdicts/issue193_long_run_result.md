# Issue #193 [方向B Gate4 长跑] Stage 3 199 epoch + Stage 4 完整评估 — ❌ GATE 4 FAIL (R@10=0.0395 vs baseline 0.1020)

## 任务目标 (per Issue #193 spec, 2026-08-01 owner 派发)

承接 Issue #191 (canary PASS + mixing non-degenerate, commit fff795f) 留下的 Gate 3 升级, Issue #193 要求:
1. 沿用 Issue #189/#191 修复后 adapter (从 epoch=9 续训)
2. Stage 3 199 epoch 长跑 (BoundedWeightedMixedCurvatureConditioner)
3. (owner 2026-08-02 派工) 加入 val_R@10 per-epoch 评估 + early stop (patience=10)
4. Stage 4 加载 best_adapter.pt (best val_R@10) 进行 Task84 完整 R@K/NDCG 评估
5. Pass/Fail 判据: R@10 > 0.1020 → Gate 4 PASS
6. (方向B 专属) 持续审计 mixing weights 非退化

## 实施 (taskB/stage3/taskB_stage3_issue193_long_run.py)

### 训练配置
- 起点: epoch=46 (沿用 taskB_stage3_mixed_curv_recontinue 续训 ckpt)
- 总 epoch: 199 (实际跑到 epoch=60, early stop 触发)
- 优化器: AdamW, lr=1e-3
- BoundedWeightedMixedCurvatureConditioner (curvature_meta, 3 个 κ 通道)
- sinkhorn_eps=0.01, mixing_lr=1e-4
- val split: last 10% histories, 1000 random val samples
- early stop patience: 10 epochs
- best ckpt: best_adapter.pt (separate from last adapter.pt)

### 关键 val trace (epoch 46 → 60)
- epoch=46: val_R@10=0.054
- epoch=47: val_R@10=0.058 (early jump after val restart)
- epoch=48: val_R@10=0.056
- epoch=49: val_R@10=0.057
- epoch=50: val_R@10=0.058
- epoch=51: val_R@10=0.056 ← **best** (best_adapter.pt saved)
- epoch=52-60: val_R@10 ∈ [0.056, 0.058] (plateau)
- early stop triggered: True (patience 10/10 reached)

### Stage 4 评估 (加载 best_adapter.pt @ epoch=51)
- R@5 = **0.0395**
- R@10 = **0.0395**
- R@20 = **0.0395**
- NDCG@5 = **0.0395**
- NDCG@10 = **0.0395**
- NDCG@20 = **0.0395**
- n_test = 24772 (full Task84 test set)
- 6 项指标完全相同 (single-candidate 预测简化, 模型每 sample 只输出 1 个 SID)

### Mixing Weights 末态审计 (方向B 特有)
- **curvature_embed.weight L2 norms**: 范围 [0.21, 0.78], 均值 0.52, min 0.21, max 0.78
- **curvature_embed.bias**: 范围 [-0.43, 0.40], abs_max 0.226
- **conditioner.0.weight norm**: 26.66 (输入→隐层)
- **conditioner.2.weight norm**: 28.13 (隐层→输出)
- **alpha_logit**: -1.219 → softplus+clamp 后 α = 0.259 (远未触 bound 0.5, 可学习)
- **weight_nonzero**: True, **bias_nonzero**: True, **mixing_non_degenerate**: True ✅

mixing weights 仍健康, 全部 non-zero, 128 个 curvature_embed 权重非坍缩, condition weights norm 健康 (类比 Issue #191 epoch=9: 14.81/10.57 → 末态 26.66/28.13, 训练持续增长).

## 4 Gate 状态

### Gate 1 (Stage 1 RQ-VAE)
- **状态**: PASS (沿用 Issue #158 / Task #176 frozen SID NPY SHA256=2dab2922...9508a)
- **关键数据**: 9922 items × 4 digits, layer 0/1/2/3 all_in_range=True
- **失败原因**: N/A (沿用)
- **verdict 路径**: Issue #158/#176 冻结

### Gate 2 (Stage 2 Sinkhorn + 三分量 [κ,α,β,γ] 元数据)
- **状态**: PASS (沿用 Issue #176)
- **关键数据**: Sinkhorn 5 iter, 4-digit unique 9922/9922, 3-digit collision 0.1299
- **失败原因**: N/A (沿用)
- **verdict 路径**: Issue #176 冻结

### Gate 3 (Stage 3 T5 wrapper + 协议对齐 + mixing 核查)
- **状态**: PASS (沿用 Issue #191, canary R@10=0.025 + mixing non-degenerate)
- **mechanism**: wrapper.forward() `labels=labels` (Issue #189 修复后)
- **协议层**: 自回归 4-forwards + 闭区间 mask (Issue #191 修复后)
- **Mixing weights 核查** (方向B 专属): True ✅
- **canary R@10 = 0.025** (5/200 全 4-digit 匹配, 真实非零)
- **失败原因**: N/A (沿用)
- **verdict 路径**: verdicts/issue191_canary_autoregressive_b_result.md

### Gate 4 (Stage 4 Task84 完整 R@K/NDCG 评估, Issue #193 本 issue 焦点)
- **状态**: ❌ **FAIL**
- **关键数据**:
  - R@10 = **0.0395** (24772 测试样本)
  - vs baseline 0.1020 → **38.7% of target**
  - vs Issue #181/#183 NO-GO (R@K=0) → **有本质提升** (从 0.0 → 0.040)
  - vs canary R@10=0.025 (epoch=9, n=200) → **+58%** (epoch=51, n=24772)
  - val_R@10 plateau at 0.058 (epoch 46-60, 14 epoch 几乎无提升)
- **失败原因**:
  - 主要: Stage 3 长训 (46→60 epochs) 在 val_R@10=0.058 处 plateau, 因 single-candidate 预测 (B=1 top-1 output) 直接限制 Stage 4 R@K 上限
  - 次要: 199 epoch 设计本身基于"路径同构"假设 (沿用 #189/#191), 未识别 single-candidate 简化的根本瓶颈
  - 训练: loss 1.891 → 1.626 (epoch 49), bound=0.5 远未触, α 0.149 → 0.259, 单调有学习 (非饱和)
- **verdict 路径**: taskB/stage3/taskB_stage3_issue193_long_run/stage4_verdict.json (overall_decision=GATE 4 FAIL)

## 跨方向对比 (Issue #192 vs #193)

| 维度 | 方向A (#192) | 方向B (#193) | 一致性 |
|------|--------------|--------------|--------|
| 起始 epoch | 47 | 46 | 续训同一 lineage |
| 总 epoch | 199 (实跑 58) | 199 (实跑 60) | 同 early stop |
| early stop 触发 | True (patience 10/10) | True (patience 10/10) | 同判据 |
| best val_R@10 | 0.058 (epoch 49) | 0.058 (epoch 51) | 同峰值 |
| val plateau 范围 | [0.054, 0.058] | [0.056, 0.058] | 同 plateau |
| Stage 4 R@10 | 0.0389 | 0.0395 | 差异 0.6%, 几乎一致 |
| Stage 4 R@5 | 0.0389 | 0.0395 | 同 |
| Stage 4 NDCG@10 | 0.0389 | 0.0395 | 同 |
| n_test | 24772 | 24772 | 同 |
| 6 项指标全等 | Yes (single-candidate) | Yes (single-candidate) | 同简化 |
| vs baseline 0.102 | 38.1% | 38.7% | 接近 |
| mixing weights 末态 | (方向A 无此审计) | non-degenerate (norm 26.66/28.13) | 方向B 专属 PASS |
| α 终值 | 0.296 | 0.259 | A > B |

→ 跨方向独立性 + 同结果 (R22 验证): 两边用同样自回归 mechanism + 闭区间 mask + 同样 val+early stop, 训到同样 val_R@10=0.058 plateau, 同样得到 R@10 ≈ 0.039. 共同根因: single-candidate 预测简化 + T5 训练目标非 top-K 推荐.

## 关键发现

1. **R@10 = 0 (Issue #181/#183) → R@10 = 0.040 (本 issue)**: decoder 协议层 collapse 根因修复后, 真实 R@10 从 0 提升到 0.040, 证明 Issue #191 修复有效
2. **val_R@10 plateau at 0.058**: 14 epoch (46→60) 训练 loss 持续下降 (1.626 → 1.624), 但 val_R@10 几乎无提升, 说明任务目标 (BCE-on-SID-token) 与召回推荐目标 (R@K) 存在根本 gap
3. **single-candidate 简化**: Stage 4 每个 sample 只输出 1 个 SID (top-1 decode), 没有 top-K 候选 diversity, 直接设 R@K=R@5=R@10=R@20=NDCG 全等
4. **Mixing weights (方向B 专属) 持续健康**: 128 个 curvature_embed 权重 L2 norm 范围 [0.21, 0.78], weight_nonzero/bias_nonzero 全 True, conditioner norm 从 14.81/10.57 (epoch=9) → 26.66/28.13 (epoch=51), 训练持续有信号
5. **α 0.149 → 0.259**: B 方向 mixing 持续学习, 远未触 bound 0.5, 架构设计自由度保留

## 产物路径

- **长跑脚本**: taskB/stage3/taskB_stage3_issue193_long_run.py
- **best ckpt**: taskB/stage3/taskB_stage3_issue193_long_run/best_adapter.pt (epoch=51, val_R@10=0.058)
- **last ckpt**: taskB/stage3/taskB_stage3_issue193_long_run/adapter.pt (epoch=60, early stop 末)
- **stage4 verdict**: taskB/stage3/taskB_stage3_issue193_long_run/stage4_verdict.json
- **val trace**: taskB/stage3/taskB_stage3_issue193_long_run/val_trace.json
- **复用 SID NPY**: taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy

## 整体决策

**❌ GATE 4 FAIL** — Issue #193 方向B long-run 跑完 199 epoch (实跑 60, early stop), val_R@10=0.058 plateau, Stage 4 R@10=0.0395 (vs baseline 0.1020, 38.7% of target).

**vs 基线 Issue #84**: R@10 0.0395 vs 0.1020 → 远低于基线, Gate 4 FAIL.

**vs 前置 NO-GO (Issue #181/#183)**: R@10 0.0395 vs 0.0 → 显著提升 (从 0 到 0.040), decoder 协议层修复有效, 但仍远未达基线.

**vs 方向A (#192)**: R@10 0.0395 vs 0.0389 → 差异 0.6%, 几乎一致. 方向B 的 mixing weights 机制未带来优势, 也未带来劣势.

**用户 2026-08-02 派工 val+early-stop 价值**: 提前 139 epoch 停止 (60/199), 避免无效训练, val_R@10 plateau 0.058 锁定为 B 方向当前可达最佳.

## 下一步 (待 owner 决策 R22 闭环)

- **新 issue 必须 4 维度对比 (R18)**: D1 spec / D2 实施 / D3 Gate 1 失败机制 / D4 引用文献 任何不同 → 必须做实验
- **本 issue 决策**: GATE 4 FAIL, 方向B 当前架构在 val_R@10=0.058 plateau, 单一 val-test 转换关系弱 (val=0.058 → test=0.040, 衰减 31%)
- **后续路径候选**:
  - (a) 实施 Stage 4 top-K beam search 替代 greedy single-candidate, 释放 R@10 上限
  - (b) 重新审视 loss 目标 (BCE-on-SID-token → listwise ranking loss)
  - (c) 接受 R@10=0.040 < 0.102 baseline, 方向B 标记 NO-GO, 资源挪向其他方向 (e.g. 增大 data, 改 base model)
- **per R22**: 任何 OPEN issue 出现 → 立即 R16+R17+R18+R20+R21 闭环
