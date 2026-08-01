# Task #472 / Issue #179 [方向A Gate4] κ-scale 有界残差 conditioner 完整 Stage3 200 epoch + Stage 4

## 任务摘要

Issue #179 owner 2026-08-01 派发方向A Gate4: 复用 #177 (Task #470) BoundedKappaScaleConditioner (α=softplus(α_logit).clamp(max=0.5)) 完整 Stage 3 200 epoch + Stage 4 R@K 双复跑 (R@5/10/20, NDCG@5/10/20). 跟 Task #450 (方向C, α exponent α_e init 不限) 平行, 实施差异 = 强制 α ≤ 0.5 有界残差.

R18 4 维度对比 vs Task #450 (方向C 200 epoch):
- **D1 spec**: Issue #179 方向A vs Issue #150 方向C — 同一 long train 协议, 不同 wrapper
- **D2 实施**: BoundedKappaScaleConditioner (α 有界 + κ+scale 元数据) vs ZeroCenteredLayerNormAdapter (无 α clamp) → **不同**
- **D3 Gate 4 决策**: 方向A R@10 > 0.1020 vs 方向C R@10 > 0.1020 — 平行对照
- **D4 引用**: 同一族 + Issue #179 spec 强制复用 #177 wrapper

→ D1/D2/D3/D4 显著不同 (wrapper 差异 + 平行对照), R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/best_loss_model.pth` (sha256=59a38fa3...)
- **SID NPY**: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy` SHA256=2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a
- **T5 ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth` (Task #84 frozen)
- **wrapper**: `HG_Rec_with_BoundedAdapter` (from #470)
  - BoundedKappaScaleConditioner: α = softplus(α_logit).clamp(max=0.5) 强制有界
  - κ + scale 元数据编码 + SID token projection + 3-layer MLP direction (tanh bounded)
  - T5 主干冻结, 仅 `encoder.block.0.layer.0.layer_norm` + conditioner 解冻
- **GPU**: GPU 1 (R7 满足, GPU 0 task450, GPU 2 留给 task473, GPU 3 留 Issue #161 backup)

## Gate 顺序 (per Issue #179 spec)

- **precheck**: 5 项 — ① SID SHA256 = #157/#177 ✅; ② wrapper 类加载成功; ③ α=0 init; ④ trainable/frozen 数合理; ⑤ Stage 3 训练开始
- **Stage 3 (200 epoch)**: 跟 task450 同规格, 完整 200 epoch. early stop patience=5
- **Stage 4 (R@K 双复跑)**: R@5/10/20, NDCG@5/10/20 全 6 项, run1 + run2 双复跑, baseline R@10=0.1020 对照
- **Gate 4 验收**: test R@10 > 0.1020 + 6 项指标齐全 → [TARGET REACHED]

## R23 强制

val_R@10=0 跨 ≥2 epoch / α 超 0.5 跨 ≥2 epoch / cond_grad=0 跨 ≥3 epoch / NaN/Inf → kill + NO-GO

## R12 强制

每 epoch 末强制存 ckpt (删旧 + 存新) + PID 文件

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案 = 严格按 Issue #179 spec:
1. seed=42 (per R5 数据集硬约束)
2. batch_size=32, 200 epoch, LR_conditioner=1e-3, LR_LN=1e-4 (跟 task450 同规格)
3. α=softplus(α_logit).clamp(max=0.5) 强制有界
4. early stop patience=5 (per owner 2026-08-01 派工)
5. Stage 4 R@K double-run (run1 + run2, 取平均)
6. baseline R@10=0.1020 对照
7. R12 强制 ckpt 落盘 (每 epoch 删旧 + 存新)
8. R143 reproducibility triangle: SHA256 ckpt + SID + parquet 三件套
9. R137 TRITON_CACHE_DIR per task 路径

## 产物路径

- **verdict**: `verdicts/task472_issue179_direction_a_gate4_200ep_result.md`
- **verdict.json**: `products/task472_issue179_direction_a_gate4_200ep/verdict.json`
- **stage4_verdict**: `products/task472_issue179_direction_a_gate4_200ep/stage4_verdict.json`
- **train_trace**: `products/task472_issue179_direction_a_gate4_200ep/train_trace_200ep.json`
- **adapter_200ep.pt**: `products/task472_issue179_direction_a_gate4_200ep/adapter_200ep.pt`
- **eval_run1.json**: `products/task472_issue179_direction_a_gate4_200ep/eval_run1.json`
- **eval_run2.json**: `products/task472_issue179_direction_a_gate4_200ep/eval_run2.json`
- **script**: `scripts/task472_issue179_direction_a_gate4_200ep_stage3_stage4.py`
- **log**: `logs/task472_issue179_direction_a_gate4_200ep.log`
