---
type: precheck
issue: 2
created: 2026-08-02
tags:
  - kappa
up: "[[index]]"
---
# Issue #2 precheck verdict — 方向A κ/尺度有效性诊断
**Generated**: 2026-08-02 (loop tick, follow loop.md R26)
**Issue**: #2 [方向A Gate4后续] κ/尺度有效性诊断与单seed重评估
**Loop §16 row**: `taskA_g4_retry`
**Commit**: `d14c5b1` (R26)

---

## 1. precheck 目的

确认 taskA 主训练路径**未**违反 Issue #2 红线：
- (a) 未引入 **global κ** (single shared kappa across layers)
- (b) 未变成 **fixed-only** (κ frozen in main path)
- (c) 未走 **纯欧氏旁路** (Euclidean-only / skip_kappa)

## 2. 扫描结果

### 2.1 禁用模式 grep (taskA + taskB, 跨方向交叉验证)

| Pattern | taskA hits | taskB hits | 评估 |
|---------|-----------|-----------|------|
| `global_kappa` / `shared_kappa` / `single_kappa` | 0 | 0 | ✅ 无 |
| `pure_euclidean` / `euclidean_only` / `no_kappa` / `skip_kappa` | 0 | 0 | ✅ 无 |
| `fixed_kappa` / `frozen_kappa` (main path) | 0 | 0 | ✅ 无 |
| `kappa.requires_grad = False` (any) | 1 (Phase 4 ablation) | 1 (Phase 4 ablation) | ⚠️ 见 §2.2 |

### 2.2 `kappa.requires_grad = False` 上下文审查

**taskA/stage2/taskA_stage2_kappa_sync.py:524**:
```python
# ── Phase 4: 对照消融 (关闭同步重校准) ──
no_recal_model = KappaAwareHRQVAE(...).to(device)
# 模拟 "不重校准" 行为: 让 c 冻结为 init=1.0 (no κ update effective)
for q in no_recal_model.vq_layers:
    q.kappa.requires_grad = False
print("Ablation: κ frozen, no sync recalibration (对照)")
```

**结论**: 出现在 **Phase 4 对照消融** (intentional ablation comparison)。Main path 训练时 κ 是 `nn.Parameter(torch.tensor(0.0, ...))` (line 111), 默认 `requires_grad=True`。✅ 不算 fixed-only。

### 2.3 κ per-layer 验证

- `taskA/stage2/taskA_stage2_kappa_sync.py:111`: `self.kappa = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))` — **每个 VQ layer 实例独立持有自己的 kappa parameter**
- `taskA/stage2/taskA_stage2_kappa_sync.py:219`: `self.vq_layers = nn.ModuleList([...])` — 3 层 VQ layer (对应 L0/L1/L2)
- `taskA/stage2/taskA_stage2_kappa_sync.py:429`: `grads_kappa = torch.autograd.grad(total_loss, [q.kappa for q in precheck_model.vq_layers], ...)` — **per-layer κ 独立计算梯度**
- `taskA/stage2/taskA_stage2_kappa_vq_fix.py:104`: `self.kappa_logit = nn.Parameter(torch.tensor(kappa_init_logit))` — 备选 logit 参数化方案, 同样是 per-layer

✅ 确认三层独立 learnable κ (L0 K64, L1 K128, L2 K256) 在 forward 路径中独立持有、独立优化。

## 3. 前序 canary Gate 3 状态（已通过）

参考 `taskA/stage3/taskA_stage3_kappa_scale_recontinue/verdict.json`:

| 指标 | 值 | 评估 |
|------|----|------|
| Gate 3 PASS | true | ✅ |
| epoch 0→9 loss | 1.8912 → 1.8878 | ✅ 下降 |
| cond_grad nonzero all | true | ✅ |
| ln_grad nonzero all | true | ✅ |
| alpha bounded all | true (max=0.5) | ✅ |
| final α | 0.0998 | ✅ 远低于 bound |
| NaN/Inf all | false (干净) | ✅ |
| SID hash match | true (`2dab29...`) | ✅ |
| labels bug fix | applied | ✅ |

参考 `taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter_init_proof.json`:
- `max_diff_x_emb = 1.81e-05` (~0, 即 init 时 adapter 输出 ≈ emb 本身, 正确)
- `alpha_value = 4.54e-05` (init 时 α 极小, 几乎纯 bypass)
- `alpha_max_bound = 0.5`

## 4. 现有产物清单

- `taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter.pt` (535615 bytes, recontinue epoch 9, α=0.0998)
- `taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt` (536217 bytes, 长跑 best ckpt, Gate 4 R@10=0.0389 FAIL)
- `taskA/stage4/taskA_stage4_canary_argmax/canary_verdict.json`
- `taskA/stage4/taskA_stage4_canary_autoregressive/canary_verdict.json`

## 5. precheck 决策

**PRECHECK: PASS**

主训练路径满足 Issue #2 红线要求：
- ✅ 三层独立 learnable κ (per-layer `nn.Parameter`, NOT global)
- ✅ Main path κ 可训练 (Phase 4 `requires_grad=False` 仅限 ablation)
- ✅ 无 pure Euclidean bypass
- ✅ κ 梯度实际进入 forward (per-layer autograd.grad 验证)
- ✅ SID 流可复现 (hash match `2dab29...`)
- ✅ α bounded + 缓慢增长 (canary 9 epoch 内 0 → 0.0998, 远低于 0.5 bound)

## 6. 下一步 → Gate 1 → Gate 2 → Gate 3 已经在 canary 通过，仅缺 Gate 4 单seed 完整评估

按 Issue #2 spec:
1. ~~precheck~~ ✅ 本 verdict
2. **Gate 1**: 验证 Stage 1 三层独立 κ/scale metadata + hash（指向 Stage 1 embedding + Stage 2 SID 流，需现场跑一次）
3. **Gate 2**: 三层 κ 梯度非零、尺度与距离同步重校准、SID 流可复现（已由 canary verdict 部分证明，需要现场再跑一次严格 Gate 2 脚本）
4. **Gate 3**: 仅 Gate 2 PASS 后接 Stage 3 T5，并保存 best_adapter.pt（canary 已有 adapter.pt）
5. **Gate 4**: 仅 Gate 3 PASS 后单seed Task84 正式评估，需报告 R@5/R@10/R@20 + NDCG@5/NDCG@10/NDCG@20；阈值 R@10 > 0.1020 才 Target reached

**当前 tick 进度**: precheck 完成。下一 tick 应启动 Gate 1 实际脚本（不是 canary recontinue，是 Issue #2 spec 要求的正式 Gate 1 验证）。

## 7. 判定

- precheck: **PASS** (红线全过, 不需修改代码)
- 可以继续: Gate 1 → Gate 2 → Gate 3 (canary 已有) → Gate 4
- 决策阈值: Gate 4 R@10 > 0.1020 = Target reached (否则 Gate 4 FAIL, 需新 issue 重新审视)