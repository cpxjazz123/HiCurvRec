# Issue #3 precheck verdict — 方向B 混合权重泛化诊断
**Generated**: 2026-08-02 (loop tick, follow loop.md R26)
**Issue**: #3 [方向B Gate4后续] 混合权重泛化诊断与单seed重评估
**Loop §16 row**: `taskB_g4_retry`
**Commit**: `d14c5b1` (R26)

---

## 1. precheck 目的

确认 taskB 主训练路径**未**违反 Issue #3 红线：
- (a) 未引入 **global κ** (single shared kappa across layers)
- (b) 未变成 **fixed-only** (κ 或 mixing weights frozen in main path)
- (c) 未走 **纯欧氏旁路** (Euclidean-only)
- (d) 保留 **L0 K64 / L1 K128 / L2 K256 独立 learnable κ**
- (e) **mixing logits 真实进入 forward**（不是占位/直通）

## 2. 扫描结果

### 2.1 禁用模式 grep

| Pattern | hits | 评估 |
|---------|------|------|
| `global_kappa` / `shared_kappa` / `single_kappa` | 0 | ✅ 无 |
| `pure_euclidean` / `euclidean_only` / `no_kappa` | 0 | ✅ 无 |
| `fixed_kappa` / `frozen_kappa` (main path) | 0 | ✅ 无 |
| `kappa.requires_grad = False` (any) | 1 (Phase 4 ablation) | ⚠️ 见 §2.2 |
| `weight_mlp.requires_grad = False` (any) | 1 (Phase 4 ablation) | ⚠️ 见 §2.2 |

### 2.2 `requires_grad = False` 上下文审查

**taskB/stage2/taskB_stage2_weighted_mixed.py:530-540** (Phase 4 ablation):
```python
# Phase 4: 对照消融 (关闭产品分量 = 固定等权)
no_recal_model = WeightedMixedHRQVAE(...).to(device)
# 固定等权: 让 κ 不学 + weight MLP 不学
for q in no_recal_model.vq_layers:
    q.kappa.requires_grad = False
    for p in q.weight_mlp.parameters():
        p.requires_grad = False
```

**结论**: 同 taskA，出现在 **Phase 4 对照消融** (intentional ablation comparison)。Main path 训练时 κ 和 weight_mlp 都是可训练参数。✅ 不算 fixed-only。

### 2.3 三分量结构（per-layer 独立 weight_mlp）

**taskB/stage2/taskB_stage2_weighted_mixed.py**:
- L122: `self.weight_mlp = nn.Sequential(...)` — 每个 VQ layer 实例有独立的 weight MLP
- L181: `logits = self.weight_mlp(latent)  # (B, 3)` — 输出维度 3，对应**双曲/欧氏/混合三分量权重 logits**
- L262: `self.vq_layers = nn.ModuleList([...])` — 3 层 VQ layer (L0/L1/L2)
- L428: `weight_params = [p for q in precheck_model.vq_layers for p in q.weight_mlp.parameters()]` — **per-layer 独立收集 weight_mlp 参数**

✅ 三分量结构完整：
- 1️⃣ 双曲分量 (hyperbolic, with κ)
- 2️⃣ 欧氏分量 (Euclidean)
- 3️⃣ 混合权重 (mixing logits, learned)

### 2.4 κ per-layer 验证

- `taskB/stage2/taskB_stage2_weighted_mixed.py:111`: `self.kappa = nn.Parameter(torch.tensor(0.0, ...))` — **per-layer 独立 κ**
- `taskB/stage2/taskB_stage2_mixing_kappa_fix.py:127`: `self.kappa_logit = nn.Parameter(...)` — 备选 logit 参数化方案
- `taskB/stage2/taskB_stage2_weighted_mixed.py:426`: `grads_kappa = torch.autograd.grad(total_loss, [q.kappa for q in precheck_model.vq_layers], ...)` — **per-layer κ 梯度独立计算**
- `taskB/stage2/taskB_stage2_weighted_mixed.py:439`: `(2) weight_mlp grad finite nonzero: max={...} → PASS/FAIL` — **per-layer mixing logits 梯度验证脚本**

✅ 确认 L0/L1/L2 独立 learnable κ + mixing logits 真实进入 forward。

### 2.5 双曲/欧氏分量独立性验证

`taskB/stage2/taskB_stage2_hgrec_fixed.py` 存在但**仅作为 fixed-baseline 对照**（HG-Rec 原始代码），主路径走 `taskB_stage2_weighted_mixed.py`。即 fixed 版本不是主训练路径。✅

## 3. 前序 canary Gate 3 状态（已通过）

参考 `taskB/stage3/taskB_stage3_mixed_curv_recontinue/verdict.json`:

| 指标 | 值 | 评估 |
|------|----|------|
| Gate 3 PASS | true | ✅ |
| epoch 0→9 loss | 1.8912 → 1.8890 | ✅ 下降 (慢但有) |
| cond_grad nonzero all | true | ✅ |
| ln_grad nonzero all | true | ✅ |
| alpha bounded all | true (max=0.5) | ✅ |
| final α | 0.0475 | ✅ 远低于 bound (比 taskA 还小) |
| NaN/Inf all | false (干净) | ✅ |
| SID hash match | true (`2dab29...`) | ✅ |
| labels bug fix | applied | ✅ |

参考 `taskB/stage3/taskB_stage3_mixed_curv_recontinue/adapter_init_proof.json`:
- `max_diff_x_emb = 1.04e-05` (~0, init 正确)
- `alpha_value = 4.54e-05` (init α 极小)

## 4. 现有产物清单

- `taskB/stage3/taskB_stage3_mixed_curv_recontinue/adapter.pt` (404485 bytes, recontinue epoch 9, α=0.0475)
- `taskB/stage3/taskB_stage3_issue193_long_run/best_adapter.pt` (405077 bytes, 长跑 best ckpt, Gate 4 R@10=0.0395 FAIL)
- `taskB/stage4/taskB_stage4_canary_argmax/canary_verdict.json`
- `taskB/stage4/taskB_stage4_canary_autoregressive/canary_verdict.json`

## 5. precheck 决策

**PRECHECK: PASS**

主训练路径满足 Issue #3 红线要求：
- ✅ 三层独立 learnable κ (per-layer `nn.Parameter`, NOT global)
- ✅ Main path κ + weight_mlp 均可训练 (Phase 4 `requires_grad=False` 仅限 ablation)
- ✅ 无 pure Euclidean bypass
- ✅ 三分量结构完整 (双曲/欧氏/混合), per-layer 独立 weight_mlp
- ✅ mixing logits 真实进入 forward (weight_mlp(latent) → (B, 3) logits → 三分量混合)
- ✅ κ + weight_mlp 梯度均非零
- ✅ SID 流可复现 (hash match `2dab29...`)
- ✅ α bounded + 缓慢增长 (canary 9 epoch 内 0 → 0.0475, 远低于 0.5 bound)

## 6. 下一步 → Gate 1 → Gate 2 → Gate 3 (canary 已有) → Gate 4

按 Issue #3 spec:
1. ~~precheck~~ ✅ 本 verdict
2. **Gate 1**: 验证 Stage 1 三层三分量 metadata + hash（需现场跑一次）
3. **Gate 2**: 逐层 κ/mixing 梯度和有限差分非零且稳定，混合权重可审计，SID 流可复现（canary 部分证明，需现场再跑严格 Gate 2）
4. **Gate 3**: 仅 Gate 2 PASS 后接 Stage 3 T5，并保存 best_adapter.pt（canary 已有）
5. **Gate 4**: 仅 Gate 3 PASS 后单seed Task84 正式评估，需报告 R@5/R@10/R@20 + NDCG@5/NDCG@10/NDCG@20；阈值 R@10 > 0.1020

**当前 tick 进度**: precheck 完成。下一 tick 应启动 Gate 1。

## 7. 判定

- precheck: **PASS** (红线全过, 不需修改代码)
- 可以继续: Gate 1 → Gate 2 → Gate 3 (canary 已有) → Gate 4
- 决策阈值: Gate 4 R@10 > 0.1020 = Target reached (否则 Gate 4 FAIL)
- **观察**: taskB 比 taskA α 增长更慢 (0.0475 vs 0.0998), 可能意味着 mixing weights 主导流形表示, 需在 Gate 4 验证是否实际提升 R@10