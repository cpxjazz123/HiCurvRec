# Task #68 result — L1 Brand-Augmented SID 完整流水线

> **任务名**: Task #68 — L1 Brand-Augmented SID on Toys (num_hierarchies=5)
> **完成日期**: 2026-07-17
> **状态**: ❌ **假设否证 — R@10=0.0792 低于 baseline 0.0971 (-18.4%)**

---

## 1. 任务目标

测试在 Stage 2 RQ-VAE 推断时追加一列 brand-augmented token（`l1_codeword = argmax(brand_codebook @ item_emb)`），下游 Stage 3 TIGER 训练是否提升推荐质量。

**假设 H1**: L1 brand 信息显式编码为 RQ-VAE 第 1 层 codeword，下游生成模型可学到"同 brand item 共现"模式 → Recall 提升

**对照基线**（Task #65）:
- Stage 2 RQ-VAE: num_hierarchies=3, codebook_width=256
- Stage 2.2 → 3 bridge: 追加 1 列 dedup digit → (4, 11924)
- Stage 3: num_hierarchies=4
- **Baseline R@10 = 0.09710**

---

## 2. 执行时间线

| 时间 | 阶段 | 关键动作 |
|------|------|---------|
| 07-17 上午 | Stage 2 准备 | brand_lookup.json + l1_codeword.pt + sid_with_l1.pt (5, 11924) 生成 |
| 07-17 11:18 | Stage 3 launch | TIGER 训练，num_hierarchies=5，sid_with_l1.pt |
| 07-17 11:19 - 17:42 | Stage 3 训练 | 5625 步，val/recall@5 在 step 5125 达到 0.04647★ |
| 07-17 17:42 | Stage 3 终止 | step 5625 val_loss=9.95, **5 连续 transient** → 立即 fire Stage 4 |
| 07-17 17:44 - 17:45 | Stage 4 inference | TIGER 推断，使用 step 5125 ckpt |
| 07-17 17:45 - 17:46 | Stage 4 eval | R@5=0.0495, R@10=0.0792 |

---

## 3. 关键结果指标

### 3.1 Stage 3 训练轨迹

| Step | val/recall@5 | val_loss | 状态 |
|------|-------------:|---------:|------|
| 3625 | 0.04209 | 9.58 | 历史高 |
| 4250 | 0.04322 | 9.61 | 突破 |
| 4625 | 0.04338 | 9.66 | 突破 |
| 4750 | 0.04482 | 9.62 | +0.00144 jump |
| **5125** | **0.04647** ★ | 9.73 | **best, +0.00165 jump** |
| 5250 | — | 9.75 | transient |
| 5375 | — | 9.80 | transient |
| 5500 | — | 9.88 | transient (4 连续) |
| 5625 | — | 9.95 | transient (5 连续) → **STOP** |

### 3.2 Stage 4 Eval（canonical）

| 指标 | Task #68 | Task #65 baseline | Δ% | 判定 |
|------|---------:|------------------:|----:|------|
| **R@5** | 0.04950 | 0.06419 | **-22.9%** | ❌ |
| **R@10** | **0.07918** | **0.09710** | **-18.4%** | ❌ |
| NDCG@5 | 0.03030 | 0.04118 | -26.4% | ❌ |
| NDCG@10 | 0.03990 | 0.05180 | -23.0% | ❌ |

---

## 4. 分析与解读

### 4.1 H1 否证

L1 brand-augmented SID **未能提升**推荐质量，反而**显著下降**：
- R@10 下降 18.4%（0.0971 → 0.0792）
- R@5 下降 22.9%（0.0642 → 0.0495）

### 4.2 失败原因分析

虽然 Stage 3 val/recall@5 在 step 5125 达到 0.04647（**显著高于 baseline val/recall@5≈0.040**），但 R@10 反而更低。可能原因：

1. **L1 brand 列引入过强先验**：TIGER 生成时过度依赖 L1 brand，但 L1 brand 只有 128 unique 值（128 brand），强制每个 item 序列第一 token 必须共享同 brand → 限制表达力
2. **bridge 处理不一致**：baseline 用 `(4, 11924)` = (L1, L2, L3, dedup)，Task #68 用 `(5, 11924)` = (L1_brand, L1_code, L2_code, L3_code, dedup) — 真实 codeword 维数没变但 token 序列变长，模型容量需求更大
3. **val/recall@5 提升 ≠ R@10 提升**：val 看到 top-5，训练目标就 5 — 强化了 brand 共现但削弱了个性化精度

### 4.3 与 Task #65 过拟合对比

Task #65 在 step 2875 达到 R@10=0.09710 后开始过拟合（3500 步跌至 0.0932）。Task #68 在 step 5125 达到 val peak，但 Stage 4 R@10=0.0792 比 val/recall@5=0.04647 的预期更低 — **模型可能学会了 brand 共现的简单模式，但牺牲了真正的 item-level 排序精度**。

---

## 5. 决策表（vs baseline R@10=0.09710）

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| R@10 ∈ [0.090, 0.105] | — | ✅ H1 confirmed（未触发）|
| R@10 ∈ [0.070, 0.090) | **0.0792** | ⚠️ H1 否证（轻微软失败）|
| R@10 < 0.070 | — | ❌ pipeline 有问题（未触发）|

**最终决策**: ⚠️ H1 否证 — L1 brand augmentation 不应用于推荐任务

---

## 6. 产物清单

| 产物 | 路径 |
|------|------|
| L1 brand lookup | `products/task68/brand_lookup.json` |
| L1 codeword per item | `products/task68/l1_codeword.pt` |
| Stage 2.2 sid_with_l1 | `products/task68/sid_with_l1.pt` (5, 11924) |
| Stage 3 best ckpt | `products/task68/s3_ckpt/best_step5125.ckpt` (154 MB) |
| Stage 4 predictions | `logs/inference/runs/2026-07-17/17-44-47/pickle/merged_predictions_tensor.pt` (19412, 10, 5) |
| Stage 4 eval (canonical) | `products/task68/s4_l5w256_canonical_eval.json` |
| Stage 3 训练日志 | `GRID/task_artifacts/scripts/logs/task68_s3.log` |
| Stage 4 推断日志 | `GRID/task_artifacts/scripts/logs/task68_s4_l5w256_l1brand.log` |
| Stage 4 launch script | `GRID/task_artifacts/scripts/task68_stage4_l5w256_l1brand.sh` |
| Eval script | `scripts/task68_s4_item_eval_l5.py` (num_hierarchies=5 patch) |
| Verdict (本文档) | `verdicts/task68_result.md` |

---

## 7. 后续建议

### 7.1 L1 brand 方向

**结论**：L1 brand-augmented SID **应放弃**作为 Stage 2 RQ-VAE 改进方向。可能改进方向：
- **L2/L3 brand augmentation**：在更细粒度的 codeword 上注入 brand 信息
- **Soft L1 brand**（embedding 而非硬 code）：用 brand embedding 替换而非 brand codeword

### 7.2 Stage 3 训练时长

Task #68 训练 5625 步（vs Task #65 baseline 2875 步）→ val/recall@5 持续上升但 R@10 反而降。**建议**: num_hierarchies=5 的 Stage 3 训练应加 `early_stopping patience=3`（val/recall@5 不再提升 3 个 cycle 就停），避免 GPU 时间浪费。

### 7.3 不要扩展到 R@20 评估

R@10 已经低于 baseline 18.4%，R@20 不会反转结论。无需进一步评估。

---

## 8. 任务完成度

- [x] Stage 2: L1 brand codebook + sid_with_l1.pt (5, 11924)
- [x] Stage 3: TIGER 训练 5625 步（val/recall@5 best=0.04647 @ step 5125）
- [x] Stage 4: TIGER 推断（17-44-47 run, 19412 users × 10 candidates × 5 digits）
- [x] R@10 eval: 0.0792（vs baseline 0.0971, -18.4%）
- [x] 写 verdict → `verdicts/task68_result.md`

**Task #68 完成 — H1 否证，L1 brand-augmented SID 不适用于推荐系统。**

---

**result**: Task #68 R@10=0.0792 低于 baseline 0.0971 (▼18.4%)，L1 brand augmentation 假设被否证。

result: Task #14 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
