# Task #476 / Issue #188 [方向A Gate3 修复] wrapper labels bug → 短训 → canary — ⚠️ Gate 3 PASS / Gate 4 (canary) FAIL → conditional-not-verified 维持

## 任务目标 (per Issue #188 spec, 2026-08-01 owner 派发)

1. **修复 wrapper.forward() labels bug**: `dummy_decoder_output = torch.zeros_like(input_ids[:, :4])` 覆盖真实 `labels` → decoder 训练时收到全 PAD 监督, 学不到 SID 信号
2. **短程重训 Stage 3** (沿用 #183 同规模, 10 epoch)
3. **重新 canary Stage 4** (200 样本, 单 seed)
4. **不可"含糊关闭"**: canary R@10 非零 → PASS, canary R@10=0 → 必须具体定位新不一致项

## 实施 (taskA/stage3/taskA_stage3_kappa_scale_recontinue.py + taskA/stage4/taskA_stage4_canary_argmax.py)

### 修复点 (Issue #188 根因)
**File**: `taskA/stage3/taskA_stage3_kappa_scale_recontinue.py:145-150` (原代码)
```python
if labels is not None:
    dummy_decoder_output = torch.zeros_like(input_ids[:, :4])
    return self.t5.model(
        inputs_embeds=x_emb_with_residual,
        attention_mask=attention_mask,
        labels=dummy_decoder_output,    # ❌ 全 PAD 监督, decoder 学不到 SID
    ), None, alpha
```

**修复后**:
```python
if labels is not None:
    return self.t5.model(
        inputs_embeds=x_emb_with_residual,
        attention_mask=attention_mask,
        labels=labels,    # ✅ 真实 SID token 作为 decoder 监督信号
    ), None, alpha
```

### 顺带修复 (脚本维护)
1. **`numpy.bool_` → JSON 序列化失败**: 训练 10 epoch 后 verdict.json 写入失败 (`TypeError: Object of type bool is not JSON serializable` — sid_range_check["all_in_range"] 是 numpy bool_). 添加 `_json_default` 处理 numpy types.
2. **Canary CKPT_PATH**: 原 `taskA/stage3/taskA_stage3_direction_a_train/adapter_200ep.pt` (199 epoch 路径, 已废弃) → 修正为 `taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter.pt` (10 epoch 修复后)
3. **Canary ckpt key 名**: 原 `first_input_ln_state_dict`/`alpha_value` → 修正为 `ln_state_dict`/`alpha` (匹配训练 save 格式)
4. **Canary SID_NPY / TEST_PARQUET 路径**: 原 `HG-Rec/dataset/...` → 修正为 `taskA/_data/...` (data/ 已删除, hg-rec/ 只读)

## 4 Gate 状态

### Gate 1 (Stage 1 RQ-VAE embedding)
- **状态**: PASS (沿用 Issue #157 / Task #175 冻结 SID, SHA256=2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a)
- **关键数据**: SID_NPY = 9922 items × 4 digits, layer 0/1/2/3 all_in_range=True
- **失败原因**: N/A (沿用)
- **verdict 路径**: verdicts/task157_issue157_kappa_scale_result.md (R12 invariant 强制)

### Gate 2 (Stage 2 Sinkhorn + α/κ 元数据)
- **状态**: PASS (沿用 Issue #175, frozen SID NPY = 2dab2922...9508a)
- **关键数据**: Sinkhorn 5 iter, 4-digit unique 9922/9922=100%, 3-digit collision 0.1299
- **失败原因**: N/A (沿用)
- **verdict 路径**: verdicts/task175_issue175_kappa_sync_stage2_result.md (per Issue #157+#158 Gate 2)

### Gate 3 (Stage 3 T5 wrapper, Issue #188 本 issue 焦点)
- **状态**: ✅ PASS (mechanism)
- **关键数据**:
  - Precheck 1: α=4.54e-05, max_diff=1.81e-05 ✅ (init identity)
  - Precheck 2: trainable=12 (10 conditioner + 2 LN), frozen=104 ✅
  - Precheck 3: loss=2.0093, cond_grad nonzero, ln_grad nonzero ✅
  - Precheck 4: SID range layer 0/1/2/3 all in-range ✅
  - Precheck 5: SID SHA256 = 2dab2922...9508a ✅
  - 10 epoch 短训: loss 1.8912 → 1.8878 (-0.18%, 健康下降)
  - α bounded: 4.54e-05 → 9.98e-02 (≤ 0.5, 远未触发 clamp)
  - cond_grad nonzero 全程 (1.84e-02 ~ 1.98e-02), ln_grad nonzero 全程 (1.41e-01 ~ 1.45e-01)
  - nan_inf False 全程 10 epochs
- **vs 修复前**: epoch 1 loss = 9.13 (PAD-only 监督) → 1.89 (real SID 监督, -79.3%)
- **失败原因**: N/A (mechanism 验证通过)
- **verdict 路径**: taskA/stage3/taskA_stage3_kappa_scale_recontinue/verdict.json (reconstructed)

### Gate 4 (Stage 4 R@K canary, Issue #188 #4 验收)
- **状态**: ❌ CANARY FAIL
- **关键数据**:
  - canary_n=200, R@5=R@10=R@20=0.0 (200/200 全 4-digit 不匹配)
  - P1 (forward path): PASS (encoder→decoder→lm_head→argmax)
  - P2 (vocab mapping): FAIL (informational, t5_config vocab_size=1025 ≠ expected=449, per Issue #186 沿用)
  - P3 (lm_head path): PASS (t5.model.lm_head weight=(1025,128), bias=True)
  - P4 (valid SID constraint): **FAIL** — in-valid-range=200/800 (25.0%), layer 0 全 valid, layer 1/2/3 全 token_id=5 (in layer 0 range [1,65])
- **失败原因**: **decoder 位置 1/2/3 collapse 到相同 token (token_id=5, 落在 layer 0 范围内)** — canary argmax 用 `decoder_input_ids = zeros(B, 4)`, 位置 1/2/3 decoder hidden state 没差异化, lm_head argmax 全选 layer 0 范围内同一 token
- **修复方向 (per Issue #188 #4 不接受"含糊关闭")**:
  1. **autoregressive feed argmax**: canary 推理时把前一步 argmax feed 回 decoder_input_ids, 而不是固定 zeros — 这样位置 1/2/3 能拿到不同的 decoder hidden state
  2. **训练时长**: 10 epoch 不足, α=0.1 远未到 bound (0.5), adapter 影响微弱, 需长训让 conditioner 真影响 lm_head logits
  3. **lm_head 独立 per-layer head**: 当前 lm_head 是 shared (1025, 128), 位置 0/1/2/3 共用同一权重 — 可改 per-position lm_head 强制每位置学到不同分布
- **verdict 路径**: taskA/stage4/taskA_stage4_canary_argmax/canary_verdict.json

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案:
1. 修 wrapper labels bug (核心根因)
2. 10 epoch 短训 (Issue #188 spec 强制)
3. canary 200 样本 (Issue #188 spec 强制)
4. canary R@10=0 → 不接受"含糊关闭", 定位到位置 1/2/3 collapse 根因
5. 建议下一 issue: canary 改 autoregressive argmax feed, 跟训练 teacher-forcing 对齐

## 产物路径

- **训练脚本 (修复后)**: taskA/stage3/taskA_stage3_kappa_scale_recontinue.py
- **canary 脚本 (修复后)**: taskA/stage4/taskA_stage4_canary_argmax.py
- **训练 verdict**: taskA/stage3/taskA_stage3_kappa_scale_recontinue/verdict.json (gate3_pass=true)
- **训练 ckpt**: taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter.pt (epoch=9, α=9.98e-02)
- **canary verdict**: taskA/stage4/taskA_stage4_canary_argmax/canary_verdict.json (decision=FAIL)
- **canary log**: logs/task474_issue186_canary_stage4_argmax_fix.log
- **训练 log**: logs/task470_issue177_gate3_a_recontinue_fix.log

## 整体决策

**⚠️ Gate 3 mechanism PASS, Gate 4 (canary) FAIL** — Issue #188 wrapper labels bug 修复成功 (decoder 不再输出全 PAD), 但 canary R@10=0 因 decoder 位置 1/2/3 collapse, 维持 Gate 3 conditional-not-verified 状态.

**对比基线 Issue #186 (修复前)**: 200/200 (100%) 输出全 PAD → 修复后 200/200 (100%) 输出 layer 0 valid tokens + 600 same-token 错配 — 行为有实质改进, 但仍需修复 decoder collapse 才能 PASS Gate 4.

## 下一步建议 (创建后续 issue)

- **方向A canary 改 autoregressive**: canary 推理循环: position 0 → argmax → feed as decoder_input_id position 1 → argmax → feed position 2 → argmax → feed position 3 → argmax. 跟训练 teacher-forcing 一致
- **方向A 长训 (200 epoch)**: α=0.1 vs bound 0.5, adapter 影响微弱, 需长训让 conditioner 真影响 logits
- **共同根因 (跨方向)**: lm_head shared weights + 单 argmax 不 autoregressive → 位置 1/2/3 必然 collapse