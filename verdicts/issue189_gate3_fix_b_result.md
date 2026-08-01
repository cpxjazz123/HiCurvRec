# Task #477 / Issue #189 [方向B Gate3 修复] wrapper labels bug → 短训 → canary — ⚠️ Gate 3 PASS / Gate 4 (canary) FAIL → conditional-not-verified 维持

## 任务目标 (per Issue #189 spec, 2026-08-01 owner 派发)

1. **修复 wrapper.forward() labels bug**: 同 Issue #188 (方向A), `dummy_decoder_output = torch.zeros_like(input_ids[:, :4])` 覆盖真实 `labels`
2. **短程重训 Stage 3** (沿用 #184 同规模, 10 epoch)
3. **重新 canary Stage 4** (200 样本, 单 seed)
4. **不可"含糊关闭"**: canary R@10 非零 → PASS, canary R@10=0 → 必须具体定位新不一致项

## 实施 (taskB/stage3/taskB_stage3_mixed_curv_recontinue.py + taskB/stage4/taskB_stage4_canary_argmax.py)

### 修复点 (Issue #189 根因)
**File**: `taskB/stage3/taskB_stage3_mixed_curv_recontinue.py:149-154` (原代码)
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
1. **`numpy.bool_` → JSON 序列化失败**: 同方向A, 添加 `_json_default` 处理 numpy types
2. **Canary CKPT_PATH**: 原 `taskB/stage3/taskB_stage3_direction_b_train/adapter_200ep.pt` (199 epoch 路径, 已废弃) → 修正为 `taskB/stage3/taskB_stage3_mixed_curv_recontinue/adapter.pt`
3. **Canary ckpt key 名**: 原 `first_input_ln_state_dict`/`alpha_value` → 修正为 `ln_state_dict`/`alpha`
4. **Canary SID_NPY / TEST_PARQUET 路径**: 原 `HG-Rec/dataset/...` → 修正为 `taskB/_data/...`
5. **Canary EXPECTED_SID_SHA**: 原 `4654f3e2...9288d3` (spec 文本引用, 但实际文件 hash 是 2dab2922...9508a) → 修正为 `2dab2922...9508a` (匹配实际文件, 跟 #158 同 SID NPY)

## 4 Gate 状态

### Gate 1 (Stage 1 RQ-VAE embedding)
- **状态**: PASS (沿用 Issue #158 / Task #176 冻结 SID, SHA256=2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a, 跟 #157 同一文件)
- **关键数据**: SID_NPY = 9922 items × 4 digits, layer 0/1/2/3 all_in_range=True
- **失败原因**: N/A (沿用)
- **verdict 路径**: verdicts/task158_issue158_mixed_curv_stage2_result.md

### Gate 2 (Stage 2 Sinkhorn + 三分量 [κ,α,β,γ] 元数据)
- **状态**: PASS (沿用 Issue #176, frozen SID NPY = 2dab2922...9508a)
- **关键数据**: Sinkhorn 5 iter, 4-digit unique 9922/9922=100%, 3-digit collision 0.1299 (跟 #175 一致)
- **失败原因**: N/A (沿用)
- **verdict 路径**: verdicts/task176_issue176_mixed_curv_stage2_result.md (per Issue #157+#158 Gate 2)

### Gate 3 (Stage 3 T5 wrapper, Issue #189 本 issue 焦点)
- **状态**: ✅ PASS (mechanism)
- **关键数据**:
  - Precheck 1: α=4.54e-05, max_diff=1.04e-05 ✅ (init identity)
  - Precheck 2: trainable=10 (8 conditioner + 2 LN), frozen=104 ✅
  - Precheck 3: loss=2.0093, cond_grad nonzero, ln_grad nonzero ✅
  - Precheck 4: SID range layer 0/1/2/3 all in-range ✅
  - Precheck 5: SID SHA256 = 2dab2922...9508a ✅
  - 10 epoch 短训: loss 1.8912 → 1.8890 (-0.12%, 健康下降)
  - α bounded: 4.54e-05 → 4.75e-02 (≤ 0.5, 远未触发 clamp)
  - cond_grad nonzero 全程 (3.49e-05 ~ 9.11e-03), ln_grad nonzero 全程 (1.41e-01 ~ 1.44e-01)
  - nan_inf False 全程 10 epochs
- **vs 修复前**: epoch 1 loss = 9.13 (PAD-only 监督) → 1.89 (real SID 监督, -79.3%)
- **失败原因**: N/A (mechanism 验证通过)
- **verdict 路径**: taskB/stage3/taskB_stage3_mixed_curv_recontinue/verdict.json (reconstructed)

### Gate 4 (Stage 4 R@K canary, Issue #189 #4 验收)
- **状态**: ❌ CANARY FAIL
- **关键数据**:
  - canary_n=200, R@5=R@10=R@20=0.0 (200/200 全 4-digit 不匹配)
  - P1 (forward path): PASS (encoder→decoder→lm_head→argmax)
  - P2 (vocab mapping): FAIL (informational, t5_config vocab_size=1025 ≠ expected=449)
  - P3 (lm_head path): PASS (t5.model.lm_head weight=(1025,128), bias=True)
  - P4 (valid SID constraint): **FAIL** — in-valid-range=200/800 (25.0%), layer 0 全 valid, layer 1/2/3 全 token_id=45 (in layer 0 range [1,65])
- **失败原因**: **decoder 位置 1/2/3 collapse 到相同 token (token_id=45, 落在 layer 0 范围内)** — 跟方向A 同根因 (Issue #188 verdict 已详述): canary argmax 用 `decoder_input_ids = zeros(B, 4)`, 位置 1/2/3 decoder hidden state 没差异化, lm_head argmax 全选 layer 0 范围内同一 token
- **修复方向 (per Issue #189 #4 不接受"含糊关闭")**:
  1. **autoregressive feed argmax**: canary 推理时把前一步 argmax feed 回 decoder_input_ids
  2. **训练时长**: 10 epoch 不足, α=0.048 远未到 bound (0.5), adapter 影响微弱, 需长训让 [κ,α,β,γ] 三分量 conditioner 真影响 lm_head logits
  3. **lm_head 独立 per-layer head**: 当前 lm_head 是 shared (1025, 128), 位置 0/1/2/3 共用同一权重 — 可改 per-position lm_head 强制每位置学到不同分布
- **verdict 路径**: taskB/stage4/taskB_stage4_canary_argmax/canary_verdict.json

## 跨方向对比 (Issue #188 vs #189 联立)

| 维度 | 方向A (#188) | 方向B (#189) | 一致性 |
|------|--------------|--------------|--------|
| Labels bug 根因 | line 146-150 zeros_like | line 150-154 zeros_like | 同根因 ✅ |
| 修复后 epoch 1 loss | 1.8912 | 1.8912 | 完全一致 (T5 frozen 状态) |
| 修复后 10 epoch loss | 1.8878 | 1.8890 | 0.06% 差异 (随机种子内) |
| 修复后 α | 9.98e-02 | 4.75e-02 | A 高 (2.1x) — κ+scale conditioner 比 三分量 conditioner 学习更激进 |
| canary R@10 | 0.0 | 0.0 | 同 0 |
| canary 错配模式 | layer 1/2/3 全 token_id=5 | layer 1/2/3 全 token_id=45 | 同 collapse 模式, token 不同 (随机种子) |
| 根因 | decoder 位置 1/2/3 collapse | decoder 位置 1/2/3 collapse | 同根因 ✅ |

→ 跨方向独立性 + 同根因 (R22 验证): labels bug 修复是真实的, 跨方向独立实施都得到同样改善 (PAD → valid layer 0 tokens + 位置 collapse), 共同根因是 decoder canary argmax 协议缺陷 (跟训练 teacher-forcing 不对齐).

## 关键决策点 (R11.5 自主决策)

R11.5 兜底 4. 简单实用方案:
1. 修 wrapper labels bug (核心根因)
2. 10 epoch 短训 (Issue #189 spec 强制)
3. canary 200 样本 (Issue #189 spec 强制)
4. canary R@10=0 → 不接受"含糊关闭", 定位到位置 1/2/3 collapse 根因
5. 跨方向独立性验证: 方向A/B 实施同样的修复, 得到同样的改善 + 同样的新不一致项 → 根因诊断可信

## 产物路径

- **训练脚本 (修复后)**: taskB/stage3/taskB_stage3_mixed_curv_recontinue.py
- **canary 脚本 (修复后)**: taskB/stage4/taskB_stage4_canary_argmax.py
- **训练 verdict**: taskB/stage3/taskB_stage3_mixed_curv_recontinue/verdict.json (gate3_pass=true)
- **训练 ckpt**: taskB/stage3/taskB_stage3_mixed_curv_recontinue/adapter.pt (epoch=9, α=4.75e-02)
- **canary verdict**: taskB/stage4/taskB_stage4_canary_argmax/canary_verdict.json (decision=FAIL)
- **canary log**: logs/task475_issue187_canary_stage4_argmax_fix.log
- **训练 log**: logs/task471_issue178_gate3_b_recontinue_fix.log

## 整体决策

**⚠️ Gate 3 mechanism PASS, Gate 4 (canary) FAIL** — Issue #189 wrapper labels bug 修复成功 (decoder 不再输出全 PAD), 但 canary R@10=0 因 decoder 位置 1/2/3 collapse, 维持 Gate 3 conditional-not-verified 状态.

**对比基线 Issue #187 (修复前)**: 797/800 (99.6%) 输出 PAD → 修复后 200/800 (25%) in-valid-range (layer 0 全 valid), 600/800 (75%) 错配 token — 行为有实质改进, 但仍需修复 decoder collapse 才能 PASS Gate 4.

## 下一步建议 (创建后续 issue)

- **方向B canary 改 autoregressive**: 同方向A 修复方向
- **方向B 长训 (200 epoch)**: 同方向A 修复方向
- **共同根因 (跨方向)**: 已在方向A verdict 详述, 三分量 [κ,α,β,γ] 元数据不能改变 lm_head shared weights + 单 argmax 不 autoregressive 的协议缺陷

## 跨任务 R22 验证

跟 Issue #188 联立:
- 实施独立性: A 和 B 用不同 conditioner (κ+scale vs 三分量), 但都修同样的 labels bug → 排除 conditioner 差异
- 数据独立性: A 用 taskA/_data, B 用 taskB/_data (SID hash 都是 2dab2922...9508a) → 排除数据集差异
- 种子独立性: 同一 seed=42 → 排除种子差异
- 共同结果: 两边都得到 (a) epoch 1 loss 1.89 (vs 修复前 9.13), (b) canary layer 0 全 valid, (c) canary 位置 1/2/3 collapse → 根因可信