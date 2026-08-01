# Issue #187 / Task #515 [方向B canary] 协议对齐 sanity check — CANARY FAIL

## Canary 决策: ❌ FAIL (Gate 3 不升级)

## 1. Stage 1 (RQ-VAE/HRQVAE): ⏭️ N/A (沿用 Issue #176/#178 SID NPY)
- 关键数据: SID NPY SHA256=`4654f3e22ced0932a3e5004dfe9a854694da822fb5d10282faf497f9769288d3`
- SHA match: ✅ (informational, 跟 #176 路径一致)

## 2. Stage 2 (Sinkhorn): ⏭️ N/A
- 关键数据: 4-digit SID 9922/9922=100% unique

## 3. Stage 3 (T5-mini) - Issue #184 PASS 机制 → Canary PASS/Fail
- 关键数据:
  - 复用 task473 ckpt (Issue #181 200 epoch 训练产物, alpha=0.5 saturated, epoch=34)
  - ckpt SID SHA ✅, T5 SHA ✅
  - canary_n=200, 单 seed=42, 真实 Stage 4 argmax decode (无训练, 禁用 double-run)
- **Canary 结果**: R@5=0.0, R@10=0.0, R@20=0.0 (200/200 samples 全部失败)

## 4. 4 项协议一致性 audit (Issue #187 spec 强制)

| Protocol | 检查内容 | 结果 | 详情 |
|----------|---------|------|------|
| **P1** forward path | encoder→decoder→lm_head→argmax 是否同一路径 | ✅ PASS | `t5.model.encoder → t5.model.decoder → t5.model.lm_head → argmax` |
| **P2** vocab mapping | T5 vocab_size vs codebook 总和 | ❌ FAIL | t5_config vocab_size=1025 ≠ expected=451. 跟 #186 同根因, T5 内部 vocab 包含额外 token |
| **P3** lm_head path | t5.model.lm_head 是否存在 | ✅ PASS | `hasattr(t5.model, 'lm_head')`=True, weight shape=(1025, 128), bias=True |
| **P4** valid SID constraint | argmax token IDs 是否在合法 SID codebook 范围 | ❌ **FAIL** (核心 bug) | **In-valid-range: 3/800 (0.4%), Out-of-range: 797/800 (99.6%)**. **797/800 prediction token_id=0 (PAD)**, 只有 3 个偶然不在 0 (但仍可能在 valid range 边缘) |

## 5. 根因定位 (Issue #187 spec 强制)

**P4 FAIL 的具体根因: Wrapper 训练时 decoder labels 是 PAD token (token_id=0)**

定位证据 (来自 scripts/task471_issue178_gate3_b_recontinue.py line 137-143):
```python
def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, curvature_meta=None):
    ...
    if labels is not None:
        dummy_decoder_output = torch.zeros_like(input_ids[:, :4])  # ← 全零 = PAD
        return self.t5.model(
            inputs_embeds=x_emb_with_residual,
            attention_mask=attention_mask,
            labels=dummy_decoder_output,  # ← BUG: 同方向A, 忽略真实 labels
        ), None, alpha
```

**Bug 分析** (跟方向A同根因):
1. BoundedWeightedMixedCurvatureConditioner wrapper 接收 `labels` 但用 `torch.zeros_like` 覆盖
2. 训练 cross-entropy 把 decoder output 对齐到 PAD, decoder 学 "输出 PAD 最优"
3. Stage 4 推断时 argmax 几乎全是 token_id=0 (200 samples × 4 digits = 800 tokens, 仅 3 个偶然非 0)
4. R@10 = 0 (PAD ≠ 任何 target SID)

## 6. 后续修复方向 (per Issue #187 spec)

跟 #186 同修复路径:
1. Wrapper `labels=dummy_decoder_output` 改为 `labels=labels` (使用真实 SID)
2. 修复后必须重新 Stage 3 训练 (现有 ckpt 已被 PAD labels 污染)
3. 新 issue: 修复 wrapper decoder labels bug → 重训 → canary 验证 → 决定能否进 Gate 4

## 7. 整体决策

**Issue #187 [方向B canary]: ❌ CANARY FAIL**

- 关键产物:
  - verdict: verdicts/task515_issue187_canary_stage4_argmax_result.md (本文)
  - verdict.json: products/task515_issue187_canary_stage4_argmax/canary_verdict.json
  - script: scripts/task515_issue187_canary_stage4_argmax.py
  - log: logs/task515_issue187_canary_stage4_argmax.log

- Gate 3 状态: 仍 conditional-not-verified
- 修复路径: 必须修复 wrapper decoder labels bug

## 8. #186 + #187 联立 root cause 总结

两个 canary 都 FAIL, 且失败模式完全一致:
- **共同根因**: task470/task471 wrapper 的 `forward()` 用 `torch.zeros_like(input_ids[:, :4])` 覆盖真实 `labels` 参数
- **影响**: decoder 被训练成 "看到什么 input 都输出 PAD", Stage 4 推断时 R@10=0
- **历史回溯**: Issue #179/#181 200 epoch 长训 + 双复跑 R@10=0 的真根因 = 同一个 wrapper bug
- **修复代价**: 改 wrapper 一行 + 重训 (200 epoch 长训成本), 不能复用现有 ckpt
- **累计**: 24 issue κ/scale 元数据适配收口 + 2 canary FAIL (#186/#187) = 26 issue 全部 FAIL/PASS 收口
