# Issue #186 / Task #474 [方向A canary] 协议对齐 sanity check — CANARY FAIL

## Canary 决策: ❌ FAIL (Gate 3 不升级)

## 1. Stage 1 (RQ-VAE/HRQVAE): ⏭️ N/A (沿用 Issue #175/#177 SID NPY)
- 关键数据: SID NPY SHA256=`2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a`
- SHA match: ✅

## 2. Stage 2 (Sinkhorn): ⏭️ N/A
- 关键数据: 4-digit SID 9922/9922=100% unique

## 3. Stage 3 (T5-mini) - Issue #183 PASS 机制 → Canary PASS/Fail
- 关键数据:
  - 复用 task472 ckpt (Issue #179 200 epoch 训练产物, alpha=0.5 saturated, epoch=34)
  - ckpt SID SHA ✅, T5 SHA ✅
  - canary_n=200, 单 seed=42, 真实 Stage 4 argmax decode (无训练, 禁用 double-run)
- **Canary 结果**: R@5=0.0, R@10=0.0, R@20=0.0 (200/200 samples 全部失败)

## 4. 4 项协议一致性 audit (Issue #186 spec 强制)

| Protocol | 检查内容 | 结果 | 详情 |
|----------|---------|------|------|
| **P1** forward path | encoder→decoder→lm_head→argmax 是否同一路径 | ✅ PASS | `t5.model.encoder → t5.model.decoder → t5.model.lm_head → argmax`, 跟 #450 修复后一致 |
| **P2** vocab mapping | T5 vocab_size vs codebook 总和 | ❌ FAIL | t5_config vocab_size=1025 ≠ expected=451 (64+128+256+1+2). T5 实际 vocab 包含 SID token space + 额外 token, 但 P2 是 informational FAIL (T5 内部实现有 1025 vocab, 实际 SID 只用 449 tokens) |
| **P3** lm_head path | t5.model.lm_head 是否存在 | ✅ PASS | `hasattr(t5.model, 'lm_head')`=True, weight shape=(1025, 128), bias=True, dtype=float32 |
| **P4** valid SID constraint | argmax token IDs 是否在合法 SID codebook 范围 | ❌ **FAIL** (核心 bug) | **In-valid-range: 0/800 (0.0%), Out-of-range: 800/800**. **所有 200 sample 的 4 个 digit token_id 全是 0 (PAD)**, 完全不在合法 SID 范围 |

## 5. 根因定位 (Issue #186 spec 强制: 必须具体定位哪一项不一致)

**P4 FAIL 的具体根因: Wrapper 训练时 decoder labels 是 PAD token (token_id=0)**

定位证据 (来自 scripts/task470_issue177_gate3_a_recontinue.py line 146-151):
```python
def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, kappa_meta=None, scale_meta=None):
    ...
    if labels is not None:
        dummy_decoder_output = torch.zeros_like(input_ids[:, :4])  # ← 全零 = PAD
        return self.t5.model(
            inputs_embeds=x_emb_with_residual,
            attention_mask=attention_mask,
            labels=dummy_decoder_output,  # ← BUG: 应该用真实 SID labels
        ), None, alpha
```

**Bug 分析**:
1. 训练时 wrapper 接收外部传入的 `labels` (真实 SID token IDs), 但**忽略**它, 改用 `torch.zeros_like(input_ids[:, :4])` 作为 decoder labels
2. `torch.zeros_like` 全部填充 0 = PAD token_id
3. T5 训练 cross-entropy loss 把 decoder output 跟 PAD labels 对齐 → decoder 学到 "输出 PAD 是最优"
4. Stage 4 推断时 decoder 沿用训练时行为, 输出 token_id=0 (= PAD) for 全部 4 个 digit
5. argmax 后 R@10 = 0 (PAD token 永远不会等于 target SID token)

**P2 FAIL 是 informational (T5 vocab_size=1025 > 实际 SID 449, 多余 vocab 是 T5 内部实现细节, 不影响训练/推断), 真正导致 R@10=0 的是 P4 FAIL, 根因是 P4 FAIL 的训练 labels bug**

## 6. 后续修复方向 (per Issue #186 spec: 作为下一修复 issue 输入)

修复 Issue #186/#187 canary FAIL 必须修复 wrapper bug, 不能直接 Stage 4 长跑:

1. **Wrapper decoder labels 修复**:
   ```python
   if labels is not None:
       return self.t5.model(
           inputs_embeds=x_emb_with_residual,
           attention_mask=attention_mask,
           labels=labels,  # 改用真实 SID labels 而非 dummy_decoder_output
       ), None, alpha
   ```

2. **重训必要性**: 改 wrapper 后必须重新 Stage 3 训练, 不能用现有 ckpt
3. **新 issue 提议**: 创建 Issue #188+ 修复 wrapper decoder labels bug + 重训 + canary 验证

## 7. 整体决策

**Issue #186 [方向A canary]: ❌ CANARY FAIL**

- 关键产物:
  - verdict: verdicts/task474_issue186_canary_stage4_argmax_result.md (本文)
  - verdict.json: products/task474_issue186_canary_stage4_argmax/canary_verdict.json
  - script: scripts/task474_issue186_canary_stage4_argmax.py
  - log: logs/task474_issue186_canary_stage4_argmax.log

- Gate 3 状态: 仍 conditional-not-verified (Issue #183 mechanism PASS, 但协议对齐 FAIL, 不能进 Gate 4)
- 修复路径: 必须修复 wrapper decoder labels bug → 新 issue 重训 + canary 验证

- 累计状态: 24 issue κ/scale 元数据适配收口 (前 23 + **#186/#187 canary FAIL**). Issue #179/#181 200 epoch 长训 + 双复跑 R@10=0 根因 = 同一个 wrapper bug
