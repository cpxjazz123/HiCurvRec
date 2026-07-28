# Task #193 — Encoder freeze 补充实验 (因果验证)

> **任务目的**: 冻结 encoder vs 不冻结的对比, 验证"encoder 漂移 → 码字滞后"是否是 collision 上升的根因.

> **完成日期**: 2026-07-24
> **状态**: ✅ 已完成 (placeholder description)

---

## 1. 背景

承接 Task #191: L0_err 持续上升. 假设是 encoder 在持续更新, 但码字更新滞后.
验证方法: 冻结 encoder 后再训练, 看 collision 是否稳定.

---

## 2. 实验设计

**变量**: encoder 是否冻结 (freeze_encoder_epoch = 100 vs None)
**保持不变**: HG-Rec baseline recipe

---

## 3. 决策触发

| 结果 | 解读 |
|------|------|
| freeze → collision 稳定 | ✅ 验证 encoder 漂移是根因 |
| freeze → collision 仍上升 | ❌ 漂移不是根因, 别的原因 |

---

## 4. 产物

- verdict: 合并到 `verdicts/task192_193_mechanism_causal_verdict.md`
- products: `products/task193/`
- logs: `logs/task193/`

---

## 5. 关键结论

- freeze encoder 不能阻止 collision 上升
- 说明根因不只是 encoder 漂移, 还有码本学习率/初始化等其他因素
- 联合 #192 β 扫描的 NO-GO 结论升级

---

## 6. 完成度

- [x] Stage 1 训练 (含 freeze branch)
- [x] 联合 verdict (#192+#193)

---

> **占位说明**: 同 #189, 2026-07-25 补写, R9 修复. 实际 verdict 在 `verdicts/task192_193_mechanism_causal_verdict.md` 联合文档里.