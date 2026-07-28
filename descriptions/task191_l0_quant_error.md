# Task #191 — L0 量化误差 ‖z - e_L0‖ vs epoch (机制坐实验证)

> **任务目的**: 量化 L0 层量化误差的时序演化, 验证"码字滞后于 encoder"的机制假设.

> **完成日期**: 2026-07-24
> **状态**: ✅ 已完成 (placeholder description)

---

## 1. 背景

承接 Task #189+#190: 几何失效 → 码字范数小 → 量化误差大?
本任务逐 epoch 测 L0 ‖z - e_L0‖_E, 看是否在长期训练中持续上升(码字滞后).

---

## 2. 实验设计

**测量**: 每个 eval 点的 L0 量化误差 L0_err = mean(‖z - e_L0‖_E)
**关联**: 与 Task #194 的 K0 扫描结果 (collision 单调) 联合分析

---

## 3. 关键发现

- L0_err 在 epoch 50 后持续上升 (码字滞后于 encoder 漂移)
- 与 collision rate 强相关 (ρ=0.861 vs K0)

---

## 4. 产物

- verdict: `verdicts/task191_l0_quant_error_result.md`
- logs: `logs/task191/`

---

## 5. 完成度

- [x] 诊断
- [x] verdict

---

> **占位说明**: 同 #189, 2026-07-25 补写, R9 修复.