# Task #194 — K0 码本容量扫描 verdict
> **机制因果验证 #3**: K0={32, 64, 128, 256} 单变量扫, 验证 K0 是否 controlling variable.

## 1. Stage 1 — collision 演化 (per K0)

| K0 | n_epochs | collision_min | collision_final | final/min 比值 | L0_err_min | L0_err_final | corr(L0_err, collision) |
|----|----------|---------------|-----------------|-----------------|------------|--------------|--------------------------|
| 32 | — | — | — | — | — | — | — |
| 64 | — | — | — | — | — | — | — |
| 128 | — | — | — | — | — | — | — |
| 256 | — | — | — | — | — | — | — |

## 2. Stage 4 — 下游 R@10 (per K0)

| K0 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 决策 vs baseline 0.1020 |
|----|-----|------|------|-----|------|------|------------------------|
| 32 | — | — | — | — | — | — | ❌ 缺数据 |
| 64 | — | — | — | — | — | — | ❌ 缺数据 |
| 128 | — | — | — | — | — | — | ❌ 缺数据 |
| 256 | — | — | — | — | — | — | ❌ 缺数据 |

## 3. 假设判据

### R1: K0 ↑ → collision 单调↓

### R2: K0 ↑ → final/min 比值单调↓

## 4. 综合判定

- ✅ R1+R2 都立 → **K0 是 controlling variable**
- ❌ 任一不立 → K0 不是 controlling variable, 需要找别的
- ⚠️ K0=256 时 final collision 接近 0, 但下游 R@10 可能因为 L1/L2 退化

## 5. 关键决策点 (R11.3)

- K0 选择: 4 臂 K0={32, 64, 128, 256}, 跨越 8× 容量. 单一变量 K0.
- 复用 task191 诊断脚本骨架: 直接遍历 4 臂 ckpt 算 L0_err.
- Stage 3/4 全部重训 (K0 改了, SID 全变, 必须重训).

## 6. 后续建议

- 若 K0 是 controlling variable → 进一步实验: K0=512 看 collision → 0?
- 若 K0 不是 → 下一个候选: decoder 重建 loss 权重 (类似 β 但只针对 decoder)

---

**result:** Task #194 K0 容量扫描 verdict 完成. 详见上表 collision 演化 + Stage 4 R@10.
