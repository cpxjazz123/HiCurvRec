# Task #177 — κ-Stereographic + β(x) = β_base/λ_κ(x) conformal factor

> **任务目的**: 验证 hypothesis"用 conformal factor λ_κ(x) 倒数校正 commitment weight, 复用 distance formula 里已经算的 λ, 自动按 Riemannian metric 逆校正原始梯度"是否能改善下游 Recall
> **承接**: Task #176 sigmoid β(x), Task #175 ORC LOCKED κ, Task #174 D 臂 gating
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**用户 hypothesis (2026-07-25 §2 方法 B)**: "直接复用你们代码里已经在算的东西: 共形因子 λ_κ(x) 本身就是'这个点附近空间被压缩/拉伸了多少倍'的精确度量, 可以直接令 β(x) = β_base / λ_κ(x) —— λ 越大 (离边界越近, 失真越严重) β 越小, 梯度步长被自动按当前的真实几何拉伸比例缩回去. 这其实就是黎曼优化里'用度量张量的逆去校正原始梯度'这个标准做法."

**理论依据**:
- Stereographic model 的 conformal factor: λ_κ(x) = 2 / (1 + κ·||x||²)
- Riemannian metric: g_κ(x) = (λ_κ(x)/2)² · I (正比于 λ²)
- Riemannian gradient correction: grad_R = grad_E / λ² (or similar)
- 用户提议: β(x) = β_base / λ_κ(x) = β_base · (1 + κ·||x||²) / 2

**Curvature-sign behavior**:
- κ < 0 (双曲): ||x||² 大 → 1+κ·||x||² < 1 → β < β_base/2 (小, 给 encoder 自由度)
- κ > 0 (球面): ||x||² 大 → 1+κ·||x||² > 1 → β > β_base/2 (大 commitment)
- 注: 球面 ||x||² 大可能让 β 爆炸, 需要 clamp 到 β_max

**Method A (#176) vs Method B (#177)**:
- A: sigmoid 压缩 + sign 控制 → bounded [0, β_base]
- B: linear (1+κ·||x||²)/2 → unbounded for 球面
- B 是直接 Riemannian metric correction, 理论更优雅
- A 是模仿 MCKG 设计, 工程上更稳定

---

## 2. 实验设计

**变量**: β(x) = β_base · (1 + κ · ||x||²) / 2 (per-component per-layer)
**保持不变**:
- 跟 #176 完全相同, 只换 β_mode = conformal_factor

**β(x) formula 详解**:
```python
# For each component m, with latent x_m and curvature κ_m:
norm_sq_m = ||x_m||²                        # (B,)
beta_m = beta_base * (1 + kappa_m * norm_sq_m) / 2   # (B,)
# Clamp to avoid explosion for κ>0 large ||x||²
beta_m = beta_m.clamp(min=0.0, max=beta_max)   # beta_max = 5.0 (R11.3)
# Then take mean across M (or max, or other aggregation)
beta_x = beta_m.mean()  # scalar per sample
```

**Default hyperparams**: β_base=1.0, β_max=5.0 (R11.3 决策)

**Stage 1 启动命令**:
```bash
python3 -u scripts/task177_posdep_beta_conformal_stage1_train.py \
    --beta_mode conformal_factor \
    --beta_base 1.0 --beta_max 5.0 \
    --M 3 --kappa_max 2.0 --epochs 200 --batch_size 256 --lr 1e-3 \
    [...跟 #176 一样...]
```

---

## 3. 决策触发 (vs baseline 0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | 🟢 GO | conformal β(x) 验证成功 |
| 0.1000 ≤ test R@10 ≤ 0.1058 | 🟡 MARGINAL | 接近 baseline, NO-GO |
| test R@10 < 0.1000 | ⛔ NO-GO | 假设证伪 |

**额外对比**:
- #176 sigmoid β(x) vs #177 conformal β(x) vs baseline 0.1058
- 哪个更接近 baseline? 如果 conformal factor 显著优于 sigmoid, 说明 Riemannian metric 理论 correction 有效

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 | ~3 min (GPU 2) |
| Stage 2 | ~3 min |
| Stage 3 | ~50 min |
| Stage 4 | ~1 min |
| **总计** | **~57 min** |

---

## 5. 风险与缓解

**风险 1**: 球面分支 ||x||² 大时 β 爆炸 → β_max=5.0 clamp (R11.3)
**风险 2**: ||x||² 在 κ<0 时 > -1/κ 会让 1+κ·||x||² < 0, β negative → clamp_min=0
**风险 3**: conformal factor 跟 distance formula 里 denom 用 ||x-y||² (diff) 不同, λ_κ(x) 用 ||x||² (point), 可能 user 描述不严格 → 仍然按 user proposal 实现 (||x||² point norm)

---

## 6. 完成度跟踪

- [ ] Stage 1 launch
- [ ] Stage 1 finish
- [ ] Stage 2 SID codebook inference
- [ ] Stage 3 T5-mini launch
- [ ] Stage 3 finish
- [ ] Stage 4 test eval
- [ ] 写 verdict (`verdicts/task177_posdep_beta_conformal_result.md`)
- [ ] 更新 loop.md §16