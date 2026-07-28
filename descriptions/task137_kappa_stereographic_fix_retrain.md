# Task #137 — κ-stereographic 硬分支梯度 bug 修复 + Task #89 Stage 1 重训

> **任务目的**: 修复 `hrqvae_free_curv.py` 中 4 处硬分支导致的 κ_m 梯度断裂 (Task #135 诊断结果), 用 `torch.where` 统一算子替换, 让 κ_m 在 κ=0 / κ<0 / κ>0 三个区间都能正常学习. 重训 Task #89 (3 臂 A/B/C × 1000 epoch) 验证修复后的 κ_m 是否仍锁 0 (→ "数据本质欧氏" 结论**干净验证**) 还是学到非 0 (→ 强推翻 paper Section 5.4).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (代码修复进行中)

---

## 1. 背景

承接 Task #135 诊断结论 (`verdicts/task135_kappa_zero_gradient_path_audit_result.md`) + Task #89 retro caveat:

- **Bug**: `HG-Rec/model/hrqvae_free_curv.py` line 152-178 (`_per_component_dist_sq`) + line 207-231 (commitment/codebook loss) + line 34-55 (dead `geodesic_distance_sq`) 共 4 处硬分支
  - `if k_m.item() == 0.0:` + `elif k_m.item() < 0:` — Python 硬分支, `.item()` detach 梯度
  - `c = (-k_m).item()` — detach κ from graph
  - 唯一能 work 的分支: κ>0 (spherical) (`theta / sqrt(kappa)`)
- **影响**: Task #89 训练时 θ_m init = [0.0] → 所有 18 个 (layer, κ_m) 进入 κ=0 Euclidean 分支 → 1000 epoch 完全锁死精确 0.000000
- **结论**: 5 重证据 "数据本质欧氏" **被 bug 污染**, 锁 0 是因为代码断层而非优化结果

## 2. 实验设计

**变量**: `_per_component_dist_sq` + commitment/codebook loss + `geodesic_distance_sq` 三处 Python 硬分支 → `torch.where` 统一算子 (autograd-safe)

**保持不变**:
- `kappa_m` 重参数化: `kappa_m = κ_max · tanh(θ_m)`, θ_m init=0
- `poincare_distance` / `expmap0` / `proj_to_ball` 函数 (utils.py) — 接受 tensor `c` 是 autograd-safe
- 训练超参 (1000 epoch, Adam, lr 等) 与 Task #89 完全一致
- Task #89 已有产物保留 (bug-polluted baseline)

**启动命令**:
```bash
# Step 1: 修复代码 (3 处)
# Step 2: 验证 — 重跑 scripts/task135_kappa_grad_diagnostic.py (4 run, 50 epoch each)
# Step 3: 重训 — 复用 scripts/task89_stage1_train_rqvae.py 3 臂 (A/B/C × M=1/2/3 × 1000 epoch)
# Step 4: 写 verdict 报告 κ_m 终值 + 与 baseline 对比
```

## 3. 决策触发 (vs Task #89 baseline)

| κ_m 终值模式 (3 臂 × 18 个 κ_m) | 解读 | 决策 |
|----------------------------------|------|------|
| 所有 18 个 κ_m 仍 ≈ 0 (within ±0.05) | **干净验证**: "数据本质欧氏" 结论**无 bug 噪声** | ✅ retro caveat 关闭, 写 Task #89 final verdict 续章 |
| 至少 1 个 κ_m 显著偏离 0 (|κ|>0.3) | **强推翻**: 修复后曲率**有真实信号** | ⚠️ 启动下游 Stage 2/3/4 (A 臂) 验证 recall 是否提升 |
| 部分 κ_m 偏离但 recall 无提升 | 过拟合式漂移 | ⚠️ 加 L2 正则拉回 0 重跑 |
| 至少 1 个 κ_m 仍 None grad | **修复未生效** | ❌ 检查 `poincare_distance(c_tensor)` 链路, 重新审查修复 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 代码修复 (3 处) + py_compile | ~5 min |
| 重跑 Task #135 验证 (4 run × 50 epoch) | ~5 min |
| Stage 1 重训 3 臂 (A/B/C × 1000 epoch) | ~6 h (顺序 GPU 1) |
| 总计 | ~6 h |

## 5. 风险与缓解

**风险 1**: `torch.where` 同时计算 3 分支, GPU 显存/计算量增加 3×
→ 缓解: 单 batch 32 item × 32 dim, 显存占用仍 <2 GB, 无 OOM 风险

**风险 2**: 修复后 θ_m 学到非 0 κ 但下游 recall 没提升 (过拟合)
→ 缓解: 决策表 §3 已包含 "part偏离 but 无 recall 提升" 路径, 加 L2 正则

**风险 3**: `torch.where` 在 κ=0 边界处不稳定 (sph_sq 含 1/√κ)
→ 缓解: `kappa_abs.clamp(min=1e-8)` 强制下限, sph_sq 与 hyp_sq 在 |κ|≈0 时数值上溢出但被 torch.where 跳过

**风险 4**: Task #89 已有产物 (18 个 ckpt + A 臂下游) 被新训练覆盖
→ 缓解: 写新产物到 `products/task137/` (R9: 任务编号隔离), Task #89 产物保留

## 6. 完成度跟踪

- [x] Task #137 description 写入
- [x] R9 contiguous 1-137
- [ ] 代码修复 3 处 + py_compile
- [ ] Task #135 验证重跑 (4 run, 全部 NON-None grad)
- [ ] Task #89 Stage 1 重训 3 臂 launch (GPU 1)
- [ ] Task #89 Stage 1 重训完成 + κ_m 终值落盘
- [ ] verdict 写入 (vs baseline)
- [ ] loop.md §16 归档

---

## 7. 关联

- 前置: Task #135 (诊断), Task #89 (原训练, 已 retro caveat)
- 关联: Task #84 baseline R@10=0.1020, Task #88 c555 R@10=0.1051
- 后续: 若 κ_m 偏离 0, 启动 Task #137+ (Stage 2/3/4 A 臂下游验证)