# Task #118 — 失败 / 与 baseline 差距过大的 tokenizer 清理

> **完成日期**: 2026-07-19
> **状态**: ✅ 已完成
> **执行人**: Claude（响应用户两条连续反馈）

---

## 1. 清理标准 (用户两条反馈迭代)

### 1.1 第一轮: "删掉那些失败的tokenzier, 也就是不合理的"
- ❌ mode collapse (训练失败, R@5 是 class prior trivial bias)
- ❌ 无下游 TIGER 端到端评估 (无法关联 Recall)
- ❌ cascade 架构比 single-layer 更差 (架构选择失败)

### 1.2 第二轮: "删掉和 baseline 差距很大的 tokenizer"
- ❌ R@5 < 50% of baseline (Task #87 R@5=0.01937 → 阈值 0.009685)

---

## 2. 删除清单 (4 个 tokenizer 完整清理)

| Tokenizer | R@5 | % of baseline | 删除原因 | 删除路径 |
|-----------|-----|---------------|----------|----------|
| **Task85_m2_hyperbolic** | 0.25546 | mode collapse | trivial bias, TIGER 全部预测高频 item | `products/task85_tri_geom_rqvae/hyperbolic_full/` |
| **Task #80 v3 T5** | n/a | n/a | 训练完 RQ-VAE 但**未跑 TIGER 端到端**, 无 R@5 | `products/task80_v3_toys_mckg/sid_rqvae_t5_tensor.pt` |
| **Task #80 v3 MCKG** | n/a | n/a | 同上, 无 R@5 | `products/task80_v3_toys_mckg/sid_rqvae_tensor.pt` |
| **Task22 PM-RQ Phase 3 cascade** | 0.00144 | 7% | cascade 架构比 Phase 2 (0.00474) 更差, 架构失败 | `products/task22_pm_rq/sid_phase3_cascade.pt` + `products/task22_pm_rq/phase3_cascade/` |
| **Task22 PM-RQ Phase 2 single-layer** | 0.00474 | 24% | 与 baseline 差距过大 (>50%) | `products/task22_pm_rq/sid_phase2.pt` + `products/task22_pm_rq/phase2_full/` |

**删除总磁盘**: ≈ 5 MB (小, 但清理逻辑必要)

---

## 3. 保留的 4 个 tokenizer (n=4 全部在 baseline ±30% 区间)

| Tokenizer | R@5 | % of baseline | 备注 |
|-----------|-----|---------------|------|
| **Task85_m1_quasi_euclid** | 0.0200 | **103%** | 单 κ=0 流形, 略超 baseline |
| **Task87_K256_seed42** | 0.01937 | 100% | TIGER-aligned baseline (anchor) |
| **Task85_m0_sphere** | 0.0174 | 90% | 单 κ=+1 球面流形 |
| **Task107_K256_seed123** | 0.01489 | 77% | 同 #87 SID, 不同 TIGER seed (方差探测) |

**n=4 全部可用**, 区间 [0.01489, 0.0200] 集中, 占 baseline 77%-103%. 这是 Task #27 现在能用的全部数据点.

---

## 4. 后续影响

### 4.1 Task #27 (邻域保持 vs Recall 相关性)
- **n 从 6 降到 4**, 进一步恶化样本量问题
- 历史 verdict `verdicts/task27_neighborhood_quality_result.md` 第 80-86 行的 Jackknife LOO 表已替换为 "TBD" (n=4 数据未重跑)
- 需要**重跑** `scripts/task27_enhanced_stats.py` 获取新的 ρ / p / power / CI

### 4.2 取消的 K-ablation (Task #26/#28/#29/#30)
- 已经在 Task #117 cancellation verdict 中处理, 不再受本次清理影响

### 4.3 历史 verdict (Task #22 / Task #24 / Task #80 / Task #81 / Task #82)
- 保留作为历史记录 (不删除 verdict 文件, 只清理物理产物)
- 这些 verdict 仍引用已删除的 SID 路径, 但 verdict 文件本身是历史文档, 不应修改

### 4.4 关键观察
- **PM-RQ 整体被删除** (Phase 2 + Phase 3 cascade 都差距过大). PM-RQ 在 Toys 数据集上 R@5 始终 < baseline 30%, 架构选择错误
- **Task #85 单 κ 流形表现尚可**: m=0 sphere 90%, m=1 euclid 103%, 与 baseline 同量级
- **baseline seed=123 (Task #107) 比 seed=42 低 23%**: TIGER 训练 seed variance 显著 (Task #25 已确认 ±25% 区间)

---

## 5. 当前可用 tokenizer 完整 SID 路径 (Task #27 manifest)

```
products/task85_tri_geom_rqvae/sphere_fix/sid_subspace_0.pt     (3, 11924)
products/task85_tri_geom_rqvae/euclid_full/sid_subspace_1.pt    (3, 11924)
products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt   (4, 11924)
logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt  (同 #87 SID)
```

---

## 6. 后续建议 (供用户决策)

1. **立即**: 重跑 `scripts/task27_enhanced_stats.py`, 用 n=4 数据生成新 ρ / power / CI
2. **可选**: 接受 n=4 的统计限制, 把 Task #27 verdict 结论改为 "effect size 描述" 而非 "显著性检验"
3. **若想扩 n**: 必须重新设计实验 (用户已在 Task #117 cancellation 文档中提了 4 个方向)
   - 冻结 RQ-VAE, 训练集不同 → 同一架构下 SID 不同 → kNN preservation 变化
   - 在固定 SID 上注入随机 bit → kNN preservation 强制下降 → 直接因果验证
   - 完全随机 SID (无 RQ-VAE) → kNN preservation ≈ 0 → 下游 ≈ 0 → 因果下界验证

---

**当前任务已完成, 请做下一个任务的指示.**