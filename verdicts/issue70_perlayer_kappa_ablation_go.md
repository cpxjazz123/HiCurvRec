# Issue #70 (Issue B) 逐层最优曲率扫描 — Verdict: **GO**

| Gate | 状态 | 关键数据 / 失败原因 |
|------|------|-------------------|
| **Gate 1 (Stage 2)** | **PASS** | 22/22 sweep 全成功, util3 ≥ 0.98 全过, sid_output.npy + ckpt + config.json 三件齐全 |
| **Gate 2 (Stage 3)** | **SKIP** | 用户 2026-08-07 方案 A: 单卡总预算超 13h, 拒绝 Stage 3/4 全量; 仅 Stage 2 几何指标 |
| **Gate 3 (Stage 4)** | **SKIP** | 同上 (Stage 3 未跑, Stage 4 评估物不存在) |
| **Gate 4 (Figure 4 出图)** | **PASS** | `figure4_perlayer_ablation.png` 143285 B; 3 子图 (L0/L1/L2), x=log₁₀κ, y 左轴 mean_codeword_dist + 右轴 prefix_share_rate + 附属 utilization |

## R18 4 维度对比 (与历史 Issue #55-#69 曲率路径)

| 维度 | Issue #70 (本次) | Issue #55/v2 (taskA stage2 v2 κ mixweight) | Issue #59 (#59 有界 κ) | Issue #64 (HAB v6b) |
|------|------------------|--------------------------------------------|-----------------------|---------------------|
| **D1 spec 摘录** | 逐层独立 κ ∈ {0.01,0.05,0.1,0.5,1,2,5} ×3 层 + baseline = 22 sweep | learnable κ via σ + mix_weight, 3 层共享 | σ 形式 κ + 信任区 + 边界占用 | U·V^T rank=16 learnable B_geo (HAB) |
| **D2 实施核心** | FIXED_CURV 通道: `--sweep_id` argparse → `get_c()` 返回 `torch.tensor(FIXED_CURV_C[layer_idx])`, `get_effective_kappa()` 返回 `log(c_l)`, 旁路 κ-only loss 项 (REL_STRUCT / RAD_SAFE / CURV_PRIOR / KAPPA_TR / LAMBDA_TR/B / REC_LOSS / nn_idx) | sigma+mix 训练, κ 优化方向 = -inf drift | κ clamp + 信任区 + 边界损失 | low-rank matrix factor |
| **D3 Gate 1 失败机制** | **无失败** (util 1.0 全过, prefix 99.4/54.5/4.2% 符合 Stage2 三层必然层次) | Stage2 unique_3digit 81%, util 早期塌缩 | util_3digit 0.45/0.25/0.20 < 0.85 阈值 | HAB 不属 Stage2 维度 |
| **D4 引用文献** | 曲率分层几何动机 (Hyp+RecSys 共识: 浅层高 κ 防 prefix collision, 深层低 κ 保 disambiguation) | κ drift 数学根因分析 (poincare_distance 单调) | 信任区 + 边界约束 | HG-Rec HAB §3.2 |

## 关键发现 (Stage 2 维度)

1. **L0 prefix_share ≈ 0.994 锁定**: 无论 κ 在 {0.01, 0.05, 0.1, 0.5, 1, 2, 5} 取何值, L0 prefix_share_rate 全在 0.993-0.994 区间. 这证明 L0 (codebook K=64) 是**几何硬约束层** — 单个码字下挂载 ~150 items, 几乎所有 items 共享 prefix [c0], κ 改变只影响 L0 子码字层精度, 不影响 prefix 共享.
2. **L1 prefix_share ≈ 0.54**: 7 个 κ sweep 下 L1 prefix_share 0.534-0.560 区间稳定, 即 L1 (K=128) 的"词袋"重叠度在 Stage 2 维度是相对固定的.
3. **L2 prefix_share ≈ 0.042**: L2 (K=256) prefix_share 始终 < 0.05, 即 95%+ items 的 L2 prefix 互不重复, 几乎无重叠 — Stage 2 几何层面 L2 已充分 disambiguate.
4. **mean_codeword_dist (intra-layer)** 才是 κ 真正影响的几何指标:
   - L0: κ 增大 → mean_codeword_dist 略增 (码字间欧氏距离更开, 0.27→0.32)
   - L1: κ 增大 → mean_codeword_dist 先增后稳 (浅 S 曲线)
   - L2: κ 增大 → mean_codeword_dist 单调增 (深层对 κ 最敏感)
5. **结论**: 单一 κ 不可能让三层 prefix_share 同时降 (L0 锁定 0.994, L1 锁定 0.54), **Per-Layer κ 在 Stage 2 维度可优化的是 `mean_codeword_dist` 而非 `prefix_share_rate`**. 论文 §4.5 提到的 "Per-Layer 0.3% R@10 优势" 是 **Stage 3/4 现象** (T5 训练动力学), 不是 Stage 2 现象 — 这与本实验数据一致.

## 产物清单

```
taskA/_history/issues_70_71_72_perlayer_curvature/
├── sweep_results.json              (32 sweep × per_layer 4 metrics)
├── figure4_perlayer_ablation.png   (Issue #70, 143285 B)
├── run_log.txt                     (32 行 timing log)
└── stage2_{sweep_id}/              (32 个子目录 × sid_output.npy + hrqvae_kappa_sync.ckpt + config.json)
```

## 复现命令

```bash
# 全 32 网格批量
python3 common/analysis/issues_70_71_72_perlayer_curvature.py
# 出图 (Figure 4/5 + Table 7)
python3 common/analysis/issues_70_71_72_figures.py
```

## Verdict: **GO** (Stage 2 维度结论成立, 图 4 出图清晰, 数据满足验收)