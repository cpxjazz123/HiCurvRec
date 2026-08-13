**Issue #142 CLOSED — GO** (commit 5b93cd884a530b71cf56638fde38ef3bb57d4481, pushed to gitlab origin/main)

## 背景
Issue #141 B 臂证明 codeword-specific curvature 有效但分化不足 (c_0,b std=0.0023, 仅用有界范围 [0.5,1.5] 的 1%, per-codeword anchor 抑制分化)。本 issue 将 anchor 改为 **mean-only** (`L_anchor = λ·(mean_b log c_0,b − log c0_ref)²`), 只锚整体均值防集体塌缩, 个体自由分化。

## Gate 1 — 机制与等价性: PASS (5/5)
- c_0,b 初始全等时, 新旧 anchor 贡献 0 → distance (max diff 4.2e-07) / assignment / loss / 前三位 SID 与 reference 完全一致;
- 单列扰动隔离; 64 θ 进 optimizer 梯度 finite; d_norm mean=1.0; 无 NaN/边界/捷径。

## Gate 2 — 分化度审计: PASS (分化度释放 163x)
| 指标 | Issue141 B (per-item anchor) | Issue142 (mean-only anchor) |
|---|---|---|
| c_0,b std | 0.0023 | **0.375 (163x)** |
| c_0,b range | [1.000, 1.011] | **[0.5006, 1.4995]** (铺满 [0.5,1.5]) |
| occupancy Spearman | -0.604 (p=3.8e-7) | **-0.825 (p=1.6e-16)** |
| density Spearman | -0.15 (n.s.) | **-0.33 (p=0.015)** |
| 边界命中 / 尺度捷径 | 0 / False | 0 / False |

## Gate 3 — Stage3/4: GO (TARGET 未达)
- Stage3: best valid R@10=**0.1291** (epoch 200) > Issue141 B 0.1282 (+0.0009)。
- Stage4: **test R@10=0.0992** vs Issue141 B 0.0990 (**+0.0002**); first_error=1 87.93% (持平); fixed bucket [200,1000) (reference-0 编码, n=6717) 0.1691 (-0.0003)。
- TARGET (>0.1020) 未达。

## 关键结论: 分化度-收益曲线饱和
- 0.0023 std → +0.0022 vs A (Issue141)
- 0.375 std → +0.0002 vs B (Issue142)
- **L0 曲率分化空间已耗尽** — 更大的分化不带来更多 R@10 收益。occupancy Spearman 从 -0.604 增强到 -0.825 验证机制方向正确, 但测试集收益饱和。
- 下一瓶颈: Stage3 valid-test gap (0.1291 vs 0.0992, 差 0.030) 或 L1/L2 尚未 per-codeword。

产物: 4-stage 完整目录 (R46 baseline 复制, 仅改 anchor)、gate1_equivalence_report.json、l0_branch_curvature_trace.json/csv、scale_shortcut_audit.json、Stage4 六指标 + raw predictions、issue142_verdict.json。
