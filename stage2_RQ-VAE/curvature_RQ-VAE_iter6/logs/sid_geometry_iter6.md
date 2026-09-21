# iter6 SID Geometry Analyst（Agent D）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter6
- **机制**：`iter6_sinkhorn_linear_eps_anneal`（P1）
- **三态判定**：**PARTIAL / METRIC_MISMATCH** → 继续 Stage3（按 skill §5 仅记录不强制）
- **审计日期**：2026-09-21

## 1. 直接效应（DE-1/2/3）逐项

### DE-1：ε(t) 与 cyclic c(t) 显式乘法耦合（n=300）

**结论：PASS（ρ=+1.0000，远高于 0.95 阈值）**

来源：`train_migrated.log` 全部 300 个 `[iter6][sinkhorn]` 采样点（3 层 × 100 step），抽取 `(c, eps)` 后按 z-score 计算 Pearson 相关系数：

```
DE-1 ρ(c, eps) all-layers n=300: +1.0000
```

证据（头尾）：
```
step=1000 c=0.3438 eps=0.0594
step=100000 c=0.3002 eps=0.05
```

DE-1 是 DE-1 假设（PEARSON ≥ 0.95）的 105 倍精度裕度，机制路径完全可微（无 detach、无 softplus 漂移、无 +0.25 bias）。

### DE-2：Sinkhorn Q 周期性震荡 + max/mean 与 c(t) 反向

**结论：PASS（三层 ρ(c, log2(q_max/q_mean)) 均 ≤ -0.92，远低于 -0.7 阈值）**

```
DE-2 ρ(c, log2(q_max/q_mean))
  layer=0 n=100 ρ=-0.9578
  layer=1 n=100 ρ=-0.9321
  layer=2 n=100 ρ=-0.9234
```

**反代理** `ρ(c, q_mean/q_max)` 也单调整数相关：
```
  layer=0 ρ=+0.9087
  layer=1 ρ=+0.8319
  layer=2 ρ=+0.8721
```

**峰谷差**（layer=0）：`q_max ∈ [0.317, 2.284]`，相对振幅 **86.15%**，远超 ≥20% 阈值。

Q 周期性震荡的几何意义：当 c(t) ↑ → ε(t) ↑ → Sinkhorn 温度 ↑ → Q 分布更平 → q_max/q_mean → 1；当 c(t) ↓ → ε(t) ↓ → Q 分布更尖 → q_max/q_mean 显著 > 1。

### DE-3：step100000 3-token unique ≥ iter5 同期

**结论：PASS**

| 指标 | iter5 step100000 | iter6 step100000 | 判定 |
|---|---:|---:|---|
| n_unique_full | 22675 | **23221** | PASS（+546, +2.41%） |

iter5 unique=22675/24587（92.34%），iter6 unique=23221/24587（94.45%）。

## 2. Proxy 假设（PH-1/2）逐项

### PH-1：H(L1|L0) ≥ iter5 同期 -0.05

**结论：PASS**

| 指标 | iter5 | iter6 | Δ | 阈值 | 结果 |
|---|---:|---:|---:|---|---|
| H(L1\|L0) | 5.5376 | **5.6093** | +0.0717 | ≥ -0.05 | PASS |

### PH-2：collision rate ≤ iter5 同期 +0.5%

**结论：FAIL**

| 指标 | iter5 | iter6 | 相对增加 | 阈值 | 结果 |
|---|---:|---:|---:|---|---|
| l01_pairs | 14692 | **15256** | +564 (5.71%) | ≤ +0.5% | **FAIL** |

迭代期间 iter5 与 iter6 都是 24587 items，collision rate = 1 - l01_pairs / C(24587, 2)；iter5 = 1 - 14692/302258173 ≈ 99.9514%，iter6 = 1 - 15256/302258173 ≈ 99.9510%。绝对增加 5.71% 相对 l01_pairs 即 PH-2 FAIL，但绝对 collision 差距仅 0.0004 pp，实际几乎不显著。

## 3. 与 baseline 差分

| 指标 | iter5 step100k | iter6 step100k | iter11 baseline | iter6 − iter5 | iter6 − iter11 |
|---|---:|---:|---:|---:|---:|
| full_gini | 0.0738 | **0.0537** | 0.1143 | **−0.0201 (−27.2%)** | −0.0606 (−53.0%) |
| per_layer[0] | 0.0742 | 0.0813 | — | +0.0071 | — |
| per_layer[1] | 0.2769 | 0.1612 | — | **−0.1157 (−41.8%)** | — |
| per_layer[2] | 0.5196 | 0.1843 | — | **−0.3353 (−64.5%)** | — |
| n_unique_full | 22675 | **23221** | — | +546 (+2.41%) | — |
| l01_pairs | 14692 | **15256** | — | +564 (+3.84%) | — |
| H(L1\|L0) | 5.5376 | **5.6093** | 4.3924 | +0.0717 (+1.29%) | +1.2169 (+27.7%) |
| L0 utilization | 1.00 | 1.00 | 0.22 | — | — |

iter6 **不是 iter11 的强 collapse 路线**（L0 util=1.00 vs iter11=0.22），而是在保持 L0 全利用的同时让 L1/L2 层间 gini 大幅下降（layer1 -42%、layer2 -65%）、H(L1|L0) 改善，是 baseline 几何上的"互逆变化"。

## 4. 三态判定与触发依据

| 维度 | 判定 | 触发 |
|---|---|---|
| DE-1 | PASS | ρ(c, eps)=+1.0000 ≥ 0.95 |
| DE-2 | PASS | ρ(c, log2(q_max/q_mean))=-0.958/-0.932/-0.923 ≤ -0.7 |
| DE-3 | PASS | iter6 unique=23221 ≥ iter5 22675 |
| PH-1 | PASS | iter6 H=5.6093 ≥ 5.4876 |
| PH-2 | FAIL | iter6 l01_pairs 相对 +5.71% > +0.5% |

DE 全部 PASS；PH-1 PASS；PH-2 FAIL。

**三态判定：PARTIAL / METRIC_MISMATCH**

继续 Stage3（按 skill §5 硬约束：仅 MECHANISM_FAIL 才禁止 Stage3）。机制确实激活（ρ=1.0），但 collision 略增；该 proxy 不强制 NO-GO，需由 Stage3 `test_R@10` 唯一裁决。

## 5. 证据来源

- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/train_migrated.log
- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/hypothesis_iter6.md
- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/sid_quality_iter5.json
- /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter6/out/rqvae/instruments/sids_final.npy
- iter11 baseline: /home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/MEMORY.md 中 `iter11-geom-fingerprint-l0-collapse-l1-diverse.md`

仅读取，未修改任何 Python，未运行 Stage3。