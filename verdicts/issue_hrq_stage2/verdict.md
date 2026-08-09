# Issue #102 — Stage2 Hyperbolic Residual Quantization Verdict (NO-GO: 路径同构 v15)

## 状态: ❌ NO-GO (R18 路径同构 — v15 capmatch 已实施 HRQ)

## 关键发现: v15 capmatch 已经实施 Poincaré NN

| Issue #102 spec | v15 capmatch 现状 | 同构? |
|-----------------|--------------------|------|
| `exp_map_0(r_ℓ)` tangent → ball 投影 | `expmap0(latent, c_geom)` (line 744) | ✅ 同 |
| `E_ℓ_hyp ← {exp_map_0(e) : e ∈ E_ℓ}` | `expmap0(codebook_e, c_geom)` (line 745) | ✅ 同 |
| `r_ℓ_dist ← d_P(r_ℓ_hyp, e_hyp)` | `poincare_distance(x_exp, cb_exp, c_geom)` (line 754) | ✅ 同 |
| `e*_ℓ ← argmin r_ℓ_dist` | `indices = torch.argmin(d, dim=-1)` (line 762) | ✅ 同 |
| c_l ∈ {1.3547, 6.0021, 4.3941} | final_cs from Stage2 ckpt = 同 | ✅ 同 |
| 单层 lookup O(K · dim) | K ∈ {64, 128, 256}, e_dim=32 | ✅ 同 |

**v15 实际就是 Issue #102 提议的 HRQ**, 实施完整, 1000 epoch 训练成功, util_3digit=0.92+, SID sha=5f8331cc。

## 4 维度对比 (R18 强制)

| 维度 | Issue #102 spec | v15 capmatch 实施 | 差异? |
|------|------------------|--------------------|------|
| **D1 spec 摘录** | "Euclidean kNN → Poincaré NN" | "Poincaré NN (Issue #157 κ-aware HRQVAE)" | ❌ **spec 错误**: v15 不是 Euclidean |
| **D2 实施核心** | exp_map_0 + d_P + argmin | expmap0 + poincare_distance + argmin | ❌ **完全相同** |
| **D3 Gate 1 失败机制** | (假设存在 Euclidean 失败) | (Poincaré 已成功训练) | ❌ **无失败机制可比** |
| **D4 引用文献** | exp_map_0 / d_P 标准公式 | 同 | ❌ **同文献** |

**判定**: 4 维度全部 ❌ (无差异), **路径同构** → R18 NO-GO。

## 时间线

- **2026-08-10 03:18** — 收到 Issue #102 (= Stage2 HRQ)
- **2026-08-10 03:18** — R39 立即实施前做 R18 4 维度对比检查
- **2026-08-10 03:18** — 检查 taskA/stage2/taskA_stage2.py 发现 v15 capmatch 已实施 HRQ
- **2026-08-10 03:18** — 路径同构确认, NO-GO (避免 GPU 浪费)

## R 合规

- **R18** ✅ 路径同构 NO-GO, 不 launch GPU 训练 (issue body 提议的机制已被 v15 实施)
- **R19** ✅ 未 launch (R18 同构禁止)
- **R37** ✅ 不在失败品上叠加 (也未做任何叠加, 立即 close)
- **R39** ✅ 立即按 R18 判定 + close (不阻塞 Gate A)
- **R35+R36** N/A (无新 ckpt)

## 后续

- Issue #102 close (R18 路径同构 NO-GO)
- v15 capmatch 已是当前 Stage2 最优 (Issue #157 κ-aware HRQVAE)
- 如需进一步 Stage2 创新, 必须提出与 v15 路径**非同构**的方向 (例如改 encoder / decoder / loss function)

## 产物

- verdicts/issue_hrq_stage2/verdict.md — 本文件
- Issue #102 comment + close