## Issue #76 Gate 4 FAIL + Phase A.1 径向扩张 NO-GO 闭环

### 关键发现 (本轮)

1. **HAB 曲率失效 bug** (commit cba50d3): `c = max(-κ, 1e-6)` 在 CURV_PRIOR 后 κ>0 时把 c 压成 1e-6, 触发 R2 fallback. 修复后端到端 test_R10 0.1062→0.1063, ratio 1.2775→1.251. c 修复影响仅 +0.0001, 印证 (2) 是主导瓶颈.

2. **REL_STRUCT 量纲错配** (commit 32beba0): 优化 `√c·r` (切空间, 0.42~0.63) 但实际 ρ_ball = `tanh(√c·‖e‖)` 仅 0.23~0.29 — tanh 压缩 38~56%. HAB 拿到的 Dbar ≈ 欧氏重参数化 → 14342 参数不携带几何归纳偏置 → 容量增益被过拟合吃掉.

3. **Poincaré 球 ρ_ball 数学上限 tanh(1)=0.76** (safe_distance u_max=0.985 截断). RHO_BALL_TARGET 不可超过 0.75.

4. **codebook norm 与 SID 健康互斥**: 要 ρ_ball=0.50+ 反解 norm=[0.69,0.97,1.26] vs kmeans 健康 0.11. rescale 强行放大 → util=0.008 SID 死掉. 不 rescale → ρ_ball≤0.10 永远到不了 target.

### Gate 4 评估 (FAIL)

| 指标 | v77 基准 | v15+c=1e-6 | v15+c 修复 (habcfix) |
|---|---|---|---|
| best test R@10 | **0.1080** | 0.1062 | 0.1063 |
| best valid R@10 | 0.1312 | **0.1358** (新高) | 0.1344 |
| ratio @best test | 1.215 | 1.2775 | 1.251 |

两个 test_R10 都 < v77 0.1080, 未达 0.1100 目标. **过拟合机制已确认 = HAB 增参数无几何信息**.

### 径向扩张三轮 NO-GO

- v2 (rescale+target=0.85): util=0.008 SID 死掉
- v3 (rescale+target=0.72): grad_κ=78401 爆炸
- v4 (温和 target, 无 rescale): ρ_ball 永远 ≤0.10

### 产物

- commit cba50d3: HAB 曲率修复 (hyperbolic_attention_bias.py + Stage3/Stage4 调用点)
- commit 32beba0: 径向量纲修复 + Stage2 主脚本保留 REL_STRUCT_ON_BALL 可选开关
- verdict verdicts/issue76_hab_curvature_bug.md + verdicts/issue76_radial_exploration_nogo.md
- 实验 _history/issue141_v15_hab_stage3/ + _history/issue76_habcfix_v15/ + _history/issue76_stage2_radball{,_v2,_v3,_v4}/

### 下一步

**Phase A.2 行为监督 κ** 才是正确路径 — 行为图 (train split only) 监督信号直接对齐"协同近邻 vs 几何近邻", 绕开径向瓶颈.

- L0/W0=10 强共现, L1/W1=3 中共现, L2 next-item
- KL(p_beh ‖ p_geo) 替代量纲不匹配的径向目标
- 交替更新 κ 与 RQ-VAE 避免量化距离作弊

需要 train.split.parquet + Stage2 架构改造 (大改动, 单独 issue).