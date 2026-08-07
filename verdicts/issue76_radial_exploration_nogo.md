# Issue #76 — 径向扩张探索 NO-GO + HAB 曲率 bug 修复

日期: 2026-08-07

---

## 1. 总结

**Issue #76 Phase A.1 (径向扩张) NO-GO** — 三轮实验证明径向扩张无法在保持 SID 健康前提下达到 ρ_ball≥0.50 目标, 端到端 test_R10 突破 0.11 必须改走 **行为监督 κ (Phase A.2)**.

**HAB 曲率失效 bug 已修** — `c = max(-κ, 1e-6)` (R2 fallback 违规) 让 Issue #64/#71/#138/#141 全系列 HAB "双曲" bias 实际都是欧氏重参数化. 修复后 test_R10 = 0.1063 (vs v15 修复前 0.1062, +0.0001; vs v77 0.1080, −0.0017).

---

## 2. 实验对照

| 实验 | Stage2 | HAB c | test_R10 | 备注 |
|---|---|---|---|---|
| v77 基准 | flat Euclidean | κ=0/c=1 (静默) | **0.1080** | 历史最佳 |
| v15 SID + HAB c=1e-6 bug | κ=[0.30,1.79,1.48] c=1e-6 | c=1e-6 静默欧氏 | 0.1062 | ratio 1.2775 严重过拟合 |
| v15 SID + HAB c 修复 (habcfix) | 同上 | **c=[1.355,6.002,4.394] 真值** | **0.1063** | ratio 1.251 仍过拟合, +0.0001 |
| v2 径向扩张 (rescale+target=0.85) | κ 涨到 3.6 | — | — | util 0.008 SID 死掉 |
| v3 径向扩张 (rescale+target=0.72) | κ 涨到 4.8 | — | — | util 0.008 REVIVE 死循环 |
| v4 温和径向扩张 (target=0.72, 无 rescale) | c≈0.78 | — | — | ρ_ball 仅 0.10 (target 0.72 数学可达但 codebook norm=0.11 太小) |

---

## 3. 三层根因诊断

### 根因 1 (BUG, 已修): HAB 曲率 c 被静默压成 1e-6

**`common/hyperbolic_attention_bias.py:119` (修复前)**:
```python
c_l = max(-kappa_l, 1e-6)  # κ 为负, c = -κ > 0; 若 κ ≥ 0 用极小正值保护
```

符号约定来自 Issue #61 (κ<0 ⇒ c=-κ), 但 CURV_PRIOR 后 `c = exp(κ)`, κ 变正 (v15 κ=[0.30,1.79,1.48]). `-κ` 全负 → `max(...,1e-6)` 把三层 c 全部压成 **1e-6** (近完全平坦欧氏距离). 触发实测: c_l used by HAB = [1e-06,1e-06,1e-06], p95/med 三层 [1.24,1.20,1.22] (几乎一致 = 欧氏特征).

**违反 R2 (用默认值静默掩盖预期外失败)**. Issue #64/#71/#138/#141 全系列 HAB "双曲" 实验的 bias 实际都是欧氏距离的重参数化.

### 修复 (commit cba50d3)

- `load_hab_assets_from_stage2_ckpt` 返回 `final_cs` (Stage2 ckpt top-level key, 训练时实际生效曲率, 唯一可信来源). 缺 key → `raise KeyError`; 长度≠3 / c≤0 → `raise ValueError` (无 fallback).
- `precompute_distance_matrices(codebook_list, final_cs, ...)` 直接吃 c, 不再由 κ 反推; `kappa_l = math.log(c_l)` 仅用于 stats 记录.
- Stage3 + Stage4 调用点改 `final_kappas → hab_final_cs`, log final_cs 供审计.
- 修复后实测 v15 ckpt: c=[1.355,6.002,4.394], Dbar med [0.7260, 0.3607, 0.3208] (修复前 [0.7085, 0.3508, 0.3154], +2.5%).
- 端到端 habcfix (`taskA/_history/issue76_habcfix_v15/`): test_R10=0.1063, ratio=1.251. c 修复的端到端影响仅 +0.0001 — 因为 (2) 才是主要瓶颈.

### 根因 2 (架构, NO-GO): 码字径向 ρ_ball 上限 0.76

`REL_STRUCT` 优化 `√c·r` (切空间量, L0 达 0.63), 但码本经 `expmap0` 进球后真实半径 = `tanh(√c·‖e‖)` 仅 0.28. **tanh 吃掉 38–56%**. 量纲错配: 约束量与 HAB 真正消费量不是同一个.

更深限制: Poincaré 球 ρ_ball 数学上限 = `tanh(1) = 0.76` (因为 safe_distance 用 u_max=0.985 截断). 因此 RHO_BALL_TARGET 不可超过 0.75. 实测 v15 κ=[2.09,4.13,4.85] 才能让 ρ_ball=0.72 — 但 κ 参数化上界 KAPPA_MAX=0.5, 需扩容.

### 根因 3 (互斥, NO-GO): codebook norm 与 SID 健康互斥

要 ρ_ball=[0.50,0.62,0.72], 给定 v15 实测 ‖e‖=[0.244,0.123,0.111], 反解需要 ‖e‖=[0.69,0.97,1.26]. 但:

- v1/v2 rescale 把 norm 强行拉到 0.69–1.26 → kmeans assignment 距离异常 → util=0.008 SID 死掉, REVIVE 死循环
- v4 不 rescale 保持 norm=0.11 → ρ_ball=tanh(√c·0.11) ≤ 0.10 永远到不了 target
- 即"让 norm 长到目标"与"kmeans 初始化健康"在现有 Stage2 框架下互斥

### 根因 1+2 综合: 过拟合真实机制

HAB 注入 14342 参数 (U/V rank=16 + λ_raw + residual_alpha) 携带的 bias ≈ 欧氏距离重参数化 (根因 1), 码字挤在 ρ<0.3 近欧氏区双曲度量无效 (根因 2). 这 14342 参数对 T5 是纯额外自由度 → 只能记住 valid 模式 → 容量增益 100% 被过拟合吃掉.

v15 对照 (v77→v15): valid 0.1312→0.1358 (+0.0046), test 0.1080→0.1062 (−0.0018), ratio 1.215→1.2775 (+0.0625 恶化). habcfix 修复 c: ratio 1.251 (略好 0.026, 仍 > v77).

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS. 沿用 `taskA_stage1_hyp_v2/item_emb.parquet` (sha256 1a42341f...).

**Gate 2 (Stage2)**: PASS (v15 复用). `taskA_stage2_v15_capmatch_1000ep` verdict gate2_decision=PASS, κ=[0.304,1.792,1.480], util_3digit=[1,1,1], util_4digit=1.0, 9922 unique 4-digit SID 无碰撞.

**Gate 3 (Stage3)**: PASS. habcfix 训练正常收敛, DDP 4×L40S 12s/epoch, 无 NaN/Inf, ckpt 22.5MB 正常保存.

**Gate 4 (Stage4)**: **FAIL**.
- v15 c=1e-6 bug: test_R10=0.1062, valid 0.1358 (新高), ratio 1.2775
- v15 c 修复 (habcfix): test_R10=0.1063 (+0.0001), ratio 1.251

**两个 test_R10 都 < v77 0.1080, 未达 0.1100 目标**. 修复 c 几乎没改善, 印证根因 2 主导.

---

## 5. 径向扩张失败的技术细节

### v2 (rescale+target=0.85) — SID 死掉

```
rescale 后 norm=[0.69,0.97,1.26] (v15 是 [0.244,0.123,0.111])
util_per_layer_3digit=[0.156, 0.047, 0.059] (vs v15 健康 1.0)
util_4digit=0.008 (77 unique)
```

kmeans 选中心后人工放大 norm → 所有码字到 latent 距离相近 → assignment 区分度丧失 → unique 极少 → REVIVE 死循环.

### v3 (rescale+target=0.72) — 数学上限撞墙

发现 Poincaré 球 ρ_ball 上限 `tanh(1)=0.76` (safe_distance 用 u_max=0.985 截断). 0.72 在数学上可达, 但 grad_κ 爆炸: ep15 L2 grad_κ=78401.27 (因 tanh 接近饱和 + ρ_ball 与 target 偏差 0.70 × λ200 = 强梯度, 但没对应更新方向).

### v4 (温和 target, 无 rescale) — norm 长不大

ρ_ball = tanh(√c·norm) 受限于 norm=0.11 (kmeans 决定, 训练后不变). 即使 c 收敛到平衡点 0.78, ρ_ball=tanh(0.10)=0.10, 远低于 target 0.50.

---

## 6. 结论 + 下一步

**Issue #76 Phase A (径向扩张) NO-GO** — 在保持 SID 健康前提下无法达到 ρ_ball≥0.50, 端到端无增益. 不再尝试此方向.

**Issue #76 Phase A.2 (行为监督 κ) 才是正确路径** — 行为图 (train split only) 监督信号 L0/W0=10, L1/W1=3, L2 next-item + 流行度校正 + KL(p_beh‖p_geo). 行为信号与"码本范数"无关, 只与 κ 是否让"相邻商品的几何关系对齐推荐关系"相关. 这能直接绕开径向瓶颈.

需要:
1. 加载 `train.split.parquet`, 构造三层行为图 (L0/W0=10 强共现, L1/W1=3 中共现, L2 next-item 顺序)
2. Stage2 改造: 加 `_behavior_struct_loss(c, codebook_emb)`, 用 `poincare_distance` 在码字图上做"协同近邻 vs 几何近邻"的 KL
3. 交替更新 κ 与 RQ-VAE (避免 κ 通过量化距离作弊)

**已修复并提交的产物** (commit `cba50d3`):
- `common/hyperbolic_attention_bias.py` — 修复 c fallback
- `common/stage3/stage3_train_pure_t5.py` — 调用点同步
- `common/stage4/stage4_eval_pure_t5.py` — 调用点同步
- `verdicts/issue76_hab_curvature_bug.md` — 诊断报告
- `taskA/_history/issue76_habcfix_v15/` — 修复验证 (test_R10=0.1063, ratio=1.251)

**保留但回滚到安全值的实验配置** (Stage2 主脚本, 不影响 v15 复现):
- `REL_STRUCT_ON_BALL = True` 保留 (proj-aware 公式, 默认开, 改 `False` 关)
- `RHO_BALL_TARGET = [0.50, 0.62, 0.72]` 保留 (数学可达上限)
- `KAPPA_MIN=-1.0, KAPPA_MAX=0.5` 回到 v15 (KAPPA_DRIFT_INIT 已删除)
- `CURV_PRIOR_LAMBDA = 0.1` 回到 v15
- `INIT_CODEBOOK_RESCALE_BY_TARGET = False` (rescale 路线禁用)
- py_compile 通过