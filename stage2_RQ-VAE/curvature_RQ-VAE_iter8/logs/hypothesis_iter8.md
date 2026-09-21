# iter8 Hypothesis Designer（Agent C，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter8（沿 BACKUP P1 路径）
- **机制**：P1 M2 Intrinsic Residual Reference Point 重构（Agent B 唯一推荐；P2 BACKUP）
- **审计基线 commit**：`097c573a11dea2613f69306efbb44ba2ab8ccab5`
- **审计日期**：2026-09-21
- **作者**：Agent C（Hypothesis Designer），只读取、不修改 Python、不调用训练

## 0. 假设一句话总述

> 若 P1 M2 reference point 从 Poincaré origin 改为 selected codebook codeword 的 Lorentz centroid（按当前 batch assignment 加权），则 Stage 2 残差向量**直接对齐** Stage 3 codeword 加权分布几何（in-loop 无 Stage3 依赖），从而尝试修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）；否则 iter8 NO-GO。

## 1. 直接效应（DE）— 必须在 Stage2 step ≤ 5000 内打印数值证据

### DE-1 codebook Lorentz centroid norm 与 cyclic c(t) 的 Pearson ρ ≥ 0.7

- **可证伪陈述**：`centroid_norm = ||centroid_e||_2`（按当前 batch assignment 加权平均后的 selected codeword Lorentz centroid）在 cyclic c(t) 的 1 个周期内（T_cycle = 50000 step）出现明显周期性，centroid_norm 与 c(t) 的 Pearson ρ ≥ 0.7（@ step5000）。
- **数值测量方法**：每 100 step 同步记录 `centroid_norm.detach().item()` + `c(t).detach().item()`；对最后 50 个采样点（step4500–step5000）计算 Pearson 相关系数；打印窗口至少覆盖 cyclic c(t) 完整 0.1 个周期。
- **PASS 阈值**：ρ(centroid_norm, c) ≥ 0.7 且符号为正。
- **FAIL 触发条件**：|ρ| < 0.4 → centroid norm 与 cyclic c(t) 几乎无关，centroid 未在 cyclic 通路中传导；机制实质是 stage 2 端常驻正则项，不修复传导路径。
- **不可证伪词扫描**：阈值/判定全部数值化。

### DE-2 残差 latent 与 selected codeword angle 余弦分布与 c(t) 单调反向 ρ ≤ -0.5

- **可证伪陈述**：`next_residual`（经过 P1 M2 重构后的 Stage 2 残差）与 selected codeword Lorentz centroid 之间的 angle 余弦分布，在 cyclic c(t) 的 1 个周期内与 c(t) 单调反向（c 越大 → angle 越大 → cosine 越小），Pearson ρ(cos, c) ≤ -0.5（@ step5000）。
- **数值测量方法**：每 100 step 同步记录 `cos_sim(next_residual, centroid_e).mean().detach().item()` + `c(t).detach().item()`；最后 50 个采样点 Pearson ρ。
- **PASS 阈值**：ρ(cos, c) ≤ -0.5。
- **FAIL 触发条件**：|ρ| < 0.3 → 残差与 centroid 未真正几何对齐；或 ρ > 0 → 反向（residual 离 centroid 越远 cos 越大，与 P1 预期方向相反）。
- **不可证伪词扫描**：阈值/判定全部数值化。

### DE-3 step5000 3-token unique ≥ iter5 同期 22675

- **可证伪陈述**：step5000 时 valid items 的 3-token SID（不含 PAD 列）unique 计数 ≥ 22675（iter5 step5000 同期 baseline）；如果 P1 让 stage 2 退化到 L0/L1 collapse，unique 会跌破 22000，机制 FAIL。
- **数值测量方法**：step5000 时一次性推理 valid items，统计 `len(set(sids[:, :3]))`。
- **PASS 阈值**：unique ≥ 22675。
- **FAIL 触发条件**：unique < 22000 → stage 2 已塌陷；iter6 baseline unique=23221（iter6 step5000）+5.3% 余量。
- **不可证伪词扫描**：阈值/判定全部数值化。

## 2. 性能阈值假设（PH）— 必须经 Stage3 完整训练验证

### PH-1 Stage3 150 epoch 后 test_R@10 > iter6 baseline 0.0534

- **可证伪陈述**：Stage3 完整 150 epoch 跑完后 `test_R@10 > 0.0534`；硬目标 `test_R@10 > 0.065`；若 P1 修复 Stage2→Stage3 几何传导路径，Stage3 T5 表征空间应能利用 P1 注入的 centroid 几何信息。
- **数值测量方法**：Stage3 trainer 150 epoch 后读 `test_final.json`。
- **PASS 阈值**：test_R@10 > 0.0534（避免 v321 lock 行为）。
- **FAIL 触发条件**：test_R@10 ≤ 0.0534 → P1 未修复传导路径；或 SID 字节级 lock baseline → v321 论证再次验证。
- **硬目标**：test_R@10 > 0.065。

### PH-2 valid→test drift ratio > 0.9（避免 iter32 drift 0.886）

- **可证伪陈述**：Stage3 150 epoch 跑完后 valid_best recall@10 / test_recall@10 > 0.9；iter32 drift ratio = 0.886（FAIL），iter28 baseline drift = 0.985（PASS）。
- **数值测量方法**：Stage3 trainer 每 5 epoch 在 test 上评估（已在 iter32 后默认开启），取 valid_best epoch 对应 test_recall@10。
- **PASS 阈值**：valid/test ratio > 0.9。
- **FAIL 触发条件**：ratio ≤ 0.886 → P1 复制了 iter32 valid 过拟合 pattern。

## 3. 如何避免 iter32 valid→test drift 与 v321 Stage3 T5 SID lock

### 3.1 iter32 drift 根因复盘

- iter32 = R36n b (per-layer 异质 c) + R36n f (per-layer 异质 sk_eps)，valid_best 0.06586 (+9.4% vs iter28) 但 test_R@10=0.05832 (-1.65%)，drift ratio 0.886 vs baseline 0.985。
- 根因：Stage 1 端越复杂 → valid 拟合越强 → test 漂移越严重。
- v321 R36n f (Angular Orthogonality) 已证 Stage 3 T5 SID 表征空间对 Stage 1 端几何变更强 lock（v319/v320/v321 连续 3 个不同机制 lock baseline）。

### 3.2 P1 避免 iter32 drift 的硬约束

| 约束 | 数值上限 | 触发条件 | 反向动作 |
|---|---|---|---|
| M2 reference point | selected codebook Lorentz centroid（不引入新 loss） | centroid 计算失败（NaN/Inf） | 立即 R50 + 回退到 origin ref point |
| centroid 反向传播 | 全部 requires_grad（不写 .detach()） | DE-1 / DE-2 / DE-3 任意 FAIL | 检查 `_poincare_to_lorentz_t` 与 `_lorentz_normalize_t` 是否阻断反向 |
| cyclic c(t) 联动 | centroid norm 随 c(t) 周期性变化 | ρ(centroid_norm, c) < 0.4 | 强制 centroid 重新计算（含 batch dependency） |

### 3.3 P1 跳出 v321 Stage3 T5 lock 的机理

- v321 lock 根因：Stage 1 端所有几何变更都被 Stage 3 T5 SID 表征空间 argmin 路径"吸收"。
- P1 不在 Stage 1 端做几何变更，而是把 Stage 2 残差 reference point 改为 selected codebook centroid，让 Stage 2 残差向量**直接对应** Stage 3 codeword 加权分布。
- 这绕开 v321 论证的"Stage 1 端变更无法传导"——P1 直接把传导链路显式插入 Stage 2 → Stage 3 codeword 加权分布。

## 4. R37 + R36p + R36r 三态预检

| 检查项 | 触发条件 | P1 当前预估 | 失败动作 |
|---|---|---|---|
| R37 SID 字节级 lock | iter8 sids_for_hgrec.npy MD5 = baseline MD5 | **不命中**（P1 改 M2 reference point，argmin 路径必变） | 立即 R50 + 检查 M2 reference point 计算是否被 `.detach()` 截断 |
| R36p util 阈值 | L0/L1/L2 utility < 75% | **不命中**（P1 in-loop 不引入新参数，codebook 几何不变） | 降低 reference point weight |
| R36r silent no-op | DE-1 / DE-2 Pearson = 0 / grad 全 zero | **不命中**（centroid 必须含 batch dependency 才能维持 DE-1） | 检查 `_lorentz_normalize_t` 是否阻断反向 |

## 5. 不可证伪词扫描自检

- DE-1 / DE-2 / DE-3 全部使用 `≥ / ≤ / < / > / =` 数值阈值 + `PASS / FAIL` 二元判定；
- PH-1 / PH-2 全部使用 ratio / weight / 占比数值判定；
- 全文不含 "可能提升" / "也许" / "或许" / "视情况而定" 等不可证伪词。

## 6. 假设验证边界

1. 本假设只对 P1 M2 reference point 重构给出可证伪条件；不覆盖 P2 / P3。
2. P1 实施需保持 `[256,256,256,1]` codebook 容量、SID 长度、item 顺序与 Stage3 输入协议不变。
3. 本文件不修改 Python 代码，不触发训练或 gradient check。
4. DE-1 / DE-2 / DE-3 必须在 Stage2 step ≤ 5000 内打印数值证据；PH-1 / PH-2 需 Stage3 150 epoch 跑完后验证。
5. 若 DE-1 / DE-2 / DE-3 任意一项 FAIL，stage2 训练必须立即 R50 + 回退到 iter6 baseline，不进入 Stage3 训练。

## 7. 关键证据来源

1. iter7 审计 commit `097c573a11dea2613f69306efbb44ba2ab8ccab5`
2. iter8 candidate pool: `stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/lit_search_iter8.md`
3. iter8 dominant bottleneck + forbidden directions: `stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/iteration_bridge.md`
4. Agent B 唯一推荐: `stage2_RQ-VAE/curvature_RQ-VAE_iter8/logs/direction_decision_iter8.md`
5. v321 memory: `v321-r37-fail-sid-locks-baseline.md`
6. iter32 memory: `iter32-r37-fail-valid-test-drift.md`

只读取，未修改 Python；未触发训练。