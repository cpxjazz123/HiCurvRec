# iter7 可证伪假设（P3 Stage3-aware Distillation Term）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter7（targeted bottleneck search）
- **机制候选**：P3 Stage3-aware Distillation Term（Agent B 唯一推荐，P1 BACKUP，P2 (e) FAIL）
- **审计基线 commit**：`e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`
- **审计日期**：2026-09-21
- **作者**：Agent C（Hypothesis Designer），只读取、不修改 Python、不调用训练

---

## 0. 假设的一句话总述

> 若 P3 蒸馏项 loss 真正把 Stage2 geometric 重构向量对齐到 Stage3 beam top-K 检索分布（且不在 Stage 1 端产生 collapse / util 暴跌 / valid 过拟合），则 Stage2→Stage3 几何传导路径修复；否则 iter6 dominant bottleneck（Stage2→Stage3 传导失效）未被修复，test_R@10 ≤ iter6 baseline 0.0534。

---

## 1. 直接效应（DE，Direct Effect）— 必须在 Stage2 step ≤ 5000 内打印数值

### DE-1 蒸馏项数值随 cyclic c(t) 周期性变化

- **可证伪陈述**：P3 `stage2_distill_loss = 1 - cos_sim(stage2_recon, stage3_beam_avg)` 在 cyclic c(t) 的 1 个周期内（T_cycle steps 内）出现 ≥ 1 个局部极小值 + ≥ 1 个局部极大值，蒸馏 loss 与 c(t) 的 Pearson ρ ≥ 0.5（@ step5000）。
- **数值测量方法**：每 100 step 同步记录 `distill_loss.detach().item()` + `c(t).detach().item()`，对最后 50 个采样点（step4500–step5000）计算 Pearson 相关系数；打印窗口至少覆盖 cyclic c(t) 完整 2 个周期（≥ T_cycle × 2 步）。
- **PASS 阈值**：|ρ(distill, c)| ≥ 0.5 且符号为正（c 越大 distill loss 越大，因为高曲率下 stage2_recon 与 stage3_beam_avg 几何距离拉伸）。
- **FAIL 触发条件**：|ρ| < 0.3 → distillation term 与 cyclic c(t) 几乎无关，蒸馏未进入 cyclic 通路；机制实质是 stage2 端常驻正则项，不修复传导路径。
- **不可证伪词扫描**：陈述仅含 "≥ 0.5 / ≤ 0.3 / PASS / FAIL" 数值判定，不含 "可能 / 也许 / 视情况而定"。

### DE-2 Stage2 geometric 重构向量与 Stage3 beam top-K 平均 embedding 的 cosine ≥ 0.4

- **可证伪陈述**：在 step5000 时，从 valid 取 200 个 batch 的 items，跑 Stage3 frozen beam=20 检索，取 beam top-20 的 token embedding 平均；与该 batch 的 stage2_reconstruction（end-to-end 经过 encoder + RQ codes + decoder）做 cosine similarity，分布的 mean ≥ 0.4，median ≥ 0.35。
- **数值测量方法**：step5000 时一次性 forward，200 个 batch；记录 `cos_sim(stage2_recon, stage3_beam_avg)` 的 mean/median/p10/p90；同时打印 `cos_sim(stage2_recon, item_emb_parquet_baseline)` 作为对照（应低于 stage3-beam-side，否则说明 stage2 仅学到"对齐 item_emb"，与 Stage3 beam 分布无关）。
- **PASS 阈值**：mean(cos) ≥ 0.4 且 median ≥ 0.35。
- **FAIL 触发条件**：mean(cos) < 0.3 → stage2_recon 与 stage3_beam_avg 几乎正交，蒸馏项未真生效；或 mean(cos_stage2_vs_itememb) > mean(cos_stage2_vs_beam) + 0.1 → distillation 退化为 stage2→stage1 embedding 对齐，不修复 Stage2→Stage3 传导路径。
- **不可证伪词扫描**：阈值/判定全部数值化。

### DE-3 step5000 3-token unique ≥ iter5 同期 22675

- **可证伪陈述**：step5000 时，valid items 的 3-token SID（不含 PAD 列）unique 计数 ≥ 22675（iter5 step5000 同期值）；如果 P3 蒸馏让 stage2 退化到 L0/L1 collapse，unique 会跌破 22000，机制 FAIL。
- **数值测量方法**：step5000 时一次性推理 valid items，统计 `len(set(sids[:, :3]))`。
- **PASS 阈值**：unique ≥ 22675。
- **FAIL 触发条件**：unique < 22000 → stage2 已塌陷；iter6 baseline unique=23221（iter6 step5000）+5.3% 余量，P3 不能跌破 iter5 同期 baseline。
- **不可证伪词扫描**：阈值/判定全部数值化。

---

## 2. 性能阈值假设（PH，Performance Hypothesis）— 必须经 Stage3 完整训练验证

### PH-1 valid→test drift ratio > 0.9

- **可证伪陈述**：Stage3 150 epoch 跑完后，valid_best recall@10 / test_recall@10 > 0.9；iter32 drift ratio = 0.886（FAIL），iter28 baseline drift = 0.985（PASS）；P3 必须不能复制 iter32 模式。
- **数值测量方法**：Stage3 trainer 每 5 epoch 在 test 上评估（已在 iter32 后默认开启），取 valid_best epoch 对应 test_recall@10。
- **PASS 阈值**：valid/test ratio > 0.9。
- **FAIL 触发条件**：ratio ≤ 0.886 → P3 复制了 iter32 valid 过拟合 pattern；或 test_R@10 ≤ iter6 baseline 0.0534 → 蒸馏未传导。
- **硬目标**：test_R@10 > 0.065（项目硬目标）。

### PH-2 distill_weight ≤ 0.1（避免 over-constraint）

- **可证伪陈述**：`distill_weight = 0.1`（硬编码，不可调）— 若 distill_weight > 0.1，stage2 端会过度拟合 stage3 当前参数（stage3 还没收敛就被 stage2 当 ground truth 对齐），导致反向过拟合 test_R@10 反而下降。
- **数值测量方法**：仅记录蒸馏 loss 在 total loss 中的占比：`distill_weight × distill_loss / total_loss`；若该占比 > 30% → distillation 已 dominate stage2 训练，stage2 失去 reconstruction / RQ-VAE 主任务动力。
- **PASS 阈值**：distill_loss 占比 ∈ [5%, 25%]。
- **FAIL 触发条件**：占比 > 30% → over-constraint，stage2 RQ-VAE 码本语义被蒸馏目标覆盖；占比 < 3% → distillation 实质为 silent no-op（与 v317/v325/v326 同路径失败）。
- **不可证伪词扫描**：阈值/判定全部数值化。

---

## 3. 如何避免 iter32 valid→test drift 与 Stage3 T5 lock（v321 56 次 lock）

### 3.1 iter32 drift 根因复盘

- iter32 = R36n b (per-layer 异质 c) + R36n f (per-layer 异质 sk_eps)，valid_best 0.06586 (+9.4% vs iter28 0.06020) 但 test_R@10=0.05832 (-1.65%)，drift ratio 0.886 vs baseline 0.985（drift 放大 10x）。
- 根因：Stage 1 端越复杂 → valid 拟合越强 → test 漂移越严重；per-layer 异质机制让 stage2 学到了 valid 集特有分布，不跨 split 泛化。
- v321 R36n f (Angular Orthogonality) 已证 Stage3 T5 SID 表征空间对 Stage 1 端几何变更强 lock（连续 3 个不同机制 v319/v320/v321 都 lock 到 baseline 或 v319 等价 SID 子空间）。

### 3.2 P3 避免 iter32 drift 的硬约束

| 约束 | 数值上限 | 触发条件 | 反向动作 |
|---|---|---|---|
| `distill_weight` | ≤ 0.1（硬编码） | distill loss / total loss > 30% | 立即降至 0.05 |
| Stage3 frozen | 推理路径完全 `torch.no_grad()`，参数不在 stage2 优化器中 | stage3 param 在 autograd graph 中 | 加 `with torch.no_grad()` 包裹 |
| beam top-K 平均 | K=20 而非 top-1 | K < 10 | 强制 K=20，避免 top-1 单点过强约束 |
| beam 缓存 TTL | 5 min 内存缓存（key=item id set） | 每次 forward 重算 | 加 LRU dict + TTL |
| valid→test drift 监控 | Stage3 每 5 epoch test 评估 | test 连续 3 次下降 | 立即 R50 + 停止 stage2 训练 |

### 3.3 P3 跳出 v321 Stage3 T5 lock 的机理

- v321 lock 根因：Stage 1 端所有几何变更都被 Stage 3 T5 SID 表征空间 argmin 路径"吸收"（Stage 1 端 ckpt Δnorm 8.5–84.88 巨大但 argmin 选中完全相同的 256 code per layer）。
- P3 不在 Stage 1 端做几何变更，而是把 Stage 2 重构目标**显式**对齐到 Stage3 检索分布（cosine similarity loss），让 Stage 2 geometric 输出不再依赖 origin，而是依赖 Stage3 当前 beam 分布。
- 这绕开了 v321 论证的"Stage 1 端变更无法传导"——P3 直接把传导链路显式插入 Stage 2 → Stage 3。
- 对比 iter32：iter32 仍属 Stage 1 端纯几何变更（per-layer 异质 c + sk_eps），未跨 stage；P3 是 v321 memory §"必须跳出的方向"第 3 条"跨 stage 联合机制"的精确实现。

### 3.4 R37 + R36p + R36r 三态预检

| 检查项 | 触发条件 | P3 当前预估 | 失败动作 |
|---|---|---|---|
| R37 SID 字节级 lock | stage2 sids_for_hgrec.npy MD5 = baseline MD5 | **不命中**（P3 显式改 stage2 reconstruction 目标，argmin 路径必变） | 立即 R50 + 检查 distillation 是否被 .detach() 截断 |
| R36p util 阈值 | L0/L1/L2 utility < 75% | **不命中**（distill_weight ≤ 0.25% of total loss，不主导 stage2 训练） | 立即降低 distill_weight 至 0.05 |
| R36r silent no-op | distill_loss 连续 100 step = 0 或常数 | **不命中**（DE-1 ρ ≥ 0.5 强制 distill_loss 数值随 c(t) 变化） | 检查 `with torch.no_grad()` 是否过早截断 stage3_beam_avg |

---

## 4. 不可证伪词扫描自检

- DE-1 / DE-2 / DE-3 全部使用 `≥ / ≤ / < / > / =` 数值阈值 + `PASS / FAIL` 二元判定；
- PH-1 / PH-2 全部使用 ratio / weight / 占比数值判定；
- 全文不含 "可能提升" / "也许" / "或许" / "视情况而定" / "理论上" 之类不可证伪词；
- "理论上" 一词出现在第 0 节与 3.3 节均为描述机理路径而非预测（后者附数值阈值）。

---

## 5. 假设验证边界

1. 本假设只对 P3 Stage3-aware Distillation Term 给出可证伪条件；不覆盖 P1（BACKUP）或 P2（已 (e) FAIL 淘汰）。
2. P3 实施需保持 `[256,256,256,1]` codebook 容量、SID 长度、item 顺序与 Stage3 输入协议不变。
3. 本文件不修改 Python 代码，不触发训练或 gradient check。
4. DE-1/DE-2/DE-3 必须在 Stage2 step ≤ 5000 内打印数值证据；PH-1/PH-2 需 Stage3 150 epoch 跑完后验证。
5. 若 DE-1 / DE-2 / DE-3 任意一项 FAIL，stage2 训练必须立即 R50 + 回退到 iter6 baseline，不进入 Stage3 训练。

---

## 6. 关键证据来源

1. iter6 审计 commit `e5b3c2a2631e013e7b6832124c3eafc1045e0dd5`（含 gate_decision_iter6.md / failure_analysis_iter6.md / failure_attribution_iter6.md）
2. iter7 candidate pool：`/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/lit_search_iter7.md`
3. iter7 dominant bottleneck + forbidden directions：`/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/iteration_bridge.md`
4. Agent B 唯一推荐：`/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter7/logs/direction_decision_iter7.md`
5. v321 memory：`/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/v321-r37-fail-sid-locks-baseline.md`（Stage 3 T5 SID 锁层面）
6. iter32 memory：`/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter32-r37-fail-valid-test-drift.md`（valid→test drift 0.886 vs baseline 0.985）

只读取，未修改 Python；未触发训练。