# iter5 → iter6 Iteration Bridge（Agent G — Root-Cause & Gap Analyst）

- **上一轮**：iter5（机制 `iter5_hyperbolic_codebook_attention`，HiHPQ 式固定容量层次双曲 product-codebook attention）
- **本桥日期**：2026-09-21
- **裁决依据**：`logs/hypothesis_iter5.md`（Agent C DE-1~3 / PH-1~4）+ `logs/sid_geometry_iter5.md`（Agent D 三态 + DE-1 反向证据）+ `logs/failure_analysis_iter5.md`（Agent E 类 1 机制未生效 + 几何差分）+ `logs/failure_attribution_iter5.md`（Agent F `ACTIVATION_FAIL`，不写 ledger）
- **基线对照**：
  - iter4 baseline（`git show 224c5f3:stage2_RQ-VAE/curvature_RQ-VAE_iter4/logs/gate_decision_iter4.md`）`test_R@10=0.054022528247358065`，Stage2 step40000 `full_gini=0.1205` / `H(L1|L0)=5.1387` / `l01_pairs=11818`
  - iter11 baseline 指纹（`iter11-geom-fingerprint-l0-collapse-l1-diverse.md`）`L0_util=0.22`（强 collapse）/ `H(L1|L0)=4.3924` / `L0_max_freq_ratio=0.0305` —— Stage1 端纯曲率路线历史 ceiling ≈0.06017
  - TIGER/curvature baseline（`/results/stage3_T5Train/test_final.json`）`test_R@10=0.0591`（Amazon-2023 Instruments）
  - v318 cyclic-c baseline（`v318-v317-cyclic-c-baseline-promoted.md`）`test_R@10=0.11791237113402062`，R51+ 2 RUN 字节级 PASS —— **另一项目历史，对本 iter5 baseline 不直接相关**

---

## 1. 机制是否成功执行（Implementation / Activation / Geometry 三层判定）

| 层 | 判定 | 证据 |
|---|---|---|
| **Implementation** | **PASS** | `train_migrated.log` Step9 `[Step9] first backward ok params_with_grad=20`，Stage2 完整跑完 100000 step（1077.0s），`rqvae_final.pt` (14.1 MB) / `sids_final.npy` (24587,3) int32 / `quality_final.json` 全部产出；无 NaN/Inf/shape 错配；counterfactual OFF attention 可恢复 baseline（v318 cyclic-c 已 R51+ 2 RUN 字节级 PASS 验证） |
| **Activation** | **DE-1 FAIL（反向）** | DE-1 反向三联：① `softplus(temperature_scale)` L0 由 1.1442 漂到 8.17（raw +7.08）；② `cyclic_factor.detach()` 阻断 c(t) → temperature 反向；③ `+0.25` bias 把 cyclic_factor ∈ [0.3, 1.0] 映射到 [0.55, 1.25] 振幅比 0.44，被 soft 巨大值掩盖；DE-2/3 命中（key_proj/temperature_scale 被更新，attention_logits 注入 distances） |
| **Geometry** | **MECHANISM_FAIL → SKIP Stage3** | Agent D 三态 `MECHANISM_FAIL`（DE-1 反向触发 Agent C 规则「任一项反向 → MECHANISM_FAIL → NO-GO 不进 Stage3」）；Agent F 标 `ACTIVATION_FAIL`；Agent E 类 1 机制未生效；iter5 L0_util=1.00 / per_layer L0 Gini=0.0742 与 baseline/v318 同质，未形成 iter11-style L0 collapse 路线（`L0_util=0.22`）；DE-3 attention_logits 量级（L0 min≈-0.0148 max≈0.0069）远小于 distances（典型 L0 距离 1~5），attention 路径对 assignments 影响极小 |

---

## 2. 与最佳 baseline 差分（iter5 final vs iter4 / iter11 / TIGER-curvature baseline）

| 指标 | iter5 final (step100000) | iter4 baseline (step40000) | iter11 baseline | TIGER/curvature baseline | Δ vs iter4 | Δ vs iter11 | Δ vs TIGER baseline |
|---|---|---|---|---|---|---|---|
| `full_gini` | **0.0738** | 0.1205 | 0.1143 | 0.0672 | **−0.0467（−38.8%）** | **−0.0405（−35.4%）** | +0.0066（+9.8%） |
| `per_layer_gini[L0]` | **0.0742** | — | collapse 强 | 0.3598 | — | iter5 与 baseline 接近 0.36，远高于 iter11 collapse | iter5 低 −0.286 |
| `per_layer_gini[L1]` | 0.2769 | — | — | 0.2467 | — | — | +0.030 |
| `per_layer_gini[L2]` | 0.5196 | — | — | 0.1656 | — | — | +0.354 |
| `n_unique_full` | 22675/24587 (92.2%) | 21361/24587 (86.9%) | — | baseline l01_pairs=13340 | **+1314 unique（+6.15pp）** | — | — |
| `l01_unique_pairs` | **14692** | 11818 | — | 13340 | **+2874（+24.3%）** | — | +1352（+10.1%） |
| `H(L1\|L0)` | **5.5376** | 5.1387 | 4.3924 | 5.6116 | **+0.3989（+7.76%）** | **+1.1452（+26.1%）** | −0.074 |
| `L0_util`（unique/256）| **1.0000** | — | **0.2227** | 1.0000 | — | **iter5 高 +0.78**（iter11 collapse 路线完全未形成）| iter5 = baseline（同质）|
| `L0_max_freq_ratio` | ≈0.0040 | — | **0.0305** | ≈0.004 | — | iter5 低 −0.027（baseline 同质）| iter5 = baseline |
| `collision rate` | 7.78% | 13.13% | ~7.8% | ~17% | **−5.35pp** | ≈持平 | iter5 显著优 |

**估算 Δtest_R@10**（未跑 Stage3，按 DE-1 反向 + 17 iter LOO predictor 预测）：

- 17 iter LOO Spearman：`geom_L0_util` ρ=−0.69（`l1-sweep-fingerprint-summary.md`），iter11 (L0_util=0.22) test_R@10=0.06017，iter4 (L0_util≈1.0) test_R@10=0.0540
- **iter5 (L0_util=1.00, H(L1|L0)=5.5376) → test_R@10 ≈ 0.054~0.060**（与 iter4/iter11 同量级）
- **结论：未突破 0.065 硬目标，按 Agent D NO-GO 提前拦截，未进入 Stage3；估算值与 iter4 baseline 0.0540 相近**

---

## 3. Dominant bottleneck（唯一一句，禁止 "可能 A/B/C 都有问题"）

**attention temperature_scale 经 softplus + detach + +0.25 bias 后，cyclic c(t) 联动降级为 softplus 单调漂移，导致 attention logits 量级（L0 min≈-0.0148 max≈0.0069）远小于 distances 量级（典型 L0 距离 1~5），机制未产生预期直接效应 DE-1。**

---

## 4. Forbidden next directions（iter6 必须禁止的方向）

### 4.1 直接重复 iter5 错误实现细节（若重做 iter5 必须先修这三条 + 量级）

| 禁止方向 | 根因 | 为什么禁止 |
|---|---|---|
| 保留 `cyclic_factor.detach()` | 切斷 c(t) → temperature 反向，temperature_scale 在 commitment loss 单调漂移 | iter5 已证 DE-1 反向根因之一 |
| 保留 `softplus(temperature_scale)` 长漂路径（不加 clamp / 不加 warmup）| L0 raw +7.08 漂移完全淹没 cyclic 振幅 | iter5 L0 temp 终值 8.17 vs 初始 1.14 已证漂移主导 |
| 用 `+0.25` bias 隐藏 cyclic 振幅 | cyclic_factor ∈ [0.3,1.0] 映射到 [0.55,1.25] 振幅比 0.44，被软化淹没 | iter5 已证 bias 完全无效（+0.25 vs soft ~7）|
| 重写 attention 时仍让 attention_logits 量级 ≪ distances | L0 logits 0.007 vs distance 1~5，attention 梯度对 assignments 影响极小 | iter5 DE-3 命中但权重不足 |

### 4.2 直接形态禁止

- **禁：再做"再加 attention"**（无论 codebook attention / cross-layer attention / 自注意力变体）——iter5 已证 attention 路径在该机制设计下权重太小，无法产生有效直接效应；Stage2 端纯曲率路线历史 ceiling 锁死 92 次（`iter32-r37-fail-valid-test-drift.md`），第 93 次（iter5）证 attention 方向继续往上叠亦不能突破
- **禁：再做"再加 softplus 长漂路径"**（任何形式的长 softplus 漂移温度 / 嵌入 softplus 调制）——iter5 已证单调漂移会淹没 cyclic 振幅
- **禁：把 `cyclic_factor` 写成 `.detach()`** ——阻断反向传播，必须保留梯度通路让 c(t) 真正驱动 temperature/attention
- **禁：用 `+0.25` 或任何常数 bias 隐藏 cyclic 振幅** ——bias 会被 softplus 漂移完全淹没，反向掩盖机制失效

### 4.3 重要历史背景提示（iter5 裁决与 ledger 关系）

iter5 机制 P2 HiHPQ 式固定容量层次双曲 product-codebook attention **被 Agent F 标 `ACTIVATION_FAIL`，不写 ledger**（仅 `TRUE_MECHANISM_FAIL` 进 `references/failed_mechanism_ledger.md`）。

**若下次重做 iter5 必须先修下列三条实现细节**：
1. 删除 `cyclic_factor.detach()`（保留梯度通路让 c(t) 反向影响 temperature）
2. 删除 `+0.25` bias（让 cyclic_factor 真实振幅直接生效）
3. 限制 `softplus(temperature_scale)` 漂移幅度（如 clamp 到 [0, 1] 或 warmup 系数 ≤ 1）

**且必须保证 attention logits 量级与 distances 同阶**（如 logits × distance_scale 至与 distance 同量级），否则即使代码对仍会 `MECHANISM_FAIL`。

---

## 5. Next iteration objective（若 iter6）

### 5.1 核心约束

- **不改变**：`codebook_size=[256,256,256,1]`、SID 长度（3-token）、Stage3 输入协议、cyclic `c(t)` 调度
- **避免**：再加 attention / 再加 softplus 长漂路径 / 把 cyclic_factor 写成 `.detach()` / 用 `+0.25` bias 隐藏 cyclic 振幅
- **必须**：所选机制的直接效应可在 Stage2 早期打印数值验证（agent 标 DE-1 的实现细节规则）

### 5.2 候选方向（任选一种，每种都必须有早期数值验证清单）

| 方向 | 简介 | 必须的早期 DE-1 数值验证 | 预期 mechanism fingerprint |
|---|---|---|---|
| A. Codebook-level 几何变换（不含 attention）| 在每层 codebook embedding 上做 per-layer 异质几何变换（如 L0↔L1 几何桥接 / 共形映射），跳过 attention 路径的 logits 注入，转而直接修改 codebook Euclidean ↔ Poincaré ↔ Lorentz 表达 | `module.layers.0.codebook.weight` L2 norm 在 step0 vs step=C_CYCLIC_PERIOD/4 差距 > 0 且方向与 c(t) 形状一致 | `L0_util` 与 iter11 baseline (0.22) 接近；`per_layer L0 Gini` ≥ 0.30 |
| B. Sinkhorn 的 c(t)-dependent epsilon（不含 attention）| Sinkhorn sk_eps = `eps_0 / c(t)`，让 L0/L1/L2 各自 epsilon 随 c(t) 联动 | `sinkhorn_eps_L0/L1/L2` 在 step0 vs step=T/4 vs step=T/2 数值走向与 c(t) 形状一致 | `L0_util` 0.3~0.7 之间；`l01_pairs` ≥ iter4 baseline +20% |
| C. Curriculum 路径（不含 attention 与 softplus）| 分阶段 curriculum（如 `c(t)` 前向 + commitment weight 反向，或 temperature 不参与 · 正向 α 衰减）| 每阶段关键超参（curriculum_step）在 step0/25000/50000/75000/100000 数值走向可打印验证 | per_layer utility 全部 ≥ 75%（R36p PASS），collision rate ≤ iter4 (13%) |
| D. Stage 0 几何桥接（R36m 边界线）| 输入端 item_emb 做非平凡几何变换（如 L0/L1/L2 共享的 per-layer Poincaré ↔ Lorentz 嵌入学习）| `module.input_projection.weight` 终值与初始 L2 差距 > 0；forward 输出与 baseline 字节级不一致但 metric 命中 | oracle top10 ≥ iter11 +5% |

**任一方向都必须满足**：直接效应可在 Stage2 step ≤ 5000 通过 `[iter6][de1]` 日志打印数值命中；attention_logits 注入 distances 路径不可使用；`detach()` 在 c(t) 联动路径上不可出现。

---

## 6. 累计失败计数与历史定位

- **iter5 是 Stage1 端纯曲率路线历史 ceiling 锁死的第 93 次失败**（参考 `iter32-r37-fail-valid-test-drift.md` 第 92 次 + 本轮）
- **5 个 mechanism 全部 oracle ≈ baseline ceiling**（0.0540~0.0602），硬目标 0.065 持续不可达：
  1. L0 collapse sweep（iter35-38）—— `sk_eps ∈ {0.30,0.40,0.50,0.60,0.70}` 全 FAIL
  2. L1-only sweep（iter39-42）—— `sk_eps_L1` 4/5 FAIL
  3. per-layer hetero c（R36n b，iter31）—— oracle 锁死
  4. attention with cyclic（iter5）—— DE-1 反向 ACTIVATION_FAIL
  5. valid→test drift（R36n b+f stack，iter32）—— valid 优但 test −1.65%
- **结论**：若 iter6 继续走 Stage1 端纯曲率路线，几乎必然继续锁死 0.065；下一轮必须给"几何指纹突破 iter11 collapse 路线"的真实可测量证据（不可数学公式预测），否则提前拦截不进 Stage3

---

## 7. 关键文件路径

- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/hypothesis_iter5.md` — Agent C 直接效应与不可证伪词清单
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/sid_geometry_iter5.md` — Agent D 三态判定 `MECHANISM_FAIL` + DE-1 反向证据
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/failure_analysis_iter5.md` — Agent E 类 1 机制未生效 + 几何差分
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/failure_attribution_iter5.md` — Agent F `ACTIVATION_FAIL`（不写 ledger）
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/train_migrated.log` — 27 个 `[iter5][attn]` 采样点 + `[Step9] first backward ok params_with_grad=20`
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:62-73` — `get_temperature()` 根因（`detach()` + `+0.25` + softplus）
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:240-303` — `distances = distances + attention_logits` 注入路径（DE-3 命中）
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/quality_final.json` — step100000 质量文件
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/sids_final.npy` — final SID (24587, 3) int32
- `git show 224c5f3:stage2_RQ-VAE/curvature_RQ-VAE_iter4/logs/gate_decision_iter4.md` — iter4 baseline `test_R@10=0.054022528247358065`
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter11-geom-fingerprint-l0-collapse-l1-diverse.md` — iter11 L0 collapse + L1 diverse 指纹
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/v318-v317-cyclic-c-baseline-promoted.md` — v318 cyclic-c baseline 数值参照
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/l1-sweep-fingerprint-summary.md` — 17 iter LOO predictor（geom_L0_util ρ=−0.69）
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter32-r37-fail-valid-test-drift.md` — 第 92 次失败对照
