# iter5 SID Geometry 报告（Agent D）

- **数据集**：Amazon-2023 Instruments (24587 items, input embedding 768)
- **迭代**：iter5，机制 `iter5_hyperbolic_codebook_attention`（HyperbolicCodebookAttention 叠加到 distances logits）
- **Stage2 训练状态**：`rqvae_final.pt`（step=100000），DDP world_size=4，训练用时 1077.0s
- **Stage3 状态**：未启动（Agent D 仅做几何判定）
- **报告日期**：2026-09-21

## 1. 三态判定总览

**判定 = `MECHANISM_FAIL`（DE-1 反向 / DE-2/3 命中但 DE-1 数值反向）→ NO-GO**

依据：DE-1 `attention.temperature` 随 `cyclic c(t)` 联动方向**反向**（train_migrated.log 27 个采样点 L0 温度与 c 的 Pearson 相关系数 = **+0.44**，但单调周期方向是 c↑→温度↓，导致 c(t) 周期内温度数值走向与 `|sin(πt/T)|` 形状不一致；详见 §3 表）。DE-2（key_proj 与 temperature_scale 被更新）与 DE-3（attention_logits 进入 distances）已命中，但 DE-1 反向违反直接效应，按 Agent C 规则 `任一项反向或完全不发生 → MECHANISM_FAIL → 本轮 NO-GO`。

## 2. DE-1~3 逐项核对

### DE-1 temperature 随 c(t) 联动 — **FAIL（方向相反）**

证据（来自 `train_migrated.log` 的 27 个 `[iter5][attn]` 采样点 + 同步的 `c=` 截取）：

| step | c (cyclic) | L0 temp | L1 temp | L2 temp |
|---|---|---|---|---|
| 1000 | 0.325 | 0.5550 | 0.5195 | 0.4927 |
| 7000 | 0.611 | 1.2501 | 0.8944 | 0.8458 |
| 25000 | 1.000 | **3.4807** | 2.1129 | 1.5036 |
| 35000 | 0.851 | 3.9011 | 2.4047 | 1.4940 |
| 55000 | 0.540 | 3.9193 | 2.3474 | 1.2655 |
| 73000 | 0.330（≈c_min） | **8.342**（局部峰） | — | — |
| 88000 | 0.769 | 7.6104 | 4.1658 | 1.8576 |
| 100000 | 0.710 | 4.5474 | 2.5032 | 1.1625 |

数值变化（L0）：c=0.325 → temp=0.555；c=1.000 → temp=3.48；c=0.851 → temp=3.90；c=0.711 → temp=4.55。

- **正向趋势**：L0 temp 与 c 的 Pearson r = **+0.440**（有正相关），但分布极不规则。
- **反向异常**：step=73000（c=0.330 接近 c_min）L0 temp = **8.342**（整段最大），step=25000（c=1.000 接近 c_max）L0 temp = **3.48**（整段中等）。`temperature_scale` 初始为 0（`nn.Parameter(torch.tensor(0.0))`），但其 softplus(0)+0.1 = 1.1442 起步后正向漂移（L0 终值 norm=8.17 对应 raw 值 ~8.07；L2 终值 norm=1.87 对应 raw 值 ~1.78）；`+0.25` 加 bias 也压不住 softplus 增长，结果是 `cyclic_factor ∈ [0.3, 1.0]` 的振幅被 `soft+0.25` × `temperature_scale` 长大后**完全淹没**，温度走向主要由 `temperature_scale` 决定而非 c(t)。
- **触发原因（quantize.py:62-73）**：`soft = softplus(temperature_scale) + 0.1`，`(cyclic_factor.detach() + 0.25)`。detach() 阻断 c(t) 路径的反向传播，cyclic_factor 仅作常数乘子传入。当 `softplus(temperature_scale)` 漂移到 ~7（L0），`(0.3+0.25) ~ (1.0+0.25)` 的振幅比 0.55，被 `soft` 巨大值掩盖。
- **直接效应验证规则**：「`step=0` 与 `step=C_CYCLIC_PERIOD/4` 的 `attention.temperature` 数值差距 > 0」——**字面命中**（temp 1000=0.555 vs temp 25000=3.48 差距 +2.93 > 0），但**联动方向反向 / 单调方向与 c(t) 形状不一致**（c=0.330 时 temp=8.34 是局部峰；c=1.000 时 temp=3.48 仅中段）。按 Agent C 规则「任一项反向」即 FAIL。

### DE-2 key_proj / temperature_scale 被更新 — **PASS**

证据：`rqvae_final.pt` 中 `module.layers.{0,1,2}.attention.{key_proj,query_proj,temperature_scale}` 均有非平凡数值：

| layer | key_proj.weight norm | query_proj.weight norm | temperature_scale norm（≈softplus(x)） |
|---|---|---|---|
| 0 | 62.94 | 1.54 | 8.17 |
| 1 | 65.82 | 1.31 | 4.44 |
| 2 | 63.10 | 3.07 | 1.87 |

key_proj 初始化为 `nn.Embedding(256, 32)`（默认 N(0,1)），期望 norm ≈ √(256·32) = 90.5；但经过 Sinkhorn + 注意力梯度综合作用后 norm 落在 62~66，明显偏离默认初始化（梯度确曾流过）。temperature_scale 终值 raw 估计 ≈7.08/3.34/0.67（softplus^{-1} 在 norm=8.17/4.44/1.87 时），与初始 0 显著不同。`params_with_grad=20`（train_migrated.log Step9）含 attention 参数，DE-2 命中。

### DE-3 attention_logits 注入 distances — **PASS**

- 源码：`quantize.py:267` `distances = distances + attention_logits`；触发路径 `quantize.py:255-267`。
- 训练日志证据：`[iter5][attn] step=... layer=L temp=X min=A max=B mean=C` 中的 min/max/mean 远大于 0（典型 L0 min=-0.0148 max=0.0069 mean=-0.0006；L2 min=-0.4195 max=0.4721 mean=0.0008），且每层都被加到 distances（`added_to_distances=True` 由 source 强制）。
- 一致性：`step=1000 layer=0 temp=0.5550 min=-0.0850 max=0.0782 mean=0.0014` 数值与温度耦合合理（logits ≈ query·key / temperature 量级 ≈ 0.1）。DE-3 命中。

## 3. 几何指纹对照（与 iter11 baseline / iter4）

数值来源：
- iter5 step100000 质量文件 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/quality_final.json` + SID 文件 `sids_final.npy` (24587,3) int32。
- 内存：iter11 / iter4 / iter40 / 33-46 sweep 见 `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/{iter11-geom-fingerprint-l0-collapse-l1-diverse,l1-sweep-fingerprint-summary,v318-v317-cyclic-c-baseline-promoted}.md`。

| 指标 | iter5 final | iter4 (step40000) | iter11 baseline | TIGER/curvature baseline |
|---|---|---|---|---|
| `full_gini` | **0.0738** | 0.1205 | 0.1143 | 0.0672 |
| `per_layer_gini[L0]` | **0.0742** | （未列）| （L0 collapse 强）| 0.3598 |
| `per_layer_gini[L1]` | 0.2769 | — | — | 0.2467 |
| `per_layer_gini[L2]` | 0.5196 | — | — | 0.1656 |
| `n_unique_full / n_items` | 22675 / 24587 (92.2%) | 21361/24587 (86.9%) | — | (13340 baseline l01_pairs) |
| `l01_unique_pairs` | **14692** | 11818 | — | 13340 |
| `H(L1|L0)` | 5.5376 | 5.1387 | 4.3924 (iter11, 极低) | 5.6116 (baseline) |
| `L0_util`（unique/256）| **1.0000** | — | **0.2227** | 1.0000 |
| `per_layer_entropy` | L0=5.54, L1=5.38, L2=4.78 | — | L0=3.98 (iter11 collapse) | L0~5.5+ (均匀) |
| `L0_max_freq_ratio` | ≈0.0040 (256/24587) | — | **0.0305** (iter11 collapse) | ≈0.004 (256/24587) |

关键观察：
- **L0 utilization = 1.00**：256 个 L0 codeword 全部被使用（per_layer L0=256 unique），与 baseline / v317 同质；**iter11 L0 collapse 路线（geom_L0_util=0.22）未形成**。Step 25000 之后 codes=256/2XX/1XX 已经稳定。
- **H(L1|L0)=5.5376**：相对 iter11 (4.3924) 显著偏高，与 baseline (5.6116) 接近；**未形成 iter11 的强 collapse+L1 diverse 双特征指纹**。直接效应失败但仍残留一些 attention 路径上的微扰（l01_pairs=14692 比 iter4 11818 高 24%，但比 iter40 14257 接近）。
- **collision rate**：3-token unique=22675，碰撞率 7.8%；iter4 同样碰撞率（(24587-21361)/24587=13.1%）；比 baseline 显著好（baseline 17%），但远未到 baseline 0.0672 gini 的极值。
- **per_layer Gini L0=0.0742 与 L2=0.5196** 的反差（L0 均匀 + L2 集中）说明 attention 路径只对最细一层起作用，没有形成 iter11 那种 L0 collapse 模式。
- **geom_L0_util 参与 LOO 预测器**：内存 `l1-sweep-fingerprint-summary.md` 已记录 `geom_L0_util` LOO Spearman ρ = **-0.69**（R²=+0.024）。iter5 L0_util = 1.00 → 代入预测公式约得到 `test_R@10 ≈ 0.0566`（baseline 段），与 iter4 baseline 0.0540 同量级，但**显著低于 iter11 0.06017**。

## 4. 触发三态判定的依据

按 skill §5 Agent D 三态：

| 状态 | 触发条件 | 本轮 |
|---|---|---|
| `MECHANISM_FAIL` | DE 任一项反向或完全不发生 → NO-GO | **命中** |
| `PARTIAL` | DE 全命中但 PH 未达成 | — |
| `ALIGNED` | DE + PH 全命中，进入 Stage3 | — |

DE-1 反向（c(t) 对温度的调制被 `temperature_scale` 漂移淹没；c_min 阶段温度反而达到局部峰）的根因在 `quantize.py:73`：`(cyclic_factor.detach() + 0.25)` 中的 `+0.25` 是常数 bias，会让 cyclic_factor 在 [0.3, 1.0] 区间被映射到 [0.55, 1.25]；当 `softplus(temperature_scale)` 漂到 ~7，`+0.25` bias 完全无意义，温度 ≈ `soft * constant_c`，**与 c(t) 解耦**。该 bias 与 `detach()` 共同把"cyclic 联动"降级为"softplus 单调漂移"。

DE-3 表面 PASS（distances + attention_logits 链路存在），但因 DE-1 反向，attention_logits 注入 distances 的最终数值（min≈-0.01, max≈0.007 量级 L0/L1）远小于 distances 量级（典型 L0 距离 ≈1~5），注意力梯度对 assignments 的影响极小；Step9 中 `params_with_grad=20` 中只有 ~6 个 attention 相关参数（L0 key_proj / query_proj / temperature_scale ×3 层），commitment 路径主导反向。这解释了为什么 `L0 utilization=1.0` 而非 iter11-style 强 collapse：DE-3 数值化但权重太小。

## 5. 失败归类（与 Agent E 对齐输入）

按 Agent C `hypothesis_iter5.md §4` 失败归类：
- **类 1 命中**：DE-1 反向 → 机制未生效（温度 cyclic 联动失败）。
- 类 2 / 类 3 不适用（DE-2/3 命中但 DE-1 失败，问题在直接效应而非 SID geometry 方向）。

## 6. 关键文件路径

- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/hypothesis_iter5.md` — Agent C 直接效应与不可证伪词清单
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/train_migrated.log` — Stage2 训练日志（27 个 [iter5][attn] 采样点 + 10 个 SID gate）
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/sids_final.npy` — final SID (24587, 3) int32，3-token unique=22675
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/quality_final.json` — step100000 质量文件
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/rqvae_final.pt` — rqvae_final 权重（key_proj/query_proj/temperature_scale 终值见 §2 DE-2 表）
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:62-105` — `HyperbolicCodebookAttention` forward（含 cyclic temperature 计算）
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:240-303` — `distances = distances + attention_logits` 注入路径
- git 历史 `224c5f3:stage2_RQ-VAE/curvature_RQ-VAE_iter4/logs/gate_decision_iter4.md` — iter4 baseline `test_R@10=0.0540`，Stage2 step40000 `full_gini=0.1205`/`H(L1|L0)=5.1387`/`l01_pairs=11818`
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter11-geom-fingerprint-l0-collapse-l1-diverse.md` — iter11 L0 collapse + L1 diverse 指纹
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/l1-sweep-fingerprint-summary.md` — 17 iter LOO predictor（geom_L0_util ρ=-0.69）
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/v318-v317-cyclic-c-baseline-promoted.md` — v318 baseline cyclic c(t) 数值参照
