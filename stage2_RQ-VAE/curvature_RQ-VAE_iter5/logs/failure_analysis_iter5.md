# iter5 Failure Analysis（Agent E）

- **迭代**：iter5 — 机制 `iter5_hyperbolic_codebook_attention`（HiHPQ 式固定容量层次双曲 product-codebook attention）
- **裁决依据**：`logs/sid_geometry_iter5.md`（Agent D 三态判定）+ `logs/hypothesis_iter5.md` DE-1~3 + PH-1~4
- **判定**：**类 1 机制未生效**（DE-1 反向）→ 本轮 NO-GO（不进入 Stage3）
- **报告日期**：2026-09-21

---

## 1. 三类失败的定义（与 Agent C §4 对齐）

按 `hypothesis_iter5.md §4` 失败归类三态：

| 类 | 触发条件 | 本轮 |
|---|---|---|
| **类 1 机制未生效** | DE-1~3 任一项反向或完全不发生 | **命中**（DE-1 反向） |
| 类 2 方向错 | DE 全命中但 PH 未达成 | — |
| 类 3 downstream 不吃 | DE + PH 全命中但 Stage3 `test_R@10 ≤ 0.065` | — |

DE-2（key_proj / temperature_scale 被更新）与 DE-3（attention_logits 注入 distances）已命中，但 DE-1 反向触发 Agent C 规则「任一项反向或完全不发生 → MECHANISM_FAIL → 本轮 NO-GO」。

---

## 2. 事实 1：机制未生效（DE-1 反向的逐项证据）

### 2.1 DE-1 反向：attention.temperature 不与 cyclic c(t) 联动

27 个 `[iter5][attn]` 采样点（`train_migrated.log`）显示 `attention.temperature` 数值由 `softplus(temperature_scale)` 单调漂移主导，而非 `|sin(πt/T)|` 形状。

| step | c (cyclic) | L0 temp | L1 temp | L2 temp | 期望单调方向 |
|---|---|---|---|---|---|
| 1000 | 0.325 | 0.5550 | 0.5195 | 0.4927 | temp ≈ `soft·(0.325+0.25)` ≈ `soft·0.575` |
| 7000 | 0.611 | 1.2501 | 0.8944 | 0.8458 | temp ≈ `soft·0.861`（c↑ 应伴随 temp↑） |
| 25000 | 1.000（c_max） | 3.4807 | 2.1129 | 1.5036 | c_max 时温度应局部峰；实测 L0=3.48 中段 |
| 35000 | 0.851 | 3.9011 | 2.4047 | 1.4940 | temp 仍在涨（+0.42） |
| 55000 | 0.540 | 3.9193 | 2.3474 | 1.2655 | temp 涨到 3.92（c↓ 但 temp↑） |
| 73000 | 0.330（≈c_min） | **8.342** | — | — | **c_min 时温度反而达局部峰 8.34** |
| 88000 | 0.769 | 7.6104 | 4.1658 | 1.8576 | c↑ 但 temp 仍 7.61（与 73k 相当） |
| 100000 | 0.710 | 4.5474 | 2.5032 | 1.1625 | temp 回落（与 c↑ 不一致） |

**Pearson r(L0 temp, c) = +0.440**（形式上正相关，但分布极不规则）：
- c_min (0.330, step 73k) → temp 8.34（局部峰）；c_max (1.000, step 25k) → temp 3.48（中等）；
- 单调方向与 `|sin(πt/T)|` 形状不一致——温度主要由 `softplus(temperature_scale)` 漂移决定，c(t) 仅作常数乘子。

### 2.2 DE-1 反向的根因（`quantize.py:62-73`）

源码（`modules/quantize.py`）：

```python
soft = torch.nn.functional.softplus(self.temperature_scale) + 0.1
return soft * (cyclic_factor.detach() + 0.25)
```

两个机制共同导致 cyclic c(t) 联动失效：

1. **`cyclic_factor.detach()`**：切断 c(t) → temperature 的反向传播，cyclic_factor 仅作常数乘子传入；Step1~100000 整个训练期 optimizer 不会因 c(t) 调整 `temperature_scale`，导致 temperature_scale 在 commitment / reconstruction loss 驱动下沿单调漂移。
2. **`softplus(temperature_scale)` 漂移主导**：L0 temperature_scale 终值 norm=8.17（`softplus⁻¹(8.17-0.1)≈7.08`），相比初始 `nn.Parameter(torch.tensor(0.0))` 的 softplus(0)+0.1=1.1442，已漂移约 +7.1 单位（L0）。
3. **`+0.25` bias 被淹没法**：cyclic_factor ∈ [0.3, 1.0] 通过 `+0.25` 映射到 [0.55, 1.25]，振幅比 0.55/1.25 ≈ 0.44（最大/最小比 ≈2.27）。当 `soft` 漂到 ~7，`(cyclic_factor+0.25)` 的振幅比相对 `soft` 已完全淹没，温度走向由 softplus 单调漂移决定。
4. **`detach()` 进一步固定 cyclic_factor**：c(t) 数值随 step 变化是 torch 数值，但 `.detach()` 后不参与反向，导致 optimizer 完全看不到 c(t) 对 loss 的影响——temperature_scale 不会因 c(t) 调整。

这三条共同将"cyclic 联动"降级为"softplus 单调漂移"。

### 2.3 DE-2 / DE-3 命中但权重不足

- **DE-2 PASS**：`rqvae_final.pt` 中 `key_proj.weight` norm=62.94/65.82/63.10（远低于默认 N(0,1) 初始化期望 √(256·32)=90.5，梯度确曾流过）；`temperature_scale` 终值 norm=8.17/4.44/1.87（对应 raw 7.08/3.34/0.67，初始 0）。
- **DE-3 PASS**：`distances = distances + attention_logits`（`quantize.py:267`）触发；但 L0/L1 attention_logits 数值（min≈-0.01, max≈0.007）远小于 distances 量级（典型 L0 distance ≈1~5），attention 路径对 assignments 影响极小——`params_with_grad=20` 中仅 ~6 个 attention 相关参数，commitment 路径主导反向。

### 2.4 三段事实陈述（按用户 (a) 任务）

1. **机制未生效**：DE-1 反向（temperature 不随 c(t) 周期变化），根因 `quantize.py:62-73` 的 `softplus(temperature_scale)` 漂到 ~7 后完全淹没 `cyclic_factor ∈[0.3,1.0]` 与 `+0.25` bias，温度走向由 softplus 单调漂移主导而非 c(t)。
2. **SID 几何方向错（不适用）**：DE-2/3 已命中，本轮 SID 几何方向问题不是根因；只在 attention 数值化但权重过小层面间接呈现（L0_util=1.0 而非 iter11-style 0.22）。
3. **SID 几何对但 downstream 不吃（不适用）**：本轮 Stage3 未启动（Agent D NO-GO 提前拦截），未发生 DE+PH 全命中但 Stage3 ≤ 0.065 的第三类失败。

按 Agent C §4 规则 + Agent D 三态表，本轮**唯一**归类为**类 1：机制未生效**。

---

## 3. 事实 2：与 iter11 baseline / iter4 的几何对照

### 3.1 描述性指标对照（stage2 step=100000）

| 指标 | iter5 final | iter4 (step40000) | iter11 baseline | TIGER/curvature baseline |
|---|---|---|---|---|
| `full_gini` | **0.0738** | 0.1205 | 0.1143 | 0.0672 |
| `per_layer_gini[L0]` | **0.0742** | — | L0 collapse 强（geom_L0_util=0.22） | 0.3598 |
| `per_layer_gini[L1]` | 0.2769 | — | — | 0.2467 |
| `per_layer_gini[L2]` | 0.5196 | — | — | 0.1656 |
| `n_unique_full / n_items` | 22675 / 24587 (92.2%) | 21361/24587 (86.9%) | — | baseline l01_pairs=13340 |
| `l01_unique_pairs` | **14692** | 11818 | — | 13340 |
| `H(L1|L0)` | 5.5376 | 5.1387 | **4.3924**（iter11 极低） | 5.6116 |
| `L0_util`（unique/256）| **1.0000** | — | **0.2227** | 1.0000 |
| `L0_max_freq_ratio` | ≈0.0040 | — | **0.0305**（iter11 collapse） | ≈0.004 |

### 3.2 Δ 计算（按用户 (b) 任务）

| 指标 | iter5 − iter11 baseline | iter5 − iter4 | 解读 |
|---|---|---|---|
| **Δfull_gini** | 0.0738 − 0.1143 = **−0.0405**（−35.4%） | 0.0738 − 0.1205 = **−0.0467**（−38.8%） | iter5 更均匀（full_gini 低 = 各 SID 长度段出现频率更平均 = 均匀利用） |
| **ΔH(L1\|L0)** | 5.5376 − 4.3924 = **+1.1452**（+26.1%） | 5.5376 − 5.1387 = **+0.3989**（+7.76%） | iter5 条件熵更高（更接近 baseline 5.6116，远高于 iter11 collapse 路线） |
| **Δcollision rate** | (24587−22675)/24587 − iter11(估) ≈ 7.78% − 7.8%(iter11 历史估) = 接近持平 | 7.78% − (24587−21361)/24587 = 7.78% − 13.13% = **−5.35pp**（iter5 更低） | iter5 collision rate 显著优于 iter4 |
| **Δoracle**（predictor 估算）| 用 `geom_L0_util + geom_h_l1_given_l0_mean` 回归（`test_R@10 ≈ 0.0237 + 0.0013·L0_util + 0.0082·H(L1\|L0)`）：iter11 (L0_util=0.22, H=4.39) 代入 → 0.0237 + 0.00029 + 0.0360 ≈ 0.0600（实测 0.0602）；iter5 (L0_util=1.00, H=5.54) 代入 → 0.0237 + 0.0013 + 0.0454 = **0.0704**（数学预测）；iter4 (L0_util≈1.0, H=5.14) → 0.0700；baseline → 0.0705 | — | 10 样本 R² < 0，预测公式仅作 ranking 用；iter5 预测绝对值高于 iter11 但仅反映"iter5 的几何指纹已偏离 iter11 的 collapse 路线" |
| **Δtest_R@10**（估算，未跑 Stage3）| iter11=0.06017；iter4=0.0540；iter5 估算：按 L1 sweep fingerprint summary（17 iter）`geom_L0_util` LOO Spearman ρ=−0.69 推断 L0_util=1.00 → test_R@10 ≈ baseline 段（iter4 0.0540 ± 0.005）；但 iter5 collision 更优（7.78% vs iter4 13.13%）+ l01_pairs 更高（14692 vs 11818）有正向贡献，**综合估算 test_R@10 ∈ [0.054, 0.060]**（仍 ≤ 0.065 硬目标，与 iter4/iter11 同量级，绝非突破） | — | 未跑 Stage3；Agent D NO-GO 提前拦截，仅做几何估算 |

**关键观察**：iter5 L0_util = 1.00（HIGH），H(L1|L0) = 5.5376（HIGH），不形成 iter11 的"L0 collapse + L1 diverse"双特征指纹（`iter11-geom-fingerprint-l0-collapse-l1-diverse.md`）；Stage1 端纯曲率 ceiling 锁死（MEMORY 多条 R36h / R36p / R36n FAIL 锁定），iter5 attention 路径权重太小，未触发新的 ceiling 突破路径。

---

## 4. 失败 fingerprint（具体引用 + 对照 iter11）

### 4.1 失败 fingerprint 摘要

| 维度 | 数值 / 引用 |
|---|---|
| **机制名** | `iter5_hyperbolic_codebook_attention`（HiHPQ 式固定容量层次双曲 product-codebook attention） |
| **触发类别** | 类 1 机制未生效（DE-1 反向） |
| **根因 1（量化路径）** | `quantize.py:62-73` `soft = softplus(temperature_scale) + 0.1`; `return soft * (cyclic_factor.detach() + 0.25)` |
| **根因 2（数值漂移）** | L0 `temperature_scale` softplus 值由 1.14（initial）漂到 8.17（final），相对 `cyclic_factor+0.25 ∈ [0.55, 1.25]` 振幅比 0.44 完全淹没 |
| **根因 3（梯度截断）** | `cyclic_factor.detach()` 阻断 c(t) → temperature 反向传播，temperature_scale 在 commitment loss 驱动下沿单调漂移而非 c(t) 形状 |
| **直接效应失败** | DE-1 反向：c_min (0.330, step 73k) → temp 8.34（局部峰）；c_max (1.000, step 25k) → temp 3.48（中等）；L0 temp 与 c 的 Pearson r=+0.440 形式正相关但分布不规则 |
| **残留命中** | DE-2（key_proj/temperature_scale 被更新，norm 偏离初始）；DE-3（attention_logits 注入 distances） |
| **DE-3 数值化但权重小** | L0 attention_logits min≈-0.01, max≈0.007 远小于 distances 量级（典型 L0 distance ≈1~5），attention 梯度对 assignments 影响极小 |
| **Step9 params_with_grad** | 20 个参数中仅 ~6 个 attention 相关（key_proj/query_proj/temperature_scale ×3 层） |
| **L0_util** | 1.00（与 baseline 同质，未触发 collapse） |
| **H(L1\|L0)** | 5.5376（与 baseline 5.6116 接近，远高于 iter11 4.3924） |
| **l01_pairs** | 14692（比 iter4 11818 高 +24%，接近 iter40 14257） |
| **Stage3 状态** | 未启动（Agent D NO-GO 提前拦截） |
| **test_R@10 估算** | ∈ [0.054, 0.060]（与 iter4/iter11 同量级，未跑 Stage3） |

### 4.2 与 iter11 baseline 的几何差距

| 维度 | iter11 baseline（成功） | iter5（失败） | 差距 |
|---|---|---|---|
| `L0_util` | **0.22**（强 collapse） | **1.00**（均匀） | **iter5 高 +0.78**——iter11 通过 L0 collapse 路径把"高频 L0 token + 低频 L1 离散"作为下游可识别结构；iter5 走 baseline 同质路径 |
| `H(L1\|L0)` | **4.3924** | 5.5376 | iter5 高 +1.145（信息量大但未形成 L0→L1 强条件结构） |
| `L0_max_freq_ratio` | **0.0305**（iter11 collapse） | ≈0.004 | iter5 与 baseline 同质（0.004），未触发 collapse |
| `h_l1_given_l0_mean` | **4.39**（高） | 5.54（更高但均匀） | iter5 L1 更分散（与 H 数字一致），未形成 iter11 的"L0 collapse 集中、L1 diverse"双特征 |
| 预测 test_R@10 | 0.0602 | 0.0704（数学公式预测，仅 ranking）| 公式仅 ranking 用；iter11 实测 0.0602 高于 iter4 0.0540 的根本原因是 L0 collapse 路径与 L1 diverse 的双特征组合，iter5 完全没有 |

**iter5 与 iter11 的几何差距**：iter11 通过 `sk_eps=0.5` 的 Sinkhorn epsilon 路径触发了 L0 collapse（L0_util=0.22），同时 L1 保持 diverse（H(L1|L0)=4.39 但 l01_pairs 充分大）；这种"L0 集中 + L1 diverse"组合给下游 T5 SID 表征空间提供了"低频 L0 + 高频 L1 条件"的层级信号。iter5 attention 路径权重过小，未能在 Stage2 触发任何 L0 collapse 路径（`L0_util=1.00`，per_layer L0 Gini=0.0742 与 baseline 0.3598 一致——但 iter5 是"过度均匀"而非 baseline 的"L0 适中"），最终指纹与 baseline/v318 同质，不构成 iter11 的"L0 collapse + L1 diverse"突破路径。

### 4.3 与 v318 cyclic-c baseline 的对照

- v318_v317_cyclic_c baseline（`v318-v317-cyclic-c-baseline-promoted.md`）使用 `c_min=0.3 / c_max=1.0 / T=50_000` 完整 cyclic 调度 + v317 Midpoint-Only commit，**未叠加任何 attention 路径**，已能在 Stage3 达到 `test_R@10=0.11791237113402062`（R51+ 2 RUN 字节级 PASS）。
- iter5 在 v318 cyclic-c 基础上叠加 `HyperbolicCodebookAttention`，但 attention 路径在 `temperature_scale` 漂移主导下实际退化为"常数乘子 × 漂移 softplus"，既未改变 cyclic c(t) 实际行为，也未引入新的几何突破；attention_logits 数值化但权重太小，对 assignments 影响极小（L0 attention_logits 量级 0.007 vs distance 量级 1~5）。
- 即便 cyclic 调度保留完整有效（v318 已证 PASS），iter5 额外的 attention 路径没有正向贡献也没有显著负向贡献——纯几何 fingerprint 与 baseline 接近（full_gini 0.0738 vs baseline 0.0672 仅微高），但**未触及** v318 已达到的 cyclic c(t) 优势。

---

## 5. 累计失败计数（与 MEMORY 对齐）

本轮 iter5 是 Stage1 端纯曲率路线历史 ceiling 锁死序列的**第 93 次**失败（参考 `iter32-r37-fail-valid-test-drift.md`「第 92 次失败」+ 本轮）。5 个 mechanism（L0 collapse sweep / L1-only sweep / per-layer hetero c / attention with cyclic / 与 stage3 联动）全部 oracle ≈ baseline ceiling（0.0540~0.0602），硬目标 0.065 持续不可达。

---

## 6. 关键文件路径

- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/hypothesis_iter5.md` — Agent C 直接效应清单 DE-1~3
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/sid_geometry_iter5.md` — Agent D 三态判定 + DE-1 反向证据
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/lit_search_iter5.md` — Agent A 候选机制 P1/P2/P3
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/direction_decision_iter5.md` — Agent B 唯一推荐 P2 HiHPQ
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/train_migrated.log` — 27 个 `[iter5][attn]` 采样点 + 10 个 SID gate
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:55-100` — `HyperbolicCodebookAttention.get_temperature`（含 softplus + detach 根因）
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:240-303` — `distances = distances + attention_logits` 注入路径（DE-3 命中）
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/quality_final.json` — step100000 质量文件
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/sids_final.npy` — final SID (24587, 3) int32
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter11-geom-fingerprint-l0-collapse-l1-diverse.md` — iter11 L0 collapse + L1 diverse 指纹
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/v318-v317-cyclic-c-baseline-promoted.md` — v318 cyclic-c baseline 数值
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/l1-sweep-fingerprint-summary.md` — 17 iter LOO predictor（geom_L0_util ρ=−0.69）
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter32-r37-fail-valid-test-drift.md` — 第 92 次失败对照

---

## 7. 结论

iter5 `iter5_hyperbolic_codebook_attention` 归类为**类 1 机制未生效**：

1. **DE-1 反向**（直接效应失败）：`quantize.py:62-73` 中 `softplus(temperature_scale)` 漂移到 ~7（L0）+ `cyclic_factor.detach()` + `+0.25` bias 三个机制共同将"cyclic c(t) 联动"降级为"softplus 单调漂移"，温度走向由 softplus 单调漂移主导而非 c(t)。
2. **DE-2/3 命中但权重不足**：`key_proj` / `temperature_scale` / `attention_logits` 路径均数值化，但 attention_logits 量级（L0 min≈-0.01 max≈0.007）远小于 distances（L0 距离 ~1~5），attention 梯度对 assignments 影响极小。
3. **Stage3 未启动**：Agent D NO-GO 提前拦截，未发生第三类 downstream 不吃失败。
4. **iter5 vs iter11 几何差距**：iter5 L0_util=1.00（均匀）vs iter11 L0_util=0.22（强 collapse），未形成 iter11 的"L0 collapse + L1 diverse"突破指纹。
5. **历史定位**：本轮是 Stage1 端纯曲率路线历史 ceiling 锁死的**第 93 次**失败；5 个 mechanism 全部 oracle ≈ baseline ceiling，0.065 硬目标持续不可达。

按用户 (c)/(d) 任务：本报告 `logs/failure_analysis_iter5.md` 已写入；下一步追加 `references/failed_mechanism_ledger.md`（按 header 注释规则新建）。

当前任务已完成，请做下一个任务的指示。
