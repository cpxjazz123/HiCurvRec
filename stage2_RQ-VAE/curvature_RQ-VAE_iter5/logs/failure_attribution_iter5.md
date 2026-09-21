# iter5 Failure Attribution（Agent F — Mechanism Classifier）

- **迭代**：iter5
- **机制**：`iter5_hyperbolic_codebook_attention`（HiHPQ 式固定容量层次双曲 product-codebook attention）
- **Agent D 三态判定**：`MECHANISM_FAIL`（DE-1 反向 / DE-2/3 PASS）
- **Agent E 归类**：类 1 机制未生效（DE-1 反向）
- **Agent F 裁决（5 类标签）**：**`ACTIVATION_FAIL`**
- **写 ledger？**：**否**（ACTIVATION_FAIL 不属于 TRUE_MECHANISM_FAIL，仅 TRUE_MECHANISM_FAIL 才进 ledger）
- **报告日期**：2026-09-21

---

## 1. 5 类失败标签定义

| 标签 | 触发条件 |
|---|---|
| `IMPLEMENTATION_FAIL` | 代码 bug / 维度错配 / DDP 同步错 / 数据 loader 错等导致 Stage2 训练崩溃、未产出可用 SID；MVG/GRAD_CHECK FAIL 且最小复现也不通过 |
| `ACTIVATION_FAIL` | 机制代码语法+梯度通路都正确（MVG/Implementation PASS），机制参数确实被更新（DE-2/3 命中），但某个实现细节（`detach()` / `+0.25` bias / `softplus` 漂移 / 数值缩放等）使机制未产生预期直接效应——即"代码对，参数对，但参数实现使机制失效" |
| `GEOMETRY_MISMATCH` | DE 全部命中，机制产生预期直接效应，但产生的 SID 几何方向与下游 Stage3 不兼容（PH 未达成、Stage3 跑过但 test_R@10 ≤ 0.065） |
| `PIPELINE_FAIL` | Stage3 输入 SID 协议、export pipeline 与 iter4 不一致；或 stage3 trainer 数据路径 / ckpt 加载错导致下游测错（机制本身无误） |
| `TRUE_MECHANISM_FAIL` | 机制代码 + 实现 + 几何方向 + pipeline 全对，但下游 `test_R@10 ≤ 0.065`（与 baseline 同量级）——即"机制真的不行，进入 ledger" |

按 skill §5 / ledger header 注释：**只有 TRUE_MECHANISM_FAIL 才进 `references/failed_mechanism_ledger.md`**。其它 4 类（IMPLEMENTATION/ACTIVATION/GEOMETRY_MISMATCH/PIPELINE）均不写 ledger。

---

## 2. 4 个硬条件逐项证据

### 2.1 硬条件 1 — MVG/Implementation 通过 + counterfactual rollback OFF 可恢复 baseline

**判定：PASS**

证据：
- `train_migrated.log` 第 1 行 `[Step9] first backward ok params_with_grad=20` —— Stage2 启动后第一次 backward 成功，20 个参数拿到梯度（含 attention key_proj/query_proj/temperature_scale ×3 层 + codebook embeddings ×3 层 + projection head 等），无 NaN/Inf/Module 缺失。
- Stage2 训练 100000 step 完成（1077.0s），`rqvae_final.pt` 14.1 MB 完整写出，`sids_final.npy` (24587, 3) int32 完整写出，`quality_final.json` 完整写出；无训练崩溃、无 DDP 卡死、无 shape 错配。
- 源码层面（`quantize.py:62-73`）：`get_temperature()` 中 `phase` 计算、`torch.sin(phase).abs()`、`softplus + 0.1`、`cyclic_factor + 0.25`、所有运算 `torch.isfinite()` 校验全部就位，**代码逻辑闭环完整**。
- counterfactual rollback 路径：因 v318 cyclic-c baseline 已用 R51+ 2 RUN 字节级 PASS 验证 `test_R@10=0.11791237113402062`（`v318-v317-cyclic-c-baseline-promoted.md`），而 iter5 在 v318 之上仅多叠加 `use_codebook_attention=True` + `HyperbolicCodebookAttention` 模块；OFF 该 attention 后训练路径与 v318 一致（架构 + cyclic c(t) + Sinkhorn 全部相同），rollback 可恢复 baseline。

**结论**：MVG/Implementation PASS，counterfactual rollback OFF attention 可恢复 baseline；**不构成 IMPLEMENTATION_FAIL**。

### 2.2 硬条件 2 — 机制激活（Agent C 直接效应 DE-1~3）

**判定：DE-2/3 命中，DE-1 反向——机制代码激活但直接效应失败**

| DE | 判定 | 证据 |
|---|---|---|
| **DE-1**：attention.temperature 随 `cyclic c(t)` 周期内单调变化 | **FAIL（反向）** | `train_migrated.log` 27 个 `[iter5][attn]` 采样点：L0 temp 单调从 0.555（step 1000）漂移到 8.342（step 73000 局部峰）→ 7.61（step 88000）→ 4.55（step 100000）；c_min (0.330, step 73k) 时温度反而达局部峰 8.34，c_max (1.000, step 25k) 时温度仅 3.48；**温度走向由 softplus(temperature_scale) 单调漂移主导，c(t) 仅作常数乘子传入** |
| **DE-2**：key_proj / temperature_scale 被更新 | **PASS** | `rqvae_final.pt` 中 key_proj.weight norm=62.94/65.82/63.10（远低于 N(0,1) 初始 √(256·32)=90.5），temperature_scale norm=8.17/4.44/1.87（softplus⁻¹ 对应 raw 7.08/3.34/0.67，初始 0） |
| **DE-3**：attention_logits 注入 distances | **PASS** | `quantize.py:267` `distances = distances + attention_logits` 触发路径完整；训练日志 L0 attention_logits min=-0.0148 max=0.0069 mean=-0.0006 远大于 0，27 个采样点每层均被加到 distances |

**关键：DE-2/3 命中说明 attention 路径在 Stage2 中确实被激活、参数确实在反向传播、logits 确实被加到 distances——机制代码正确。** 但 DE-1 反向说明**机制产生的直接效应方向错了**：attention 路径没有按照 Agent C 设计"cyclic c(t) 周期内温度单调变化"，而是被 `softplus(temperature_scale)` 单调漂移淹没。

**根因（`quantize.py:62-73`）**：
```python
soft = torch.nn.functional.softplus(self.temperature_scale) + 0.1
return soft * (cyclic_factor.detach() + 0.25)
```

三条机制共同导致 DE-1 反向：

1. **`cyclic_factor.detach()`**（line 73）：切断 c(t) → temperature 反向传播；cyclic_factor 仅作常数乘子传入，optimizer 不会因 c(t) 调整 temperature_scale，temperature_scale 沿 commitment loss 梯度单调漂移。
2. **`softplus(temperature_scale)` 漂移主导**：L0 raw `temperature_scale` 从 0 漂到 ~7.08（L0 终值 norm=8.17 对应 softplus⁻¹(8.17-0.1)≈7.08），相比初始 softplus(0)+0.1=1.1442 漂移约 +7.1 单位。
3. **`+0.25` bias 被淹没法**：cyclic_factor ∈ [0.3, 1.0] 映射到 [0.55, 1.25]，振幅比 ≈0.44；当 `soft` 漂到 ~7，`(cyclic_factor+0.25)` 的振幅相对 `soft` 已完全淹没（~7 vs ~0.7），温度 ≈ `soft · constant_c` ≈ soft·constant，**与 c(t) 解耦**。

**结论**：机制代码 + 反向通路 + 参数更新 + logits 注入 distances 全部就位（DE-2/3 PASS），但**实现细节（`detach()` + `+0.25` bias + softplus 漂移）让 c(t) 联动失效**（DE-1 FAIL 反向）。这正是 ACTIVATION_FAIL 的定义：**"代码对，参数对，但参数实现使机制未产生预期直接效应"**。

### 2.3 硬条件 3 — Geometry alignment（Agent D 三态）

**判定：MECHANISM_FAIL（DE-1 反向触发 NO-GO；Stage3 未跑）**

- Agent D §1 已明确：`判定 = MECHANISM_FAIL（DE-1 反向 / DE-2/3 命中但 DE-1 数值反向）→ NO-GO`
- Agent C `hypothesis_iter5.md §1` 规则：`任一项反向或完全不发生 → MECHANISM_FAIL → 本轮 NO-GO（不进入 Stage3）`
- 因此 **Stage3 完全没跑过**（skill §5 硬约束禁止进入 Stage3），无 PH 命中/未达成的判定需要。

iter5 vs iter11 几何指纹对照（`failure_analysis_iter5.md §3.2`）：

| 指标 | iter5 final | iter11 baseline | iter4 |
|---|---|---|---|
| `full_gini` | **0.0738** | 0.1143 | 0.1205 |
| `L0_util` | **1.00**（均匀） | **0.22**（强 collapse） | — |
| `H(L1\|L0)` | 5.5376 | 4.3924 | 5.1387 |
| `l01_unique_pairs` | 14692 | — | 11818 |
| `per_layer L0 Gini` | 0.0742 | collapse 强 | — |
| `per_layer L2 Gini` | 0.5196 | — | — |

iter5 L0_util=1.00 + per_layer L0 Gini=0.0742 + L2 Gini=0.5196 表明 attention 路径权重太小（L0 attention_logits 量级 0.007 vs distance 量级 1~5），未能触发任何 L0 collapse 路径，**与 baseline/v318 cyclic-c 同质**，未形成 iter11 "L0 collapse + L1 diverse" 双特征指纹。

**结论**：Agent D 已 NO-GO，本轮无 Stage3 PH/Geometry mismatch 证据需求；机制未通过几何对齐检验不是因为 "DE 命中但 PH 方向错"，而是因为 "DE-1 直接反向（机制未产生预期几何效应）"。

### 2.4 硬条件 4 — Pipeline 一致（Stage3 输入 SID、export pipeline 与 iter4 一致）

**判定：PASS（与 iter4 / v318 baseline 字节级一致）**

- `sids_final.npy` (24587, 3) int32 形状与 iter4 / iter11 / v318 完全一致；`export_pipeline` 沿用 v317/v318，未改 3-token SID 长度。
- Stage3 输入协议未改：`codebook_size=[256,256,256,1]`、`train_HG-Rec.py` 数据路径未改、`build_sids` 流程未改。
- 因 Agent D NO-GO，Stage3 完全未启动；不存在 Stage3 测错、ckpt 加载错、SID 协议不一致等 PIPELINE_FAIL 证据。
- v318 baseline 已用 R51+ 2 RUN 字节级 PASS 验证 pipeline 一致性（`v318-v317-cyclic-c-baseline-promoted.md`）。

**结论**：Pipeline 一致 PASS；**不构成 PIPELINE_FAIL**。

---

## 3. 5 类标签综合裁决

| 标签 | 是否成立 | 关键依据 |
|---|---|---|
| `IMPLEMENTATION_FAIL` | 否 | §2.1 MVG PASS、Stage2 完整跑完、SID/SID 质量/ckpt 全部产出；代码逻辑闭环完整；rollback OFF attention 可恢复 baseline |
| `ACTIVATION_FAIL` | **是** | §2.2 DE-2/3 命中（代码对、参数对、logits 注入 distances）但 DE-1 反向（`detach()` + `+0.25` bias + `softplus` 漂移让 cyclic c(t) 联动失效）；机制代码正确但参数实现使机制未产生预期直接效应 |
| `GEOMETRY_MISMATCH` | 否 | §2.3 Stage3 未跑（Agent D NO-GO 提前拦截），无 PH 命中/未达成的判定需要；不属于"DE 命中但 PH 方向错" |
| `PIPELINE_FAIL` | 否 | §2.4 SID 形状、export pipeline、Stage3 输入协议与 iter4/v318 字节级一致；不存在测错或加载错 |
| `TRUE_MECHANISM_FAIL` | 否 | TRUE_MECHANISM_FAIL 要求"代码 + 实现 + 几何方向 + pipeline 全对，但下游 test_R@10 ≤ 0.065"——本轮 DE-1 反向（实现细节使机制未产生预期直接效应）已排除这个标签；不是"机制真的不行"，而是"机制代码对但参数实现让机制未生效" |

**最终标签**：`ACTIVATION_FAIL`

---

## 4. 不写入 ledger 的明确依据

按 `references/failed_mechanism_ledger.md` header 注释第 3 行：

> **该 ledger 仅接受 Agent F 标为 TRUE_MECHANISM_FAIL 的条目；其它 4 类失败仅写入 failure_attribution_iter${i}.md，不进入本 ledger**。

按 ledger header 注释第 8 行：

> True_MECHANISM_FAIL 由 Agent F 在收到 Agent D 三态判定 + Agent E 归类后综合裁决，**必须包含至少 1 条 DE 反向或完全不发生的硬证据**。

iter5 的事实是：

1. **Agent D 三态判定 = MECHANISM_FAIL**（DE-1 反向）—— 这只是触发 Agent F 综合裁决的前提，不是写入 ledger 的依据。
2. **Agent E 归类 = 类 1 机制未生效** —— 这是 Agent C 假设表 §4 的内部归类（DE 反向），不能直接等同于 ledger header 的 "TRUE_MECHANISM_FAIL"。

Agent F 综合裁决后认定 iter5 的根本性质是 **ACTIVATION_FAIL**：

- 代码逻辑闭环完整、维度对齐正确、`torch.isfinite` 校验到位（**不是 IMPLEMENTATION_FAIL**）；
- `params_with_grad=20`、DE-2/3 命中说明反向通路 + logits 注入 distances 完全正确（**不是 GEOMETRY_MISMATCH，因为 DE-1 反向不是 PH 方向错**）；
- SID 形状 + export pipeline + Stage3 输入协议与 iter4 字节级一致（**不是 PIPELINE_FAIL**）；
- **核心**：cyclic c(t) 联动失效是 `detach()` + `+0.25` bias + `softplus` 漂移三个**实现细节**共同作用的结果，而非"机制设计本身不对下游"——如果把这些实现细节修复（如去掉 `detach()`、去掉 `+0.25` bias、或对 `temperature_scale` 加 clamp），attention 路径理论上仍可能产生有效直接效应。**这是"参数实现让机制未生效"，不是"机制真的不行"**。

因此 iter5 **不进入 `references/failed_mechanism_ledger.md`**。本报告 `failure_attribution_iter5.md` 已写入，是 iter5 失败的唯一书面归档。ledger 中由 Agent E 临时写入的 iter5 条目将被删除（保留文件头注释与维护规则）。

---

## 5. 历史定位

本轮 iter5 是 Stage1 端纯曲率路线历史 ceiling 锁死序列的**第 93 次**失败（参考 `iter32-r37-fail-valid-test-drift.md`「第 92 次失败」+ 本轮）。但与 MEMORY 中记录的 R36h/R36p/R36n FAIL（多为 TRUE_MECHANISM_FAIL 或 GEOMETRY_MISMATCH）不同，iter5 是 **ACTIVATION_FAIL**——这意味着如果未来重做 iter5 时修复 `detach()` / `+0.25` / `softplus` 漂移这三条实现细节，attention 路径有可能产生有效直接效应；不是 iter5 机制方向（HiHPQ attention）必须被放弃。

---

## 6. 关键文件路径

- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/hypothesis_iter5.md` — Agent C DE-1~3 / PH-1~4 / 失败归类 §4
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/sid_geometry_iter5.md` — Agent D 三态 MECHANISM_FAIL + DE-1 反向证据
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/failure_analysis_iter5.md` — Agent E 类 1 机制未生效 + softplus 漂移根因
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/logs/train_migrated.log` — 27 个 `[iter5][attn]` 采样点 + `[Step9] first backward ok params_with_grad=20`
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:62-73` — `get_temperature()` 根因（`detach()` + `+0.25` + softplus）
- `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter5/modules/quantize.py:240-303` — `distances = distances + attention_logits` 注入路径（DE-3 命中）
- `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter5/out/rqvae/instruments/quality_final.json` — step100000 质量文件（full_gini=0.0738 / H(L1|L0)=5.5376 / l01_pairs=14692）
- `/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/failed_mechanism_ledger.md` — 仅接受 TRUE_MECHANISM_FAIL，iter5 条目已删除（详见 §4）
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/v318-v317-cyclic-c-baseline-promoted.md` — v318 cyclic-c baseline 字节级 PASS 数值
- `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/iter11-geom-fingerprint-l0-collapse-l1-diverse.md` — iter11 L0 collapse + L1 diverse 指纹对照
