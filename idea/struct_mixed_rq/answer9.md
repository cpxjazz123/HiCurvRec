# answer9 — 单次采样身份一致、完整报告与旧链退役；规格审查被现有工具缺失阻塞

**状态：CLOSED（样本身份/报告/旧链退役已完成；原生残差参考与真实 RQLayer 组合未实现；规格审查 BLOCKED）**　完成：2026-11-10　对应 `question9.md`

---

## 1. 单次采样与身份一致

新生产者 `probe_q9.run` 只采样一次，**同一份三元组列表直接进入结构损失**（`H._positions` + `S.structure_loss`），不再调用会二次采样的 `structure_term`：

| 项 | q8（旧） | **q9（新）** |
|---|---|---|
| 请求数 | 64 | 64 |
| 接受数 | 45（记录） | **45（实际使用）** |
| `z_struct` 行数 | 99 | **135 = 3×45** |
| 身份一致性 | **不一致**（99 ≠ 135） | 一致 |

行为回归 `tests/test_q9_sampling_identity.py`：

- `consistent(report)` 同时兼容 q8 的 `shapes.z_struct_rows` 旧布局；
- 对 q8 历史 JSON 断言"不一致"→ 检测器具备抓住该缺陷的能力（实测通过，输出 `historical inconsistency detected`）；
- 对 q9 的新运行断言 `z_struct_rows == 3×triples_accepted`、`requested ≥ accepted`、两条同图合成偏差为 0。

---

## 2. 完整报告（`q9/run-q9/factors.json`）

同一批 64 商品、接受 45 个三元组、旧 checkpoint、`s=0.04444597661495209`、`λ=9.226019983543665e-07`、mode=train、`behaviour_on=True`、`geometry_on=True`。

**损失**：recon `1.0601e-04`、quant `1.6660e-04`、λ·structure `3.96e-06`、total `2.7657e-04`。

**潜变量 E/H（按各自来源）**：

| 损失 | `z_base`（64 行） | `z_struct`（135 行） |
|---|---|---|
| recon | E `2.4668e-04` / H `1.9557e-04` | **None**（无依赖） |
| quant | E `8.1057e-05` / H `8.3537e-05` | **None** |
| structure | **None** | E `3.8047e-05` / H `4.2535e-05` |
| total | E `2.2550e-04` / H `1.7657e-04` | E `3.8047e-05` / H `4.2535e-05` |

交叉分量的 `None` 是**图依赖缺失**，与采集失败（会抛错或返回带原因的 `NOT_MEASURED`）分开记账。

**同图合成**：`∇z_base L_total = ∇z_base L_base`、`∇z_struct L_total = λ∇z_struct L_struct`，最大偏差均为 **0.0**。

**参数块状态**（每块 `none/zero/nonzero`）：

| 损失 | encoder | C1/C2/C3 | decoder |
|---|---|---|---|
| recon | 非零 8 | 非零 1 | **非零 8** |
| structure | 非零 8 | 非零 1 | **none 8** |

checkpoint 加载经**实际消费路径**（`probe_q9.run` 内 `strict=False` + missing/unexpected 硬失败），不兼容状态会 `SystemExit`。

---

## 3. 旧调用链退役

- 删除 `tests/test_q7_latent_capture.py`（它导入 `probe_q6_factors`，后者在模块顶层测量并写 q7）；
- q5–q8 的历史产物保持只读：`q7/factors.json` 的 SHA 仍为 `bfb8253c…`（本轮未写入）；
- 当前活动报告生成路径只有 `probe_q9.py`（导入无副作用，`run()` 才写盘）；
- 测试输出使用 `tempfile`，不写历史证据根。

边界：`probe_q6_factors.py` 文件仍留在树中作为历史，但已无测试/调用者引用。

---

## 4. 数学验证状态

| 项 | 状态 |
|---|---|
| 累计回读独立前向 + 1/2/3 层完整 Jacobian + 两侧 gradcheck | **DONE**（q8 已有，保留） |
| 原生残差更新的独立连续参考 | **NOT_IMPLEMENTED** |
| 实际 `RQLayer` 固定选码的 H 码字梯度与 encoder 单位 STE 组合 | **NOT_IMPLEMENTED**（现有固定选码测试只读 E 坐标） |

离散选择/STE 仍在排除面内，不能用硬前向有限差分冒充代理梯度。

---

## 5. 规格审查：实际阻塞

`question8.md` §4 给出的候选（三层等权平均、`s` 沿用末层初始化表面、`λ` 对完整前缀目标一次块预算）已 DOCUMENTED，但**无法沿现有研究 DAG 推进**：

- 用户已明确删除 DAG 与全部子 agent 派发链路（`2f07400c` 删除 `core/dag.py`、`workflows/pipeline.py` 等；`bdf93104` 删除 rounds 十个派发模块与验证作者），
- 因此不存在可用的 `research_start/resume`、公式 Agent、Math Review 或差分/冻结工具，**没有 run_id 或工具回执可以产生**；
- 这不是"未尝试"，而是入口已被移除；按 `question9` §5.2 记为 **BLOCKED（工具缺失）**，不手工把 DRAFT 改成 FROZEN，也不另建调度器。

前缀实现因此仍为 **NOT_IMPLEMENTED**。

---

## 6. 逐层验收表

| 项 | 状态 | 证据 |
|---|---|---|
| 实际样本与身份一致 | **DONE** | 请求 64 / 接受 45 / 行数 135；回归含 q8 反例检测 |
| 真实报告完整 | **DONE** | 四项损失、潜变量 E/H、状态、合成偏差、身份全在 JSON |
| 旧链不再写历史证据 | **DONE** | 旧测试退役；q7 SHA 未变；活动路径仅 q9 |
| 连续回读数学 | **DONE** | q8 的 Jacobian/gradcheck 保留 |
| 原生残差独立参考 | **NOT_IMPLEMENTED** | 见 §4 |
| 真实 RQLayer H+STE 组合 | **NOT_IMPLEMENTED** | 见 §4 |
| 规格审查/冻结 | **BLOCKED** | DAG 与审查工具已被 owner 删除，无回执可产 |
| 前缀实现 | **NOT_IMPLEMENTED** | 待冻结 |
| 框架运行绑定 | **NOT_RUN** | `LOG_DIR` 仍在源码树 |
| 机器准入 / 正式训练 / 收益 | **NOT_RUN** | 旧 `0.05790490781524748` 不继承 |

测试：`files pass=22 fail=0`（`q9/tests.log`）。

---

## 7. 身份与产物

| 项 | 路径 | SHA-256（前 20 位） |
|---|---|---|
| 生产者 | `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `10b5fef4e14ed130914c` |
| 身份回归 | `tests/test_q9_sampling_identity.py` | `4d3a78276c3140c7fc69` |
| 新报告 | `results/research/struct_mixed_rq/q9/run-q9/factors.json` | `bdf41e9b0a57243afdd0` |
| 测试日志 | `results/research/struct_mixed_rq/q9/tests.log` | `6a0dda015132256176ab` |
| q7 历史 JSON（未变） | `q7/factors.json` | `bfb8253ca690f66d359c` |

---

## 8. 边界

本问不派发正式训练、不覆盖原实验与 R2 编号 57；修复版与前缀版收益均为 `NOT_RUN`；机制未完整实现前不新建 idea。
