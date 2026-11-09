# answer8 — 状态进入真实报告、双潜变量同图核对、逐层 H Jacobian；三层前缀规格候选待审

**状态：CLOSED（测量/数学/证据生成完成；规格冻结、前缀实现、准入、训练、收益均未运行）**　完成：2026-11-10　对应 `question8.md`

---

## 1. 真实报告保留梯度三态

`probe_factors.block_status` 此前只被单测调用，真实报告仍把 `None` 填零。本轮改为：

- `probe_q8.report_blocks(blocks, grads)` 为报告生产者，逐块输出 `tensors / none_tensors / zero_tensors / nonzero_tensors / norm`；
- 复用的真实 TRAIN 运行把它写入 `q8/factors.json` 的 `block_states`；
- 回归 `tests/test_q8_report.py`：`report_blocks` 对 `[None, 零, 非零]` 给出 `1/1/1`；导入该模块不写任何文件（子进程在临时目录 import 后目录为空），修复了 q7 测试导入即测量并写盘的问题。

真实报告（`q8/factors.json`，同一批 64 商品、接受 45 三元组、`s=0.04444597661495209`、`λ=9.226019983543665e-07`、旧 checkpoint）：

| 损失 | encoder | C1 | C2 | C3 | decoder |
|---|---|---|---|---|---|
| base | 非零 8 | 非零 1 | 非零 1 | 非零 1 | **非零 8** |
| structure | 非零 8 | 非零 1 | 非零 1 | 非零 1 | **none 8** |
| total | 非零 8 | 非零 1 | 非零 1 | 非零 1 | **非零 8** |

即 decoder 的"结构项未连接"与"连接但为零"在最终 JSON 中已可区分（前者 `none_tensors=8`，后者会记 `zero_tensors`）。

checkpoint 的拒绝路径已通过实际消费函数覆盖：`probe_q8.main` 用 `strict=False` 后检查 `missing/unexpected` 并 `SystemExit`；回归还验证了权重被真实恢复（`torch.equal`）与截断 state 被 `strict=True` 拒绝。

---

## 2. 同图双潜变量：`z_base` 与 `z_struct`

在构建实际损失的同一张图中用 hook 捕获两个张量（不再按行数猜、不再另算一个未参与损失的 `encoded`）：

| 张量 | 形状 | 来源 |
|---|---|---|
| `z_base` | `(64, 32)` | `model.forward(batch)` 的 encoder 输出 |
| `z_struct` | `(99, 32)` | 结构项自身 forward（3 × 33 个三元组） |

注意：请求 45 个三元组时 `structure_term` 实际接受 **33** 个（`3×33=99`），与 q7 的 45 不同；该接受数是本次实际值，已记录，不事后修正。

恒等核对（`q8/factors.json`）：

| 关系 | 最大绝对偏差 |
|---|---|
| `∇_{z_base} L_total` vs `∇_{z_base} L_base` | **0.0** |
| `∇_{z_struct} L_total` vs `λ ∇_{z_struct} L_struct` | **0.0** |

两个梯度张量均存在（`z_base_grad_present=true`、`z_struct_grad_present=true`）。交叉项（如 `∇_{z_base} L_struct`）不拼接成同一批，也不冒充"未连接"。

---

## 3. H 对照：逐层完整 Jacobian 与双侧 gradcheck

`tests/test_q7_h_reference.py` 已从单一 VJP 升级为：

- 对累计层 **1、2、3** 分别用 `torch.autograd.functional.jacobian` 比较完整 Jacobian（候选实现 vs 测试内独立公式），`rtol 1e-8 / atol 1e-10` 全过；
- 对**两侧实现**分别执行 `torch.autograd.gradcheck`（双精度、非共线、球内安全输入），均通过；
- 前向取值对照保留（`rtol 1e-10`）。

边界：这些是连续面与接线的数学证据，不是推荐收益；离散 argmin 与 STE 代理仍由固定选码测试覆盖，未用硬前向有限差分冒充 STE 验证。

---

## 4. 三层前缀规格候选（待 Formula/Math Review，**未冻结**）

按 `question8.md` §4 提出唯一定义，不试版本、不按结果选优：

1. **等权平均，不加层权重**：同一次量化 forward 的 `return_levels` 给出 `q^(1),q^(2),q^(3)`，
   `L_l = E_T[⅓ Σ_{a<b}(d_M²(q_a^(l),q_b^(l))/(s²h_ab²) − 1)²]`，
   `L_prefix = (L_1+L_2+L_3)/3`，`L = L_base + λ L_prefix`；同一三元组、同一 `s`、同一 `λ`。
2. **`s` 沿用末层初始化标定表面**：保留 `calibrate_structure_scale` 的"初始化后末层"位置与现有数值约定（`sqrt(clamp_min(d_M², 1e-8))` 后取 `torch.median`），在任何 optimizer step 前一次冻结。既有 `not distances: return 1.0` 的隐式回退需在审查中与本契约核对，若改须先有行为回归。
3. **`λ` 对完整前缀目标一次块预算**：`λ = min_{b∈{encoder, 全部可训练码本}} ‖∇_b L_base‖ / ‖∇_b L_prefix‖`；decoder 不入最小值；零/非有限按既有 `CALIBRATION_DEGENERATE` 拒绝。
4. **等价性（推导，不是训练记录）**：在相同 s、状态、样本、参数块且无非线性 λ 裁剪下，`λ_mean = 3 λ_sum`，`λ_mean L_mean = λ_sum L_sum`；这不表示末层监督与三层监督等价，也禁止改变目标后沿用旧 λ。

状态：候选 **DOCUMENTED**，独立公式审查与冻结 **NOT_RUN**；前缀实现 **NOT_IMPLEMENTED**。

---

## 5. 逐项验收表

| 项 | 状态 | 证据 / 缺口 |
|---|---|---|
| 真实 JSON 状态修复 | **DONE** | `q8/factors.json` 的 `block_states`；`test_q8_report.py`；导入无副作用 |
| 同图多潜变量诊断 | **DONE** | `z_base (64,32)`、`z_struct (99,32)`，两条恒等偏差 0.0 |
| 无副作用证据生成 | **DONE** | 子进程 import 后临时目录为空；`main()` 才写盘 |
| 逐层 H 数学验证 | **DONE** | 1/2/3 层完整 Jacobian + 双侧 gradcheck |
| 规格候选独立审查/冻结 | **NOT_RUN** | 第 4 节仅为候选 |
| 前缀实现 | **NOT_IMPLEMENTED** | 待冻结 |
| 框架运行绑定 | **NOT_RUN** | `LOG_DIR` 仍在源码树；`ARM_NAME` 已分离 |
| 机器准入（smoke/DDP） | **NOT_RUN** | 未走 MCP |
| 正式训练（修复版） | **NOT_RUN** | 未重训 |
| 下游收益 | **NOT_RUN** | 旧 `0.05790490781524748` 属旧源码图，不继承 |

测试：`files pass=22 fail=0`（`q8/tests.log`，含逐文件 `rc`）。

---

## 6. 身份与产物

| 项 | 路径 | SHA-256（前 20 位） |
|---|---|---|
| 报告脚本 | `probe_q8.py` | `f0363482b30bdcc9d2b0` |
| 状态助手 | `probe_factors.py` | `cf00209435d31ef2e9b0` |
| 报告测试 | `tests/test_q8_report.py` | `7e46a48a56f3dad21670` |
| H 参考测试 | `tests/test_q7_h_reference.py` | `932281bdb6ae8341f5c0` |
| 真实报告 | `q8/factors.json` | `f295c286cacc7817a0aa` |
| 运行日志 | `q8/probe_run.log` | `3ef59113b30cc7dbf6b6` |
| 测试日志 | `q8/tests.log` | `358fad99efc93cc4135a` |

`q5/`、`q6/`、`q7/` 目录保持原样，未被本轮覆盖。

---

## 7. 边界

- 本问不派发正式训练，不覆盖原实验与 R2 编号 57；
- 曲率进入选码/残差/回读的数学正确性、机制完整性与真实下游增益是三个不同结论：本问只完成前两者中的测量与数学部分，第三者为 `NOT_RUN`；
- 不新建 idea。
