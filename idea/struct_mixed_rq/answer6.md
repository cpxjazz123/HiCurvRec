# answer6 — 测试真正执行、因子按几何拆分、修复副本身份独立

**状态：CLOSED（可复验性、因子定义与身份隔离已完成；规格冻结/准入/训练/收益均为 NOT_RUN）**　完成：2026-11-10　对应 `question6.md`

---

## 1. 逐项验收表

| 项 | 状态 | 证据 |
|---|---|---|
| 源码已修（隔离树） | **DONE** | `layers.py` 为 `reconstruct(codes) + (x - x.detach())`；SHA `18e5d620…711da` |
| 测试真正执行 | **DONE** | 修复树 18 个测试文件、`rc=0` 18、`rc=1` 0（`q6/tests.log`） |
| 可收集的解析回归 | **DONE** | `test_ste_target_jacobian.py::test_cumulative_levels_match_the_target_jacobian`（含 `__main__`），红→绿已有记录 |
| 原生路径 3 个测试实际执行 | **DONE** | 补 `__main__` 后实跑通过（此前只 import 不执行） |
| 因子按几何坐标拆分 | **DONE** | `probe_factors.py` + 3 条反例测试（参数块 `NOT_APPLICABLE`） |
| 修复副本身份独立 | **DONE（测试无红）** | `ARM_NAME = TIGER_struct_mixed_rq_ste_fix`；`test_arm_isolation` 断言与旧树不同 |
| 数学规格冻结 | **NOT_RUN** | `math_spec.json` 仍 DRAFT；列为 `SPECIFICATION_INCOMPLETE` |
| 机器准入 | **NOT_RUN** | 未走 `stage2_start` |
| 正式训练（修复版） | **NOT_RUN** | 未重训、未换种子 |
| 下游收益 | **NOT_RUN** | 旧 `0.05790490781524748` 属旧源码图，**不沿用**为修复版收益 |

---

## 2. 测试执行（不是"文件数=通过"）

命令：`for f in tests/test_*.py; do python3.9 $f; echo rc=$?; done`（解释器 `genrec_env_v2/bin/python3.9`）

- 结果：`files pass=18 fail=0`，逐文件 `rc` 与 stdout/stderr 见 `q6/tests.log`；
- 新增/修正的收集点：
  - `test_rqlayer_native_path.py`：补 `__main__`，3 个函数实际执行（此前该文件只定义函数，`python file.py` 不执行，故 `answer5` 的"全绿"包含这一虚计）；
  - `test_ste_target_jacobian.py`：抽出 `test_*` 函数供收集，`__main__` 仍执行三层断言；
  - 新增 `test_factor_measurement.py`（红=模块缺失，绿=3 条）、`test_arm_isolation.py`。

诚实边界：`test_arm_isolation.py` 是在改完 `ARM_NAME` **之后**写的，**没有红灯记录**；它只能防回归，不能声称 TDD 红→绿。旧 `q5` 的 `red.log/green.log` 原样保留，未被追改。

---

## 3. 因子定义修正与反例

`probe_factors.factor_norms(name, grad, kind)`：

- `kind="parameter"`（encoder/decoder 参数矩阵）→ `split=NOT_APPLICABLE`，只报整块范数；
- `kind="code"`（256×32 码本）→ 沿最后一维切 E16/H16；
- `kind="latent"`（B×32 潜变量）→ 沿最后一维切 E16/H16。

反例（`tests/test_factor_measurement.py`）：梯度只落在 encoder 输出矩阵的第 20 输出行、第 0 输入列 → 旧列切法会报"E 分量"，现函数必须返回 `NOT_APPLICABLE`（测试已通过）。码本矩阵与潜向量的切分同时用手算期望核对。

---

## 4. 真实 TRAIN batch 的修正后记录（修复树，旧 checkpoint）

身份：`ARM_NAME=TIGER_struct_mixed_rq_ste_fix`，checkpoint SHA `84f05d46…4672`，SID SHA `7c147fc6…90a1`，TRAIN 前 64 个商品（0–63），三元组接受 45/64，`s=0.04444597661495209`、`λ=9.226019983543665e-07`（**旧 checkpoint 上测试新接线，非修复版标定**），mode=train，`behaviour_on=True`、`geometry_on=True`。

**损失**：recon `1.0601e-04`、quant `1.6660e-04`、λ·structure `3.96e-06`、total `2.7657e-04`（结构项原值 ≈4.29，与 q3 一致；此前因 `load_state_dict` 未解包 `state_dict` 导致随机初始化、结构项 1.1e9 的错误已修正，并在脚本中加了 missing/unexpected 硬失败）。

**加权结构项梯度**：

| 块 | 拆分 | 范数 |
|---|---|---|
| encoder | NOT_APPLICABLE | 4.7198e-05 |
| C1 | E16 / H16 | 3.0060e-05 / 3.0355e-05 |
| C2 | E16 / H16 | 3.9891e-05 / 4.4939e-05 |
| C3 | E16 / H16 | 3.8190e-05 / 4.2658e-05 |
| decoder | NOT_APPLICABLE | 0（未连接） |

**潜变量（结构路径）**：E `3.8047e-05` / H `4.2535e-05`。recon/quant/total 的潜变量分解本次记为 **NOT_MEASURED**（hook 捕获未取到 `.grad`），不冒充已测。

**同图合成**：`base + λ·structure` 与 total 的逐张量最大偏差 **2.9e-11**（float32）。

**关于一步更新的限定**：`answer5` 的码本位移 `[0.00979, 0.04243, 0.06693]` 是 **total-loss** 单步，不能单独证明结构项的优化贡献；本轮以"结构项逐块梯度 + 同图合成"支撑接线，未训练关闭臂。

---

## 5. 身份隔离（先证明不冲突）

- 修复树 `train_rqvae.py` 的 `ARM_NAME` 由 `TIGER_struct_mixed_rq` 改为 `TIGER_struct_mixed_rq_ste_fix`（SHA `db6ee335…38e9f95`）；
- 由此派生的结果目录、日志、checkpoint、SID 名称均不再等于旧实验路径；`test_arm_isolation.py` 断言两棵树的 `ARM_NAME` 不同；
- 旧源码、SID、checkpoint、Stage3 结果与 R2 `57` 未覆盖、未重命名；
- 路径与启动仍归 `tools/research/stage2/`，候选内未新增启动逻辑。

---

## 6. 仍未完成（明确保留）

1. **规格冻结 `NOT_RUN`**：`math_spec.json` 仍 DRAFT（旧 `ratio=0.1`、cumulative codeword sum 描述），未走真实冻结/审查/差分流程；按 `question6` 要求记为 `SPECIFICATION_INCOMPLETE`，不把 DRAFT 改成 FROZEN。
2. **机器准入 `NOT_RUN`**：未调用 `stage2_start`，未做一 epoch smoke 与 DDP 检验。
3. **正式训练 `NOT_RUN`**：修复版无 Stage2/Stage3 成绩；旧 `test_R@10=0.05790490781524748`（低于 0.065 差 0.7095pp）属旧源码图，**不是**修复版收益。
4. **前缀监督未实现**：主说明 §4.1 仍为设计项。
5. 潜变量 recon/quant 的 E/H 分解 `NOT_MEASURED`。

---

## 7. 证据路径与哈希

| 内容 | 路径 | SHA-256（前 24 位） |
|---|---|---|
| 修复 `layers.py` | `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/model/layers.py` | `18e5d62005aabb65c3f2bebb…` |
| 隔离 `train_rqvae.py` | 同树 | `db6ee33503a037448d7d6991…` |
| 因子辅助 | 同树 `probe_factors.py` | `e3b95c37ac2973137b0f866a…` |
| 因子测量 | 同树 `probe_q6_factors.py` | `b8a3b5d9b16ce0a0deb4b7ba…` |
| 因子测试 | `tests/test_factor_measurement.py` | `a67e9f2c64473adc70d6da9b…` |
| 身份测试 | `tests/test_arm_isolation.py` | `6ea5ff9ed64714907f328f1c…` |
| 目标测试 | `tests/test_ste_target_jacobian.py` | `a1e51f027d7f1e47ed1c4141…` |
| 原生路径测试 | `tests/test_rqlayer_native_path.py` | `2a575628ce92c58587b62af6…` |
| 因子结果 | `results/research/struct_mixed_rq/q6/factors.json` | `b1675fb192b2e2ca809818d6…` |
| 全套日志 | `results/research/struct_mixed_rq/q6/tests.log` | `a509e55a30f74dc35766941e…` |

（完整哈希由 `sha256sum` 生成，见上表路径；`q5/red.log`、`q5/green.log` 保持原样。）

---

## 8. 关闭条件核对

| 条件 | 状态 |
|---|---|
| 两种测试都由实际执行器收集并执行 | ✅ 18 文件 / 0 失败；新增目标与因子测试均可收集 |
| E/H 按真实几何坐标 | ✅ 码本/潜变量按最后一维；参数块 `NOT_APPLICABLE`；含轴混淆反例 |
| 修复副本身份独立 | ✅ `ARM_NAME` 独立 + 回归测试（无红灯，已注明） |
| 规格冻结/准入/训练/收益 | ❌ `NOT_RUN` / `SPECIFICATION_INCOMPLETE`，如实保留 |

**结论**：`question6` 要求的"可复验性与身份隔离"已完成；机制仍未完成规格冻结与准入，修复版没有下游收益。不新建 idea。
