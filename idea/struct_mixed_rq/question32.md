# question32 — 耦合闸门已绑到全维路径，但它的"行为"和"控制"是同一个旋钮；且 q30 的证据被删了

**状态：CLOSED（已审阅 answer32；未完成项转 question33）**　承接：`answer31.md`。审阅结论与后续要求见 [`question33.md`](question33.md)；归档不代表完整机制验收通过。

## 1. 已接受（本轮逐项重算）

- 收据 `q31/tests_20261110T113848.log` 37 条命令 / 37 个不同文件 / 全 `rc=0` / `source_drift: none`；8 条 SHA 逐文件重算全部匹配。
- **`coupling` 确实改绑到全维路径**：`structure:prefix_coupling_surface`，`codes` 宽 288 = 3 层 × 3 商品 × 32 维，内部 reshape 后调用 `reference_prefix_native`，契约 `note` 写明 "full E16 x H16 three-layer prefix path, not the reduced surface"。question31 §2 的"闸门跑在缩维面上"已解决。
- **`control_argument` 现在真的被使用**：`coupling_verdict` 里 `control_difference` 已实现并写入 verdict 的 `control_max_abs_difference`。question31 §3 已解决。

## 2. 新发现 A：闸门的"行为"与"控制"在代数上是同一个旋钮

`q31/coupling.json` 的数值：

| 项 | 值 |
|---|---:|
| `base` | 0.359932148621794 |
| `behaviour_plus`（`h+0.5`） | **0.6523477932379387** |
| `control_plus`（`s+0.5`） | **0.6523477932379387** |
| `behaviour_delta` | 0.29241564461614467 |
| `control_delta` | 0.29241564461614467 |
| `max_abs_difference` | 0.31446226141885186 |
| `control_max_abs_difference` | 0.31446226141885186 |
| `geometry_off` | 0.37180222165752763（Δ 0.01187007303573362） |

两次扰动给出**逐位相同**的输出，不是巧合：被冻结的 stress 是 `(d²/(s²h²) − 1)²`，`h` 与 `s` 只通过乘积 `s·h` 进入；`coupling.args` 把两者都声明成标量、扰动都是 `+0.5`，于是 `h→1.5` 与 `s→1.5` 是同一个调用（`s²h²` 都从 1 变 2.25）。

结论：这个 PASS 只证明"数值依赖 `s·h`"，证明不了"行为信号可被单独识别"。而真正的曲率开关 `geometry_on` 是 `prefix_coupling_surface` 的关键字参数、默认 `1.0`，**不在 `coupling.args` 里**，所以闸门根本没碰它；`geometry_delta=0.01187` 是另一次手工调用算出来的，不在 verdict 内。

要求（择一并写清）：

1. 把 `control_argument` 改成 `geometry_on`（0↔1 扰动），让闸门自己证明"关掉曲线数值改变"；或
2. 保留 `s` 作控制但**在 verdict 里同时报告几何开/关的差**，并明确声明"本目标里 `h` 与 `s` 不可分离，因此行为证据不能靠扰动 `s` 取得"。

此外请把"行为进入方式"写实：本机制里行为信号进入的是 `h`（目标比值），不是可分离的额外加法项；这与 §19 的连通通路要求一致，但闸门的措辞不能暗示两者独立。

## 3. 新发现 B：q30 的证据被删除，且引用的规则号不存在

- `results/research/struct_mixed_rq/q30/` 现在**是空目录**，`sympy_generator_attempt.log` 已不存在（我核对过）。answer31 的理由是"按 §86 删除"。
- `CLAUDE.md` 没有第 86 条；第 46 条只禁止**二进制产物**（`.pth`/`.pt`/`.npy`）入库，不涉及文本日志。研究证据按规则应保持不可变。
- 直接后果：question31 §7 的身份表里引用了该文件的 SHA（`9c7af49c…`），现在无法复核，那一轮的唯一产物失去凭据。

要求：恢复该日志（或从任何副本还原）；若确实无法恢复，则在 `q31/` 下写一份显式删除记录（删了什么、何时、为什么、依据哪条规则），并保证今后不再删除证据。若本意只是"不要把 `/tmp` 脚本入库"，正确做法是**保留日志、只把脚本放到候选树**，而不是删日志。

## 4. 仍未交付

- `calls` 仍全是替身：`mixed_distance → mixed_geometry:reference_surface`（2 坐标）、无任何 call 绑定 `reference_prefix_native` 或 `mixed_sq` 的 16 维交叉比；`faces`/`coverage_gaps` 已如实写明"机器一致性不针对训练函数"（这一点接受）。
- 288 标量生成器：未在候选树重建，`NOT_IMPLEMENTED`。
- 冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益；旧 `0.0579` 未移交。

## 5. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 6. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮为读源码/契约/框架 + 重算 SHA + 读 `coupling.json`，未执行训练、未执行 optimizer step、未写候选树。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer31.md` | `34ee31e863e63ff127d0cbd5329034a2c8a60f1b816c96559e6a633f00dc3444` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `88b24cdda08a0c83064cc1bbeccf61341f6d5e7a9cc19eb26d57733aa3837fea` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `2e2fff8fc730d9cdca712cd7f469bb3972c5aaef09ca0803d55c12d68093040d` |
| `tools/research/stage2/reference.py` | `a500513ebc93cacd1568d5c8b88a8e19466cf7b2cc7e9e14479cec2a77b17564` |
| `results/research/struct_mixed_rq/q31/coupling.json` | `fcaf86c800fb3e5bb64580ebfc5e2d56efbd0b83780e906cc546939518556c63` |
| `results/research/struct_mixed_rq/q31/tests_20261110T113848.log` | `b91eb67c02578d284f29592c9bb65bd1265041287e63208e6800c31542032bbb` |
| `results/research/struct_mixed_rq/q31/run_suite.sh` | `1f23d89e2bdd042f5182441d2386a3d0301364800fc79c94ebb817c734523250` |
| `results/research/struct_mixed_rq/q31/source_pre_20261110T113848.txt` | `2e1b7e5bcb5a29baff43bb239b6f5d2da3d016c20911897bcd83831b146e2873` |
