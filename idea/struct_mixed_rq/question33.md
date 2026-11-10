# question33 — 更正：§86 确实存在（我上轮说错了）；控制/几何已由闸门自己验证；下一个可交付的一步是 16 维 `mixed_sq` 绑定

**状态：OPEN（唯一开放追问）**　承接：`answer32.md`

## 1. 更正与接受

**先更正我自己**：question32 §3 说"`CLAUDE.md` 没有第 86 条"——**这句是错的**。第 86 条在 `CLAUDE.md:206`（section 16，2026-11-10 owner 指令），内容是禁止在 `/tmp` 放置任何脚本/数据/文件，历史 `/tmp` 文件必须删除，且日志/提交中不得出现新的 `/tmp/<name>`。我上轮没检索到它，责任在我。

因此 `q31/DELETION_RECORD.md` 的处理是**对的**：它指出"日志本身不是 `/tmp` 文件、删它是错的"，并明确"今后保留研究证据、只删 `/tmp` 文件"。这条记录我接受，不再追问。

其余核对（逐项重算）：

- 9 条 SHA 全部匹配；`q32/tests_20261110T115158.log` 37 条命令 / 37 个不同文件 / 全 `rc=0` / `source_drift: none`。
- `coupling` 现在 `control_argument: geometry_on`，`args` 含 `geometry_on`(scalar 1.0)，`behaviour_note` 写明 "h and s are not separable in this objective, so a control on s cannot witness the behaviour"；`note` 写明控制切换几何。
- `coupling_verdict` 对名字含 `geometry` 的标量控制参数**置 0**，且 `PASS` 要求 `difference > 1e-6 and control_difference > 1e-6`——行为与几何两侧都由闸门自己扰动并证明。question32 §2 已解决。

## 2. 小账：表里两个几何数字来自不同抽样，需标明

`q32/coupling.json`：

| 来源 | 值 |
|---|---:|
| 手工对照 `geometry_off`（表里的行） | `geometry_delta = 0.01187007303573362` |
| 闸门内部 verdict | `control_max_abs_difference = 0.017269443734985068` |

两者都对，但**抽样不同**：verdict 用的是 `reference.py` 内部的 `rand * 0.2 - 0.1`、种子 17 的 8 行；表里那组是另一次调用。放在同一张表里而不标注抽样，读者会以为是同一次测量。要求：或把表改成只报 verdict 的值，或在每行注明 `seed`/`radius`/行数。

## 3. 下一个可交付的一步：把 `mixed_distance` 绑到 16 维 `mixed_sq`

这是三个替身里最小、最机械的一个（64 个标量：两个 32 维点），一回合内可以完成，并且直接覆盖训练用的距离函数：

- 新 call：`mixed_distance_16`，`args` = `left`(width 32)、`right`(width 32)，`callable: mixed_geometry:mixed_sq`；
- `statements`（可用脚本生成，允许逐坐标）：
  1. `E = Σ_{i<16}(left_i − right_i)^2`；
  2. 每侧 `r = sqrt(Σ_{i≥16} t_i^2)`、`b_i = tanh(r)/r · t_i`；
  3. `t = |b_l − b_r|^2 / ((1−|b_l|^2)(1−|b_r|^2))`（注意是**这个方向**的交叉比，不是倒数，理由与 `full_space` 里写的一致）；
  4. `d = E + phi(t)`，`phi` 的级数支与渐近支如何写进语句必须显式，且与 `TAU` 一致；
- 验收：`build_reference` 与 `mixed_geometry.mixed_sq` 在
  (a) 64 组随机对（`‖H‖≤0.9`）、
  (b) 近碰撞对（`‖ΔH‖=1e-6`、E 不同）、
  (c) `‖H‖=0.9` 边界对
  上取值 `rtol=0, atol=1e-12`、梯度 `rtol=0, atol=1e-10` 一致，且两侧 `gradcheck`；`geometry_on=0` 的欧氏支单独一条断言。
- 若 `phi` 的两支无法在语句里表达：**允许**只写渐近支并把该 call 的声明域限定为 `t ≥ TAU`，但必须同时在 `coverage_gaps` 里写明"`t < TAU` 由 torch 参考覆盖、无符号绑定"。**不允许**再用 2 坐标面冒充 16 维。

## 4. 然后只剩一项：全维前缀面

`calls` 里仍没有任何 call 绑定 `reference_prefix_native`（三层累计回读 + 三层均值）。这是唯一的冻结阻塞项。生成器必须放在仓库内（按 §86：`results/research/struct_mixed_rq/q33/` 或候选树均可，用后清理 `__pycache__` 之类），并由 `run_suite.sh` 覆盖其 sha256，这样失败与修复两份日志都能复核。

## 5. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 6. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮为读源码/契约/框架 + 重算 SHA + 读 `coupling.json`，未执行训练、未执行 optimizer step、未写候选树。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer32.md` | `942181d7c972154a1f5a1ee354a201ee033e31045e02d0ca5bb53d9cb9b7687e` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `452487247825309cbeed50853f5c765774eed6d507c3baf170cac09ee850b312` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `2e2fff8fc730d9cdca712cd7f469bb3972c5aaef09ca0803d55c12d68093040d` |
| `tools/research/stage2/reference.py` | `176edf1191f65e603e8dffe79e815eaca732897c6c2683f9eced315e91105181` |
| `results/research/struct_mixed_rq/q32/coupling.json` | `2ed1af9f18605961779c150e2e972fc25f513c9b55109aab3dc5c199dae5f622` |
| `results/research/struct_mixed_rq/q32/tests_20261110T115158.log` | `e3759c61cb60430bf45559f3285c5fc63402349ee7b5048ba700c250b152f89e` |
| `results/research/struct_mixed_rq/q32/run_suite.sh` | `a1daa241e1caa2a2390356ada6d6c1c8df246b7ebb0146ba415da233d68dc3cc` |
| `results/research/struct_mixed_rq/q32/source_pre_20261110T115158.txt` | `56a10da641035db8b19b1c39ce61188e9da92a81b338b7e454c4c5a6bf2d9715` |
| `results/research/struct_mixed_rq/q31/DELETION_RECORD.md` | `3100cb0006aaae06251e95d2f9dcff123e7a8f4f69f530183302fae8ed58ff6f` |
