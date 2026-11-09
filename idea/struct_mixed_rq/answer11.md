# answer11 — 真实 H 码字 Jacobian 已对照数值；残差覆盖扩到各层；诊断交接与待审稿仍未交付

**状态：PARTIAL CLOSED（数学覆盖修正完成；交接/待审稿/冻结/准入/训练/收益未完成）**　完成：2026-11-10　对应 `question11.md`

---

## 1. 真实 RQLayer 的 H 码字导数（DONE，验证覆盖修正）

`tests/test_q10_native_and_rqlayer.py` 中原先只断言"参与层梯度非零"。本轮改为**数值对照**：

- 对每个累计层 `k=0,1,2`、每个参与码字（层 0..k），计算输出对该码本**参数行**的**完整 32×32 Jacobian**（逐输出基方向求导），并与独立 `ref_reconstruct` 的 Jacobian 比较：`rtol 1e-6 / atol 1e-8` 全部一致，含 E 与 H 半区（覆盖符号、倍率，而不是只看范数）；
- 未参与该前缀的码字/层：Jacobian 全零；
- **H 断路的反例 fixture**：把累计输出的 H 半区 `.detach()` 后，E 仍有梯度而 H 梯度为 0——新断言会拒绝这种"E 通过、H 已断"的实现，证明检测能力；
- encoder 的单位 STE 断言（完整 32×32 = I）保留。

边界：正确实现本身未失败，因此本轮是**验证覆盖修正**，不冒充修复了模型数学。`answer10` 所称"首次参考符号错误被抓住"没有对应失败日志，按规则标 **NOT_RECORDED**，不重演补造。

---

## 2. 原生残差：各累计层的完整 Jacobian（DONE）

新增 `test_native_residual_full_jacobian_each_level`：

- 对 `1/2/3` 层、**每个累计输出**（`residuals(...)[l]`，l=0..level），比较对 latent 与各参与码字的完整 Jacobian（`rtol 1e-7`）；
- 对每个层级的最终输出，**被测实现与独立参考各自 `gradcheck`**；
- 零码字、近原点、非共线非零 H 情形保留；参考实现不调用被测函数生成期望值。

原两层 `.sum()` VJP 证据不删除，但其覆盖范围现明确为"部分方向"，不再称为完整 Jacobian。

---

## 3. 诊断交接：仍未完成（逐条）

| 项 | 状态 |
|---|---|
| 单一活动实现（迁移 `test_q8_report`/`test_q7_states` 与 `probe_q6_factors` 顶层写盘） | **未做** |
| 实际有序三元组/类型/目标/规范摘要 + TRAIN 与 embedding 与完整源码身份 | **未写入报告** |
| 等数量但商品/类型不同的身份反例 | **未实现** |
| 训练与诊断共享 `structure_term` 的 behaviour/空样本语义 | **未做** |
| cwd 外写入审计与负对照 | **未实现** |
| 独立运行目录不可覆盖（q9 仍固定 `run-q9` 且 `exist_ok`） | **未做** |
| checkpoint 缺键/多键拒绝经实际消费函数 | **未测试** |

---

## 4. 前缀待审稿与冻结

- `math_spec.json` 仍为 DRAFT，**仍含 `ratio=0.1` 与累计码字求和描述**；question8 §4 的候选（等权三层平均、共享尺度、完整前缀目标一次两块预算）**尚未落盘为待审稿**。
- 独立审查/冻结：**BLOCKED**。具体缺失前提：**符合当前规则、可用且有独立执行身份的替代审查入口**——研究 DAG（`2f07400c`）与公式/推导/实现 Agent（`bdf93104`）已被 owner 删除，相关 MCP 工具已反注册；不恢复调度器、不以主 Agent 自审或内部 `freeze` 代替。
- 前缀训练实现：**NOT_IMPLEMENTED**（未冻结前不做）。

---

## 5. 其余状态

| 项 | 状态 |
|---|---|
| 框架运行绑定（`LOG_DIR` 仍在源码树） | `NOT_RUN` |
| 机器准入（smoke/DDP） | `NOT_RUN` |
| 正式训练（修复版/前缀版） | `NOT_RUN` |
| 下游收益 | `NOT_RUN`；旧 `0.05790490781524748` 不继承 |

测试：`files pass=23 fail=0`（`q11/tests.log`）。

---

## 6. 身份

| 项 | 路径 | SHA-256（前 20 位） |
|---|---|---|
| 测试文件（本轮修改） | `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q10_native_and_rqlayer.py` | `ffe20b6514cb52fc24ed` |
| 套件日志 | `results/research/struct_mixed_rq/q11/tests.log` | `ee20fe7a5ef7acaed787` |

q5–q10 产物未覆盖；不派发正式训练、不覆盖原实验与 R2 编号 57；不新建 idea。
