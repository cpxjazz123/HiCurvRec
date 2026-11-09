# answer10 — 原生残差与真实 H/STE 组合已验证；诊断交接只完成一部分

**状态：PARTIAL CLOSED（两项数学验证完成；交接迁移、前缀待审稿与冻结、准入/训练/收益未完成）**　完成：2026-11-10　对应 `question10.md`

---

## 1. 原生残差更新：独立参考 + 双侧 gradcheck（DONE）

新增 `tests/test_q10_native_and_rqlayer.py`：参考实现完全独立于被测函数，按已声明定义写出

- E 半区：`r_e ← r_e − c_e`；
- H 半区：`r_h ← ball(z_H)`，随后 `r_h ← (−ball(c_H)) ⊕_M r_h`（**Möbius 左逆**；测试首次实现把符号写反，被取值对照当场抓住并修正）；
- 输出：`[r_e, log0(r_h)]`。

覆盖：

| 情形 | 结果 |
|---|---|
| 1/2/3 层、非共线非零码字 | 取值一致（`rtol 1e-9`） |
| 1/2/3 层、零码字（近原点） | 取值一致 |
| latent 与码字梯度 vs 参考 | 一致（`rtol 1e-8`） |
| 被测实现 / 参考实现各自 `gradcheck` | **均通过** |

离散选码未混入该连续对照。

---

## 2. 真实 `RQLayer` 的 H 真梯度与单位 STE（DONE）

同一文件、固定离散选码（三层分别选第 1/2/3 号码）、float64：

1. **H 前向**：每个累计层的完整 32 维输出等于独立右嵌套回读（`rtol 1e-8`）；
2. **encoder 代理**：`d(level_k)/dz` 的 **完整 32×32 Jacobian 等于单位阵**（`atol 1e-9`）——即修复后的 `q = R + z − sg(z)`，每层都是 I，不是 `kI`；
3. **真实码字梯度与 STE 分离**：对码本参数的梯度只在前缀层 ≤k 出现，未参与该前缀的码字梯度为零；这部分是连续回读的真梯度，与 encoder 的代理单位映射分开核对；
4. 未对含 `detach`/硬选择的整条前向做有限差分来冒充 STE 验证；STE 只按声明的代理检查。

保留 q8 的累计回读独立 Jacobian 与三层双侧 gradcheck，不重复。

---

## 3. 诊断交接：部分完成，缺口逐条列明

已接受并保留：q9 的单次采样与数量不变量（64/45/135），四项损失/参数三态/双潜变量 E/H 已入 JSON。

**未完成（不得记 CLOSED）**：

| 缺口 | 现状 |
|---|---|
| 有序三元组、类型/目标、TRAIN 与 embedding 输入摘要、完整候选源码清单的**可复算身份** | 未写入 q9 报告；`identity` 只有 IDs/计数/行数与部分源码摘要 |
| 数量相同但**商品或类型不同**的负对照 | 未实现；现有 `consistent()` 只比较 `rows == 3×accepted` |
| 复用 `structure_term` 的行为开关与空样本语义 | 诊断自行编排 `_positions`/`structure_loss`，未覆盖 behaviour-off 与空样本分支 |
| 单一活动实现 | `probe_q6_factors.py` 顶层仍写 q7；`test_q8_report.py` 仍导入 `probe_q8`，`test_q7_states_and_checkpoint.py` 仍测 `probe_factors` |
| cwd 外写入审计与负对照 | 未实现（现有 import 检查只看临时 cwd） |
| 运行目录不可覆盖 | q9 固定 `run-q9/factors.json` 且 `exist_ok=True`，重复执行会覆盖，不是不可变记录 |
| checkpoint 拒绝分支经实际消费者 | 未测试；`probe_q9.run` 的 `SystemExit` 分支没有执行证据 |

这些是下一轮的实际工作项，不能以本文件的部分进展代替。

---

## 4. 前缀待审稿与冻结

- `question8.md` §4 的候选（等权三层平均、`s` 沿用末层初始化表面、`λ` 对完整前缀目标一次块预算、`λ_mean=3λ_sum`）仍未写进 `math_spec.json`；后者仍为 DRAFT 且保留 `ratio=0.1` 与累计码字求和描述——**待审稿尚未落盘，旧描述未清理**。
- 独立审查/冻结：**BLOCKED**。具体缺失前提是**符合当前规则、可用且有独立执行身份的替代审查入口**：研究 DAG（`2f07400c`）与公式/推导/实现 Agent（`bdf93104`）已被 owner 删除，`research_start/resume` 等工具已反注册；本次审阅不授权恢复调度器，也不把工具移除当准入豁免。
- 前缀训练实现：**NOT_IMPLEMENTED**（未冻结前不做）。

---

## 5. 其余状态

| 项 | 状态 |
|---|---|
| 框架运行绑定 | `NOT_RUN`（`LOG_DIR` 仍在源码树） |
| 机器准入（smoke/DDP） | `NOT_RUN` |
| 正式训练（修复版/前缀版） | `NOT_RUN` |
| 下游收益 | `NOT_RUN`；旧 `test_R@10=0.05790490781524748` 不继承 |

测试：`files pass=23 fail=0`（`q10/tests.log`）。

---

## 6. 身份

| 项 | 路径 | SHA-256（前 20 位） |
|---|---|---|
| 新数学/接线测试 | `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q10_native_and_rqlayer.py` | `94d61555d469b6d0ec09` |
| 套件日志 | `results/research/struct_mixed_rq/q10/tests.log` | `0fd22bdb43f4c5e52aa1` |

q5–q9 产物未覆盖；不派发正式训练，不覆盖原实验与 R2 编号 57；不新建 idea。
