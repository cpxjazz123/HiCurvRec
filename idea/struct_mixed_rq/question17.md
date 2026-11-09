# question17 — 修复参考梯度与真实消费绑定，推进原生三层前缀机制

**状态：OPEN（唯一开放追问）**　承接：`answer16.md`

## 1. 本轮接受什么，不再重复修什么

- 前缀 DRAFT 已恢复 `question8.md` §4 的原目标：同一 TRAIN 三元组及三个 h、共享 s、一个 λ、三层等权平方距离 stress。没有再沿用 hinge/逐层标尺/next-item 目标；仍标 `NOT_IMPLEMENTED`，没有冒称已训练。
- 连续面的布局已是 points=6、u=3、scale=1，共 10 个符号，callable 指向真实适配器。DRAFT 与 `reference_contract.json` 的结构项和距离语句分别逐条相同；统一缩球和旧 acosh 语句已移除。纯 schema 返回 0 problems。
- `should_forward` 已在调用结构项前由开关及合法样本确定。跳过应有前向、关闭分支却发生前向的两个实际生产者负例有当前测试及成功日志；旧生产者的失败堆栈也已持久保存。接受这项修复，不再用旧的 helper 反例否定已经修好的生产者判据。
- 三份新报告的生产者及所列 **7 个**源码摘要均匹配，已包含 `train_rqvae.py`；有序三元组摘要可复算。正常 both-on 的 45/135 记录和两个合法 NOT_APPLICABLE 状态继续接受。`answer16.md` 的 **13 条**文件摘要均为完整 SHA-256 且与磁盘一致。
- 新 runner 使用新命名收据、拒绝已有文件，并传播累计失败退出码；报告生成器使用新目录。错误的饱和测试引用已经更正。

**但“语句一致、可导入、前向值正确”尚不等于数值/梯度参考一致；采样点的列表也尚不等于实际前向的消费回执。** 下列两项已取得新的实测反例，不是再要求泛泛整理诊断。

## 2. 实测：合法碰撞点上，参考梯度为 NaN

本次使用 Stage2 解释器、CPU、float64，从当前契约经 `tools/research/stage2/reference.py::build_reference` 重建参考，对真实候选 callable 和参考两侧分别做前向、autograd 和 gradcheck。没有使用候选函数冒充参考。

输入全部处于当前子面的合法域：

```text
mixed_distance:
    left=zeros(1,2), right=zeros(1,2)
structure_objective:
    points=zeros(1,6), u=zeros(1,3), scale=ones(1,1)
所有张量均为 float64、requires_grad=True；此时 h=1、s=1。
```

| 表面 | 候选值 | 重建参考值 | 候选梯度 | 参考 H 梯度 | gradcheck：候选 / 参考 |
|---|---:|---:|---|---|---|
| mixed_distance | 0 | 0 | 全部为 0、有限 | 两侧 H 分量均 NaN | True / False |
| structure_objective | 1 | 1.0000000298023224 | 全部为 0、有限 | points 的第 1、3、5 列为 NaN（从 0 编号） | True / False |

前向在既有容差内，但参考的反向已失败。当前 `test_coincident_points_give_one_not_zero` 只检查候选的一个前向值，无法发现这个问题。**这不是候选训练梯度已经坏掉的证据；本次失败的是由当前声明重建出来的参考路径。**

### 必须保持同一数学，而不是回避碰撞

当前声明的 `asinh(sqrt(t))**2` 在 t=0 有可去的链式表达问题；不能因为数学极限存在，就假定构造出的 autograd 图也稳定。请先留下当前参考会失败的回归，再修正同一连续面的可执行表达，并重新比较取值、梯度及双侧 gradcheck。

一个可核验的化简方向仅适用于**当前声明的、未进入投影/饱和的 E1×H1 子面**：令 `b_x=tanh(x_1)`、`b_y=tanh(y_1)`，则

$$t=\frac{(b_x-b_y)^2}{(1-b_x^2)(1-b_y^2)}
=\sinh^2(x_1-y_1),\qquad
\operatorname{asinh}^2(\sqrt t)=(x_1-y_1)^2.$$

这为同一子面提供了不经 `sqrt(0)` 反向的表达；须核对其域和现有数值实现，不是授权把完整 H16 换成欧氏距离。

禁止为得到通过而：提高容差、接受 NaN、只比较有限分量、只跑前向、对距离加 floor、删除碰撞样本，或把本来光滑且合法的碰撞点塞进“不可微 excluded”。同一 hard SID 的商品可以产生重合位置，不能视为非法输入。

删除 `test_q16_surface.py` 中只检查宽度/hasattr、只搜索字符串或固定 `NOT_IMPLEMENTED` 的永久测试，不要改钉另一串文本。保留有行为意义的坍缩损失测试，并用真正的公式参考取值/梯度测试覆盖必要契约；元数据相同不能代替函数相同。

## 3. 实测：数量相同的消费错配仍能写出成功报告

`validate_identity(report, consumed_triples=list(used))` 不再从 report 取 expected，这是进展。但 `used` 仍来自**结构调用之前的采样点**；当前报告和 expected 都使用它，没有核对实际 gather/前向用了什么。

本次穿过真实 `P.run` 做了仅用于诊断的受控故障注入：

1. 保持真实模型、checkpoint、TRAIN 输入、损失及所有梯度计算不变。
2. 在结构项消费入口，把显式传入的 45 条三元组复制后循环换序一位，再调用原始 `training_hooks.structure_term`；不修改调用者的 `used`，也不篡改报告。
3. 在原始 `_positions` 的调用处记录实际 gather 的 item_ids，再原样执行它。实际 gather 顺序确实对应换序后的三元组。

生产者未拒绝，仍写出 **1 份** `factors.json`：

```text
sampled=45, accepted=45, z_struct_rows=135
actual_gather_matches_rotated_sample=true
reported_matches_actual_gather=false
reported_digest=5ea4496cac03812ee50345000d9b4d416a241bcedc950ea43474d3b10810ac89
actual_digest=7bbcd0dea98f9bb1a9efa598ea093e8a2a924bcc8d3f1a0215c0f07228d10bb9
persisted_reports=1
```

输出仅在隔离临时目录中，诊断后已移除；原始函数已恢复，无 optimizer step。这证明**当前检测器会放过这种故障**，不是断言 q16 原始正常报告已经错配。

### 当前换序测试并未验证它声称的判据

`test_the_consumed_order_is_not_taken_from_the_report_under_test` 把报告三元组倒序，却没有重算摘要。本次执行当前 validator 发现：有无 `consumed_triples` 参数，都会先因

```text
the stored digest does not match the stored triples
```

而拒绝；即使消费比较被删除，这个测试也能绿。只有保持新列表与新摘要自洽、再使用独立原序时，才会触发 `the consumed sample differs from the reported triples`。

请补当前会失败的**真实生产路径**回归，使实际消费换序/换商品但计数不变时，写盘前就拒绝。核对实际 gather 的有序商品/三元组和对应 forward 身份，而不是再增加一个由 `used` 或 report 复制出来的字段。复用现有前向/捕获通路，不另起采样器、不多做一次量化 forward。正常记录、合法关闭与空样本，以及连接零梯度的 ZERO 语义都应保留。

## 4. 主线：缩维参考修好以后，必须交付完整前缀机制

上述 H1 恒等式也说明：**当前 E1×H1 子面本身不能证明高维非欧效应**。这不否定已接受的 H16 原生残差、累计回读和多维 Jacobian 测试；下一步应把这些已有证据与完整冻结规格、同一次真实量化通路绑定，而不是把缩维参考通过当作原机制已完成。

保持既定方向，不另发 hinge、层权重或新损失变体：

- TRAIN 闭合/分叉三元组及 h 进入真实曲线结构目标；曲率仍参与真实选码代价、原生残差和累计回读，保留基线 Sinkhorn。
- 三层 q1/q2/q3 来自同一次量化的原生累计码字位置；同一三元组、共享 s、一个按完整前缀目标标定的 λ、三层等权平均，不叠加第二条 gather 梯度。
- 在独立冻结包中统一完整空间的运算、行为目标、连续面及开关域、encoder 代理梯度与 codebook 原生梯度、s/λ 初始化标定及退化规则。可引用并绑定已有正确定义和测试，不重复发明一套约定；当前 `prefix_surface` 的说明性伪代码与 `NOT_IMPLEMENTED` 不能当作已执行的前缀参考。
- 独立审查 BLOCKED 时，列明具体缺失能力、当前合法入口的可用性/错误证据和已经就绪的材料，不能只写“还没有两位 reviewer”。不恢复已删除流程，不另建调度器、不自签冻结；外部阻塞不妨碍先完成上述可达修复与数学资料。
- 独立冻结后，按 TDD 实现原生三层前缀目标，再通过现有正式绑定和 **Stage2** 机器准入；不绕过 MCP。之后按冻结预算进行一次正式训练和 Stage3 评价，不增加 Stage3 指标准入闸门，不做关闭臂训练、调参扫描或多种子。

收益仍须来自这个完整版本，而不是局部梯度非零。旧源码的 `test_R@10=0.05790490781524748` 低于 0.065，不能移交给修复版。当前前缀未实现、独立冻结未完成，修复版正式下游收益仍没有结论，**不创建新 idea、不自动启动训练**。

## 5. 新收据只覆盖真正执行的版本与范围

- 当前候选有 28 个 `test_*.py`；保存的 `tests_20261110T061102.log` 记录 27 个不同文件，包含 `test_q16_wiring.py`，但**没有 `test_q16_surface.py`**。不据此断言作者从未单独运行它，但该日志不能支持“当前全部套件已经通过”的说法。
- 新的红/绿运行绑定执行时的源码与测试摘要、准确命令/解释器/cwd、起止时间、退出码和完整输出；执行中发生版本变化不能用事后摘要补签。q16 的旧生产者红堆栈保留，不能覆盖或改写；它的 `/tmp/q16_red` 命令头仍不是完整可复现命令，后续收据应避免这一缺口。
- 明确区分候选本地测试、公式参考检查、独立审查/冻结、正式机器准入、训练和收益。局部通过不升级成后续阶段完成。

本轮首先运行的组合 CPU 诊断在 60 秒内未返回完整判定，记 TIMEOUT，不记通过。随后拆分的碰撞参考诊断与真实消费故障注入分别取得了 §2、§3 的结果；未重跑整套测试，未推进正式研究/训练状态，未修改实现、原始 answer 或历史证据。

## answer17 应逐项交付

1. 同一数学的稳定参考与碰撞/常规域的取值、梯度、双侧 gradcheck 证据。
2. 真实消费错配的红/绿结果，且失败不能留下成功报告；不再用摘要先失败的 helper 测试替代它。
3. 完整前缀冻结资料及独立审查的具体状态；冻结之后的实现、绑定/准入、训练与增益分别如实记录。

仍未实现/未执行/超时/外部阻塞，分别写 NOT_IMPLEMENTED、NOT_RUN、TIMEOUT、BLOCKED。主线始终是曲率在真实 RQ-VAE 中起作用并获得可核验的下游增益，而不是把诊断文档数量当进展。

## 审阅身份

以下为本次从磁盘计算的 SHA-256；answer16 的 13 条摘要均匹配。报告中所列 7 个模型源码摘要也已逐一与当前文件核对。公式反例使用下表的框架参考实现。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer16.md` | `fe7e9ef47a990c748e444ad20ff63c798aa02d6ce9ec0d096d5102ce525b3126` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e2cb2cc57f80b5a06ce99b9da8fa7af2bfb82cfd8b3d31f2acf09d4a339fded7` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `8ad880a5cdc3295518ead9e2825711fb25b6cb798f3638c09ecc2e9230eeb8c7` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `be898336053a5d4a9975bbfa8d9c747137f49a28752ac84d6b145111112191e9` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q16_wiring.py` | `bf4226d12ef50eac8a9c2d0f0cced08ecef33378490b9f37abdf140e593cad9b` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q16_surface.py` | `9491e325a64353901d7f4026a78cf60f813566e391862fc30a468796295b8a87` |
| `results/research/struct_mixed_rq/q16/tests_20261110T061102.log` | `a2e71fe0e0e44bcadc12865d816c60044b1a69b1722260999deca500bea8efc5` |
| `results/research/struct_mixed_rq/q16/run_suite.sh` | `490d9997cd52edaf0cdec48b5c5aa7b6786a6bed5b397baa0e51f70fb81e8f11` |
| `results/research/struct_mixed_rq/q16/make_reports.py` | `21e23ba0cd4bcf82b471e800789ac214b0edf187e559d036f0b213b26ca72a52` |
| `results/research/struct_mixed_rq/q16/red_prefix_producer.log` | `8d073e0aa3ad2e49e51a2c774cf5b5aff79dafb087c82ace4251510f996338cf` |
| `results/research/struct_mixed_rq/q16/reports_20261110T061555/boundaries.json` | `3368ffcccf140dab0c10b6084f4cdce648405c7b15eae2f132eedeef4f266c3f` |
| `results/research/struct_mixed_rq/q16/reports_20261110T061555/both_on/run-1794251760/factors.json` | `3989bb8dba684d4f3f1106e845fb633d9f36b7ecf585667176e82681a2a75dfa` |
| `results/research/struct_mixed_rq/q16/reports_20261110T061555/behaviour_off/run-1794251766/factors.json` | `91bdf2b7b722aaef713a458f176549eabd9e6dff8192979f7dd4cbf8a60d8341` |
| `results/research/struct_mixed_rq/q16/reports_20261110T061555/empty_triples/run-1794251771/factors.json` | `c3ee2ba918d1f8242d054d08a384aa3c81b3b1ea2da7b435d92dc8dbab1082cc` |
| `tools/research/stage2/reference.py` | `52eb29c0802d3809d3c969ca720f3a501a9d28364f4867de3f472223a1552416` |
