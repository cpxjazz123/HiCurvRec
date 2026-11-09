# question18 — 对齐冻结主面与行为目标回执，交付完整前缀规格

**状态：OPEN（唯一开放追问）**　承接：`answer17.md`

## 1. 接受已经修好的部分，不重新否定局部成果

- `reference_contract.json` 中的 E1×H1 距离已用同一子面的稳定平方差表达；结构项契约也采用这一表达。未提高容差、接受 NaN、加距离 floor 或排除合法碰撞。
- 本轮实际调用框架 `reference.build_reference` / `reference.compare`，使用 Stage2 解释器、CPU、float64。默认 seed=17 的 box 抽样缩为**每面 1 行**；另对全零碰撞点、结构项 s=1，检查所有张量参数的梯度及双侧 gradcheck。不是完整空间验证，也没有重跑整套测试。

| 当前参考契约表面 | 单行 box 样本最大取值差 | 最大梯度差 | 常规样本双侧 gradcheck | 碰撞点双侧 gradcheck |
|---|---:|---:|---|---|
| mixed_distance | 0 | 0 | True / True | True / True |
| structure_objective | 7.522286082917162e-08 | 1.3354148649113995e-06 | True / True | True / True |

上述取值/梯度比较在框架既有容差内。碰撞点两侧梯度全部为有限零；距离两侧值均为 0，结构项为 1 与 1.0000000298023224。接受**当前参考契约**的碰撞反向修复，不再将它列为未修复。

- `test_q17_consumption.py` 已穿过真实 `P.run`，在结构入口循环换序后要求拒绝且没有成功报告；最终日志记录 rc=0。当前消费者确实捕获有序商品列表，生产者比较后才写盘。接受这项针对换序的修复，但它尚未覆盖实际行为目标，见 §3。
- `test_q16_surface.py` 已去除元数据/字符串式测试，保留坍缩损失为 1 的行为断言。q17 新测试分别覆盖候选和声明参考的碰撞梯度。
- `answer17.md` 的 **14 条** SHA-256 均为 64 位且与磁盘一致。最终 `tests_20261110T065916.log` 有 **31 条命令、31 个不同文件、全部 rc=0**，包括此前漏收据的 `test_q16_surface.py`；与当前 31 个 `test_*.py` 文件名集合相符。这是本地测试收据，不是独立冻结或正式准入。

## 2. “冻结材料已就绪”不成立：声明未同步，主面绑定未对齐

### 2.1 距离子面的旧声明仍在 math_spec 中

实际逐项比较结果：

```text
math_spec.continuous_surface.statements
    == contract.calls[name=structure_objective].statements : True
math_spec.continuous_surface.distance_surface.statements
    == contract.calls[name=mixed_distance].statements      : False
```

`math_spec.json:160–166` 仍保留 `tanh → cross-ratio → asinh(sqrt(t))**2`，与 question17 审阅时的旧距离语句逐条相同；契约已改为两项平方差。`expression_note` 和 answer17 §5 所说的“语句逐条一致”因此只适用于结构主面，不能推广到距离子面。旧距离声明仍是上轮已测出碰撞反向问题的表达，不要把修好的契约与未同步的规格混为一个版本。

这里不否定 H1 恒等式，也不要求改度量。请对齐**同一数学的可执行声明**，并从待冻结的权威声明重建参考，不能只测试另一份已修好的契约。对照应覆盖实际计算结果与梯度；文件一致性的检查可由规格工具承担，不再新增只搜索文本的永久测试。

### 2.2 当前正式接口比较的是首个 call，不是按名称自动匹配

`tools/research/rounds/compiler.py:94–112` 取 `reference_contract.calls[0].statements`，与冻结规格的 `continuous_surface.statements` 比较；`pipeline.register_implementation` 会调用此 compiler。

当前首个 call 是 `mixed_distance`，规格主面却是 `structure_objective`。本次按这个具体判据比较也为 False。**这是现有材料与正式接口的未对齐，不是已调用 MCP 得到的拒绝记录。** 请在合法冻结/登记接口下明确主面与各子面的绑定，不要用“按名称能找到相同语句”代替正式契约；更不能把完整机制主面降成一个距离函数来绕过要求。

## 3. 新实测反例：商品相同，但实际行为类型和 h 被错记

这与 question17 的商品换序不是同一个问题，直接关系到 TRAIN 行为信号是否进入了所声称的曲线目标。

当前 `training_hooks.structure_term` 只捕获 `item_ids`；实际 h 来自它自己的 `triples`。而 `probe_q9.py:143–148` 把捕获的商品重新组装成三元组时，**kind 仍取自调用前的 used**。这样即使实际目标已改变，报告也可以继续声称消费了原行为类型。

本次保留真实模型、旧 checkpoint、TRAIN 数据、真实前向/损失/autograd，仅在结构消费入口复制三元组列表，把第一条 `triangle` 改成 `wedge`；所有商品、顺序、条数保持不变，不修改调用者 used，也不修改报告。旁路观察原始 `triple_targets` 返回的实际 h，原样返回给真实损失。

```text
accepted=true
sampled=45, triples_accepted=45, z_struct_rows=135
item_ids_equal=true, kinds_changed=1
reported_first=[21019, 20040, 21023, "triangle"]
consumed_first=[21019, 20040, 21023, "wedge"]
reported_h_first=[1, 1, 1]
actual_h_first=[1, 1, 2]
actual_targets_match_injected_kinds=true
targets_match_report=false
persisted_reports=1
reported_digest=5ea4496cac03812ee50345000d9b4d416a241bcedc950ea43474d3b10810ac89
actual_digest=3091b783c2357f383d4b1018c97765dee57d72157ee2fc8b123d1f98d43e3cb6
```

这是检测器盲点的受控反例，**不是说原始正常报告已被证明错误**。临时报告已随隔离目录清除，替换函数已恢复，没有 optimizer step 或正式训练。

请先写能重现该形态的真实路径失败测试，再修复：回执须绑定参与同一次损失的有序商品、行为类型/三对实际 h，以及对应量化前向，不能用 used 补造缺失的消费字段。失败必须发生在写成功报告之前。复用现有采样/前向，不额外量化、不新增第二条 gather 梯度；保留正常、合法关闭/空样本和连接零梯度的语义。

旧 `test_q16_wiring.py` 的换序 helper 仍未重算摘要，仍会先因自摘要不符拒绝；新的真实换序测试已补上那一路径，旧测试不应继续被声称为消费比较证据。删除重复的虚假覆盖，或改成摘要自洽后确实针对独立消费记录的行为用例，不改钉异常文案。

## 4. 主线交付：不是再积累一轮诊断，而是完成可审阅的原生前缀规格

`prefix_surface.callable` 仍为 `NOT_IMPLEMENTED`，其 statements 仍是说明性伪代码；真实 `training_hooks._positions` 仍返回 `levels[-1]`。独立冻结、完整前缀实现和修复版收益均未完成，**不创建新 idea**。

下一次应交付可由独立 Formula/Math Review 核验的完整材料，而不是把当前 DRAFT 和局部参考再列一遍为“已就绪”：

1. 沿用 `question8.md` §4 的唯一目标：同一次量化产生三层原生累计码字位置；同一 TRAIN 三元组、同三个 h、共享 s、单个 λ；先三对平均、再三层等权平均。不得换成 hinge、next-item 或层权重选择。
2. 明确完整 E16×H16 距离的 cross-ratio、曲率/距离归一化、原生残差、右嵌套累计回读及各开关的计算域；把已有多维 H/Jacobian 正确证据与这些具体公式和同一次量化通路绑定。H1 平方差恒等式不是 H16 非欧效应或整个机制正确性的证明。
3. 区分已实现的末层项与待实现的前缀项；明确各层 encoder 单位 STE 与码本原生梯度的边界。离散选码仍交给基线 Sinkhorn 和既有接线检查，不用普通 argmin 替代均衡分配，不对 STE 伪造普通有限差分证明。
4. s 仍从初始化后的末层 TRAIN h=1 对一次标定，供所有层共用；λ 对**完整前缀目标**按 encoder / 合并全部码本两个块一次标定。写清退化规则及当前源码与规则的差异：`structure.calibrate_scale` 的空样本 `return 1.0` 仍在，不能遗漏为“均已闭合”。不沿用旧 checkpoint 的 s/λ、不引入试值或隐藏回退。
5. 连续参考必须能够对应实际目标域，包括闭合和分叉的三对 h。当前缩维适配器的 `h=1+tanh(u)` 与有限 box 抽样不覆盖精确 h=2；保留现有真实 h 的测试，把这项覆盖边界写清，不能据缩维适配器通过宣称完整行为目标已核对。

### 独立审查阻塞要有可核验的接口事实

本次只读核对到：

- `tools/research/agents/runtime.py` 仍有 Formula 的 `formulate` 与 `review_math` 角色和独立 MathReview payload；因此不能笼统声称仓库只剩 `specification.validate`。
- `server.py::_round_phase` 仍引用 `tools.research.rounds.worker`，其 runner 分支引用 `tools.research.agent_flow.run`；当前磁盘未见前者的模块/包或后者的包。这是具体的源端路由缺口线索，不是已发起真实 MCP 请求后获得的错误，也不证明所有合法 DAG 入口均不可用。
- 当前挂载的 round implementation/registration 工具要求真实 `round_id`，并不直接接受“两位 Math Reviewer”的请求。answer17 未提供本候选对应的实际 DAG/round 身份、合法请求与返回或两位审查收据；本轮也未启动这些请求。

请将状态拆开：角色实现存在、当前候选是否绑定合法入口、入口是否实测可用、独立审查是否执行。可达者经现有唯一 DAG/MCP 推进；确实不可达者保留 BLOCKED，给出准确入口、缺失前提和实际错误证据。**不恢复已删除流程，不自签冻结、不直接调用内部函数推进正式状态、不另建调度器。** 外部通道阻塞不妨碍先补齐上述可达数学材料；局部 schema 的 0 problems 也不能证明这些材料齐全。

独立审查和真实冻结完成后，再按 TDD 实现原生三层前缀，走现有绑定与 Stage2 机器准入。训练和收益分开记录：只跑一次冻结预算，不增加 Stage3 指标准入闸门，不做关闭臂训练、调参或多种子；本问本身不自动启动训练。

最终必须来自完整修复版本的真实 Stage3 评价。旧 `test_R@10=0.05790490781524748 < 0.065` 不移交给前缀版；局部参考或梯度通过不等于推荐增益。

## 5. 收据与结论边界

- 最终 31 文件全绿收据接受。第二份失败日志实际上已执行完 31 文件，汇总为 **30 pass / 1 fail**；“24 文件后失败”描述不完整。第一份为 29 pass / 2 fail。两份都保留，不重写历史。
- `red_reference_nan.log` 保存了旧公式 NaN 的诊断，rc=0 表示诊断脚本退出成功，不是失败断言的退出码。答复引用的修复前 `GradcheckError` 与换序 `AssertionError`，当前 q17 目录中未找到对应的独立执行收据；标明已有记录与 NOT_RECORDED，不事后补造。
- runner 目前只记录自身 SHA，没有执行时的模型源码/测试摘要。answer17 的事后摘要与当前磁盘相符，但不能倒推两份失败收据和最终收据各自精确执行了哪版源码。后续新收据绑定执行前/后的源码与测试身份，发生漂移即记录，不补签旧日志。
- 本轮两次初始 CPU 诊断各在 60 秒超时，均记 TIMEOUT、未获得判定；随后有界重试分别取得 §1 和 §3 结果。参考重试缩为每面一个 box 样本；真实消费重试未缩减模型/数据路径。未运行整个候选套件、独立审查、正式准入或训练，未修改实现、原始 answer 或历史证据。

## answer18 应交付

1. 待冻结权威声明、候选契约主面/子面和实际接口一致；同一数学的取值/梯度及合法碰撞证据。
2. 行为类型/h 的真实消费错配红→绿，拒绝时无成功报告；已修好的商品换序与合法分支不回退。
3. 完整前缀可审阅材料，以及独立审查入口、候选身份和真实状态；冻结、实现、准入、训练、增益逐项分开。未实现/未跑/缺收据/超时/外部阻塞分别注明。

## 审阅身份

以下摘要从本轮磁盘计算；它们固定审阅对象，不是正式冻结或准入凭据。answer17 的 14 条摘要全部匹配。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer17.md` | `4041067975274d562a459e353e2bbb6e0a0416d92cb159c30ce7ff74fd1770e8` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e04a17d8bad6160db886041206124f7a41d403cb04ad4a71dd0c2b3f40eff00b` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `e60b3eafabce613c05786cf7781e3c36530b2a20b716d99523c2fdbedcb4adb9` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `c91ed0d5bb72b1f6847f442a94aa7d3b422f8614d130b864e5c4beda310e6ff7` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `bff0ef340ad4cdfd85ba2cfb5168d7ec3702c293c0f00a25f449453c116b20c5` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `bbbb79caf9392829a87ff9d5494a113502445eabc8bd0831d652ee1149bfc3d6` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q17_consumption.py` | `de0904bb3ad446914fe310137a7353ced25b39f577069a93d53cf2a63b4e9c84` |
| `results/research/struct_mixed_rq/q17/tests_20261110T065916.log` | `30a9a7fe3e28436ecc7c2efa50538fad8846173b4e8603e07be8b73fc5c3f970` |
| `tools/research/stage2/reference.py` | `52eb29c0802d3809d3c969ca720f3a501a9d28364f4867de3f472223a1552416` |
| `tools/research/rounds/compiler.py` | `4fc4ef0dfc741deeb00a5fb89b983c752ecdbbd8361c2c1de4d211235ff15fcc` |
| `tools/research/agents/runtime.py` | `31a3ca4950c302803fb9bb171e77ce7c109eaf0a9fc4e58780b176af3047e642` |
