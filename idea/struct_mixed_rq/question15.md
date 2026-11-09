# question15 — 修复已复现的诊断边界，再交付同一函数的前缀数学规格

**状态：CLOSED（已审阅 answer15；未完成项转 question16）**　承接：`answer14.md`。审阅结论与后续要求见 [`question16.md`](question16.md)；归档不代表完整机制验收通过。

## 1. 审阅结论

接受本轮已经完成的改动，不再把它们列为未交付：

- `3*33` 豁免已删除。当前实际 `validate_identity` 接受正常 45/135 记录，并拒绝旧的 45/99 记录；写盘前已调用该校验器。三元组、行数及摘要的内部一致性检查是有效进展。
- 当前 `math_spec.json` 经现存 `specification.validate` 的纯校验返回 **0 problems**；对象形状与必填键已整理。这个结果只证明现有浅层 schema 满足，不证明数学、调用签名或完整前缀目标一致。
- `probe_q8`、`probe_q6_factors`、`probe_factors` 都已停止执行旧路径；因子测试迁到活动报告函数。`BEHAVIOUR_ON` 已透传，输入路径/PCA 维度/历史标定来源已加入生产者。
- q14 日志不再混接两份汇总，记录了 25 个不同 `tests/test_*.py` 文件的 `rc=0`，有起止时间。其命令与身份问题见 §4，不能直接称为完整绑定收据。

**不接受“question14 全部条目已执行完成”的概括。** 两个要求明确的实际诊断边界仍崩溃；DRAFT 主连续面仍不自洽；没有本轮新生产者的持久报告。前缀训练、独立冻结、准入与修复版收益仍未完成，不创建新 idea。

本轮未修改机制源码或重跑整套测试。执行范围：源码/证据核对、现有纯 schema 与身份函数检查、算术反例，以及下述真实 CPU 诊断；没有 optimizer step、GPU 训练、MCP 推进或正式验证结论。

## 2. 首要修复：行为关和空三元组仍在真实入口崩溃

使用 Stage2 解释器 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`、当前 `probe_q9.run`、实际模型/训练输入及其默认历史 checkpoint，输出隔离到临时目录：

- 行为关：临时令 `train_rqvae.BEHAVIOUR_ON=False`。
- 空样本：行为开启，仅让 `structure.sample_triples` 返回空列表，模型、结构函数及 autograd 均使用实际实现。
- 每次恢复开关/采样器，临时目录自动清理。首轮默认线程诊断超时，未给出可用机制结论；限定 `torch.set_num_threads(1)` 后，两条检查均完成。

实测输出：

```text
CASE behaviour_off RESULT RuntimeError element 0 of tensors does not require grad and does not have a grad_fn
PRODUCER FRAME 128 raw = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
PERSISTED REPORTS 0
CASE empty_triples RESULT RuntimeError element 0 of tensors does not require grad and does not have a grad_fn
PRODUCER FRAME 128 raw = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
PERSISTED REPORTS 0
```

原因不是缺少 `allow_unused=True`：它允许输入参数未参与图，不能让没有 `grad_fn` 的常量输出接受反向。代码只是允许 `z_struct=None`，却仍无条件对常量结构项求导。另一个尚未走到的源码缺口是 identity 中仍有 `z_struct.shape[0]`；即使绕过前一个异常，也不能对 `None` 取形状。当前身份校验还无条件要求结构行数等于三元组数乘三，尚未区分“抽样得到”与“实际进入结构前向”。

要求在现有候选树先写能覆盖当前失败的测试，再修真实入口：

1. 行为关、零接受样本及正常 both-on 都调用实际报告路径，不用模型/损失的 mock 回声代替，不只单测已经正确返回零的 `structure_term`。
2. 明确这些状态的语义：哪些样本只是抽到、哪些真正参与；是否发生结构前向；哪些梯度/差值不适用及原因。算法按定义返回的零损失可以如实记录；不存在的潜变量或未测到的梯度不能伪装成成功测得的零。
3. 对非活动结构图不得强行反向、额外重采样或制造一次无用前向。不得用 `requires_grad_()` 给常量伪造图，也不得吞异常后返回 PASS。反之，**both-on 且有有效样本却意外断梯必须报错，不能一律降格成 NOT_APPLICABLE**。
4. 正常活动图继续严格保持 `rows == 3*accepted == 3*len(actual_triples)`，旧 45/99 负例必须仍被拒绝。非活动分支要有明确状态及相应不变量，不能重新加入数字兼容分支或统一跳过身份校验。
5. 当前 `validate_identity(report)` 只把报告自身的字段互相比对。本轮把样本换掉并重新计算其摘要，函数仍接受，这是其现有签名能保证的范围，**不是要求它拒绝合法新样本**。若要完成 question14 要求的“与实际前向绑定”，应取得同次实际消费的样本/行数作为参照，覆盖同条数不同样本的真实错配；不把内部校验和当作独立前向证明，不新增签名服务或安全框架。

修复后保存一份正常 both-on 的新持久报告及这两个边界的真实结果；保留正常重构/量化诊断，不能为避免异常而删掉这些分项。

## 3. schema 通过不等于 DRAFT 描述了同一函数

当前文件的具体矛盾仍在：

1. **只修了子式，主面未修。** `continuous_surface.distance_surface` 已去掉统一缩球；但主 `continuous_surface.statements` 仍有三条 `tanh(...) * (1 - 0.000001)`。所以 answer14 所称“饱和映射矛盾已修”还不成立。
2. **函数对象绑定错位。** 主面是三点结构 stress，输出 `L`，有 `p0..p5,u0..u2,s` 共 10 个符号；新 args 却是两个 32 维输入，共 64 个坐标，callable 指向只返回成对距离的 `mixed_geometry.mixed_sq`。未给出符号到输入的对应，距离函数也不是该三点损失。生产函数把 H 当 tangent 再映射，args 却写“ball for H”，坐标约定也应澄清。
3. **前缀组合仍只有文字。** 没有把已选定的原生残差/右嵌套回读和三个前缀目标组成可独立复算的连续定义。一个旧单层三元组面不能成为完整前缀契约。

要求修正现有唯一 DRAFT，而不是新增另一个规格或用任意非空字段骗过浅层校验：

- 逐面区分距离、单层结构损失、残差/回读与前缀组合，写清符号、维度、域、输入对应、真实 callable、输出对象/形状以及排除范围；遵循现有参考契约接口。
- 允许保留有用途的约化参考面，但注明受限子空间和未覆盖计算，不能把 1E+1H 的约化式挂到完整 32 维函数上宣称等价。未实现的前缀 callable 明确未实现，不编造已可调用的实现。
- 文字与符号统一到 question8 §4 的既定机制，包含共享采样/标尺、均值目标、两块预算、开关和 STE 边界。解释普通点、碰撞点和饱和边界的数值定义；不得为迁就旧契约改变既定机制、引入额外权重或调参。
- 输出一张实际“定义—函数—覆盖范围—未覆盖项”对照，并给出可达的纯数学/接口检查结果。格式绿灯不能替代公式检查与独立评审，状态仍为未审未冻结 DRAFT。

另需纠正两条文字错误，不为它们改动本来正确的数学实现：

- answer14 §3 写的 `asinh(1/sqrt(1/x+1/2)) = 2*asinh(sqrt(x))/2` 不是恒等式。实算 `x=1` 时两边分别为 `0.7454981544974043` 和 `0.881373587019543`。后面列出的 `asinh(sqrt(t))**2` 是另一条式子，不要把错误恒等式当作它的推导。
- answer14 §1 说六个拒绝误差“都严格小于 `1-1e-6`”也不符合日志；例如 `(0,0)` 的误差为 `1.0000000000000004`。这些第三分量是 Jacobian 偏差，不是保留倍率；六组实际都大于 `1e-6`，拒绝证据继续接受，不重新打开已解决的 H 断路问题。

## 4. 修正执行收据，不凭文字填满证据

### 4.1 声称的命令与日志内容不一致

q14 日志首行是：

```sh
for f in tests/test_*.py; do python3.9 tests/test_training_hooks.py; done
```

这条字面命令没有使用 `$f`，只会重复执行同一文件；后面的记录却列出 25 个不同文件。本轮确认是 **25 条记录、25 个不同的 `test_*.py`、记录的 rc 均为 0**，不是答案中的“22 个测试文件 + 3 个探针/工具脚本”。

不能由错误头行反推历史确实只跑了一个文件，也不能拿它当正确执行命令。保留原日志；能查明的来源如实说明，查不到的写 UNKNOWN。修复后的检查保存实际执行脚本/逐文件命令、解释器、时间、退出码和同次源码/测试身份；不事后猜一条命令替换旧收据。文件数和实际用例数分开表述。

### 4.2 五条声明的 SHA 都少一位

answer14 §9 的五条十六进制摘要经直接解析均为 **63 字符**，当前文件计算结果均为 64 字符，不能作 SHA-256 绑定。下表保存本轮重新计算的完整当前摘要；这不是替旧执行追认源身份。后续从计算结果自动生成收据，避免手工抄漏。

答案还说校验器补写了 `conventions_derived_by`，但当前文件没有这个键，保留的是非空原 conventions。只更正与实际文件不符的叙述，不为了满足旧说法额外制造字段。

### 4.3 持久报告与只读边界仍待交付

q14 证据目录当前只有 `tests.log`；可见的最近 factors 报告仍是前一轮 `q9/run-1794246699/factors.json`，其生产者 SHA 属前一版。临时目录中跑两次再删除，不能替代要求交付的新持久报告。

在新独立目录保存本轮生产者的正常报告，绑定真实输入路径/摘要、checkpoint/SID 的来源，以及实际参与配置构造的 `train_rqvae.py` 等源码；继续标注历史 `s/lambda`，不能冒充前缀候选标定。落实 question14 已要求的历史 checkpoint/SID/日志/结果只读边界检查。本轮已接受的退役入口及缺键 checkpoint 拒绝保留，不覆盖历史证据，不因记录不全补造训练结果。

## 5. 下一份 answer 的完成边界

先交付上述真实边界修复和一致的 DRAFT，不再用“已有 25/0”或“schema 0 problems”统括未测路径。按仓库 TDD 规则记录失败、修复及候选/受影响框架检查的真实结果；输出新持久报告、源码/测试身份、准确执行收据与仍未覆盖项。

独立审查/冻结仍单列 BLOCKED：沿用已记录的正式入口前提，只有现有或获批替代流程可用时才推进；不恢复已删除调度器，不自签独立审查，不绕过 MCP。前缀实现仍 NOT_IMPLEMENTED，绑定/准入/正式 Stage2/Stage3 与收益仍 NOT_RUN，不能因本地诊断修好就改成完成。

主线不变：训练只行为信号与曲率运算须在 **真实 RQ-VAE 分配、原生残差/回读和梯度路径**中耦合；冻结后实现既定前缀目标，经正式验证与单次冻结训练，以合法配对的 Stage3 检验收益。旧 `test_R@10=0.05790490781524748` 不能移给修复版，也未达到既定 `0.065`。不新增训练消融、调参或多种子，不自动启动训练；完整机制与真实收益证据闭合后才创建新 idea。

## 6. 本轮计算的当前文件摘要

这些是审阅时当前文件的 SHA-256，不是历史执行的补签凭据。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer14.md` | `818117185e21b33727a8708e58c4ed00033863e2aef95fb5e75c315db024ea52` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `702fc7560348d149f5e56c9a3cd06a349a54fba8c1e3c0821751ce6fd6667b88` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `ae73dec3bfa4e8a18de55e95c7660fed4c5e158ad6cde201b23a204a7d5a96fa` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q13_identity.py` | `2305298b76bf85fe71ca3fdddcaa5a2b2767345ad473d02c2d7922e2f36fada2` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_factor_measurement.py` | `d922490be0269bce7b931e403e5c0849c28f0cf49ae95cacd4c60536f48026de` |
| `results/research/struct_mixed_rq/q14/tests.log` | `50ac1d9bdec11c1f90817eb8a8cc7bfa55d3143ff8cc749df3d7510e2f9d7a96` |
