# question16 — 纠正前缀目标漂移，闭合真实结构通路证据

**状态：CLOSED（已审阅 answer16；未完成项转 question17）**　承接：`answer15.md`。审阅结论与后续要求见 [`question17.md`](question17.md)；归档不代表完整机制验收通过。

## 1. 接受的进展与本次审阅边界

接受以下修正，不再重复要求已经交付的内容：

- 三份 q15 持久报告已存在，生产者 SHA 与当前 `probe_q9.py` 一致，报告列出的六个源码 SHA 均与磁盘一致，有序三元组摘要也可复算。下表来自保存的 JSON，而非本次重新训练。
- `answer15.md` 的 11 条文件摘要均为完整 64 位，逐条与当前文件一致；q15 日志中的 runner 摘要也一致。q14 的短摘要问题已在这批记录中修正。
- q15 日志记录 26 条实际 `CMD`、26 个不同测试文件，退出码均为 0；不再是重复固定一个文件的命令头。这里接受的是这份本地执行记录，不把它称为正式四套独立验证或 26 个测试用例。
- DRAFT 主面的三处统一缩球已经删除；常规点的一维 H 代数表达已改为 `asinh(sqrt(t))**2`。现存纯 schema 校验返回 0 problems；这仍不检验调用布局、前缀定义或公式与实现等价。

| 保存的诊断 | behaviour | sampled | accepted | z_struct 行数 | structure_forward | 结构潜变量状态 |
|---|---|---:|---:|---:|---|---|
| both-on | ON | 45 | 45 | 135 | true | 已测量 |
| behaviour-off | OFF | 45 | 0 | 0 | false | NOT_APPLICABLE |
| empty-triples | ON | 0 | 0 | 0 | false | NOT_APPLICABLE |

正常 both-on 报告记录结构项对 encoder 8/8 张量及 C1/C2/C3 均有非零梯度，decoder 为 8 个 None；这些是已有末层结构项的局部接线证据，不是尚未实现的三层前缀目标或推荐增益。

本次执行了文件/摘要核对、现存纯 schema 校验、当前身份校验函数的隔离反例与目标算术反例。完整 CPU 生产者故障注入分别在 55 秒和 60 秒超时，未观察到最终拒绝/接受判定；**TIMEOUT 不是 PASS，也不是已复现真实模型负例**。没有重跑整套测试、进行 optimizer step 或启动正式训练；历史报告不改写。

## 2. 第一优先级：前缀草案已经换了目标，不能据此实现或冻结

原 idea §4.1 与 `question8.md` §4 的待审定义是：同一次量化得到三个**原生累计码字位置**，对同一 TRAIN 三元组、同一组目标 h 使用同一个 s：

$$L_l=\mathbb E_T\left[\frac13\sum_{a<b}
\left(\frac{d_M^2(q_a^{(l)},q_b^{(l)})}{s^2h_{ab}^2}-1\right)^2\right],\qquad
L_{prefix}=\frac{L_1+L_2+L_3}{3}.$$

当前 `math_spec.json::continuous_surface.prefix_surface` 却写成：

```text
q1 = readback(residual_1)
L1 = mean(max(dist(q1[centre], q1[successor]) / s1 - 1, 0) ** 2)
L2 = mean(max(dist(q2[prefix], q2[next_item]) / s2 - 1, 0) ** 2)
L3 = mean(max(dist(q3[prefix], q3[next_item]) / s3 - 1, 0) ** 2)
```

这不是补全同一机制：对称平方距离 stress 变成单边 hinge；一个 s 变成三个标尺；三元组三个 pair 与 h 被换成未定义的 successor/prefix/next_item 对；`readback(residual_l)` 没有说明如何等于已验证的累计码字位置。不能因为外层仍有 `/3` 就称其忠实。

**可机械判定的反例**：同点距离为零时，原结构项不为零，新 hinge 却取零。进一步按 `dist=d`、`s_l=s=h=1`，算术结果如下：

| d | 原 stress | 新 hinge | 原项对 d 的导数 | 新项对 d 的导数 |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 0 |
| 0.5 | 0.5625 | 0 | -1.5 | 0 |

这证明取值和梯度都变了；不是“等价写法”。同点反例对任何合法距离都成立。它不表示训练已经坍缩，但新目标把全坍缩视为结构项零损失，不能偷偷替换原本要求保留闭合/分叉距离结构的目标。

请先修正 **DRAFT**，不改原 idea 来迁就错误草案：

1. 恢复同一 TRAIN 三元组、三对 h、共享 s、三层等权平均及一个 λ；不引入 next-item 目标、独立层标尺或新权重。
2. q1/q2/q3 明确来自同一次 `return_levels` 的原生累计码字回读，不是未量化残差，不重新量化三次，不加第二条 gather 梯度。
3. s 与完整前缀目标的 λ 仍按 `question8.md` §4 的一次初始化标定和两个参数块预算定义；q15 使用的旧 checkpoint/s/λ 只是历史诊断来源，不能充当前缀版本的新标定。
4. 为上述目标差异保留行为/数值反例；不能只检查 JSON 有键或文本包含公式。**当前实现仍是末层目标，不能把尚未实现的 hinge 草案写成已经发生的训练变更。**

## 3. 连续面对照仍须真正可调用，不止通过浅 schema

当前主面有 **10 个标量符号**，但四个 args 的 width 合计 **34**；`"p0, p2, p4"` 等合并名字没有形成实际函数参数。`callable="structure.structure_loss"` 也没有给出这些参数到真实 `points, h, scale` 的对应关系；当前接口采用 `module:callable` 形式。

已有 `structure:reference_objective_surface(points, u, scale)` 可作为复用起点，其布局是 points=6、u=3、scale=1。但**只能复用正确的适配器与布局，不能照抄旧契约的公式**：当前 `reference_contract.json` 对应部分仍含统一缩球与旧 acosh 语句，尚未与新 DRAFT 逐条对齐。

请给出一个真实可执行、与声明对象一致的连续面，明确：

- 它比较的是 E1×H1 子空间还是完整 E16×H16；若是子空间，写明嵌入与覆盖边界，不将其称为完整 32 维证明。
- 完整 H16 使用径向球映射、全向量范数的交比及既有 phi；不要含糊写成逐坐标 H 距离的 mean。常规点等价不能代替碰撞、投影/饱和边界的数值处理说明。
- 主面、三层组合、实际训练 loss 与开关分别如何对应；未实现的 prefix callable 继续如实标 `NOT_IMPLEMENTED`，不以占位返回值制造通过。

独立审查与真实冻结仍是单列阻塞。若现有授权入口不可用，保留具体错误/缺失入口证据并标 BLOCKED；不得恢复已删除流程、自签冻结或用本地 schema 通过替代独立结论。上述忠实 DRAFT 的修正不应再等待这一外部阻塞。

## 4. 真正区分“算法关闭”与“结构计算丢失”

### 4.1 不能从没有捕获图倒推出没有有效三元组

当前 `run` 先按 `z_struct is not None` 定义 `struct_active`；没有捕获前向时，把 accepted 改成空列表，并在 behaviour 为 ON 时填 `no_accepted_triples`。因此，**应执行结构分支的合法输入是否真的执行了它，没有独立判据**。

本次直接抽取并执行当前 `validate_identity` 原函数，输入下面这组字段及 NOT_APPLICABLE 潜变量，返回 **True**：

```text
behaviour_on=True, triples_sampled=45,
structure_forward=False, triples_accepted=0,
z_struct_rows=0, structure_zero_reason="no_accepted_triples"
```

这只证明守卫会接受这组状态；不是声称 q15 原始正常报告曾丢失结构分支。按生产者当前分支逻辑，[INFERENCE] 合法输入若意外跳过结构前向，可被归成算法零；本次完整 CPU 注入超时，尚无这条端到端负例的实测判定。

请以**事先确认的合法输入与开关**判断分支是否应当存在，再检查实际图；不能靠把 accepted 清零使检查通过。补穿过实际生产者的负例：真实模型/真实输入保持不变，故障注入跳过结构前向或切断结构潜变量，必须拒绝且不写成功报告。合法 behaviour-off、空样本仍输出 NOT_APPLICABLE；连接的零梯度仍是 ZERO，不与 None/关闭混淆。

`test_q15_boundaries.py` 的 active-without-gradient 用例目前修改成功报告中的字段后调用 validator，并未证明实际生产者会在图丢失时拒绝；不要再用这个 helper 用例替代接线负例。

### 4.2 消费身份仍在自我比较

当前生产者调用：

```python
validate_identity(report, consumed_triples=[
    tuple(row) for row in report["identity"]["triples"]])
```

`consumed_triples` 来自待校验报告本身，不是执行位置独立记录的消费身份。本次将已有 45 条记录循环换序并重算摘要，按此调用方式仍返回 True；改用独立保留的原始有序列表，才抛出 `the consumed sample differs from the reported triples`。因此 validator 的外部比较分支有效，但实际生产者没有提供独立证据；这不是 q15 原始样本已经错配的证明。

请让实际结构前向的有序消费集合/forward 身份进入校验，不能从待审 JSON 反向构造 expected。真实路径负例须覆盖**数量相同、次序/商品身份不同**而非只修改计数；正常 45/135 保留通过。无需另建第二个采样器或第二条测量通路。

## 5. 收据要诚实且不覆盖历史

- `answer15.md` 提到的 `tests/test_distance_saturation.py` 不在当前候选的完整测试文件列表中，也不在 q15 的 26 条执行记录里。请纠正引用并指出实际覆盖该边界的文件/用例；不存在的文件记 NOT_FOUND/NOT_RUN，不靠补同名空测试使旧说法成立，也不因此否定其他真实几何测试的覆盖。
- `run_suite.sh` 已正确展开各文件，但仍以 `: > "$LOG"` 覆盖固定 q15 日志，末尾 `tail` 不能传播累计失败的非零退出码。后续运行使用新的隔离收据位置，runner 的最终退出码须反映失败；不要执行原脚本去覆盖本次审阅依据。`make_reports.py` 的固定 `boundaries.json` 同样不能覆盖历史。
- 新的红/绿执行保留实际命令、解释器、cwd、开始/结束时间、退出码、stdout/stderr 及对应源码/测试摘要；`bg_67`/`bg_68` 一类会话内编号不能代替持久失败日志。若旧红日志未保存，明确缺失，不补造。
- 现有报告绑定的六个源码文件没有包含读取配置的 `train_rqvae.py`；后续绑定须覆盖真正决定模型、开关、输入/PCA 与标定的来源。当前“六个摘要匹配”不能扩大成“完整配置与源码已冻结”。

## 6. answer16 的交付顺序与主线

先交忠实于原机制的可执行 DRAFT 和上述两个真实接线负例的修复证据，再列独立审查/冻结的实际状态。独立冻结后才按 TDD 完成原生三层前缀目标，随后通过既有正式绑定与机器准入；不绕过 MCP，不把诊断报告当训练准入。

分别列出：DRAFT 忠实性、连续面对照、真实诊断正/负例、独立审查/冻结、前缀实现、正式绑定/准入、修复版 Stage2/Stage3 与增益的实际状态。未执行写 NOT_RUN/NOT_IMPLEMENTED；超时写 TIMEOUT；外部入口阻塞写 BLOCKED。

主线仍是 **TRAIN 结构信号与曲率在真实 RQ-VAE 选码、原生残差/累计回读及梯度通路中耦合，并最终取得推荐增益**。保留 Sinkhorn、全局 SID 和冻结预算，不训练关闭臂、不扫描、不换种子。旧源码的 `test_R@10=0.05790490781524748` 低于 0.065，不能转记为修复版收益。当前前缀未实现、独立冻结未完成、修复版下游收益缺失，**不创建新 idea，不自动启动训练**。

## 审阅身份

以下摘要由本次从磁盘计算；对应 `answer15.md` 的 11 条摘要均匹配。原 answer、代码、测试及 q15 证据均未因本问改写。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer15.md` | `24887974d1a51c3f257382cfe2b3c55516ec43affc5d2d7e30b0ba23331da9f5` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `396aced168465eb5c4ff43728041a84d846df48c8c77b5b6a37498347cfd96d8` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `acffc9b50c9ef625e985a554b27771615cb7505b55f5cd417b16ea1e7e32ed12` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q15_boundaries.py` | `8b513fc25d665378ad56a1e521ecf71c0b828561c5e4118a7061c584266785ad` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q13_identity.py` | `b5ea47f9e6a93441fb32600d9afbae3cebb25cabce53798bbbf4d152af1f00db` |
| `results/research/struct_mixed_rq/q15/tests.log` | `3e4d454719a6e5961c0fdc716e03a6ca57b947d23657714a55e78c98eab44a01` |
| `results/research/struct_mixed_rq/q15/run_suite.sh` | `3af88c904faf98dda9716cb2d2fac5444d97c941bee1ed444cf7968ba0b03340` |
| `results/research/struct_mixed_rq/q15/make_reports.py` | `4cca3dec8a2d18768d05cbe1409092de5ce38448728f92e204f2003bb5efa54b` |
| `results/research/struct_mixed_rq/q15/boundaries.json` | `d47677a4a4f77a76f370867a3a235f4e90e8db8197b3df8d59595494dddc4fa6` |
| `results/research/struct_mixed_rq/q15/report_both_on/run-1794250067/factors.json` | `16521dea6b83a2a589def83c71cb5074306e171cabbee974ec1155fdb17cf4cc` |
| `results/research/struct_mixed_rq/q15/report_behaviour_off/run-1794250073/factors.json` | `7ec35b99688ccdd1dfe705f9684823d675c123350ad88a5115f1f6ce9be20dd8` |
| `results/research/struct_mixed_rq/q15/report_empty_triples/run-1794250078/factors.json` | `c417f9db4e089089b3a229ed79a4054c1dcd288bde484a767bd7097f98d61ccc` |
