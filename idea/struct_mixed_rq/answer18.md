# answer18 — 权威面与契约同一、行为类型/目标绑定到实际消费，前缀材料补齐
**状态:DRAFT 仍未冻结;prefix NOT_IMPLEMENTED;独立审查 BLOCKED(入口缺失);绑定/准入/训练/收益 NOT_RUN**

承接 `answer17.md`。接受 question18 §1 对参考碰撞修复、消费换序测试、收据 31/31 与摘要一致性的核对。

## 1. 权威声明、主面与接口已对齐
逐项实测(本轮):
```
contract.calls[0].name == "structure_objective" : True
contract.calls[0].statements == math_spec.continuous_surface.statements : True
math_spec.continuous_surface.distance_surface.statements
    == contract.calls[name=mixed_distance].statements : True
```
`tools/research/rounds/compiler.py` 取 `calls[0]` 与 `specification.statements(spec)` 比较;此前 `calls[0]` 是 `mixed_distance` 而主面是结构项,故判 False。现 `calls[0]` 为 `structure_objective`,两句同一文本,按该判据比较为 True(未调用 MCP,这是纯函数检查)。距离子面此前仍留 `tanh→cross-ratio→asinh` 旧声明,现已同步为两项平方差,与契约逐条相同。
新增 `tests/test_q18_contract.py` 固定这两个等式(非文本搜索、非 schema),`rc=0`。

## 2. 行为类型与 h 的消费错配
**红**(修复前):`tests/test_q18_consumption_kind.py` 在结构入口保持商品与顺序不变,仅把第一条 `triangle` 改为 `wedge`,再调用原函数。生产者未拒绝、仍写出报告:
`AssertionError: a kind-only consumption change was accepted`(两次运行均如此,79s/155s)。这复现 question18 §3 的 `item_ids_equal=true, kinds_changed=1, actual_h_first=[1,1,2]` 形态。

**修复**:`structure_term(..., capture=...)` 现在一次记录**实际消费**的三元组、商品序列与 `triple_targets` 返回值:
```
capture.append({"items": item_ids,
                "triples": copy of the triples it iterated,
                "targets": h.detach().tolist()})
```
`probe_q9` 把捕获的三元组(含 kind)与采样 `used` 比较,不一致即
`RuntimeError("the gathered items differ from the sampled triples")`,**写盘前失败**;报告 `identity.triples` 与新增 `identity.targets` 均取自实际消费,不再用 `used` 补造 kind。
**绿**:kind-only 改动被拒且无 `factors.json`;商品换序(`test_q17_consumption.py`)与 `test_q16_wiring.py` 仍绿。已修好的换序、合法 `behaviour_off`/空样本与连接零梯度语义未回退。

**本轮工程缺陷**:首次接入 `capture` 后,`targets` 变量在 autograd 段被同名覆盖,导致 `identity.targets` 含 Tensor、`json.dumps` 失败;该收据(`tests_20261110T071811.log`,2 fails)原样保留。重命名后 33/33 通过。

## 3. 前缀可审阅材料已补齐(仍 NOT_IMPLEMENTED)
`math_spec.continuous_surface.prefix_surface` 新增:
- `full_space`:E16×H16 的径向球映射、向量交比与 `phi` 的完整描述;
- `residual_and_readback`:同一次量化、running residual 与右嵌套累计回读,禁止重新量化或第二条 gather;
- `ste_boundaries`:encoder 的 straight-through 梯度 vs 码本原生回读梯度,Sinkhorn 仍为唯一选码;
- `calibration`:`s` 是初始化后末层 h=1 对的中位距离、全层共用;`λ` 对完整前缀目标按 encoder / 合并码本两块一次标定;并明确 `structure.calibrate_scale` 空样本仍 `return 1.0` 是已知退化默认,不是已闭合;
- `coverage_gaps`:缩维适配器的 `h=1+tanh(u)` 与 box 抽样不覆盖精确 `h=2`(由 `test_structure.py` 的真实目标覆盖);H1 恒等式不证明 H16 非欧效应;该前缀在训练器中未实现。
`specification.validate` 仍 0 problems(浅层),不当作材料齐全的证明。

## 4. 独立审查入口的可核验事实
| 项 | 状态 |
|---|---|
| 角色实现 | 存在:`tools/research/agents/runtime.py` 仍有 `formulate` 与 `review_math` 分支 |
| 当前候选绑定合法入口 | `NOT_RUN`:没有本候选的 round_id,未发起 MCP 请求 |
| 入口实测可用性 | 失败:`server._round_phase` 执行 `from tools.research.rounds import worker` 得 `ModuleNotFoundError: No module named 'tools.research.rounds.worker'`;runner 分支引用 `tools.research.agent_flow.run`,该路径不存在 |
| 独立审查执行 | `BLOCKED`:无两位审查执行收据,不自签、不恢复已删流程、不直接调用内部函数推进状态 |

冻结材料(`math_spec`、`reference_contract`、碰撞与梯度测试、双类消费测试、q18 收据)已列齐;冻结本身仍 `BLOCKED`。

## 5. 收据、来源与更正
- 最终收据 `q18/tests_20261110T072312.log`:33 条命令、33 个不同文件、**33 pass / 0 fail**,runner `rc=0`;`source_drift: none`;执行前/后源码清单保存为 `q18/source_pre_*.txt` 与 `source_post_*.txt`,runner 记录自身 SHA。
- 失败收据 `tests_20261110T071811.log` 保留(33 文件 31 pass / 2 fail,原因为 `targets` 覆盖导致的序列化错误)。
- **更正 answer17 §4**:第二份 q17 失败收据实际是 **31 文件 30 pass / 1 fail**,不是"24 文件后失败";第一份为 29 pass / 2 fail。
- `red_reference_nan.log` 的 `rc=0` 是诊断脚本退出码,不是断言失败码;修复前的 `GradcheckError` 与换序 `AssertionError` 只见于会话输出与测试重跑,未保存独立收据,记 `NOT_RECORDED`。

## 6. 状态表
| 项 | 状态 |
|---|---|
| 权威声明/契约/接口一致 | 已对齐(纯函数检查);DRAFT |
| 碰撞参考取值/梯度 | 已修并复跑 |
| 行为 kind/h 消费绑定 | 已修(red→green),商品换序不回退 |
| 前缀实现 | `NOT_IMPLEMENTED`(末层结构项仍在跑) |
| 独立审查/冻结 | `BLOCKED`(worker/agent_flow 缺失,无 round 身份与审查收据) |
| 正式绑定/准入 | `NOT_RUN` |
| Stage2/Stage3 与增益 | `NOT_RUN`;旧 `0.0579` 未移交、未达 `0.065` |

## 7. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `8ef9d712f1784919e1b249dd2e59df86f4e768c61f2daa5a4a364ba2e1432451` |
| 候选 `reference_contract.json` | `41a92152df616189f3dfa36d33a9a2512ba876c703559da35a600248eadf7e51` |
| 候选 `probe_q9.py` | `98106737c7458ae940774ad447f917a55482f533e155e9f936809a88a9cafdf3` |
| 候选 `training_hooks.py` | `a846028d5c4f1fcb7137891872802e71f9b9a5133cb32f8f79de96c434ac1395` |
| 候选 `tests/test_q18_contract.py` | `82964bf1d44b0540d510088c4ab728a3fdb89ab51404b92d0a4604fabb9a7eca` |
| 候选 `tests/test_q18_consumption_kind.py` | `7e036cf241fc025f3446ef957136f34ba22df66192a7f871795c123c630c55f9` |
| 证据 `q18/tests_20261110T072312.log` | `709dad7f6d3353ab99d292386fe52ffc7a8eeb4ca016fc9e8cd058d920d49564` |
| 证据 `q18/run_suite.sh` | `bda40f6d27b03a6ef476d63f81ff0fc3f9e2199d3c534b86a96bcbc613c67b4d` |
| 证据 `q18/tests_20261110T071811.log`(失败,保留) | `56b117aaa66fc925dd1890d2324537e6423cc1374c2807b353d277d13ac101bb` |
| 证据 `q18/source_pre_20261110T072312.txt` | `359b3f083763d1e5acc992fdd9262f5570550f1fa38757419c99f98c9b7cfc0c` |
