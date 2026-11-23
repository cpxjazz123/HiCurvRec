# Issue 251 — BLOCKED：冻结资产与规范图合同不相容

## 判定与已交付内容

**不是COMPLETED，不声明准入通过。** 实现、真实入口、四卡/全目标回传、独立评估、重导出与可修复工程故障均已完成；唯一未满足的C7是机器行为图门槛。继续满足它必须由owner修改冻结合同：不能自行重建#248/#249资产，也不能绕过§89b。

源码与实际运行证据已提交/推送：`a529589d71bb936cba244cc62d9f5a6fbab51362`；推送后本地main与远端main实测同SHA，远端为`https://github.com/cpxjazz123/HiCurvRec.git`。模型源为cf0fd50c，MCP缓存修复为860c9ffc；最终代码身份按`FINAL_SOURCE_BINDINGS.json:de1cf7273fb2c31e433e7779e4efa98940701af33321f122ff321401cffc8c6a`的135个逐文件SHA判断，不用分支名或旧结果替代。

**无正式训练**：issue目标明确“仅实现、验证与交付可训练候选，不启动正式Stage2/Stage3，不重训TIGER”；C7只允许验证性短运行。已完成四卡受控单微步及真实一epoch入口，没有3000-epoch派发、Stage3或R2四件套上传任务。

## C1–C9逐条对账

下表候选源码前缀均为`stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/`。实际命令、原始stdout、退出码/机器判决见对应日志、COMMENT_001–023和CHECK_4。

| 条目 | 实际值/行为 | 最终生产代码与证据绑定 | 判定 |
|---|---|---|---|
| C1 审计/冻结 | 9项原文；67既有资产复用，仅新增v6声明式token schema；未改冻结基线/原损失/容量 | CONTRACT SHA `4c4958596a2b40a2bf177e29780d5eeda199e5d2e3495aeafafd4c265b204892`；manifest `23a0a6e2c28795c9b768d702e048c5bbd1fe31f749ad0ed3860709f2ef457463`；INFRA `f9110db2b690cc357e4a80d5bc760f402edb1872f761b6ce1b8da68df4fecef9` | PASS |
| C2 多token | 全24587商品、每项两个完整有向ego readout，覆盖1；padding/null/重复/孤立点/分片有实际断言；未称全部Graphlet | `model/structure.py:09d59072ce77e869b505423dbd00e9c529bd97f1cb8f1a1d6691c71f9fdc8064`；RUNTIME_SUMMARY `d00f5d545e7f6cf2777378fcc17a8326829747ed2e4a9b5dedcd3ae4738a5e72` | PASS |
| C3 Cross-Attention | 独立手算Q/K/V、两个Softmax轴、mask/顺序/单token/独立Query梯度与有限差分通过；实际保存attention和w | `model/attention.py:a227155453e1d7c79cb9cf5c59a8a5e3f7defa88895d784ae499a34d25d42ac6`；`model/structure_gate.py:74e307fb00f72cc847c2ed66f3bcfa329e55243960537a6fec89ec4ccdf50802`；UNIT_MEMORY_BOUND `fc669b7600e32624f9a72d227aaad2f0a26ec78efbe9852ebf7eeb7659bc1635` | PASS |
| C4 几何/硬码 | 五几何独立参考、零/重合/饱和边界、原硬规则/平衡分配、行为关/曲率关、实际attention干预改变硬码字；q逐位等于三个选中码字和 | `model/geometry.py:1478e2cc92dbfa13478b3eecc9a74c15a1e33a0338c8fdb0b72339324d6c1d9e`；`model/selection.py:aaf40e7d14eced220dae00f6f2aa14cb33f5d911c41fb94b8d868dcecf32d524`；SID `8ce6603faf07d7c0519722525d37e3c65c25fe42d092746400a726769835a339` | PASS |
| C5 全参考损失 | 全28391948合法行/24326中心，完整组件参考，实际回传loss0.9066628217697144；208.705秒，0优化步；DDP实际不同中心梯度另做均值对账 | `model/curvature.py:6a4204bca7915ec58f679db2554631c5e9bb40af59a21488b452f59e275bc9ad`；RUNTIME_SUMMARY SHA同C2 | PASS |
| C6 真实梯度/更新 | 32行分模块原/曲率/加权范数与方向、裁剪和更新；8组均更新；曲率到7组非零，Decoder0/None；成本切断负向通过；微步改变24580项硬SID | `model/mechanism.py:337765e1a081f5778db1f5588cea5a697073e4f28ed21ace1cc8efd1fc224b19`；MODULE_CHECKS `815c8baf6de3b1907490dd760a9abb2e0ad4150e7b4509b56e40f9d21de1e0e8` | PASS |
| C7 全类验证/准入 | 数学/四卡/真实入口/导出/数值/饱和/分块/源与数据漂移/#249均已执行；公开entry PASS，但behavior_graph FAIL，整体passed=false | `tools/research/stage2/graph_contract.py:f849b585c7fcaddb8a2c9a5e125865d04996928ff057976db92874be1aafb907`；ENTRY_PUBLIC_RECEIPT `12675d046954f62e3dab7ed81fae02a29a3b14111ea857b436c2bb47c8e6f87f` | BLOCKED |
| C8 独立评估 | q/z各全量28391948行、10个strict bank与各自decoder重建；宏/微/分符号/每商品分布/失真/三码/占用俱全；历史TIGER几何不可溯源标NA | `tools/issue251/evaluation.py:7ed9a6c3b26fe121f0c0fea737b200fe4c738f812832e4b252945cf568f8fb83`；runner `2e5605196fda424f0c6b03f76ebdbf10fff24dad521fdb624c96e05b42e51803`；EVALUATION_SUMMARY `0592fc47c94ab30d1dba3b4c9e2f7ffffdbf0fb3016778b10869db925c0ccaef` | PASS |
| C9 证据/交付 | 配置、源、契约、原始rank报告/SID、源/工件SHA、日志、完整指标摘要及重载证据已推送；生成二进制/完整每商品表保留共享盘，摘要绑定SHA与重算命令 | code/evidence commit `a529589d71bb936cba244cc62d9f5a6fbab51362`；REPORT `34249b676816a787c1d25771de659d8c270b1f9a4d90befee6919e19373590fa`；REPORT_RECHECK `9d96fe1a2928ff66a67e3865e2ee7003272022ba0c6d5115fcd3efe216bb5cf3` | PASS |

计数：**8 PASS、1 BLOCKED、0 NOT_RUN**。C7没有改写成PASS或NA；关闭采用硬合同冲突的BLOCKED路径。

## 七项机制与真实执行

| 机制 | 可执行/持久化证据 |
|---|---|
| 数学契约 | CONTRACT.md及其明确纳入的PAPER_COMPARISON.md；公式/shape/dtype/边界/容差在实现前冻结 |
| 论文/官方对照 | PAPER_COMPARISON.md逐模块章节/公式/官方实现/取舍；未因实现改动放宽契约 |
| 独立参考 | 候选`tests/reference.py`、`test_reference.py`：Lorentz/球面ambient推导，不import生产几何；全量评估另用NumPy ambient实现 |
| 确定性答案 | `test_known_answers.py`、`test_curvature_loss.py`、`test_attention.py`；期望值写死，非生产输出自证 |
| 变异 | `tests/test_mutations.py`（既有复数命名）、MUTATIONS.md/JSON：9/9由指定断言捕获，覆盖数值/结构/路由；未计崩溃/超时/导入错误 |
| 分模块验证 | MODULE_CHECKS.json：四rank×8模块，实际原损失/实际A.1损失及真实商品；没有用临时简单loss替代 |
| 结论复核 | 独立全原始资产评估后，`tools/issue251/recompute_report.py`不import模型/评估器，用fsum从每商品分布独立重聚合宏/微，并从原始SID/入口日志重算前三码碰撞；REPORT_RECHECK.json实际PASS |

最终源候选45项通过，评估/框架30项通过；后续评估补项由22项专门测试覆盖，报告复算器2项、显存回归1项、launcher既有6项、MCP缓存7项通过。#249既有79项测试及24模块检查通过。各轮重复项不相加冒充独立用例数。真实四rank的梯度/参数差0，对独立rank均值相对误差1.1781930986387579e-9；checkpoint/SID重导出逐项相等，rank0推理无collective。

运行身份：`issue251_validation_2660533`，workers2660570–2660573，四卡/seed42/port50202；1085.948秒、rank0峰值9144988160字节。全目标回传在rank0，四rank各中心另做DDP对账，未谎称四rank均重复完整目标。与入口smoke并行，耗时只作诊断，不作独占GPU性能比较。

## 复核与结论边界

- 执行前独立复核绑定最终源码；执行后收集器与报告复算器再次逐SHA确认生产/测试/配置/infra未漂移。FinalGradient251无代码发现；FinalAdmission251也无代码发现，但因合同冲突给整体incorrect，保留其原判定。
- 独立全量评估实际访问全部13分片；q/z宏MAE为2.6846592811951826/2.7511175786802333，微MAE为1.744486568784263/1.7864154747500234。fsum独立重聚合与报告最大差约1.2e-14，小于声明1e-9容差；10个strict bank、原始前三码指标和报告逐项一致。
- 受控微步SID碰撞0.5949485500467726超过Stage3的0.4门槛，不能晋级；首次/修复后入口SID碰撞分别0.04681335665188924/0.047464107048440285，属于不同权重/产物，不能混用。未把指标低判作bug或“噪声”，未调参重训。
- 单微步训练图拟合诊断不等于泛化；strict bank关联不等于推荐效果。历史实际TIGER SID按SHA冻结，其checkpoint不同代，历史几何/重建NA；bank无二元label时AUC为NA，不伪造负例。

## 阻塞、已排除解释与最小放宽

失败请求与原文：

```text
verification_check({"source_dir":"TIGER_gnn_attention_sectional_rq","verification_id":"machine:issue251"})
entry: PASS
behavior_graph: FAIL
reads a graph outside the canonical source in model/mechanism.py, tests/test_structure.py
passed: false
```

`GRAPH_AUDIT.json`由`tools.issue251.audit_graphs`读取原始CSR复算，训练输入SHA同为`80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`，但冻结图是规范图的严格子图：62501有向边、120318无向弧仅在规范图。冻结图没有额外边。同训练来源不代表同图；重命名路径不能修复拓扑。

已排除：不是数据/凭证/GPU缺失（实际四卡和全目标均运行）；不是未实现或可修复OOM（真实4096入口完成，显存红→绿）；不是端口漂移（实际50202）；不是MCP超时当失败（真实入口结束后0.179秒经公开工具取回持久化回执，当前图仍重新检查）；不是缓存陈旧或伪造NA（源码/配置/共享运行时绑定，图明确FAIL）。所有工程故障均有回归；工具提前完成通知、过早收集和mkdir竞争均未计作模型通过证据。

只有owner可选择：

- **A**：明确批准issue251按当前冻结#248/#249图/ego/曲率目标执行的具名§89b例外，再按新规则重新验准入；不允许Agent自行白名单放行。
- **B**：批准新的规范图一致图/ego/全曲率目标与strict bank版本，并重新冻结公式、合法行计数和校准契约，再重新实现/验证；不允许静默替换现有资产。

这不是表示容量不足或机制数学不可行的证明，也不是工作量/训练时长阻塞。原合同下，修改图违反冻结资产要求，放行原图违反准入；Agent不能自行选择放宽哪一条。

## 关闭前五问

1. **是**：生产候选与必要工具均已实现，无stub/TODO；提交a529589d已推送。
2. **是**：所有必要测试/短验证实际执行，有原始输出和退出码/机器判决；C7判决确为FAIL，未伪装通过。
3. **是**：本issue所需短验证全部结束；正式训练按issue明确禁止，数量0，不存在待完成正式训练。
4. **是**：67既有资产原位按hash复用；唯一新增静态资产为`assets/v6_attention_tokens/token_schema.json:a721a7e29e8b8a5bced06235f8033c92174b29997b3405a571259cca5e7721e8`，已登记v6 manifest；没有重建图/目标或复制重命名infra。
5. **否**：没有可自行继续的实现、测试、评估或训练业务任务；唯一未满足项需要owner的冻结合同变更。最终关闭证据提交/推送与关闭命令属于本结论的交付步骤，不冒充解除阻塞。
