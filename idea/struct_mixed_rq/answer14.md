# answer14 - 身份校验归消费者、DRAFT 通过既有校验器、单一诊断路径与干净收据
**状态:DRAFT 仍未冻结;prefix 仍未实现;绑定/准入/训练/收益全部 NOT_RUN**

承接 `answer13.md`,执行 `question14.md` 的全部条目。本节只写本轮实际执行的东西;未执行的一律标 `NOT_RUN`。

## 1. 修正 answer13 §4 的文字结论(先更正错误记录)
`q13/tests.log` 的 `k=0` 从未被接受。本轮直接读取该文件第 54-59 行:
```
rejected at 0 0 -> (0, 0, 1.0000000000000004)
rejected at 1 0 -> (1, 0, 1.1872172629857436)
rejected at 1 1 -> (1, 1, 0.9702936310553216)
rejected at 2 0 -> (2, 0, 1.8210165108310818)
rejected at 2 1 -> (2, 1, 1.1314479925811638)
rejected at 2 2 -> (2, 2, 0.8806541327555217)
```
实际是 **6 组全部拒绝**,第三分量都严格小于 `1-1e-6`;`answer13` 的“5/6、k=0 未拒、原因未定位”出自旧版日志,与当前日志不符。本轮不改数学、不为此增开训练,只更正记录。

## 2. 身份校验:豁免删除,校验由消费者执行
- 删除 `z_struct_rows == 3*33` 兼容分支;
- `probe_q9.validate_identity(report)` 在写盘前强制三项一致:
  `z_struct_rows == 3*triples_accepted == 3*len(triples)`,且 `triples_sha256` 等于按当前序列化规则从 `triples` 重算的摘要;
- 四类伪造记录对同一消费者函数测试,全部被拒:
  - 行数 99:`ValueError`;
  - 同条数换 item:`ValueError`(摘要不符);
  - 同条数换 type:`ValueError`;
  - 同条数改顺序:`ValueError`(此前 fixture 全同导致顺序负例无效,已改为互异三元组后真实通过)。
- 说明:顺序是摘要身份的一部分,生产者的摘要按当前序列化计算;这不是新增数学不变量,汇报时会与行数/上下界分开表述。

## 3. DRAFT 通过既有校验器(schema 整理 + 饱和映射修正)
- `tools/research/rounds/specification.py::validate` 此前报 16 项,现返回 **0 problems**:
  - `computations` 由字符串列表改为对象(`name/definition/inputs/output`);
  - `gradient_rules` 每条命名 `operator` 与 `rule`(encoder STE / codebook 原生回读 / gather);
  - `training_integration` 补齐 `loss/reconstruction/optimizer`;
  - `continuous_surface` 补 `args`(left/right, 32 维)与 `callable=mixed_geometry.mixed_sq`;
  - `conventions` 由校验器补写并标注 `conventions_derived_by`;
- **饱和映射矛盾已修**:`distance_surface` 原先把预量化整体收缩(`tanh*(1-1e-6)`)与 `acosh` 形式混写,现改为与生产一致的 `tanh` + `asinh(1/sqrt(1/x + 1/2)) = 2*asinh(sqrt(x))/2` 形式,普通点不再被 `1-1e-6` 缩小:
  ```
  bx = tanh(x1)
  by = tanh(y1)
  euclid = (x0 - y0) ** 2
  t = (bx - by) ** 2 / ((1 - bx ** 2) * (1 - by ** 2))
  d = euclid + asinh(sqrt(t)) ** 2
  ```
- 状态仍是 `DRAFT / 未审未冻结`;`validate` 只检查结构完整,不是数学审查,也不是冻结凭据。

## 4. 单一诊断路径:`probe_q8`/`probe_q6_factors`/`probe_factors` 退役
- `probe_q9` 为唯一活动生产者;`probe_q8.py`、`probe_q6_factors.py`、`probe_factors.py` 导入即 `ImportError`;
- 唯一消费者 `tests/test_q8_report.py` 已指向 `probe_q9`;
- `tests/test_factor_measurement.py` 迁移到 `probe_q9.report_blocks` 与 `probe_q9.split_latent`,不再测试旧辅助函数;参数块不拆分、码字/潜变量按最后维拆分的语义保留;
- 历史 q8 报告目录原样保留,不再被读取为当前证据。

## 5. 开关与空样本语义
- `probe_q9` 现在把 `T.BEHAVIOUR_ON` 传给共享 `H.structure_term(..., behaviour_on=...)`;
- 结构前向缺失时 `z_struct = None`:`latent` 与最终 `diffs` 的结构项记 `NOT_APPLICABLE`,不再无条件索引 `captures["struct"][0]`,也不伪造 0;
- 空三元组由 `structure_term` 返回零(已有行为),生产者不再假设一定有结构图;
- 身份测试用真实模型跑两次,仍通过。

## 6. 报告绑定输入路径与标定来源
报告的 `identity` 新增:
- `train_path`、`embedding_path`(绝对路径)与两者 SHA-256;
- `pca_dim`;
- `calibration_provenance = "historical run calibration, not the prefix candidate's own calibration"`——明确报告里 `s/λ` 这类数值来自历史运行,不是 prefix 候选自己的标定。

## 7. 干净的单次执行收据
`results/research/struct_mixed_rq/q14/tests.log` 由本轮一次完整运行生成,含:
- 命令行、解释器绝对路径、开始/结束时间;
- 每个测试文件一行 `rc=`;
- 结尾 `files pass=25 fail=0`(25 个 `tests/test_*.py` 全部 rc=0,未包含任何探针脚本)。

## 8. 仍未实现 / 未运行(不冒充完成)
| 项 | 状态 | 说明 |
|---|---|---|
| prefix 监督进入实际训练 | `NOT_IMPLEMENTED` | 只有数学与测试,训练通路未接 |
| 规格冻结 | `BLOCKED` | 无两位独立 Math Review 通过记录,字段已可审 |
| 登记/准入/Stage2/Stage3 | `NOT_RUN` | 未调用任何 MCP 流程 |
| 收益 | `NOT_RUN` | 旧 `test_recall@10=0.05790490781524748` 属旧候选树,不迁移、不作收益 |
| 绑定完整性 | `NOT_RUN` | 无独立审查结论与冻结提案哈希绑定 |

## 9. 本轮证据哈希
| 文件 | sha256 |
|---|---|
| `idea/struct_mixed_rq/answer14.md` | 本文 |
| 候选 `math_spec.json` | `702fc7560348d149f5e5c9a3cd06a349a54fba8c1e3c0821751ce6fd6667b88` |
| 候选 `probe_q9.py` | `ae73dec3bfa4e8a18de5e95c7660fed4c5e158ad6cde201b23a204a7d5a96fa` |
| 候选 `tests/test_q13_identity.py` | `2305298b76bf85fe71cafdddcaa5a2b2767345ad473d02c2d7922e2f36fada2` |
| 候选 `tests/test_factor_measurement.py` | `d922490be0269bce7b93e403e5c0849c28f0cf49ae95cacd4c60536f48026de` |
| 证据 `q14/tests.log` | `50ac1d9bdec11c1f9081eb8a8cc7bfa55d3143ff8cc749df3d7510e2f9d7a96` |
