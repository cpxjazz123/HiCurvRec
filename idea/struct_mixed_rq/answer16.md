# answer16 — 前缀草案恢复原目标，连续面可执行，接线负例由真实生产者判定
**状态:DRAFT 忠实性已修、仍未冻结;prefix 仍未实现;绑定/准入/训练/收益全部 NOT_RUN**

承接 `answer15.md`。先接受 question16 §1 的核对:q15 持久报告、11 条 64 位摘要、26 条 `CMD` 收据均被承认;本轮只修前缀草案、连续面对照、接线判据与收据,不重述已接受内容。

## 1. 前缀草案已恢复原机制(不是等价改写)
q15 的 `prefix_surface` 写成了单边 hinge + 三个标尺 + `successor/prefix/next_item` 对,偏离 `idea §4.1` 与 `question8.md §4` 的对称平方距离 stress。现按 **原机制** 改为:

```
q1/q2/q3 = 同一次 return_levels 的原生累计码字回读(同一次量化)
e_l = (mixed_sq(q_l[centre], q_l[a]) / (s*s*h_ab*h_ab) - 1) ** 2
L_l = mean over TRAIN triples of (e_l(centre,a) + e_l(centre,b) + e_l(a,b)) / 3
L_prefix = (L1 + L2 + L3) / 3
callable = NOT_IMPLEMENTED
```

- 同一 TRAIN 三元组、三对 `h`、**共享 s**、等权 `/3`、**一个 λ**;
- 无 next-item 目标、无逐层标尺、无第二 gather;
- `q*` 明确来自同一次 `return_levels` 的原生累计码字,不是未量化残差;
- **实现仍是末层结构项**,因此 `callable` 如实标 `NOT_IMPLEMENTED`,不把草案写成已发生的训练变更。

可机械判定的反例(同点距离为零,`d=s=h=1`):

| d | 原 stress | 旧 hinge 草案 | 原项导数 | 旧 hinge 导数 |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 0 |
| 0.5 | 0.5625 | 0 | -1.5 | 0 |

取值与梯度都不同;全坍缩在原目标是 `L=1`,在旧草案是 `0`。该表随本轮 DRAFT 保留。

## 2. 连续面对照已可执行
- `continuous_surface.args` 改为适配器真实布局:`points width=6`、`u width=3`、`scale width=1`,合计 10 = 符号数;
- `callable` 改为 `structure:reference_objective_surface`(实际函数,重塑 `(batch,3,2)` 后调用 `mixed_geometry.reference_surface`);
- 主面保留 9 个坐标 + `s` 的约化式,并写明这是 **E1×H1 子空间**,不代表完整 E16×H16;16 维径向球映射、全向量范数交比与既有 `phi` 由完整距离函数承担,不在该子面声明等价;
- `reference_contract.json` 的 `mixed_distance` 与 `structure_objective` 语句已从旧的 `tanh*(1-1e-6)`/`acosh` 改为 `tanh` + `asinh(sqrt(t))**2`,与当前实现逐条一致;
- 新增 `tests/test_q16_surface.py` 三例(全绿):args 宽度 `[6,3,1]` 且 callable 可导入;三点重合时 `reference_objective_surface=1.0`(不是 hinge 的 0);前缀草案含单标尺、无 `next_item`、`L_prefix=(L1+L2+L3)/3`。

`specification.validate` 仍返回 0 problems,但这只是浅层 schema;独立审查/冻结仍是 `BLOCKED`。

## 3. 真实接线:先判分支,再看图
生产者现在在调用结构项**之前**计算 `should_forward = BEHAVIOUR_ON and len(used)>0`,随后:
- `should_forward` 为真而 `z_struct is None` → `RuntimeError("a legal sample was given but the structure forward did not run")`,**不写报告**;
- `should_forward` 为假而捕获到前向 → `RuntimeError("the structure forward ran although the branch is inactive")`;
- 活动图但损失无图或无结构潜梯度 → `RuntimeError`(保持 q15 的两条);
- 合法 behaviour-off / 空样本 → `NOT_APPLICABLE` + 明确原因(`behaviour_off` / `no_accepted_triples`),不伪造前向。

`validate_identity(report, consumed_triples=list(used))` 现在使用**执行位置的有序消费集合**,不再从待审 JSON 反向构造。`tests/test_q16_wiring.py` 三例(全绿):
| 用例 | 结果 |
|---|---|
| 合法样本被跳过结构前向(替换 `structure_term` 为常量) | `RuntimeError`,且临时目录无 `factors.json` |
| 开关关闭却发生前向(强制 `behaviour_on=True`) | `RuntimeError`,且无报告 |
| 报告内三元组换序后仍以执行侧原始顺序校验 | `ValueError` |

**红证据**:用 `14b6c747` 的生产者运行同一测试,得到
`AssertionError: a skipped structure forward was accepted`,`rc=1`,已持久保存为 `q16/red_prefix_producer.log`。这取代 q15 的 helper 用例作为接线负例;question16 中 `TIMEOUT` 的完整 CPU 注入不再作为通过。

## 4. 收据与引用更正
- 新收据 `q16/tests_20261110T061102.log` 由 `q16/run_suite.sh` 生成:记录脚本 SHA、解释器、cwd、起止时间,每个文件一条 `CMD:` 与 `rc=`,以及 stdout;runner **拒绝覆盖已存在日志**,失败时退出码非零;本轮 `files=27 pass=27 fail=0`,runner 退出码 0。
- `q16/make_reports.py` 写入**新的时间戳目录** `reports_20261110T061555/`,不覆盖 q15 的 `boundaries.json`。
- **更正 answer15 的引用**:`tests/test_distance_saturation.py` 在当前候选树中 **NOT_FOUND**,也不在 q15 的 26 条记录里。实际覆盖球内投影与近原点代数的是 `tests/test_mixed_geometry.py`(`test_ball_projection_is_inside_the_ball` 等);不补同名空测试,也不否定其他几何测试的真实覆盖。
- 报告绑定的 `source_sha256` 已补 `train_rqvae.py`,当前 both-on 报告列出 **7 个**源码文件,含读取配置/开关/输入与标定的入口;这仍不构成"完整配置已冻结"。

## 5. 本轮持久报告(新生产者)
| 文件 | 内容 |
|---|---|
| `q16/reports_20261110T061555/both_on/.../factors.json` | 45 采样/45 接受/135 行、`structure_forward=true`、`E_norm≈3.8e-5`、`H_norm≈4.3e-5` |
| `.../behaviour_off/.../factors.json` | 45 采样/0 接受/0 行、原因 `behaviour_off`、潜变量 `NOT_APPLICABLE` |
| `.../empty_triples/.../factors.json` | 0/0/0、原因 `no_accepted_triples`、`NOT_APPLICABLE` |
| `.../boundaries.json` | 三者状态摘要 |

## 6. 状态表(实际状态)
| 项 | 状态 |
|---|---|
| DRAFT 忠实性(前缀目标) | 已按 `idea §4.1` / `question8 §4` 恢复;仍是 DRAFT |
| 连续面对照 | 可执行子面 + 契约语句对齐;仅为 E1×H1 子空间 |
| 真实诊断正/负例 | 正例与两个接线负例已执行;红日志已保存 |
| 独立审查/冻结 | `BLOCKED`(无两位独立 Math Review;不恢复已删流程) |
| 前缀实现 | `NOT_IMPLEMENTED` |
| 正式绑定/准入 | `NOT_RUN` |
| 修复版 Stage2/Stage3 与增益 | `NOT_RUN`;旧 `0.0579` 未移交、未达 `0.065` |

## 7. 本轮摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e2cb2cc57f80b5a06ce99b9da8fa7af2bfb82cfd8b3d31f2acf09d4a339fded7` |
| 候选 `reference_contract.json` | `8ad880a5cdc3295518ead9e2825711fb25b6cb798f3638c09ecc2e9230eeb8c7` |
| 候选 `probe_q9.py` | `be898336053a5d4a9975bbfa8d9c747137f49a28752ac84d6b145111112191e9` |
| 候选 `tests/test_q16_wiring.py` | `bf4226d12ef50eac8a9c2d0f0cced08ecef33378490b9f37abdf140e593cad9b` |
| 候选 `tests/test_q16_surface.py` | `9491e325a64353901d7f4026a78cf60f813566e391862fc30a468796295b8a87` |
| 证据 `q16/tests_20261110T061102.log` | `a2e71fe0e0e44bcadc12865d816c60044b1a69b1722260999deca500bea8efc5` |
| 证据 `q16/run_suite.sh` | `490d9997cd52edaf0cdec48b5c5aa7b6786a6bed5b397baa0e51f70fb81e8f11` |
| 证据 `q16/make_reports.py` | `21e23ba0cd4bcf82b471e800789ac214b0edf187e559d036f0b213b26ca72a52` |
| 证据 `q16/red_prefix_producer.log` | `8d073e0aa3ad2e49e51a2c774cf5b5aff79dafb087c82ace4251510f996338cf` |
| 报告 `boundaries.json` | `3368ffcccf140dab0c10b6084f4cdce648405c7b15eae2f132eedeef4f266c3f` |
| 报告 `both_on/.../factors.json` | `3989bb8dba684d4f3f1106e845fb633d9f36b7ecf585667176e82681a2a75dfa` |
| 报告 `behaviour_off/.../factors.json` | `91bdf2b7b722aaef713a458f176549eabd9e6dff8192979f7dd4cbf8a60d8341` |
| 报告 `empty_triples/.../factors.json` | `c3ee2ba918d1f8242d054d08a384aa3c81b3b1ea2da7b435d92dc8dbab1082cc` |
