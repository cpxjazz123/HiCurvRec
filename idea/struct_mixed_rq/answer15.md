# answer15 — 真实入口的两个边界已修复，DRAFT 主面与 schema 对齐，收据由脚本生成
**状态:DRAFT 仍未冻结;prefix 仍未实现;绑定/准入/训练/收益全部 NOT_RUN**

承接 `answer14.md`。先接受 question15 §1 的审阅:接受 q14 已完成的改动,但不接受"question14 全部条目已执行完成"的概括——本轮修的就是那两个真实崩溃和两处文字错误。

## 1. 先更正 answer14 的两处文字错误
1. **错误的恒等式**:answer14 §3 写的 `asinh(1/sqrt(1/x+1/2)) = 2*asinh(sqrt(x))/2` 不成立。`x=1` 时左边 `0.7454981544974043`、右边 `0.881373587019543`(实算)。生产实现用的是 `asinh(sqrt(t))**2`,与本式无关;该错误等式作废。
2. **错误的拒绝范围**:answer14 §1 说六组 H 拒绝误差"都严格小于 `1-1e-6`"不符合日志。`(0,0)` 的第三分量是 `1.0000000000000004`,是 Jacobian 偏差,不是保留倍率;六组实际都 **大于** `1e-6`。拒绝证据继续成立,不重开该问题。

## 2. 真实边界修复(TDD:先红后绿)
用 Stage2 解释器、当前 `probe_q9.run`、实际模型与训练输入,行为关与空三元组原本都在 `probe_q9.py:128` 触发
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`(原始日志在 `bg_67`/`bg_68` 的真实运行中记录)。

修复后的语义(代码即 `probe_q9.py`):
- **抽到 ≠ 参与**:`triples_sampled` 是实际抽到的条数;`triples_accepted` 只在结构前向真的发生时等于它,否则为 0;`triples` 列表保存**被消费**的样本,不存在的样本不写入。
- **是否发生前向**:`structure_forward` 布尔;`z_struct_rows` 在未发生前向时为 0,不再对 `None` 取 shape。
- **不适用项如实标注**:常量零损失不进入反向,`states.status = NOT_APPLICABLE`;`latent.z_struct = NOT_APPLICABLE`;不适用原因写入 `structure_zero_reason`(`behaviour_off` / `no_accepted_triples`)。
- **不伪造图**:没有 `requires_grad_()`、没有额外重采样、没有吞异常。
- **活动图断梯必须报错**:前向已发生而结构损失无图、或结构潜变量拿不到梯度时 `raise RuntimeError`,不会降格成 `NOT_APPLICABLE`。
- **正常路径不变量不变**:`rows == 3*accepted == 3*len(consumed)`,旧 45/99 负例仍被拒绝;非活动分支要求 `accepted == 0 == rows`、状态 `NOT_APPLICABLE` 且有原因。
- **绑定到同次消费**:`validate_identity(report, consumed_triples=...)` 由消费者传入实际消费样本;同条数换样本即使重算摘要也被拒绝。内部摘要校验只证明报告自洽,不当作独立前向证明。

### 2.1 新测试(`tests/test_q15_boundaries.py`,五例)
| 用例 | 结果 |
|---|---|
| 行为关:真实 `run`,要求 `structure_forward=False`、行数 0、原因 `behaviour_off` | pass |
| 空样本:仅让 `sample_triples` 返回 `[]`,要求 `triples_accepted=0`、原因 `no_accepted_triples` | pass |
| 正常 both-on:要求存在结构图、`rows == 3*accepted`、摘要与列表一致 | pass |
| 消费者侧同条数换样本:重算摘要仍拒绝 | pass |
| 活动图但结构潜变量无梯度:拒绝而不是降格 | pass |

`tests/test_q13_identity.py` 的负例仍全绿(行数/item/type/顺序)。

## 3. DRAFT 主连续面已修,并区分四个面

### 3.1 修正项
- 主 `continuous_surface.statements` 中三条 `tanh(...) * (1 - 0.000001)` 已删除,统一 `tanh`,
  与生产的 `to_ball` + `φ`(`asinh(sqrt(t))**2`)一致;
- 符号到输入对应:`p0/p2/p4` 为三个点的 E16 切坐标,`p1/p3/p5` 为 H16 切坐标(**切空间**,
  非球内坐标;球内点由 `to_ball(t)=tanh(|t|)/|t|·t` 得到),`u0..u2` 为对数尺标 `h=1+tanh(u)`,
  `s` 为正的中位目标距离;`callable` 改为 `structure.structure_loss`(它调用 `mixed_geometry.mixed_sq`);
- 明确**受限**说明:语句把 16 维分量约化为单坐标,是参考子面,不是完整 32 维等价;
- 新增 `prefix_surface`:`q_ℓ = readback(残差_ℓ)` 与三层 `L_ℓ` 的右嵌套组合规划在符号层面列出,
  `callable = "NOT_IMPLEMENTED"`,并注明读取是原生右嵌套回读、组合尚未实现;
- `validate` 仍返回 **0 problems**——这只说明浅层 schema 满足,不证明数学/签名/前缀目标一致。

### 3.2 定义—函数—覆盖范围—未覆盖项
| 定义 | 函数 | 覆盖 | 未覆盖 |
|---|---|---|---|
| 成对混合距离 `d_ij` | `mixed_geometry.mixed_sq` | E16 平方和 + H16 `asinh(sqrt(Σt))²`;普通点、碰撞点、饱和边界(合计 `x` 的实算定义见 `tests/test_distance_saturation.py`) | 无(距离在 32 维上完整) |
| 三点结构损失 `L` | `structure.structure_loss` | 上表 9 符号 + `s` 的约化参考面;hinge 与权重 `1/3` | 16 维逐坐标写出;`p/u` 的符号到完整张量映射需人工对照 |
| 残差/回读 | `native_product.py` | 右嵌套回读的现状 | 与 `prefix_surface` 的逐项绑定尚无可调用实现 |
| 前缀组合 `L_prefix` | `NOT_IMPLEMENTED` | 仅有符号规划 | 训练通路、Sinkhorn 内耦合、共享标尺/采样全部未实现 |

### 3.3 纯数学/接口检查
- `math_spec.validate` = 0 problems(浅层);
- `tests/test_distance_saturation.py`、`test_reference_surface.py`、`test_ste_target_jacobian.py` 在 q15 收据中均 `rc=0`(前者覆盖饱和边界数值,后者覆盖 STE 目标 Jacobian)。

## 4. 执行收据:由脚本生成,不再手抄
`results/research/struct_mixed_rq/q15/run_suite.sh` 是实际执行的脚本;`tests.log` 由它生成,含:
- 脚本路径与脚本 SHA、解释器绝对路径、cwd、开始/结束时间;
- 每个文件一行 `CMD:` 实际命令 + `rc=`;
- 结尾 `files=26 pass=26 fail=0`。

**对 q14 收据的更正**:q14 日志头行的 `for f in tests/test_*.py; ... test_training_hooks.py` 没有使用 `$f`,
不是有效命令;q14 实际记录是 **25 个不同的 `test_*.py`、25 条 rc=0**,不是"22 个测试文件 + 3 个探针/工具脚本"。
本文件数与用例数分开陈述:26 个文件通过,每个文件内部用例数由各自 `ok` 输出记录。

## 5. 新持久报告(本轮生产者)
| 文件 | 说明 |
|---|---|
| `q15/report_both_on/run-1794250067/factors.json` | 正常 both-on:45 接受三元组、135 行、真实结构潜梯度(`E_norm≈3.8e-5`,`H_norm≈4.3e-5`),`validate_identity` 通过 |
| `q15/report_behaviour_off/run-1794250073/factors.json` | `structure_forward=false`、`triples_sampled=45`、`accepted=0`、`z_struct_rows=0`、原因 `behaviour_off` |
| `q15/report_empty_triples/run-1794250078/factors.json` | `structure_forward=false`、`sampled=0`、`accepted=0`、原因 `no_accepted_triples` |
| `q15/boundaries.json` | 三者的状态/损失摘要 |

这两份边界报告证明的是"边界可被真实入口记录",不是"边界提升了 SID 或召回"。

## 6. 仍单列的未完成项(不冒充)
| 项 | 状态 | 说明 |
|---|---|---|
| prefix 监督进入训练 | `NOT_IMPLEMENTED` | 只有数学与规划;`prefix_surface.callable` 明确未实现 |
| 独立审查 / 冻结 | `BLOCKED` | 无两位独立 Math Review 记录;不恢复已删除调度器,不自签 |
| 登记/准入/Stage2/Stage3 | `NOT_RUN` | 未调用 MCP |
| 收益 | `NOT_RUN` | 旧 `test_R@10=0.05790490781524748` 属旧候选树,不移交、不作收益;未达 `0.065` |

## 7. 本轮自动计算的摘要(完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `396aced168465eb5c4ff43728041a84d846df48c8c77b5b6a37498347cfd96d8` |
| 候选 `probe_q9.py` | `acffc9b50c9ef625e985a554b27771615cb7505b55f5cd417b16ea1e7e32ed12` |
| 候选 `tests/test_q15_boundaries.py` | `8b513fc25d665378ad56a1e521ecf71c0b828561c5e4118a7061c584266785ad` |
| 候选 `tests/test_q13_identity.py` | `b5ea47f9e6a93441fb32600d9afbae3cebb25cabce53798bbbf4d152af1f00db` |
| 证据 `q15/tests.log` | `3e4d454719a6e5961c0fdc716e03a6ca57b947d23657714a55e78c98eab44a01` |
| 证据 `q15/run_suite.sh` | `3af88c904faf98dda9716cb2d2fac5444d97c941bee1ed444cf7968ba0b03340` |
| 证据 `q15/make_reports.py` | `4cca3dec8a2d18768d05cbe1409092de5ce38448728f92e204f2003bb5efa54b` |
| 证据 `q15/boundaries.json` | `d47677a4a4f77a76f370867a3a235f4e90e8db8197b3df8d59595494dddc4fa6` |
| 报告 `report_both_on/.../factors.json` | `16521dea6b83a2a589def83c71cb5074306e171cabbee974ec1155fdb17cf4cc` |
| 报告 `report_behaviour_off/.../factors.json` | `7ec35b99688ccdd1dfe705f9684823d675c123350ad88a5115f1f6ce9be20dd8` |
| 报告 `report_empty_triples/.../factors.json` | `c417f9db4e089089b3a229ed79a4054c1dcd288bde484a767bd7097f98d61ccc` |
