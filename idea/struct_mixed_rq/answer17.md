# answer17 — 碰撞参考已稳定、消费绑定到实际 gather，完整前缀仍未冻结
**状态:DRAFT 仍未冻结;prefix 仍未实现(NOT_IMPLEMENTED);正式绑定/准入/训练/收益 NOT_RUN**

承接 `answer16.md`。先接受 question17 §1 的核对:前缀 DRAFT 已按 `question8.md §4` 恢复;连续面布局与契约语句一致;`should_forward` 的两个真实生产者负例已接受;报告 7 个源码摘要含 `train_rqvae.py`;answer16 的 13 条摘要完整。本轮只修两处新实测缺口。

## 1. 合法碰撞点上声明参考的 NaN 已修(同一数学)
**红**:从 `57bfdb62` 的契约语句重建参考,输入全零、float64、`requires_grad=True`:
```
grad tensor([[0., nan]], dtype=torch.float64)
finite False
struct grad finite False
```
持久记录:`q17/red_reference_nan.log`(命令、契约来源、时间、输出、rc)。声明的 `asinh(sqrt(t))**2` 在 `t=0` 的反向产生 NaN;`test_q17_reference_build.py` 在修复前给出
`GradcheckError: Jacobian mismatch ... analytical:tensor([[0.],[nan]])`。

**修复(同一函数,非新度量)**:在 E1×H1 子面上 `b=tanh(x)`,`t=(b_x-b_y)^2/((1-b_x^2)(1-b_y^2))=sinh^2(x_1-y_1)`,因此
`asinh^2(sqrt(t))=(x_1-y_1)^2`。据此:
- `reference_contract.json` 的 `mixed_distance` 改为 `d = (x0-y0)^2 + (x1-y1)^2`(并注明恒等式来源);
- `structure_objective` 改为三对 `d_ab=(p_{2k}-p_{2k+2})^2+(p_{2k+1}-p_{2k+3})^2`;
- `math_spec.continuous_surface.statements` 同步为同一形式,并写入 `expression_note`;
- `mixed_geometry.reference_surface`(1 个 H 坐标)直接返回 `(left_h-right_h)^2`,注释说明这是同一函数的稳定反向。

**未做**:未提高容差、未接受 NaN、未加 floor、未删除碰撞样本、未把碰撞点塞进 excluded。取值不变(距离 0;坍缩 stress 1),只是反向有限。

**绿**:
| 测试 | 结果 |
|---|---|
| `tests/test_q17_reference.py`:Python 参考在碰撞点 `gradcheck` 且梯度有限 | `rc=0` |
| `tests/test_q17_reference_build.py`:由声明语句重建的参考在碰撞点 `gradcheck` | `rc=0` |
| `tests/test_q16_surface.py`:坍缩点 stress=1(行为值) | `rc=0` |

## 2. 消费身份绑定到实际 gather
**红**(修复前):`test_q17_consumption.py` 在结构消费入口把 45 条三元组循环换序一位后调用原函数,生产者接受并写出报告:
`AssertionError: a rotated consumption was accepted`。这复现了 question17 §3 的 `sampled=45 / accepted=45 / actual_gather_matches_rotated_sample=true / persisted_reports=1` 形态。

**修复**:`training_hooks.structure_term(..., capture=...)` 把实际 gather 的 `item_ids` 写入调用方列表;`probe_q9` 在写报告前把这些 item 依三元组分组,与采样 `used` 逐条、逐序比较;不一致即
`RuntimeError("the gathered items differ from the sampled triples")`,**不写 `factors.json`**。报告与校验使用**实际消费**的三元组(而非采样列表),`validate_identity(consumed_triples=consumed)`。
**绿**:换序注入现在被拒绝且目录内无报告(`rc=0`)。正常 45/135、合法 `behaviour_off`、`empty_triples` 的记录与 `NOT_APPLICABLE` 语义保持不变。

**测试更正**:旧的 `test_the_consumed_order_is_not_taken_from_the_report_under_test` 先因摘要失败而绿,不能证明消费比较;新测试断言的是生产者的 `RuntimeError` 文本与"无报告",走真实 `P.run`。

## 3. 测试卫生
`tests/test_q16_surface.py` 删除了只查宽度/`hasattr`/字符串/固定 `NOT_IMPLEMENTED` 的三条永久测试,仅保留有行为意义的坍缩值测试;公式覆盖由 `test_q17_reference.py` 与 `test_q17_reference_build.py` 的取值/梯度/双侧 `gradcheck` 承担。

## 4. 本轮工程故障(如实保留)
改 `training_hooks` 时有两份失败收据,均保留未覆盖:
| 收据 | 结果 | 原因 |
|---|---|---|
| `q17/tests_20261110T064853.log` | 31 文件 29 pass / 2 fail | `capture` 被误插入 `calibrate_structure_scale`,两处 `NameError: capture` |
| `q17/tests_20261110T065344.log` | 24 文件后失败 | 修复时删掉了校准路径的 `item_ids`,触发 `NameError: item_ids` |
| `q17/tests_20261110T065916.log` | **31 文件 31 pass / 0 fail**,runner `rc=0` | 最终绿色收据,含 `test_q16_surface.py` |

两次失败都属工程接线错误,未改数学、未放宽判定;修复后单独复跑成功才做最终收据。

## 5. 完整前缀与独立审查的实际状态
| 项 | 状态 | 具体内容 |
|---|---|---|
| 前缀目标 | `NOT_IMPLEMENTED` | DRAFT 已恢复原 stress(同三元组、三对 h、共享 s、单 λ、等权 `/3`);训练通路仍未实现 |
| 独立审查 / 冻结 | `BLOCKED` | 当前可用入口只有本地 `specification.validate`(0 problems);没有可用的两条独立 Math Review 执行通道,未自签、未恢复已删除流程 |
| 冻结材料 | 已就绪 | `math_spec` DRAFT、`reference_contract`(与语句逐条一致)、碰撞参考与梯度测试、消费绑定测试、q17 收据 |
| 登记 / 准入 | `NOT_RUN` | 未调用 MCP |
| 修复版 Stage2 / Stage3 | `NOT_RUN` | 未训练 |
| 增益 | 无结论 | 旧 `0.0579` 未移交、未达 `0.065` |

§2 的恒等式只证明 E1×H1 参考可执行,**不证明高维非欧效应**;H16 的原生残差、累计回读与 Jacobian 测试继续作为既有证据,完整空间与行为目标的统一仍属冻结包内容。

## 6. 本轮摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e04a17d8bad6160db886041206124f7a41d403cb04ad4a71dd0c2b3f40eff00b` |
| 候选 `reference_contract.json` | `e60b3eafabce613c05786cf7781e3c36530b2a20b716d99523c2fdbedcb4adb9` |
| 候选 `mixed_geometry.py` | `8842df9b04622f26ce9c6adb3d45f69cf7d86fac1fdb98fc407047c22ebf01b0` |
| 候选 `training_hooks.py` | `bff0ef340ad4cdfd85ba2cfb5168d7ec3702c293c0f00a25f449453c116b20c5` |
| 候选 `probe_q9.py` | `c91ed0d5bb72b1f6847f442a94aa7d3b422f8614d130b864e5c4beda310e6ff7` |
| 候选 `tests/test_q17_reference.py` | `e9ee69b29db2aaea691c1b112507f60e273a58a357d01bcb79b8dae86d698242` |
| 候选 `tests/test_q17_reference_build.py` | `127c440e31a498cd6d04dc0ffce38dd43dbfb09f2903f89d80f58cdb622ce273` |
| 候选 `tests/test_q17_consumption.py` | `de0904bb3ad446914fe310137a7353ced25b39f577069a93d53cf2a63b4e9c84` |
| 候选 `tests/test_q16_surface.py` | `056ead8974ceb771dade72658df00094e65fdf0544c14d9a4ceaa154baafebce` |
| 证据 `q17/tests_20261110T065916.log` | `30a9a7fe3e28436ecc7c2efa50538fad8846173b4e8603e07be8b73fc5c3f970` |
| 证据 `q17/run_suite.sh` | `e79d773900ac97c7562bdc9d838b5cfe374acb75563207d7fdf75886b4a0f436` |
| 证据 `q17/red_reference_nan.log` | `3a0ba8883855b45b72082e238272e826ca6e4fc2f22e6966167b6c634457c1c3` |
| 证据 `q17/tests_20261110T064853.log`(失败,保留) | `963a3fd23977b6d5096f941b8d5fed58c041f131cda2aa2a7080bd496424f5de` |
| 证据 `q17/tests_20261110T065344.log`(失败,保留) | `7186d08efa4bf7dd1f9d88df7d0a1dfff5afde91873ec947eae845c9a948a030` |
