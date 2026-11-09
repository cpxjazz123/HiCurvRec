# answer19 — 全维前缀数学的两处错误已纠正，前缀材料与收据边界重写
**状态:DRAFT 仍未冻结;prefix NOT_IMPLEMENTED;冻结 BLOCKED(无候选入口/审查收据);绑定/准入/训练/收益 NOT_RUN**

承接 `answer18.md`。接受 question19 §1 对契约/子面一致、kind 消费绑定与 33/33 收据的核对。本轮只做可达的数学纠正、测试卫生与收据更正。

## 1. cross-ratio 方向已纠正(不是等价写法)
`math_spec.continuous_surface.prefix_surface.full_space` 原先把 `t` 写成倒数
`(1-|a|^2)(1-|b|^2)/|a-b|^2`,在合法碰撞 `a=b` 除零、趋近碰撞时反而发散。现改为与生产一致:
```
t = |a-b|^2 / ((1-|a|^2)(1-|b|^2))
phi(t) = asinh(sqrt(t))^2
d_M^2 = |e-e'|^2 + phi(t)
```
并写明标准双曲平方距离是 `4*phi`、乘积项用的是 `phi`、`c=1` 固定。
`prefix_surface.rejected_draft_forms.cross_ratio_reciprocal` 记录本节实测:同一球坐标下
`t=0.05260942760942763` 对应 `phi=0.051711885865117525`,而倒数草稿给出 `t=19.007999999999996`、`phi=4.745763624866576`。
`tests/test_q19_full_space.py` 断言生产 `phi` 与正确分式一致、且倒数明显不同(不是文本搜索)。

## 2. 原生残差已按 Möbius 定义重写
`residual_and_readback` 改为 question19 §2.2 的同一机制:
```
B_l = code_ball(C_l^H); r_0^E=z^E, r_0^H=to_ball(z^H)
r_l^E = r_{l-1}^E - C_l^E
r_l^H = (-B_l) mobius_add r_{l-1}^H
R_l   = (sum_{j<=l} C_j^E, log_0(B_1 mobius_add (B_2 mobius_add ... B_l)))
q_l   = R_l + z - sg(z)
```
写明选码输入是当前残差的 E 坐标与 H 的 `log_0`,不是全空间减法;Möbius 定义域为开球,边界点排除;encoder STE 与码本 native Jacobian 在 `ste_boundaries` 分开陈述。
`rejected_draft_forms.euclidean_residual` 记录实测:`z_H-C_H` 给 `(0.2,-0.2)`,native 给
`(0.1868567811974333,-0.22350722788306665)`;native 回读恢复误差 `8.3e-17`,欧氏相减配 native 回读为 `0.023069610924273662`。
`tests/test_q19_full_space.py` 断言 native 值、与欧氏差、以及回读恢复误差。

## 3. 前缀材料的实际覆盖与未覆盖
`prefix_surface` 现含:`full_space`(径向 b、`log_0`、Möbius、正确交比、`phi`、`c=1` 与 `4phi`/`phi` 归一化)、`residual_and_readback`(上述公式)、`ste_boundaries`、`calibration`、`coverage_gaps`。

标定规则写实:
- `s`:初始化后末层 TRAIN h=1 对的 `sqrt(clamp_min(d_M^2,1e-8))` 的中位数;`structure.calibrate_scale` 在 `value<=1e-6` 或空样本时返回 `1.0`,`training_hooks.calibrate_structure_scale` 在无 triples 时也 `return 1.0`——两处都是待按冻结规格修正的差异,不是已校准值。
- `λ`:完整前缀目标的 encoder / 合并全部码本两块范数预算一次确定;不沿用旧 checkpoint,不引入 epsilon 或扫参。

覆盖缺口明确保留:缩维 `1+tanh(u)` 与 box 抽样不覆盖精确 `h=2`(由 `test_structure.py` 的真实目标覆盖);H1 平方差恒等式不证明 H16 非欧效应;`prefix_surface.callable=NOT_IMPLEMENTED`,真实 `_positions` 仍取 `levels[-1]`;`calls[0]` 对齐的是单层 E1×H1 结构项,不是完整 E16×H16 三层前缀,不以该相等为主面冻结背书。

## 4. 测试与收据更正
### 4.1 q18 失败收据的真实数字
`q18/tests_20261110T071811.log` 实际为 **33 文件、29 pass / 4 fail**,`source_drift: CHANGED`,执行前后只有 `probe_q9.py` 摘要改变;失败文件为
`test_q13_identity.py`、`test_q15_boundaries.py`、`test_q16_wiring.py`、`test_q9_sampling_identity.py`(均为 `identity.targets` 含 Tensor 的序列化错误)。
**更正 answer18 §2/§5 的"31 pass / 2 fail"与"drift 未提及"**;最终 `q18/tests_20261110T072312.log` 为 33/33、drift none,两项分开记录。
answer18 声称的修复前 kind-only 两次断言红,其原始收据未保存在 q18 目录,记 `NOT_RECORDED`;不以本轮重跑来补造修复前日志。

### 4.2 删除不证明行为的永久测试
- 删除 `tests/test_q18_contract.py`(只复制并比较 JSON 名称/语句);材料一致性仍由清理后的规格文件与既有工具核对,数值/梯度与上面的反理由真实数学测试承担。
- 删除 `tests/test_q16_wiring.py` 中反转三元组却不更新摘要的 helper 用例;替换/换 kind 已由 `test_q17_consumption.py`、`test_q18_consumption_kind.py` 的真实 `P.run` 覆盖,不钉异常文案。
- 两个真实生产者负例(跳过前向、关闭却前向)保留。

## 5. 审查入口的状态拆分
| 状态 | 事实 |
|---|---|
| 角色实现存在 | `tools/research/agents/runtime.py` 有 `formulate`、`review_math` 分支 |
| 候选的合法 round/DAG 绑定 | 无(未提供 round_id) |
| MCP 请求 | 未发起 |
| 本地导入 | `from tools.research.rounds import worker` → `ModuleNotFoundError`;`tools/research/agent_flow.run` 路径不存在。这是本地导入事实,不足以断言全部合法 DAG 入口不可达 |
| 独立审查/冻结 | `BLOCKED`:无两位审查执行收据,不自签、不恢复已删流程、不对内部函数推进正式状态 |

## 6. 收据
`q19/tests_20261110T073704.log`:33 条命令、33 个不同文件、**33 pass / 0 fail**,runner `rc=0`,`source_drift: none`。
执行前/后清单现在同时绑定 `*.py`、`math_spec.json`、`reference_contract.json`(q18 只收 `*.py`,本轮已补),分别保存为
`q19/source_pre_*.txt` 与 `source_post_*.txt`。旧收据不补签。

## 7. 状态表
| 项 | 状态 |
|---|---|
| 全维交比/残差数学 | 已纠正并有数值反例;DRAFT |
| 单层折叠结构项 | 已实现(末层),接线测试通过 |
| 完整三层前缀 | `NOT_IMPLEMENTED` |
| 独立审查/冻结 | `BLOCKED`(无候选入口与审查收据;本地导入缺失仅记事实) |
| 正式绑定/准入 | `NOT_RUN` |
| Stage2/Stage3 与增益 | `NOT_RUN`;旧 `0.0579` 未移交、未达 `0.065` |

## 8. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `9b748dee5c7cea0aa9f9329e7c44eef005fcf7f621ddddb6b682a74572b7ec56` |
| 候选 `reference_contract.json` | `41a92152df616189f3dfa36d33a9a2512ba876c703559da35a600248eadf7e51` |
| 候选 `tests/test_q19_full_space.py` | `fae94221e264427afa0977bafc7a34e43bdfb070e46c4f5b5b5f5e7b08baa6d2` |
| 候选 `tests/test_q16_wiring.py`(已删 helper) | `24de76492246df033755a44ff5cdc1b8702e677249df9b3c3f4209ebc7b2bf60` |
| 证据 `q19/tests_20261110T073704.log` | `2d70ed905fee163c668402f25c5dff59da953d65b1432317f74a55cfc7e64d29` |
| 证据 `q19/run_suite.sh` | `283b61c1b7df4c8472db55bc526114a243cbb8eb7bd8de080d67aecf7709f401` |
| 证据 `q19/source_pre_*.txt` | `52d18298a4ddab245d05ab3ab7ed2539e01c7b9f383f50c140f270244c81eabd` |
