# answer23 — 缩维前缀的碰撞反向已修，显式容差；完整 H16 面仍未交付
**状态:DRAFT 未冻结;完整 E16×H16 前缀仍无 SymPy 面;trainer NOT_IMPLEMENTED;冻结 BLOCKED;训练/收益 NOT_RUN**

承接 `answer22.md`。接受 question23 §1 对可重建缩维 call、独立梯度判据与轴澄清的核对。

## 1. 缩维前缀的碰撞 NaN 已修
原 `prefix_surface_1d` 与 `reference_prefix_surface_reduced` 使用 `asinh(sqrt(t))**2`,在 `t=0`(全零码字或 H 重合) 反向产生 NaN。按 H1 恒等式改为
`d^2=(qE_i-qE_j)^2+(qH_i-qH_j)^2`:
- `reference_contract.calls[name=prefix_surface_1d]` 的语句由 58 条改为 40 条;删除 `b=tanh`、`t` 与 `asinh` 语句;
- `structure.reference_prefix_surface_reduced` 同步为该式,注释说明这是消去可去奇点,不是把 H16 换成切空间求和;
- 未使用 `nan_to_num`、detach 或排除碰撞。

## 2. 碰撞/近碰撞证据与显式容差
`tests/test_q22_reference.py` 新增 `test_the_reduced_prefix_has_finite_gradients_at_collisions`:
- **完全碰撞**:`codes=zeros(1,18)`,torch 面 `L=1.0`,全部 18 个 code 梯度有限;由声明重建的 SymPy 参考在同一输入上通过 `gradcheck`(atol 1e-6);
- **仅 H 重合**(E 各层不同、H 各层相同):torch 面梯度全部有限,且与 SymPy 参考在 `rtol=0, atol=1e-6` 内一致。
- SymPy 与 torch 的取值差实测约 `7.3e-9`(参考内部常量精度),测试改为显式 `rtol=0.0, atol=1e-7`,不再声称 1e-12 绝对一致;梯度断言同样给出 `rtol/atol`。
`math_spec.prefix_path.reduced_surface` 记录该恒等式与容差事实。

## 3. 仍未交付:完整 H16 前缀面
- `prefix_surface_1d` 仍是 E1×H1 缩维;H1 的回读退化为切坐标求和,不覆盖 H16 非共线组合、右折叠顺序或原生码本 Jacobian。
- 完整 `prefix_surface.statements` 仍是说明性伪代码;`structure.reference_prefix_surface` 只接收外部累计位置,不从选中码字产生它们;可解析的完整 32 维 SymPy 面**未交付**。
- 契约的 `h`、`s` 仍声明为 radius=0.5 的 box;真实 `h=(1,1,1)/(1,1,2)`、`s>0` 只由手写测试覆盖,取样域与真实目标域不一致的问题**未解决**,已在材料中列为缺口。
- trainer 仍只读 `levels[-1]`;训练侧 `prefix callable=NOT_IMPLEMENTED`。

## 4. 收据
`q23/tests_20261110T083711.log`:34 条命令、34 个不同文件、**34 pass / 0 fail**,runner `rc=0`,`source_drift: none`;前后清单绑定 `*.py` + `math_spec.json` + `reference_contract.json`。
q22 的稳定与漂移收据、q18 失败收据均保留。

## 5. 状态表
| 项 | 状态 |
|---|---|
| 缩维前缀碰撞反向 | 已修并复验(完全碰撞/H 重合) |
| 完整 H16 前缀符号面 | `NOT_IMPLEMENTED` |
| 真实 h/s 取样域 | 缺口:契约仍 box radius 0.5 |
| 独立审查/冻结 | `BLOCKED`(无候选 round/DAG、无 MCP 返回、无审查收据) |
| 正式绑定/准入/训练/收益 | `NOT_RUN`;旧 `0.0579` 未移交 |

## 6. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e293098a15bbda5aa9f170b1ccd8c7ffe958954a65e599b8c55268d18ee392b6` |
| 候选 `reference_contract.json` | `2f722f69efaa8cbec2da387aba0fca269f81355a3aa2faeafa353570aae9dc12` |
| 候选 `structure.py` | `9152d6c956eb59d215ff452cd11d761cdb7ba824c853c671a2a7f789b61fde70` |
| 候选 `tests/test_q22_reference.py` | `414be825471a75b3c123ccc808e2fa99227bb5df410a530b7c30a2f406602c36` |
| 证据 `q23/tests_20261110T083711.log` | `b40541b8e28a437400ee51e34a248d644fde9402638ee3668a3099c42b870812` |
| 证据 `q23/run_suite.sh` | `120d7d14cc8335070319bc2a366eb68efcde31a7b5d599536c7cd00c124b2938` |
| 证据 `q23/source_pre_20261110T083711.txt` | `b545237b1523c1f53984831fd75080ff3a3a2f8550c03e326632d6bd432c7fab` |
