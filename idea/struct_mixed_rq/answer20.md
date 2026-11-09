# answer20 — 前缀数学成为可执行定义，真实距离用例改走 mixed_sq
**状态:DRAFT 仍未冻结;完整前缀 NOT_IMPLEMENTED;冻结 BLOCKED;绑定/准入/训练/收益 NOT_RUN**

承接 `answer19.md`。接受 question20 §1 对交比方向、Möbius 残差、测试清理与 q19 收据身份的核对。

## 1. `B_l` 的接口映射已纠正
草稿写 `B_l = code_ball(C_l^H)`,与真实接口不符。规格改为:
```
C_l 是完整 32 维码字
B_l = code_ball(C_l),  code_ball 取 C_l[16:32] 再 to_ball
```
并记录:把 16 维 H 子向量直接传入 `code_ball` 得到空切片;完整 `[1,32]` 输入得到 `[1,16]`,与 `to_ball(C_l^H)` 逐元素相等。这是接口映射更正,未改生产接口。

## 2. 前缀定义展开为可复算的连续对象
`math_spec.continuous_surface.prefix_surface` 现含:
- `definitions`:`to_ball`(零点延拓)、`log_zero`、`mobius_add` 的显式分式、`mobius_left_inverse`、`nested_compose` 的括号次序、`code_mapping`、距离与 `c=1`、`4phi`/`phi` 归一化,以及 `phi_implementation` 的分段实现(小 t 用 `t-t²/3+8t³/45` 级数,与精确 `asinh²` 在 t=0.0526 相差约 1.3e-9)。
- `prefix_path`:`r_0^E=z^E, r_0^H=b(z^H)`;`r_l^E=r_{l-1}^E-C_l^E`;`r_l^H=(-B_l)(+)r_{l-1}^H`;`R_l=(Σ C_j^E, log_0(B_1(+)…(+)B_l))`;`q_l=R_l+z-sg(z)`;同一三元组的三对 h、`L_l` 与 `L=(L1+L2+L3)/3`;decoder 输入 `(Σ C_l^E, log_0(B_1(+)B_2(+)B_3))`;`L=L_base+λ·L_prefix`;闭合 `h=(1,1,1)`、分叉 `h=(1,1,2)` 精确进入同一距离/损失,不以 `1+tanh(u)` 适配器冒充。
- `faces`:`implemented`(末层结构项,唯一在训练中生效)、`to_freeze`(完整 E16×H16 三层前缀)、`to_implement`(`_positions` 仍 `levels[-1]`,必须走同一量化/回读)。三者不再混用。
- `calibration`:`s` 为初始化后末层 h=1 对的 `sqrt(clamp_min(d_M²,1e-8))` 中位数;`structure.calibrate_scale` 空/≤1e-6 与 `training_hooks.calibrate_structure_scale` 无 triples 时两处 `return 1.0` 均列为待修默认;`λ` 为完整前缀目标的 encoder / 合并码本两块预算一次确定,decoder 排除。
- `excluded`:Sinkhorn/导出分支为离散选择、STE 只声明接收侧 Jacobian(普通有限差分不构成证明)、H 多维与逐层 Jacobian 证据继续由既有原生测试承担。

## 3. 距离用例现在检查真实函数
`tests/test_q19_full_space.py` 重写:
- `test_the_real_distance_matches_the_independent_cross_ratio`:用切坐标输入调用 **`G.mixed_sq`**,独立按 `t=|Δb|²/((1-|b|²)(1-|b'|²))`、`phi` 计算期望;断言生产值 `0.05052627184756516`、球坐标静态 `phi=0.05171188712120056`(与精确 `asinh²` 的 1.3e-9 级数差单独记录)。
- `test_a_reciprocal_distance_variant_is_rejected_by_the_same_check`:仅在该测试内把 `G.poincare_sq` 替换为倒数变体,同一调用路径得到与期望差 `>1e-2`,`finally` 恢复;生产实现通过。
- 残差用例继续调用 `N.residuals`;既有 H 多维/逐层 Jacobian 测试未改。

修复前该文件只自算 num/den 并调用 `G.phi`,不触碰 `mixed_sq`(question20 的负对照已指出);本轮直接修正并复跑。

## 4. 收据
`q20/tests_20261110T074933.log`:33 条命令、33 个不同文件、**33 pass / 0 fail**,runner `rc=0`,`source_drift: none`;前后清单同时绑定 `*.py`、`math_spec.json`、`reference_contract.json`(57 项)。旧 33/33 与 q18 失败收据均保留,不补签。

## 5. 状态表
| 项 | 状态 |
|---|---|
| 全维前缀数学定义 | 已展开为方程/形状/域与实现映射;DRAFT |
| 单层折叠结构项 | 已实现(末层),接线测试通过 |
| 完整三层前缀 | `NOT_IMPLEMENTED`;`_positions` 仍 `levels[-1]` |
| 独立审查/冻结 | `BLOCKED`:无候选 round/DAG 绑定、无 MCP 请求返回、无两位审查收据;本地导入缺口仅记事实 |
| 正式绑定/准入 | `NOT_RUN` |
| Stage2/Stage3 与增益 | `NOT_RUN`;旧 `0.0579` 未移交、未达 `0.065` |

## 6. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `93741eaab9b1b53d57242681b21126578563008a4ffab38a543404e2ef99650f` |
| 候选 `reference_contract.json` | `41a92152df616189f3dfa36d33a9a2512ba876c703559da35a600248eadf7e51` |
| 候选 `tests/test_q19_full_space.py` | `7ea4b56b89ed459ac680effcf493e5e8f4c583bd6e433f713c967b25cc254194` |
| 证据 `q20/tests_20261110T074933.log` | `bcbede24a78731e3bae0797dd52c43ec82967b4f4f4e09a2c4f3d6daebade79b` |
| 证据 `q20/run_suite.sh` | `ae63c0d9a1ee7f79edeb48a6d3936fc393a94fd349d9910f5429b74cdf5378d2` |
| 证据 `q20/source_pre_20261110T074933.txt` | `49cd4263dfd23d637d7856b06b4e479fb6114f9a6447d708acea68ad7db2ce90` |
