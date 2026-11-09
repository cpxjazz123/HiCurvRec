# answer21 — 前缀参考面可执行并经梯度检查，距离反向补上 H 判据
**状态:DRAFT 未冻结;前缀参考面已实现,trainer 仍 NOT_IMPLEMENTED;冻结 BLOCKED;训练/收益 NOT_RUN**

承接 `answer20.md`。接受 question21 §1 的核对;本轮修 §2 的三处文字矛盾、§3 的误差归因、§4 的 H 梯度缺口,并交付一个可执行的完整前缀**参考面**。

## 1. §2 三处矛盾已原位修正
1. `residual_and_readback` 不再写 `B_l=code_ball(C_l^H)`;统一为
   `B_l = code_ball(C_l) = b(C_l^H)`,输入是完整 32 维码字;旧的矛盾句已删除,不再并存两个定义。
2. `definitions.nested_compose` 改为**右折叠**:先取最后一个 `B_L`,再从后向前 `B_j (+) accumulated`;
   `prefix_path.cumulative` 与 `reconstruction` 的括号写成
   `log_0( B_1 (+) ( B_2 (+) ( ... (+) B_l ) ) )` 与 `log_0( B_1 (+) ( B_2 (+) B_3 ) )`。
   明确 Möbius 加法不满足普通结合律,不允许左到右重排。
3. `phi_implementation` 的误差归因改正:该点 `t=0.0526 > TAU=0.001`,走 `asinh` 分支;
   1.256e-9 差异来自诊断张量的 float32 默认类型,不是级数截断。解析对照固定 float64
   (`math.asinh²=0.051711885865117525`),测试已用 `dtype=D`。

## 2. 前缀参考面:`structure.reference_prefix_surface`
沿用 `question8.md §4` 的唯一目标,新增纯参考函数(不是 trainer 接线):
```
reference_prefix_surface(qs, h, s)
  qs: 三层累计码字位置 (batch,3,32); h: (batch,3) 对应 (ca),(cb),(ab); s: (batch,1)
  d01,d02,d12 = mixed_sq(...) 在完整 32 维位置
  e = (d/(s^2 h^2)-1)^2, L_l = (e01+e02+e12)/3, L = (L1+L2+L3)/3
```
`math_spec.prefix_path.program` 写同一顺序的赋值;`prefix_path.reference` 指向该函数。
新增测试(`tests/test_q19_full_space.py`):
- 与手写同式循环逐元素一致(1e-12);
- 对三层位置的 H 分量梯度范数非零(完整 32 维通路);
- 全零碰撞位置给出 `L=1`(合法坍缩值)。

## 3. 距离反向的 H 判据
`tests/test_q19_full_space.py` 新增:
- 真实 `mixed_sq` 对左右输入的 H 梯度范数 `>1e-6`;
- 同一进程内把 `G.poincare_sq` 换成 `saved(x,y).detach()`,前向不变而 H 梯度范数 `<1e-12`,
  同一判据拒绝"取值正确、H 反向被切断"的变体;`finally` 恢复。
倒数变体负例保留;三个用例均使用 float64 解析期望;不在 STE/硬选择上使用有限差分。

## 4. 仍未交付的项(如实)
- **全维前缀的 SymPy 契约调用未新增**:`reference_contract.calls` 仍是单层 E1×H1 的两个面;
  前缀的权威声明目前是 `prefix_path.program` 的有序赋值 + 可执行参考函数与测试,
  尚不是可由 `tools/research/stage2/reference.build_reference` 重建的符号语句。这是本轮未闭合项。
- trainer 仍只应用末层结构项;`_positions` 仍 `levels[-1]`;`prefix_surface.callable=NOT_IMPLEMENTED`。
- 独立审查/冻结:`BLOCKED`(无候选 round/DAG 绑定、无 MCP 请求返回、无两位审查收据);
  本地导入缺口只记事实。
- Stage2/Stage3、收益:`NOT_RUN`;旧 `0.0579` 未移交、未达 `0.065`。

## 5. 收据
`q21/tests_20261110T080322.log`:33 条命令、33 个不同文件、**33 pass / 0 fail**,runner `rc=0`,
`source_drift: none`;前后清单继续绑定 `*.py` + `math_spec.json` + `reference_contract.json`。
q20 的 33/33 与 q18 失败收据保留,本轮未重跑旧失败。

## 6. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `486b696659e7a3457f911129dee9dd441ae29e8d1d302ad8959f18ac59dd0ff5` |
| 候选 `structure.py`(新增参考面) | `b0e0adf4f5a5a2092c389949b10bac4015b084a00781fb8a239bb7fbef47c753` |
| 候选 `tests/test_q19_full_space.py` | `d017c77ac61d7983e27adeaec8241ed5f53edc6e8a3393ce39507538f24e3c10` |
| 证据 `q21/tests_20261110T080322.log` | `6dd26503dfd7b47c4b9ed3a7872772b9d1934431a3a3e28898faceb56635d465` |
| 证据 `q21/run_suite.sh` | `47001808e0478c72714467c0c36bfaca649d1af5c93fb97764feeb672597f7eb` |
| 证据 `q21/source_pre_20261110T080322.txt` | `495b5709cb490c8b8ff2daaa9e794b656bb1f347e48188ca9e23b594b76e5805` |
