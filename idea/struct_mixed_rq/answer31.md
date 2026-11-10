# answer31 — 耦合闸门改绑全维前缀路径并检查控制参数;生成器与 mixed_sq 绑定仍未交付
**状态:coupling 已重绑且有数值;`mixed_distance` 16 维绑定与可重建 288 面 NOT_IMPLEMENTED;冻结 BLOCKED**

承接 `answer30.md`。接受 question31 §1 的核对(注:q30 临时日志已按 §86 删除,本答案不再引用)。

## 1. §19 耦合闸门改绑全维前缀路径
- 新增 `structure.prefix_coupling_surface(codes,h,s,geometry_on)`:`codes` 为 `(B,288)`=3 层×3 商品×32 维,
  内部 reshape 后调用 `reference_prefix_native`(全 E16×H16 三层累计回读),不是缩维面;
- `reference_contract.coupling` 改为该 callable,`behaviour_argument=h`、`control_argument=s`,
  `h`/`s` 以标量给出;
- `tools/research/stage2/reference.py::coupling_verdict` 现在**真正使用** `control_argument`:
  控制参数 +0.5 后要求输出改变,否则 FAIL,并把 `control_max_abs_difference` 写入 verdict;
- `reference_prefix_native` 增加 `geometry_on` 开关,用于数值对照。

## 2. 实际对比数值(CPU、float64、seed 0;`q31/coupling.json`)
| 项 | 数值 |
|---|---:|
| base | 0.359932148621794 |
| behaviour `h+0.5` | 0.6523477932379387 |
| control `s+0.5` | 0.6523477932379387 |
| geometry-off | 0.37180222165752763 |
| behaviour Δ | 0.29241564461614467 |
| control Δ | 0.29241564461614467 |
| geometry Δ | 0.01187007303573362 |
`coupling_verdict` 返回 `PASS`,`max_abs_difference=0.31446226141885186`,
`control_max_abs_difference=0.31446226141885186`。行为关、控制参数、几何关三项都能改变数值。

## 3. 规格措辞
`faces.to_freeze` 与 `coverage_gaps` 新增:契约仍只绑缩维面,训练的三层 H16 项与 `mixed_sq` 的 16 维交叉比
没有符号绑定,因此机器一致性不针对训练函数。

## 4. 仍未交付(question31 §4)
- 288 标量生成器:临时脚本按 §86 删除,未在候选树重建/修正,`NOT_IMPLEMENTED`;
- `mixed_distance` 仍绑 2 坐标 `reference_surface`,未绑 16 维 `mixed_sq`;
- 可重建完整面、冻结、准入、正式 Stage2→Stage3 与增益均未发生;旧 `0.0579` 未移交。

## 5. 收据
`q31/tests_20261110T113848.log`:37 条命令、37 个不同文件、**37 pass / 0 fail**,runner `rc=0`,
`source_drift: none`。

## 6. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| 候选 `math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| 候选 `reference_contract.json` | `88b24cdda08a0c83064cc1bbeccf61341f6d5e7a9cc19eb26d57733aa3837fea` |
| 候选 `structure.py` | `2e2fff8fc730d9cdca712cd7f469bb3972c5aaef09ca0803d55c12d68093040d` |
| 框架 `tools/research/stage2/reference.py` | `a500513ebc93cacd1568d5c8b88a8e19466cf7b2cc7e9e14479cec2a77b17564` |
| 证据 `q31/coupling.json` | `fcaf86c800fb3e5bb64580ebfc5e2d56efbd0b83780e906cc546939518556c63` |
| 证据 `q31/tests_20261110T113848.log` | `b91eb67c02578d284f29592c9bb65bd1265041287e63208e6800c31542032bbb` |
| 证据 `q31/run_suite.sh` | `1f23d89e2bdd042f5182441d2386a3d0301364800fc79c94ebb817c734523250` |
| 证据 `q31/source_pre_20261110T113848.txt` | `2e1b7e5bcb5a29baff43bb239b6f5d2da3d016c20911897bcd83831b146e2873` |
