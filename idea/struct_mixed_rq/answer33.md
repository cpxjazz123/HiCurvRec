# answer33 — 16 维 `mixed_sq` 已按 64 标量精确绑定;全维前缀面仍未绑定
**状态:§3 已交付并通过 1e-12/1e-10 验收;§4 仍 NOT_IMPLEMENTED;冻结 BLOCKED**

承接 `answer32.md`。接受 §1 的更正(§86 在 `CLAUDE.md:206`)与复核。

## 1. 先记账:q32 表里两个几何数字的抽样不同(question33 §2)
| 来源 | seed | 行数 | codes 抽样 | h / s | 几何差 |
|---|---:|---:|---|---:|---:|
| q32 手工 `geometry_off` 行 | 0 | 4 | uniform(−0.1, 0.1) 每坐标 | 1 / 1 | 0.01187007303573362 |
| q32 verdict `control_max_abs_difference` | 17 | 16 | box, radius 0.2 | 1 / 1 | 0.017269443734985068 |

两行不是同一次测量;数值都对,但不可横向当同一抽样比较。

## 2. `mixed_distance` 绑到 16 维 `mixed_sq`(question33 §3)
契约新增两个 call(每个 64 个标量符号 `L_0..L_31,R_0..R_31`):
- `mixed_distance_16_series`,`t < 1e-3`,`d = E + t - t²/3 + 8t³/45`;
- `mixed_distance_16_asym`,`t >= 1e-3`,`d = E + asinh(sqrt(t))²`;
- `E = Σ_{i<16}(L_i−R_i)²`,`rl=sqrt(Σ L_i²)`,`bl_i=tanh(rl)·L_i/rl`,
  `t = Σ(bl_i−br_i)² / ((1−Σbl_i²)(1−Σbr_i²))`(方向与 `full_space` 一致)。

`sympytorch` 无 `Piecewise`,故按两支拆成两个 call;两支合起来覆盖 `phi` 的全域。

### 实测(`build_reference` vs `mixed_geometry.mixed_sq`,float64)
| 场景 | 取值 \|Δ\|max | 梯度 \|Δ\|max |
|---|---:|---:|
| 64 组随机对,`‖H‖≤0.9` | 7.105e-15 | 1.332e-15 |
| 近碰撞 `‖ΔH‖=1e-6`、E 不同 | 2.132e-14 | 9.588e-16 |
| 边界 `‖H‖=0.9` | 7.105e-15 | 2.220e-15 |

要求 `rtol=0, atol=1e-12`(值)/`1e-10`(梯度)`→ 全部通过;`gradcheck` 两侧通过;
`geometry_on=0` 欧氏支与 `E + Σ(HΔ)²` 逐位相等(差 0.0)。

## 3. 生成器与测试归属
- 生成器:`results/research/struct_mixed_rq/q33/gen_mixed_distance.py`(仓库内,无 `/tmp`);
- 候选测试:`tests/test_q33_mixed_distance_binding.py`(两支取值/梯度、欧氏支、64 标量宽度);
- `run_suite.sh` 记录生成器 sha256;本轮收据 `q33/tests_20261110T121515.log`。

## 4. 仍未交付
- **全维前缀面**:`calls` 里仍没有绑定 `reference_prefix_native`(三层累计回读+三层均值)的 call;
  契约里 `structure_objective`/`prefix_surface_1d` 仍是缩维替身。这是冻结的唯一阻塞项;
- 相应的 288 标量生成器未重建 → `NOT_IMPLEMENTED`;
- 冻结、准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益均未发生;旧 `0.0579` 未移交。

## 5. 与准入的关系(本轮之后发生)
owner 于 2026-11-10 明确要求删除“申报公式 vs 训练公式”的自动比对:commit `0b88835b` 已把
`check_reference_agreement` 移出 `run_all`/round 登记/服务端 `admissible`,并以 `CLAUDE.md` §87 记录;
随后 `9f7afdcd` 把 `run_all` 缩到只剩 `check_entry`(§88)。因此本节的 16 维绑定现在**不再是准入条件**,
仅作为可复算的公式证据保留。

## 6. 收据
`q33/tests_20261110T121515.log`:38 条命令、38 个不同文件、**38 pass / 0 fail**,runner `rc=0`,
`source_drift: none`。

## 7. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| 候选 `math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| 候选 `reference_contract.json` | `6b2a4187f99694343b1012f302995150519b740e481937d062a6f28fab07d31d` |
| 候选 `structure.py` | `2e2fff8fc730d9cdca712cd7f469bb3972c5aaef09ca0803d55c12d68093040d` |
| 候选 `tests/test_q33_mixed_distance_binding.py` | `68ec34e9d55137d35d5dbeb3104e11ae73e167abfc3f5ccbdf4b594204eccc9b` |
| 生成器 `q33/gen_mixed_distance.py` | `dbb40f7ad5252dc054eefd4f18421695cf5929262a8d85eb76f846833465658b` |
| 收据 `q33/tests_20261110T121515.log` | `2ded928a01f48a30c8ee4551eea0d0eba4c09968e60f9ad145fe9de2dcf74f30` |
| `q33/run_suite.sh` | `bb8a289d9cf160fb77173062b3ddbc11f5805adf56fc605af6cce4fdba1b6633` |
| 源快照 `q33/source_pre_20261110T121515.txt` | `97c03e21f31f325d770ed73ba080e0e28544904e66591d4aadf7a68a6f972337` |
| 框架 `tools/research/stage2/checks.py` | `297238f6f9f83bdb8308cd4008d96201d0c8700a79f0c09d20ed102073bf3916` |
| 框架 `tools/research/stage2/reference.py` | `8bb7f495ec86359413a5af1c430f08b91e39129cad378fc055bfbcbcb215d50c` |
