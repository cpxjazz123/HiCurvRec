# answer25 — 声明适用域并统一饱和策略，近碰撞用例改为真实商品
**状态:DRAFT 未冻结;完整符号面/取样域/生产接线仍未交付;冻结 BLOCKED;训练/收益 NOT_RUN**

承接 `answer24.md`。接受 question25 §1 对 q24 收据与在域内一致性的核对。

## 1. 适用域与唯一饱和策略已写入权威材料
`math_spec.prefix_surface.domain` 与 `reference_contract.calls[*].domain`:
- **声明域**:每个码字 `‖C_l^H‖ ≤ 1`;记录的检查点实测 L1/L2/L3 H 范数为 `0.63 / 0.27 / 0.04`(均值),离边界尚远,但源码没有范数约束,不据此声称"不会越界";
- **唯一饱和策略**(采用生产实现,不保留两套 epsilon):`to_ball` 只在饱和尾部乘 `(1-1e-6)`,`log_0` 用 `atanh(norm.clamp(max=1-1e-9))` 且 `norm>1e-12` 时除以范数;
- **域外行为**:参考复现同一饱和映射,两侧返回同一有限值(不再一套 `1e-12`、一套 `1e-9`);超出 `‖C^H‖>1` 的记录是诊断,不是训练门槛。

## 2. 参考的径向映射已对齐(先红后绿)
新增 `test_the_reference_radial_map_matches_production_over_the_measured_norms`(8 点:0.16,1,5,10,15,19,20,30):
- **红**(修复前):`‖t‖=15` 时参考回读 `14.1621`、生产 `10.7082`,差值来自参考的 `1e-12` clamp 与缺少尾部收缩;
- **绿**(修复后):`structure._ball/_log0` 改用生产的 `1-1e-6` 尾部收缩与 `1-1e-9` clamp,8 点全部在 `rtol=0, atol=1e-9` 内一致。
原 `_phi_stable` 的 `t<1e-6` 分支保留;它不与生产阈值冲突,question25 §1 已实测差值在 1e-13 量级。

## 3. 近碰撞用例改为真实商品
`test_the_near_collision_uses_distinct_products_and_covers_the_h_domain`:
- 三个商品不再共用同一常数张量:第 1 层 H 取 `0.5, 0.5+1e-5, 0.5-2e-5`,E 取 `0.01 / 0 / -0.02`;
- loss 在 `[0.98, 1.0]`(实测 0.9966),全部梯度有限;
- 另取 `1.8×` 幅度(H 范数接近 0.9,仍在声明域内)验证取值与梯度有限;
- 断言显式给出 `rtol/atol`;原 `torch.full` 的三点完全重合用例不再被称作"近碰撞"。

## 4. 仍未交付(与 q24 相同)
- 完整 E16×H16 的 288 标量 SymPy 面仍未可重建;`reference_contract` 只有三个 call;
- 契约 `h/s` 仍是 `radius=0.5` box,真实 `h=(1,1,1)/(1,1,2)`、`s>0` 未进入取样域;
- `training_hooks._positions` 仍只返回 `levels[-1]`,三层前缀监督未接线;
- 每层 `max ‖C_l^H‖` 的训练诊断**未加入**(代码改动未做,如实记 NOT_IMPLEMENTED)。

## 5. 收据
`q25/tests_20261110T102115.log`:35 条命令、35 个不同文件、**35 pass / 0 fail**,runner `rc=0`,
`source_drift: none`;绑定 `*.py` + `math_spec.json` + `reference_contract.json`。

## 6. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e6303ae9e0cf8fc0c1903a49dba4718c5441f26b74d646d4b8f84b8b3ce1a6a1` |
| 候选 `reference_contract.json` | `02f759fbe4c1712c789e76d4386f1960b97a6287b101e6d9283a27c93a392fdc` |
| 候选 `structure.py` | `a6014949e72852b224195e86688c4a2222a0b89d0c9a413c4152d85c9720c762` |
| 候选 `tests/test_q24_full_prefix.py` | `526a55737384ad62c5d156cdc8e9ee8bc303b2f1e84d96696f5a5d9bfaa6080e` |
| 证据 `q25/tests_20261110T102115.log` | `ee4268706fb28ba8d8459b4d699532ff2e2a0f60c9008369548cbc64398b2ccd` |
| 证据 `q25/run_suite.sh` | `875e62cd3625328cfadfa65c62776a8173f4abbb83218d43365cada6fb08190d` |
| 证据 `q25/source_pre_20261110T102115.txt` | `8fdef32ef7957048b47f2783a52215f933697e75dceaa363c9aac532e8416074` |
