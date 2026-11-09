# answer24 — 完整 E16×H16 前缀参考面实现，独立公式与碰撞梯度
**状态:DRAFT 未冻结;完整符号面未交付;trainer NOT_IMPLEMENTED;冻结 BLOCKED;训练/收益 NOT_RUN**

承接 `answer23.md`。接受 question24 §1 对缩维修复、§2 对测试表述边界的核对。

## 1. 完整 H16 前缀参考面已实现
`structure.reference_prefix_native(codes, h, s)`:
- 输入:长度 3 的选中码字序列,每项 `(B,3,32)`;商品轴=3、层轴=外层 3、坐标轴 E16/H16;
- 独立实现(不调用 `mixed_geometry`/`native_product`):径向球映射、`log0`、Möbius 加法、右折叠
  `R_l=(Σ C_j^E, log0(B_1⊕(B_2⊕…⊕B_l)))`、三对距离、逐层 stress、三层均值;
- 零点安全:范数用 `sqrt(·²+1e-24)`、球映射/log0 的零分支用 clamp,`phi` 在 `t<1e-6` 用
  `t-t²/3` 级数(与生产的 phi 分段一致),碰撞处梯度有限;
- 输出 `(B,)` 逐三元组值;样本均值不在该函数内。

`tests/test_q24_full_prefix.py` 三例(全绿):
| 用例 | 结果 |
|---|---|
| 与测试内独立手写右折叠对照(非调用候选实现),`rtol=0, atol=1e-10` | 一致 |
| 全零碰撞与近碰撞(1e-5) | `L` 有限、全部梯度有限 |
| 非碰撞双侧 `gradcheck`(atol 1e-8, rtol 1e-6) | 通过 |

## 2. 仍未交付
- **完整符号面**:`reference_contract` 仍只有缩维 `prefix_surface_1d`;完整 E16×H16 的 288 标量语句
  未能写成可重建 SymPy 面。当前完整面只有 torch 参考 + 测试,不是冻结材料。
- **真实取样域**:契约 `h/s` 仍是 `radius=0.5` 的 box;真实 `h=(1,1,1)/(1,1,2)`、`s>0` 只由手写测试覆盖。
- **生产接线**:`training_hooks._positions` 仍只取 `levels[-1]`;训练侧 `prefix callable=NOT_IMPLEMENTED`。
- **编码器/码本梯度分账、Sinkhorn 分支、冻结与正式准入**:未在真实模型上执行。

## 3. 收据
`q24/tests_20261110T085008.log`:35 条命令、35 个不同文件、**35 pass / 0 fail**,runner `rc=0`,
`source_drift: none`;绑定 `*.py` + `math_spec.json` + `reference_contract.json`。q23/q22/q18 收据保留。

## 4. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `87fd1379a6d30dfb4a3849c8c66f3deee99fcefc0c16c006d63e639ae7c5c5c5` |
| 候选 `math_spec.json` | `e293098a15bbda5aa9f170b1ccd8c7ffe958954a65e599b8c55268d18ee392b6` |
| 候选 `reference_contract.json` | `2f722f69efaa8cbec2da387aba0fca269f81355a3aa2faeafa353570aae9dc12` |
| 候选 `tests/test_q24_full_prefix.py` | `bc17b919340d98d6ba16bd6e18e87fa70263c7aa0c332c09c9a0123b09009ec4` |
| 证据 `q24/tests_20261110T085008.log` | `93097dc0b609fe286b8e0352161893676795f826c96fa10395f42b1be546fdcb` |
| 证据 `q24/run_suite.sh` | `5d085f703bbb71bc24e69e4aaca7ca30a9877837abe554d65f259b63d1997c76` |
| 证据 `q24/source_pre_20261110T085008.txt` | `bcb225be32c8481d5411a21ad30fe0157e619b7f0e43ebfa526918bf608b6288` |
