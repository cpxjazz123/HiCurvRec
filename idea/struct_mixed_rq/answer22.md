# answer22 — 可重建的缩维前缀符号面，距离梯度用独立解析判据
**状态:DRAFT 未冻结;完整 32 维前缀仍无 SymPy 契约;trainer NOT_IMPLEMENTED;冻结 BLOCKED;训练/收益 NOT_RUN**

承接 `answer21.md`。接受 question22 §1 的核对。

## 1. 新增可由构建器重建的前缀符号面
`reference_contract.calls` 新增 `prefix_surface_1d`(第三个 call):
- 参数:`codes` 宽 18 = 3 层 × 3 商品 × (E,H) 各 1 维;`h` 宽 3;`s` 宽 1;
- 语句:三层累计 `qe/qh = sum_{j<=l} c`、`b=tanh(qh)`、交比 `t=(Δb)^2/((1-b^2)(1-b'^2))`、
  `d=(Δqe)^2+asinh(sqrt(t))^2`、`e=(d/(s^2 h^2)-1)^2`、`L_l=(e01+e02+e12)/3`、`L=(L1+L2+L3)/3`;
- 一层 H 坐标下 `log_0(B_1(+)...(+)B_l)` 恰等于切坐标求和,故累计回读即前缀和;
- `callable=structure:reference_prefix_surface_reduced`,实现同一算式;`h` 轴为三元组的
  `(centre,a),(centre,b),(a,b)`,返回 `(B,)` 逐三元组值,样本平均不在该面内。
`tests/test_q22_reference.py`:
- 由 `tools/research/stage2/reference.build_reference` **实际重建**该 call,与 torch 实现逐元素一致(1e-12);
- 两侧均通过 `gradcheck`(float64, eps 1e-6);
- 缩维面不是 32 维几何,`full_space`/`prefix_path.function` 已注明;完整 E16×H16 前缀仍是
  `structure.reference_prefix_surface` 的 torch 参考 + `prefix_path.program`,**尚无可解析的 SymPy 契约调用**。

## 2. 距离梯度改用独立解析对照
`tests/test_q22_reference.py`:
- 解析式自行实现径向球映射 `b(h)=tanh(|h|)/|h|·h`、交比与 `asinh²(√t)`,不调用 `G.mixed_sq`/`G.phi`;
- 生产 `mixed_sq` 对该非共线点的 H 梯度与解析梯度在 1e-10 内一致;
- 负对照 `2*v - v.detach()`(前向同、反向倍率 2)在同一判据下被拒绝(差异 0.20 vs 0.40 量级);
- 保留 q21 的 detach 与倒数负例。

## 3. 接口与轴澄清
`prefix_path.reference` 写明:`qs` 是长度 3 的序列,每项 `(B,3,32)`;第二轴是三元组商品,不是层轴;
返回 `(B,)`。上一版把 qs 写成单个 `(batch,3,32)` 的描述已更正。

## 4. 收据
`q22/tests_20261110T082222.log`:34 条命令、34 个不同文件、**34 pass / 0 fail**,runner `rc=0`,
`source_drift: none`,前后清单绑定 `*.py` + `math_spec.json` + `reference_contract.json`。
(q22 另有一份运行中途规格被修改的收据,drift 非 none,已保留未删。)

## 5. 仍未交付
- 完整 32 维前缀没有 SymPy 可重建契约;当前可重建面是 1 维缩维,覆盖累计回读/三对/三层算术,
  不覆盖 H16 向量交比与多维 Jacobian。
- trainer 仍只读取 `levels[-1]`;`prefix callable` 在训练侧 `NOT_IMPLEMENTED`。
- 独立审查/冻结 `BLOCKED`(无候选 round/DAG、无 MCP 返回、无两位审查收据);
  Stage2/Stage3/增益 `NOT_RUN`;旧 `0.0579` 未移交。

## 6. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `56faad1d0708d118444dee9fe9aa977478f2f43fc488996cd7f013e2da92ff55` |
| 候选 `reference_contract.json`(新增缩维 call) | `2c9bc9f0ccd6b366bb6231c19d4f96e17a2c616bd5dc4a9043daa0931ed7c648` |
| 候选 `structure.py` | `262d90eed4a3af5be7a8099d780cafda0a4fefa51bdd1bf69af34c0d8fd2ced8` |
| 候选 `tests/test_q22_reference.py` | `46caf990f31d2a263853019cd9c90f9c91383a08d3e1466d290439ade252dc3c` |
| 证据 `q22/tests_20261110T082222.log` | `ba1683199801da345f61bb1801afd9e521c49aa2f30fa575dd14155c9301c019` |
| 证据 `q22/run_suite.sh` | `d7cc473fd556e0efa8cd8cd630c4446566af9c84ef8872b0acbd82c3450f24e7` |
| 证据 `q22/source_pre_20261110T082222.txt` | `a0f3d16e1720020a995c9e73c94ca0608dff28840ebcf1444d96e9857a6891ac` |
