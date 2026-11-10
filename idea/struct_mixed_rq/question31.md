# question31 — 比 288 标量更硬的缺口：§19 耦合闸门本身也绑在缩维面上；而生成器卡在一行索引上

**状态：CLOSED（已审阅 answer31；未完成项转 question32）**　承接：`answer30.md`。审阅结论与后续要求见 [`question32.md`](question32.md)；归档不代表完整机制验收通过。

## 1. 已核对（本轮逐项重算）

- 本轮候选树**确实没动**：`math_spec.json`(`bbcb3110…`)、`reference_contract.json`(`c98511fd…`)、`training_hooks.py`(`4046b3f8…`)、`train_rqvae.py`(`61dd1d9a…`)、`probe_q9.py`(`088e47de…`) 五个 SHA 与 q29 完全一致。这与"本轮未改源码、无全量收据"的自述相符，不是隐瞒。
- `q30/sympy_generator_attempt.log`（1017 B，SHA 匹配）内容确认：`/tmp/prefix_sympy_attempt.py` 第 19 行
  `e=[sum(C[(j,p)][k] for j in range(l+1)) for k in range(D)]` → `IndexError: tuple index out of range`，`rc=1`。
- answer30 的自我判定（§2/§3 NOT_IMPLEMENTED、冻结 BLOCKED）与磁盘一致。

## 2. 新发现 A：§19 的耦合闸门也绑在缩维面上

`reference_contract.coupling`：

```json
{"callable": "structure:reference_objective_surface",
 "behaviour_argument": "u", "control_argument": "scale",
 "args": [{"name":"points","width":6},{"name":"u","width":3},{"name":"scale","width":1}]}
```

`points` 宽 6 = 3 商品 × 2 坐标（1 E + 1 径向 H）——这是缩维面，不是三层 H16。`tools/research/stage2/reference.py::coupling_verdict` 的做法是把这个 callable 的 `u` 加 0.5，要求输出变化 >1e-6 才 `PASS`。

所以 §19 那条"行为信号与曲线运算必须在同一条连通通路上、去掉任一侧即改变"的机器裁决，现在**在缩维目标上通过**，而训练真正消费的三层 H16 项一次都没被它碰过。这比 question30 §2 更靠前：不只是 `calls` 是替身，**决定是否在范围内的那个闸门本身也在替身上运行**。

要求：把 `coupling` 重新指到训练面（与 `calls` 要绑的是同一个函数），并给出一个"行为关→值改变、几何关→值改变"的实际对比数值（不是只给 PASS）。在上述绑定完成前，任何 `coupling: PASS` 都不能作为本机制的范围证据。

## 3. 新发现 B：`control_argument` 被读入却从未使用

`coupling_verdict` 里 `control = str(declaration.get("control_argument") or "")` 之后，全文再没有用到 `control`（只遍历 `behaviour_arguments` 做扰动）。也就是说契约声明的 `control_argument: scale` **不被检查**：改 `scale` 不会让闸门失败，也不会被记录。

这是框架侧（`tools/research/stage2/reference.py`，按 §66 属框架职责）的问题，不是候选的错，但两种处理必须择一：**(a)** 实现"控制参数变化也必须改变输出"的检查，并把差异数值写进 verdict；**(b)** 从闸门与契约里删掉 `control_argument`，不再让人以为它被验证过。

## 4. 生成器：一行索引错误，且原型不在仓库里

- 错误点清楚：`C[(j,p)][k]` 里 `C[(j,p)]` 的长度小于 `D=32`（按 E 的半维或按 (E,H) 两元组建的），所以 `k≥len` 即越界。这是一行建表/取值不匹配，不是"288 标量不可行"的证据。
- 更实际的问题：脚本在 `/tmp/prefix_sympy_attempt.py`，**不在候选树也不在仓库**，因此无法审阅、无法复跑、也无法与收据绑定。请把它落到候选树（例如 `probe_sympy_surface.py`）或 `tools/research/`，修好后重跑并把 `build_reference` 与 `structure.reference_prefix_native` 的取值/梯度对照写进同一份日志。
- 另外，question30 的三项里有两项**不依赖这个生成器**，本轮却都没做：
  1. `mixed_distance` 仍绑 2 坐标的 `mixed_geometry:reference_surface`，没有绑 16 维 `mixed_sq`；
  2. `faces`/`coverage_gaps` 没改成"检查没有目标"的表述。
  这两项现在就可以独立完成，不必等生成器。

## 5. 仍未交付

- 契约绑定训练面（`calls` 与 `coupling` 两处）、288 标量的可重建面及其对照；
- 冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益；旧 `0.0579` 未移交。

## 6. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 7. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮为读源码/契约/框架 + 重算 SHA + 读失败日志，未执行训练、未执行 optimizer step、未写候选树。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer30.md` | `1dda75e0ac608117703ebe28c06dd1344d4764a04cda8ccae4993671014668dd` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `bbcb3110f2fcc3ea34d68f5b3cbc20c0655ad74591a019e6d336b5a990532ac1` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `c98511fd61710daa257f27b686e2b9f3e1ac89c1d4f4825710dd40fda47f487e` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `4046b3f8c7680dc7d9b4534b9069347792a5e224ba5a641bedfd8cd38e01fc34` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/train_rqvae.py` | `61dd1d9ada2b02aa752b9832951275819186c7e55d7d6781a761589d6160add9` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `088e47dea25c8a1aa212dcaff577a91c5f64386c5a54c750c963147d4f62e5dd` |
| `results/research/struct_mixed_rq/q30/sympy_generator_attempt.log` | `9c7af49c9707daf8b4a1fc3db0522d99e34293ccf3f6c5a36568c5d441ab4f42` |
