# question26 — 饱和域已闭合，但"声明域"无人读取：`h`/`s` 仍是有符号 box、`s=0` 只在没抽到时不炸；`_positions` 的单层理由不成立

**状态：CLOSED（已审阅 answer26；未完成项转 question27）**　承接：`answer25.md`。审阅结论与后续要求见 [`question27.md`](question27.md)；归档不代表完整机制验收通过。

## 1. 已接受（本轮逐项重算，不是读日志）

- **收据**：`q25/tests_20261110T102115.log` 35 条命令、35 个不同文件、全部 `rc=0`、`source_drift: none`；7 条 SHA 我逐文件重算全部匹配（含 `math_spec.json`、`reference_contract.json`、`structure.py`、`test_q24_full_prefix.py` 与三份证据）。
- **饱和策略已统一到生产**：`structure._ball` 用 `(1-1e-6)` 尾部收缩、`_log0` 用 `atanh(clamp(max=1-1e-9))`，与 `mixed_geometry` 一致，不再有第二套 `1e-12`。
- **我做了组合面差分，不只是径向基元**：把 `reference_prefix_native` 与生产原生通路 `reference_prefix_surface([reconstruct(codes[:l+1])])` 对齐，同 `h=[[1,1,2]]`、`s=[[1]]`：

| 用例 | 参考 | 生产 | 最大取值差 | 最大梯度差 |
|---|---:|---:|---:|---:|
| 域内 ‖·‖≈0.16 | 0.8265806468482665 | 0.8265806468482665 | 0.0 | 5.55e-17 |
| 声明域上沿 ‖·‖≈0.9 | 0.3574 | 0.3574 | 0.0 | 5.55e-17 |
| 饱和域 ‖·‖=20 | **14212.993641245414** | **14212.993641245414** | **0.0** | **0.0** |

question25 里同一 ‖·‖=20 输入曾是 `217789.82213240952` vs `14212.993641245414`（取值差 203576.83、梯度差 3269.00）；现在两侧完全一致。**"参考就是生产那个函数"在 0.16 / 0.9 / 20 三点成立，含饱和域。** 这一点不再追问。
- **近碰撞已改为真实商品**：`0.5 / 0.5+1e-5 / 0.5-2e-5` 的 H 加 `0.01 / 0 / -0.02` 的 E，三点互不相同；另有 1.8×（H 范数≈0.9）覆盖声明域上沿；容差显式。
- **域确实写进了权威材料**：`math_spec.continuous_surface.prefix_surface.domain`（codewords / saturation / outside 三段）与 `reference_contract.calls[*].domain`。question25 §2 的"未声明"已关闭。

## 2. 新发现 A：声明域没有任何读取路径，机器实际采样的仍是会除零的 box

`tools/research/stage2/reference.py::_inputs` 只读 `args[].{width,radius,sampler,scalar,samples}`；`domain` 在该文件里只出现在 docstring 一句话，`tools/research/stage2/checks.py` 完全不读它。也就是说：

- 契约真正被采样的参数是 `codes` box r=0.2、`h` box r=0.5、`s` box r=0.5 —— **h、s 都是带符号的**；
- 而被冻结的面是 `(d_M²/(s²h²)-1)²`：`s=0` 或 `h=0` 是**除零**，`s<0`、`h<0` 在机制自身定义里不存在（真实取值是 `h=(1,1,1)/(1,1,2)`、`s>0`）；
- 现在能过只是因为 16 行均匀采样没抽到精确 0，**不是被拒绝**。声明域既不约束采样，也不拒绝样本，它只是 JSON 里的一段文字。

同一份材料内部还自相矛盾：`math_spec.continuous_surface.prefix_surface.coverage_gaps[0]` 写"reduced adapter uses `h=1+tanh(u)`"，而 `reference_contract.calls[prefix_surface_1d].args` 写 `h` 是 box r=0.5。机器按后者采样，前者是错的。

要求（同一批交付）：

1. `h`/`s` 改成**构造上为正**（`h=1+tanh(u)`、`s=softplus(v)` 或 log 参数化），不再靠"抽样不命中"；
2. `domain` 必须被机器消费：`_inputs` 对域外样本拒绝或重采样，或 `checks.py` 增一条断言；
3. 补一条**先红后绿**的测试：喂 `‖C^H‖=2`、`s=0`、`h=0` 这类域外/退化样本，期望被拒绝，而不是静默返回 inf 或 NaN；
4. `coverage_gaps[0]` 与 `args` 的口径二选一对齐。

注：既然饱和域两侧现在数值一致，域的作用不再是"让参考成立"，而是声明"映射在哪一段仍是可逆反演"（实测 ‖·‖≲10 保真、≈19 起 `tanh` 饱和后回读塌到 7.25）。这两个数值应写进 `domain.outside`，而不是只留 `≤1`。

## 3. 新发现 B：`_positions` 的"会改坏机制"理由不成立

`training_hooks._positions` 的 docstring 说：加 `codes_to_positions` "会引入第二条码本路径并静默改变机制"。前半句对，但它与当前事实无关：

- `model.forward_with_codes` 每层 `levels.append(quantized_x)`，而 `quantized_x = reconstruct(codes 到该层) + STE`，所以 `levels` 里**已经就是** `q_1, q_2, q_3` 三个累计回读，出自同一次前向、同一次量化、同一次回读；
- 用 `levels[0] / levels[1] / levels[2]` 既不新增 gather，也不新增码本路径，正是规格 `prefix_path.cumulative` 写的 `R_l`；
- 规格自己的 `prefix_path.loss` 写 `L_prefix=(L1+L2+L3)/3`，`faces.to_implement` 也写"wiring: `_positions` still returns `levels[-1]`"。

现在只算 `L3`，等于把机制定义里的两层监督丢掉。这不是"接线风格"，是机制没实现完。

要求：同一前向取三层 `q_l`，对**同一组** TRAIN 三元组分别算三对 stress 再等权平均；不新增码本路径、不动 Sinkhorn 末层分支、不改导出。TDD：先写一个改动前会失败的测试（例如三层损失与仅末层损失在同一批三元组上不相等，且三层版对 `levels[1]` 的梯度非零）。

## 4. 仍未交付：可重建的完整符号面

`math_spec.continuous_surface.prefix_surface.statements` 只有 6 条，内容是 `q1 = native_cumulative_codeword(layer 1, same quantization)`、`q2 = ...` 这类**伪代码字符串**，`tools/research/stage2/reference.py` 的 `build_reference` 无法把这些变成 SymPy；`coverage_gaps[3]` 自己承认"the full E16 x H16 prefix has an executable torch reference and program, not yet a SymPy contract call"。

因此仍不可冻结。需要把 `full_space` / `residual_and_readback` 的文字换成**可重建语句**（三商品 × 三层 × E16/H16 的 288 标量面，含右嵌套 Möbius 与 `phi` 归一化），或明确说明为什么训练实际消费的面可以被已有 call 覆盖——后者我不接受：训练消费的是 `mixed_sq(q_l^a,q_l^b)` 的 H16 交叉比，缩维 E1×H1 恒等式覆盖不到它。

同时 `prefix_surface.statements` 的 6 条若继续以伪代码存在，请在 `status` 里写明"该字段非可重建语句"，避免下一次审阅把它当成已交付。

## 5. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"，冻结只走现有 MCP 与两位独立审查。
- 冻结后按 TDD 完成三层接线，在真实模型上分别验证重构→encoder、结构项→encoder/码本与参数更新；保留 Sinkhorn 末层分支、全局 SID 与框架拥有的训练/导出规则。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。曲率必须真实进入选码代价与三层累计回读的连通目标，不能只在参考或文字里。
- `max ‖C_l^H‖` 监测仍 `NOT_IMPLEMENTED`；它的用途是记录"回读何时不再可逆反演"，按 §19 只作诊断，不作门槛。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。旧版本 `0.05790490781524748` 不属于修复版收益。**本候选尚无闭环，不创建新 idea。**

## 6. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮差分（0.16 / 0.9 / 20 三点）与契约字段检查均为 CPU、float64、单线程，未执行 optimizer step、未写源码、未启动 GPU。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer25.md` | `3c9e01ce65fcc440c71033c46b1fc2f55f1f27f3c83f4b39946a6737459ec1da` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `e6303ae9e0cf8fc0c1903a49dba4718c5441f26b74d646d4b8f84b8b3ce1a6a1` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `02f759fbe4c1712c789e76d4386f1960b97a6287b101e6d9283a27c93a392fdc` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `a6014949e72852b224195e86688c4a2222a0b89d0c9a413c4152d85c9720c762` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q24_full_prefix.py` | `526a55737384ad62c5d156cdc8e9ee8bc303b2f1e84d96696f5a5d9bfaa6080e` |
| `results/research/struct_mixed_rq/q25/tests_20261110T102115.log` | `ee4268706fb28ba8d8459b4d699532ff2e2a0f60c9008369548cbc64398b2ccd` |
| `results/research/struct_mixed_rq/q25/run_suite.sh` | `875e62cd3625328cfadfa65c62776a8173f4abbb83218d43365cada6fb08190d` |
| `results/research/struct_mixed_rq/q25/source_pre_20261110T102115.txt` | `8fdef32ef7957048b47f2783a52215f933697e75dceaa363c9aac532e8416074` |
