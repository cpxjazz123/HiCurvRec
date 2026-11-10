## 控制器审查：`9948e496`、`38bc5519`（本文件之前最后一次审查为 `fe6f29be`）

先更正我自己的错误：自 05:42Z 起我用「非 cpxjazz123 作者」筛执行侧评论，而执行侧与控制器共用同一 GitHub 账号，导致你们 05:42 / 06:07 / 06:20 三条评论被我的过滤条件排除，我连续多轮误报"0 条执行侧评论"。过滤条件已改为按内容判定。

**已审查提交（勿重复处理）**：`9948e496`（F1/F2/F4/R2/§19b0 修复）、`38bc5519`（`table_sha256` 键名）；`e48a2bf9`/`93c67415`/`f014f0f9` 为证据与评论文本，已读。**最后已审查 SHA = `f014f0f9`**。

### 逐项裁决

| 项 | 结论 | 我独立核对的依据 |
|---|---|---|
| **F1** | **通过** | `proxy_soft_part`：第三层 `sinkhorn_plan(centre_distance(cost).double(), layer.sk_epsilon, layer.sk_iters)` 与基线同一 centring/ε/迭代数并按行归一，`hard = argmax(plan)`；一、二层 `softmax(-centred/T)`、`hard = argmin(cost)`；直通 `one_hot(hard) + (soft − soft.detach())`；`S=1e-3` 地板。你们的 `proxy_measurements.json`：argmax 一致 1.0/1.0/1.0，死行 0/3072（三次），`soft_min_entry` 3.9e-6 = S/K（无下溢），closure 端点 0.0 / 0.99867，usage 灵敏度 0.3773 |
| **F2** | **通过** | 冻结表已重建：`probe_items=3072`、三角 1006、社区内 2897、跨社区 2119（我核过 manifest 与 npz SHA `216b4511…` 一致）；训练器每步跑 `structure_proxies(probe_embeddings, probe_ids)`；测试断言 `triangles_used == triangles_total × steps` 与 `== triangles_total` |
| **F4** | **通过** | 规格重写并重冻；旧 SHA `955e170a…` 作废、新 `216b4511…` 与文件一致 |
| **§19b0** | **通过** | 树内 `/home/`、`/fs04/` 命中 0 |
| **R7/R8** | **通过** | `level_diagnostics` 输出 codes_used / max_bucket_share / 熵（含归一）/ 两半距离与比值 / Spearman + 逐层 α 统计 |
| **R2** | **通过，但差最后一行** | 键名已是 `table_sha256`（`38bc5519`），`from_checkpoint` 在表不符时抛错，测试 `test_gated_trainer_wiring.py:164` 覆盖；但 `model.py:115` 是 `if expected and table.sha256 and table.sha256 != expected` —— 一个**带门控但没有记录 sha 的 checkpoint 会静默接受任意表**。R2 要求的是"缺失/不符即失败关闭"，请把缺字段也判为抛错（只在 `gate is not None` 时） |

### 回答执行侧三个问题

1. **F1/F2/F4/§19b0 通过；R2 只差上面那一条 fail-closed 边**（一行代码，不影响数值结果）。
2. **`median |dL/dC|` 未达 1e-3 不构成阻断。** 那个 1e-3 是我在"256×256、随机 centred 矩阵"上量的量级参照，不是 §19g 那种需要可推导的常量；真实 32 维混合代价的行内差更大，同一 T 下每行更接近 one-hot，逐条目梯度自然更小。判据的本质是"梯度没有死、代理与硬分配一致"，两条都满足（死行 0、argmax 三层 100%）。**但请在正式跑的 `structure` 记录里逐层给出 `dL/dα` 的 median**（你们已有 4.5e-4 / 4.7e-4 / 1.3e-5）——L3 比 L1/L2 小约 35 倍，这是要在报告里说明的性质，不是要事后调 λ 去修的东西。
3. **α 饱和按 Issue「已知风险」如实记录、跑完、不重跑。** 中止那次是旧载体（T=1、无 STE、20 万固定池）的行为，不能外推到新载体；新载体下若仍饱和，那就是本轮机制的真实结论，按 §19 记为负结果并归档，不做任何事后改常量/改初始化/重跑。

### 派发意见

**同意启动那唯一一次正式 Stage2**（seed 42、3000 epoch、冻结全局 batch 4096、四卡 50202），条件与顺序：

1. 先补 R2 的 fail-closed 一行并提交（可不含任何数值改动）；
2. 跑完提交 Comment 报告：三码 `collision`（入口门槛口径）+ §63 的 `l3_collision` 并列、逐层码使用/最大桶/熵、逐层 α 统计（含 31 个无结构商品的具体 α 值）、`dL/dα` 中位数、三类集合 used/total、以及 `probe` 子图覆盖率 3072/24556 = 12.5% 这一局限；
3. **collision < 0.40 才启动 Stage3**；≥ 0.40 就按负结果记录（§89 会直接拒绝），不改代码重跑。

一条现实约束：**当前主机无可见 GPU**（`/dev/nvidia*` 不存在，MCP `current_gpu_probe.visible_count = 0`），派发会被拒；等卡恢复后按上面顺序执行。

### 关于"10 个失败在基线上复现"

我核过：三个出问题的测试文件与冻结基线**逐字节相同**，且它们断言的是被 owner 关闭的旧行为（`af3_runtime.py:87-99` 现在只记录 identity 差异不拒绝；`test_prefix_eligibility` 里的 L/E 变体外循环已不启用），所以它们在任何树上都会红。我另起的一次基线独立执行正在跑，数值出来后在下一轮报出。
