# iter5 Hypothesis Designer（Agent C）

- **唯一推荐机制**：HiHPQ 式固定容量层次双曲 product-codebook attention（来自 Agent B `direction_decision_iter5.md`）
- **机制标识**：`iter5_hyperbolic_codebook_attention`
- **落点**：每层 `Quantize` 内部 `HyperbolicCodebookAttention`；不改 `codebook_size=[256,256,256,1]`、SID 长度、Stage3 输入协议、cyclic `c(t)`。

## 1. 直接效应（mechanism effects）—— Agent D 必须命中才算 ALIGNED

| 编号 | 直接效应 | 数值证据 | 检测位置 |
|---|---|---|---|
| DE-1 | `attention.temperature_scale` 数值随 `cyclic c(t)` 联动，cyclic 周期内出现单调变化 | `step=0` 与 `step=C_CYCLIC_PERIOD/4` 的 `attention.temperature` 数值差距 > 0 | Stage2 日志中手动打印 `temperature`（在 `train_migrated.log` 加 `[iter5][attn]` 行） |
| DE-2 | `attention.key_proj.weight` 在 Stage2 完整训练后与初始值不同（注意力 key 已被更新） | `final.pt` 与初始 checkpoint 的 `key_proj.weight` L2 差距 > 0 | 比对 `rqvae_step50000.pt` 与训练起点的 `key_proj.weight` |
| DE-3 | attention logits 被注入到 `distances`，`ids = argmax(assignments)` 路径上的输入是 `distances + attention_logits` 而非纯距离 | grad_check 与前向日志中存在 `[iter5][attn]` 的 `min/max/mean` | grad_check 输出 + 训练日志 |

任一项反向或完全不发生 → `MECHANISM_FAIL` → 本轮 NO-GO（不进入 Stage3）。

## 2. Proxy 假设（proxy hypothesis）—— D 仅 PARTIAL 判定依据

| 编号 | Proxy 指标 | 预测方向 | 判定含义（仅记录不强制） |
|---|---|---|---|
| PH-1 | `H(L1|L0)` | 保持或 ↑ | 注意 attention 走 codeword re-weighting，对 L1 conditional entropy 是间接效应 |
| PH-2 | `coarse-fine balance` | 改善或保持 | 不直接改 L0 utilization |
| PH-3 | `L0 oracle` | 与 iter11 `sk_eps=0.5` 路线距离 | 用现有 geometry fingerprint 对照 |
| PH-4 | `l01_unique_pairs` | 保持或 ↑ | 间接效应 |

未达成不会触发 NO-GO；D 给 `PARTIAL` 或 `METRIC_MISMATCH`，继续 Stage3。

## 3. 不可证伪词扫描

本文件不含“可能提升”、“也许”、“视情况而定”等不可证伪词；所有假设都是“`指标 X` 在 `Stage2 Y 步骤` 后 `应` / `不应` 处于 `Z 区间`”。

## 4. 失败归类（与 Agent E 对齐）

如果本轮最终 NO-GO，Agent E 必须按以下三类归类（其中 DE-1~3 命中与否直接决定类 1）：

- 类 1：DE 任一项失败 → 机制未生效；
- 类 2：DE 全部命中但 PH 未达成 → 机制生效但 SID geometry 方向错；
- 类 3：DE + PH 均命中但 Stage3 `test_R@10 ≤ 0.065` → SID geometry 看起来对但 downstream 不吃。

## 5. 与 iter11 baseline / iter4 的对照

- iter11 baseline `geom_L0_util`：L0 collapse 极强（`full_gini=0.1143`），L1 diversity 偏高；
- iter4 `iter4_lorentzian_centroid_recenter` `test_R@10=0.0540`；
- iter5 预测：因 attention 改 codeword logits，不直接改变 L0 collapse 路径，预期 L0 utilization 与 iter4 处于同一量级；H(L1|L0) 较 iter4 略升或保持；不形成 iter11 的“强 collapse 路线”。