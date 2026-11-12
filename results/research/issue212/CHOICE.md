# 标定用 SID 的选取依据（Issue 正文路径不可用）

## 正文指定的路径不存在

`results/stage2_RQ-VAE/TIGER_mixed_rq/item_sids.json` **不存在**（该目录下只有 `TIGER_mixed_rq_A/_E/_F0/_G/_L` 等派生臂）。
`TIGER_RQ-VAE` 的 Stage3 历史记录有两条，读数分别为 **0.0401991678** 与 **0.0555371786**，**都不是 0.056216**。

## 采用的可复现对象

| 项 | 值 |
|---|---|
| SID | `results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json` |
| sha256 | `afa89a10b1b197a7b56da4ad322c6ddd96a3c26c04238af326d88747378bb848` |
| 已记录 Stage3 | `results/stage3_T5Train/TIGER_mixed_rq_F0/logs/Amazon_2023_Instruments/Nov-09-2026_08-07-21/test_final.json` |
| 已记录 `test_recall@10` | **0.057278155956754125** |
| §89 门 `collision` | 0.257168（< 0.40，允许进 Stage3） |

理由：正文要的是"同一 SID、同一 seed、冻结协议的重复性测量"，而 Δ 必须有**已知参照**才有意义。F0 是唯一同时满足"SID 文件在库、Stage3 读数在库、且该读数就是本线最关键数字"的对象；用它的 `Δ_same_sid` 直接度量 0.057278156 这个数自身的协议噪声，比对一个无法复算的 0.056216 更符合本 Issue 的判读规则（§19g 单次结论所依赖的前提）。

**协议一切常量不动**（150 epoch、`NO_EVAL=True`、`SKIP_TEST=False`、`SEED=42`、`BEAM_SIZE=20`、端口 50201），**只跑一次**。
