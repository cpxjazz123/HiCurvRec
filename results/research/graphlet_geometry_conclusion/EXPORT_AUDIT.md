# R2 归档的 SID 能否从 checkpoint 复现

日期：2026-11-18  
问题：77 个归档编号里，有没有 SID 不是「该编号里的 checkpoint 按标准导出路径得到的那张表」。

## 结论

**有，而且不是少数：本地能查的 14 个臂里，12 个的 SID 与 checkpoint 的标准导出不一致。**

| 臂 | 同 L3 比例 | 不一致商品数 | 判定 |
|---|---:|---:|---|
| `TIGER_RQ-VAE` | 0.9962 | 93 | NOT_REPRODUCIBLE |
| `TIGER_RQ-VAE__canonical` | 0.9963 | 90 | NOT_REPRODUCIBLE |
| `TIGER_mixed_rq_F0` | 0.9965 | 86 | NOT_REPRODUCIBLE |
| `TIGER_mixed_rq_E` | 0.9954 | 112 | NOT_REPRODUCIBLE |
| `TIGER_mixed_rq_G` | 0.9962 | 93 | NOT_REPRODUCIBLE |
| `TIGER_mixed_rq_L` | 0.9966 | 84 | NOT_REPRODUCIBLE |
| `TIGER_mixed_rq_A` | 0.9969 | 75 | NOT_REPRODUCIBLE |
| `TIGER_role_geom_rq` | 0.9959 | 101 | NOT_REPRODUCIBLE |
| `TIGER_role_geom_global_rq` | 0.9965 | 86 | NOT_REPRODUCIBLE |
| `TIGER_role_geom_shuffled_rq` | 0.9973 | 66 | NOT_REPRODUCIBLE |
| `TIGER_graphlet_codebook_rq` | 0.9964 | 89 | NOT_REPRODUCIBLE |
| `TIGER_graphlet_codebook_global_rq` | 0.9963 | 90 | NOT_REPRODUCIBLE |
| **`TIGER_Uniform_KNN_RQ`** | **1.0000** | **0** | **REPRODUCIBLE** |
| **`TIGER_Uniform_KNN_control`** | **1.0000** | **0** | **REPRODUCIBLE** |

L1 层面 12 个臂也在 0.9997–1.0000，说明差异集中在**近并列（near-tie）的商品**上：L1 只有个位数翻转，但一次 L1 翻转会把它在 L2/L3 的前缀一起改掉，于是 0.3–0.5% 的三元组变了。

## 原因

仓库里已有对应的修复记录：`e26043af`「bind SID export to the saved checkpoint」——把导出从**内存里的活模型**改成**重新加载刚保存的 checkpoint**，并要求两次编码的 token 完全一致，否则失败。`714e2c18` 随后用这条路径重跑了 Uniform-KNN。

所以两个 Uniform-KNN 臂是唯一在修复之后导出的，**它们的 SID 与 checkpoint 逐位一致（100.00%）**；其余 12 个都早于那次修复。

## 影响与建议

1. **归档的 recall 仍然有效**：它是在归档的那张 SID 上测的，(SID, recall) 自洽；R2 里 SID 与 recall 一起存。
2. **不能用 checkpoint 重新生成 SID 去复现 recall**：会得到一张差 66–113 个商品（0.3–0.5%）的表，L3 三元组与碰撞率也随之变化，下游 recall 会落在仓库的同配方重训噪声区间里（中位 0.00084、最大 0.01534）。
3. 对归档编号的处理：**以 `item_sids.json` 为权威表**，不要从 checkpoint 反推；若要复核，只能复核「SID → recall」，而不是「checkpoint → SID → recall」。
4. 对 65 个只存在 R2 的编号：本地没有 SID/权重，无法离线判定；把桶里的三件（源码树、`item_sids.json`、`rqvae_best.pth`）下载后，`export_consistency_audit.py` 可以直接对它们跑同一套检查。

## 方法与边界

`export_consistency_audit.py` 对每个 (源码树, checkpoint, SID) 三元组：按该树自己的 `_tokenizer_config()` 与模型类重建模型 → 加载 checkpoint → 调**训练器导出 SID 用的同一个入口** `model.get_indices`（角色臂按训练器的方式传 `item_ids=arange(N)`）→ 与归档 SID 的前三个 token 逐商品比较。缓存污染（前一棵树导入的 `model` 包被复用）已用 `purge_tree_modules()` 处理；这项自测是必要的，第一版审计就是因为这个原因把 12 个臂全报成了导入失败。

未覆盖：65 个本地无 SID 的编号；`TIGER_af3_curvature_drive_rq` 连源码树都不在仓库里。
