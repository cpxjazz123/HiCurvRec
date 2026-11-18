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

---

# 追加：把桶里 65 个编号下载下来逐个复核

上一节只覆盖了本地还留有 SID 的 12 个编号。这一节把**其余 65 个**的源码树、`item_sids.json`、`rqvae_best.pth` 从桶里下载到 `results/research/issue139/_r2_download/<NN>/`（`--exclude HG_Rec_best.pth`，共 374 MB），用同一个审计脚本跑了一遍。

## 结果

| 档位 | 数量 | 含义 |
|---|---:|---|
| 完全一致（1.0000） | **0** | — |
| 近似（同 L3 ≥ 0.98） | 36 | 差 0.3%–2% 的商品，和本地那 12 个同型 |
| 偏离（0.5–0.98） | 6 | 差 4,000–15,000 个商品 |
| 基本不相干（< 0.5） | 6 | 差 15,000–24,587 个商品 |
| 无法审计 | 17 | 归档的源码树或 checkpoint 本身就不完整 |

### 偏离与不相干的 12 个

| 编号 | 臂 | 同 L3 | 不一致商品 |
|---|---|---:|---:|
| 07 | `TIGER_successor_entropy_curvature` | 0.0000 | 24,587 |
| 08 | `TIGER_curvature-embedding-rq` | 0.0000 | 24,587 |
| 28 | `TIGER_cogeo_region_argmax` | 0.0133 | 24,259 |
| 31 | `TIGER_interest_signature_hypball` | 0.0241 | 23,994 |
| 29 | `TIGER_COOC_RANK_FRECHET` | 0.0493 | 23,376 |
| 30 | `TIGER_BRANCH_CONDITIONED_FRECHET_RESIDUAL` | 0.3692 | 15,510 |
| 23 | `TIGER_curvature-stratum-sinkhorn-v1` | 0.6409 | 8,829 |
| 44 | `TIGER_community-pairs-guarded-rq` | 0.6977 | 7,432 |
| 05 | `TIGER_tangent_successor_residual` | 0.7759 | 5,510 |
| 25 | `TIGER_behaviour_code_cohesion` | 0.8313 | 4,148 |
| 06 | `TIGER_successor-geodesic-coarse` | 0.9235 | 1,880 |
| 36 | `TIGER_HGS-TREE` | 0.9758 | 594 |

`07` 与 `08` 的 0.0000 不是「互换」造成的：把 07 的 SID 配 08 的 checkpoint、以及反向配对，结果仍是 24,587 全不同。所以这两张表与桶里任何一份权重都不对应（可能是不同 epoch、不同代码路径，或 SID 经过整体重写）。

### 17 个无法审计的原因

| 类别 | 数量 | 具体 |
|---|---:|---|
| 归档源码缺模块 | 7 | `curvature_prefix`（4 个 prefix-ranking）、`mixed_geometry`（3 个） |
| 归档源码缺文件 | 5 | `FileNotFoundError`：上传器把 `dataset/` 排除在外，而这些 trainer 要读它 |
| checkpoint 与归档模型对不上 | 4 | `state_dict` size mismatch（`poincare-frechet`、`BRad-RQ`、`GDELTA`、`CURVALLOC`） |
| 键不匹配 | 1 | `TIGER_af3_gate_mixed_rq`（missing/unexpected keys） |

## 结论

1. **“四件套”对绝大多数编号不是一个可复现单元**：65 个里 0 个能用归档的 checkpoint 逐位重建归档的 SID；36 个在 0.3%–2% 内、12 个在 2%–99% 外、17 个连编译/加载都过不去。
2. **归档的 recall 仍然可信**，因为它是**在那张 SID 上**测的，而 SID 与 recall 存在同一个编号里。被破坏的是「checkpoint → SID」这条链，不是「SID → recall」。
3. **不能用归档 checkpoint 反推 SID**：轻则换掉几十个近并列商品（落在同配方重训噪声内），重则整张表不相关。
4. 修复面：4 个 size mismatch 说明 checkpoint 与源码不是同一次运行的产物；7 个缺模块、5 个缺文件说明上传器的白名单（排除 `dataset/`）与部分 trainer 的实际依赖不匹配。这两类是**可以重传修好的**——只要本地或历史上还有对应的树。
5. 对未来的臂：`e26043af` 引入的 `export_guard`（从磁盘 checkpoint 重新加载导出、两次编码必须一致、写 `export_manifest.json`）应当成为**所有**树的默认导出路径，而不只是被重跑的 Uniform-KNN。两个 Uniform-KNN 臂 100.00% 一致，就是这条路径有效的证据。

---

# 已接入：Stage 3 入口现在强制复核导出

审计之后的修复（2026-11-18）：

| 组件 | 作用 |
|---|---|
| `tools/research/stage2/export_check.py` | 在**新进程**里从磁盘 checkpoint 重新导出，与 `item_sids.json` 逐项比对，写 `export_manifest.json`（两个 sha256、`tokens_match`、`differing_items`、`checked_at`）。`--rewrite` 时才用 checkpoint 的表替换 SID，并把训练器那张留成 `item_sids.trainer.json`。 |
| `tools/research/stage2/export_policy.py` | 纯标准库的判决：manifest 指同一张表且 `tokens_match` → `VERIFIED`；指别的表/不可读/不匹配 → `REFUSED`；无 manifest 但在 R2 账本里 → `UNVERIFIED` 放行；无 manifest 且不在账本 → `REFUSED`。 |
| `tools/genrec-mcp/server.py::stage3_export_gate` | `stage3_start` 的**第一步**：没有 manifest 且不在账本时，用树的解释器跑 `export_check --rewrite`，把 SID 变成 checkpoint 自己的表；失败就拒绝启动，并把原因写进返回。 |

实测（本轮）：

- 拿 `TIGER_graphlet_codebook_rq` 的真实权重跑 `export_check --rewrite`：**89 / 24,587** 个商品与训练器那张表不同（与审计数字一致），SID 被替换为 checkpoint 的表，训练器那张存为 `item_sids.trainer.json`，manifest 记 `rewritten_from_checkpoint=true`。
- MCP 重载后用一个没有源码树的探针 SID 调 `stage3_start`：被拒，返回 `export gate: the export check wrote no manifest: cannot find the Stage 2 source tree of ...`。
- 两个已归档臂（F0、76）过闸门得到 `UNVERIFIED`（放行，不重跑），因为它们已在 R2 账本里、recall 已按原表测过。

测试：`tools/research/stage2/tests/test_export_policy.py`（9 个判决用例，含「归档表 + 已记录不匹配 → UNVERIFIED 而非拒绝」）与 `test_export_check.py`（4 个端到端用例：匹配放行、不匹配如实报告且默认不改表、`--rewrite` 替换并留底、checkpoint 与模型不匹配时报错）。14 个全过。

仍未覆盖：**已归档的 77 个编号不会被自动修**——它们放行但标 `UNVERIFIED`；要修只能重跑导出并重新测 recall。
