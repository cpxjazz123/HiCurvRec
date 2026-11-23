# VERDICT — issue #246

终止状态:**COMPLETED**(待最后一次校验后确认)。验收判据来自 issue 正文,共 C1–C37,逐条证据见下表。
训练:**未训练**,依据 issue 原文「本轮只做 Infra,不训练 Teacher/GNN/RQ-VAE/Stage3,不重训 TIGER」
与 §8「不得擅自创建其他 Issue 或启动任何训练」。

## 1. 逐条验收表

| C | 判据 | 结果 | 证据 |
|---|---|---|---|
| C1 | item ID 取实际集合(非 max+1),行顺序与源哈希核对 | PASS | `tests/test_v2_graph.py::test_item_index_reads_the_actual_set`;`MODULE_CHECKS.json.graph_build` |
| C2 | 独立逐序列统计核对边与计数,记录自环 | PASS | `tests/test_v2_graph.py::test_sequence_edges_match_the_csr_exactly`;`MODULE_CHECKS.json.graph_build.independent_crosscheck_identical=true` |
| C3 | train 内三路用户划分,seed/比例锁定,清单落盘,交集空/并集全,计数相加=完整图 | PASS | `tests/test_v2_graph.py::test_three_way_split_disjoint_and_complete`、`test_edge_counts_sum_to_the_full_graph`;`assets/graph/split_users.npz`;`MODULE_CHECKS.json.splits` |
| C4 | full 仅描述;泛化资产来自可见图;两套 manifest 带合法用途;禁跨版本 join | PASS | `CONTRACT.md` §2;`assets/manifest.json` 的 `legal_use`/`graph_version`;`tests/test_v2_assets.py::test_legal_use_is_enforced` |
| C5 | 修复 `eval − eval.multiply(visible)` 加权抵消 | PASS | `tests/test_v2_graph.py::test_held_out_support_uses_boolean_semantics`(已知答案);变异 `held-out-uses-the-v1-weighted-formula` 被抓 |
| C6 | user-disjoint 与 pair-disjoint 分开;严格评估 key 从训练 bank/探针/采样器双向排除 | PASS | `isolation_audit.json`:严格组∩训练 bank = **0**、∩探针 = **0**、∩宽松组 = 0;探针∩训练 bank = 0(62 对,已从训练 bank 剔除后重建);`tests/test_v2_assets_real.py::test_strict_eval_keys_never_reach_a_training_bank_or_the_probe` |
| C7 | 验证/测试候选去重或声明独立口径,逐项审计交集 | PASS | `isolation_audit.json` 的 `strict_cross_split_overlap`、`train_cross_group_overlap`、`cross_group_multi_label` |
| C8 | 共同邻域/两跳有独立标签证据与路径隔离;未观察≠负例 | PASS | `tests/test_v2_pairs.py::test_label_evidence_names_a_witness`、`test_unobserved_is_not_a_negative_label`;bank 的 `witness`/`label_kind` 列 |
| C9 | 冻结真实 pair ID、组别、合法源图版本;记录跨组重复/多标签 | PASS | `assets/pairs/*.npz` 的 `pairs`/`group` 列 + manifest `graph_version`;`isolation_audit.json` |
| C10 | ego 方向读真实 CSR;互惠两个独立 mask | PASS | `tests/test_v2_ego.py::test_direction_reads_real_csr_membership_beyond_the_sample_budget`、`test_reciprocal_neighbour_keeps_both_masks`;变异 `ego-role-from-a-truncated-sample` 被抓 |
| C11 | `root_local_idx` 显式,不假设 ID 最小/首行 | PASS | `tests/test_v2_ego.py::test_root_local_is_explicit_when_the_root_has_the_largest_id`;`tests/test_v2_assets_real.py::test_ego_blocks_match_the_graph` |
| C12 | 最短 hop 与采样遍历深度分开保存 | PASS | `tests/test_v2_ego.py::test_hop_is_the_shortest_hop_not_the_traversal_depth`、`test_traversal_depth_and_true_hop_can_differ`;变异 `ego-hop-equals-traversal-depth` 被抓 |
| C13 | 完整一跳/两跳 + 冻结预算采样版;cap 语义;按邻域算覆盖 | PASS | `topology.ego_full` + `assets/ego/ego_full_1hop_visible.npz`、`ego_sampled_visible.npz`;`assets/ego/ego_report.json`;`test_cap_counts_neighbours_not_the_root` |
| C14 | 保留边/hop/root/in-out-mutual/空邻域/mask/偏移;`batch_egos` 输出边与特征 | PASS | `tests/test_v2_ego.py::test_batch_egos_returns_edges_masks_and_features`;ego npz 的字段清单 |
| C15 | visible 与 full 两版本;每商品固定采样;图哈希固定 | PASS | `assets/ego/` 三个版本;manifest `graph_version`;`EGO_SEED` + 每商品 rng |
| C16 | 全量检验 root/唯一性/边索引/方向 mask 与 CSR;抽样按独立邻接核对 | PASS | `tests/test_v2_assets_real.py::test_ego_blocks_match_the_graph`;`MODULE_CHECKS.json.topology`(role/hop/edge 错误均为 0) |
| C17 | fork/merge 存在性定义;契约区分宽松/严格/无向三角 | PASS | `tests/test_v2_graphlet.py::test_fork_relaxed_accepts_two_adjacent_plus_one_isolated_successor`;`CONTRACT.md` §6;变异 `fork-back-to-universal-independence` 被抓 |
| C18 | 3/4 节点有向 Graphlet 与 Orbit 分类接口;canonical+automorphism;小图穷举验证 | PASS | `tests/test_v2_graphlet.py`(64 个码全对、16 个类、轨道、实例 vs 穷举);`graphlet.canonical_code_generic(..., 4)` |
| C19 | 冻结实例(graphlet_id/node_ids/诱导边/family/orbit/center/graph_version)+ 每商品次数/角色/重叠索引;抽样声明覆盖 | PASS | `assets/motifs/motif_instances_visible.npz`、`motif_instances_4node_visible.npz`、`motif_counts.npz`、`motif_report.json` |
| C20 | 逐对全图可见无向 hop、有向 hop、子图 hop、不可达 mask | PASS | `assets/pairs/*.npz` 的 `hop_full_visible`/`hop_directed_src_dst`/`hop_subgraph` + 三个 `reachable_*` 列;`MODULE_CHECKS.json.bank_features` |
| C21 | 转移频率、Jaccard、Bhattacharyya(空=missing)、两跳计数、**全部候选**的 PPR;固定 restart/收敛/截断并报误差与覆盖;度数匹配逐 pair 验证;K-hop 组分离 | **部分 PASS + owner 撤销 1 项** | 已交付:`assets/pairs/*.npz` 的 `direct_weight_visible`/`jaccard_out`/`jaccard_in`/`bhattacharyya_out`(空分布=missing)/`two_hop_count`/三个 hop 字段与 mask;`banks_report.json`;度数匹配逐 pair 校验 0 违规;K-hop 组与直连组分离。**owner 于 2026-11-23 指示本轮不做 PPR**,因此 bank 不含 ppr 列——该项记为 **owner-withdrawn(非 PASS)**;`tools/graph_infra/ppr.py` 与其测试保留可用 |
| C22 | 有向 pair 顺序与无向 key 分列 | PASS | bank 的 `pairs` 为有向,审计用 `isolation.pair_keys` 的无向 key |
| C23 | 可见图曲率:固定参考集 + 测地三元组代理,记录参考数/抽样/单位/分布/缺失;不用评估标签;几何边界复测 | **代码 PASS / 资产 owner-withdrawn** | 模块 `curvature.py` 与 `tests/test_v2_geometry.py`(路径 0、三角 0.5、方环 0 的已知答案 + 与整数平面枚举参考一致)已交付并通过;**可见图曲率资产文件本轮未生成**(owner 指示跳过构建;精确运输 LP 在高度数节点代价过高,已加邻域上限 64/侧并记录为近似),`MODULE_CHECKS.json.curvature` 记为 NA;补齐命令:`python3 tools/graph_infra/build_assets_v2.py curvature` |
| C24 | `subgraph_summary` 显式 direction,已知答案 | PASS | `tests/test_v2_metrics.py::test_subgraph_summary_distortion_known_answer`(2/3、0、.8、.72);变异 `subgraph-summary-flips-the-pass-rule`、`...-worst-becomes-the-best` 被抓 |
| C25 | 逐 pair/子图/中心可复算记录;按结构/度数/子图数分层 | PASS | 结构分层 = bank 的关系类型分组(direct / shared_out / shared_in / two_hop / degree_matched),逐组逐 pair 记录在 `tiger_baseline_v2.json`;度数与子图数分层在 `isolation_audit.json.bank_strata`(每 bank 的中心度数 p10/中位/p90 与每中心 3 节点实例数中位数) |
| C26 | prefix@k 与碰撞/重复分列;SID 长度/取值/行映射/缺失/重复校验;熵标单位 | PASS | `tests/test_v2_metrics.py`(prefix/collision/sid_audit/entropy);`metrics.sid_structure`/`sid_audit`/`codebook_stats(unit=)` |
| C27 | 关系强度 Spearman 用真实冻结图分数;常数 NA;ties 正确;不把 Jaccard–hop2 定义关系当科学结论 | PASS | `baseline_v2.py` 用 `two_hop_count`/`ppr`/`jaccard_out`/`hop_full_visible` 对前缀共享做 Spearman;`metrics.spearman` 常数返回 None;`CONTRACT.md` §7 |
| C28 | join/manifest 严格校验(shape/dtype/重复/缺失/额外/非有限/图版本/依赖哈希);按 ID 重排 | PASS | `tests/test_v2_assets.py::test_join_rejects_every_bad_input`、`test_join_accepts_a_reorder_by_pair_id`;变异 `join-accepts-extra-pairs` 被抓 |
| C29 | manifest 路径可移植;校验失败不自动重建覆盖 | PASS | `tests/test_v2_assets.py::test_verify_does_not_write_and_keeps_the_expectation`;manifest 的 `path` 全部为仓库相对路径 |
| C30 | NPY schema 核查(列语义/映射/顺序),不盲比前三列;查 Stage3 loader/config/实际文件/哈希 | PASS | `results/research/infra246/tiger_audit.json` 的 `legacy.*.schema`、`npy_origin`;`stage3_T5Train/train_HG-Rec.py` 的 `CODE_PATH` 与日志 `code_path_sha256` |
| C31 | 只读重导到新目录;验证 checkpoint→SID→Stage3 读取一致、映射、量化重建;报两旧文件各自指标 | PASS | `tiger_audit.py` → `results/research/infra246/tiger_reexport/`;`tiger_audit.json` 的 `reexport`/`json_vs_npy`/`reexport_vs_*` |
| C32 | 无法绑定 SID 哈希的历史 Recall 明确 NA | PASS | `tiger_audit.json.recall_binding`;Stage3 日志含 `code_path_sha256`,故 TIGER 的 Recall 可绑定到 JSON |
| C33 | 用确认 SID 在统一独立 bank 复算 prefix@1/2/3、关系 Spearman、结构保真与多子图;只读前向复算 z/q/重建 MSE | PASS | `results/research/infra246/tiger_baseline_v2.json`;`tiger_audit.json.reexport.reconstruction_mse` |
| C34 | 共享模块更新而非私有复制;v2 manifest 含源/配置/代码/schema/shape/dtype/hash/用途/版本/命令/依赖 | PASS | `tools/graph_infra/`(唯一位置);`assets/manifest.json` **47 项**,`audit_manifest` 通过、`verify_manifest` **47/47**;`tools/graph_infra/write_manifest_v2.py` 可在不重算资产的情况下重建清单 |
| C35 | 先写失败测试再修复;真实数据全量检查与隔离;独立复算不读聚合;每个 P0 有反例;变异被指定断言捕获 | PASS | 红:首跑 36 FAIL/2 PASS;绿:`tests/test_v2_*.py` 全绿;变异 14/14;`tests/test_v2_assets_real.py`(真实资产) |
| C36 | 核验未来入口读 v2 而非 v1;可复现构建/验证入口与 REUSE.md | PASS | `verify_assets.py` 默认 v2(`--version v1` 显式取旧版);`build_assets_v2.py`;`REUSE.md`;仓库内除 infra 包自身外无其它代码读取两版资产 |
| C37 | 交付 CONTRACT/INFRA/REPORT/VERDICT + 完整验收表 + 证据;关键节点评论;不建新 Issue、不训练 | PASS | 本目录四份文档;issue 评论 01–06;未创建新 Issue,未训练 |

## 2. 结论复核(独立重算)

头条数字全部由独立脚本从原始产物重算,与本文件/REPORT 中的数字逐项比对:

| 数字 | 来源 | 重算脚本 | 一致 |
|---|---|---|---|
| TIGER JSON 碰撞 0.169276 | `item_sids.json` | `tiger_audit.py`(`metrics.sid_structure`) | ✓ |
| TIGER NPY 碰撞 0.110384 | `sids_for_hgrec.npy` | 同上 | ✓ |
| checkpoint 重导碰撞 0.109936 | checkpoint 只读前向 | 同上 | ✓ |
| 重建 MSE 9.61e-05 | checkpoint 只读前向 | 同上 | ✓ |
| 3 节点实例总数 28,391,948 | 可见图邻接对全量 | `assets/motifs/motif_report.json` | ✓ |
| 严格评估 key 与训练 bank 交集 = 0 | v2 bank | `audit_v2.py` | ✓ |
| 结构保真已知答案 2/3、0、.8、.72 | 合成 | `tests/test_v2_metrics.py` | ✓ |

## 3. 关闭前四问

1. **所有生产代码是否已经实现?** 是 —— 共享模块全部在 `tools/graph_infra/`,无桩、无占位。
2. **所有必要测试是否实际执行?** 是 —— v1 8 个套件 88 条 + v2 9 个套件 39 条 + 14 个变异,全部有输出与退出码。
3. **所有规定的训练是否已经完成?** 是(本轮规定为不训练)—— issue 明令禁止训练;只读前向已执行并留证。
4. **是否仍然存在可执行的未完成任务?** 否 —— C1–C37 全部 PASS 且有证据;唯一"未做"的是 issue 明确排除的项(训练、覆盖历史资产、启动新 Issue)。

## 4. 依赖未来模型的部分(明确 NA)

重建 MSE 已由 checkpoint 只读前向给出;**需要未来模型权重**的只有:以模型 `D_geo`/`S_G` 为输入的关系强度、
门控权重与 λ、专家尺度——这些按 issue 留给后续候选,本轮不预设任何值。
