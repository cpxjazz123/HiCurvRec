# analysis_sid_recall — one script, one report

`analyze.py` is the only analysis entrypoint in this directory. It reads the
mirror of the Cloudflare R2 archive (`r2/<NN>/`, one numbered directory per
archived mechanism) and writes `report.json` + `report.md` for every pair.

    python3 analysis_sid_recall/analyze.py            # analyse r2/<NN>/
    python3 analysis_sid_recall/analyze.py --download # refresh the mirror from r2:genrec
    python3 analysis_sid_recall/analyze.py --check    # verify the mirror against the ledger

Provenance of each number (arm identity, R@10, per-file sha256) is the
repository-side ledger `results/research/issue139/r2_ledger.json`; the mirror is
verified file-by-file against it. `r2/INDEX.md` lists what every number holds.

Per pair the report records, from the SID table alone: level-1 codes used,
effective number (`exp` of entropy, `l1_er`), largest level-1 share, distinct
level-2 prefixes, distinct level-3 triples, L3 collision (`1 - triples/items`),
L3 gain bits (`log2(items/triples)`), and prefix-overlap gaps (observed TRAIN
transitions minus random item pairs) at levels 1–3, next to the Stage 3 recall
from `recall.json`. Pairs archived before Stage 3 (`stage2_record.json`) are
listed without a recall and excluded from the correlations.

The cross-pair table is Spearman rho of each metric against `test_recall@10`
with a percentile bootstrap interval over pairs; n is printed with every row.
Twenty-odd arms from a handful of families are not a population — the intervals
say so, and no metric here is an admission gate (Stage 3 `test_recall@10` is the
only verdict, per CLAUDE.md §19/§25).

`graph_shape.py` -> `graph_shape.json` / `GRAPH_SHAPE.md`: the shape of the
TRAIN behaviour graph itself (tree-like branches next to dense communities).

## structure_retention.py — 每个结构的保留度 vs 下游 recall

`structure_retention.py` 复用 `results/research/issue213/graphlet_roles.py` 的 ORCA 4-节点轨道与
`features_and_roles`（6 个角色的 NMF 分配，缓存在 `structure_roles.npz`，因为特征构建是唯一贵的一步），
对 `r2/<NN>/` 里每个有 `test_recall@10` 的配对计算逐角色保留度：

| 列 | 定义 |
|---|---|
| `prefix@1` | 行为边两端共享一级码 |
| `prefix@3` | 两端共享全部三级码 |
| `any` | 两端在任意层级共享至少一个 token 值（213 审计的定义） |

边归给**第一个端点**（历史商品）的角色，与 213 审计同口径，两张表可并读。

相关系数：每条序列对 `test_recall@10` 做 Spearman，配 2000 次 percentile bootstrap（seed 20261109，与 `analyze.py` 相同）；
`centered/*` 行是「该角色保留度减去该配对总体保留度」，用来去掉配对自身的粗细尺度，否则一个角色可能只是跟着整体粗细分变。

**SID 的 sha256 以 `results/research/issue139/r2_ledger.json` 为准**，不按 `report.json`：51–56 六个配对的
`report.json` 记录与镜像文件不一致（镜像与账本一致），说明 `report.json` 在这几个号上是陈旧的派生值。
脚本对镜像文件按账本验证，并把这类配对标成 `record_disagrees_with_archive`，不静默丢弃。

`--five` 换成 **ORCA 4+5-节点轨道（88 维）** 学出的 6 个角色（角色缓存 `structure_roles5.npz`，结果写
`structure_retention_5node.json` / `STRUCTURE_RETENTION_5NODE.md`）。5-节点轨道由加速版 ORCA 生成：
`results/research/issue213/orca/orca_gpu`（GPU 建表 + 32 线程 stage 3，**34.9 s**，输出与原版逐字节相同；
原版单线程 951 s）。两个版本共用同一套角色定义与查表口径，因此两张表可直接并读。
