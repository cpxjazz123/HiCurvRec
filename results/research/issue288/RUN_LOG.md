# Issue #288 — RUN_LOG (frozen 2026-11-28, before R01)

* Issue: #288 `Stage2-only: curvature-guided pair ranking — minimum 10 evidence-driven iterations unless all P/N/C targets pass`
* Body sha256: `c6d1c4485bffec6df641838e444677499eebd19249782c35c39ea72cf6cde5f1`
* Predecessor: #287 (7 rounds, none met all three; not counted here), #274 (R2/R3 reference arms), #277 (evaluator)

## INFRA_MODE = FROZEN

The issue body states verbatim: "单Agent，**INFRA_MODE=FROZEN**；复用已有全量训练图、三元组/曲率/Graphlet资产、#274候选实现、#277评估器与原训练入口。不修改公共infra，不重建公共资产，不重新训练TIGER、不新增训练基线或多种子实验。"
No clause authorizes building or repairing public infrastructure → FROZEN.

## Does this issue require training? YES

The acceptance metrics (P/N/C of the exported SID) can only be produced by new checkpoint weights:
the issue requires "至少10轮新的、有效的、3000 epoch正式Stage2机制迭代" and "一轮必须具备…完整训练、
checkpoint-bound SID和统一P/N/C评估及归档". Offline diagnosis does not count as a round.

## The frozen acceptance checklist

| id | the issue's criterion (verbatim) | threshold |
|---|---|---|
| C1 | "同一最终checkpoint导出的SID必须同时满足" — one checkpoint export, no stitching | all of C2–C4 on one SID |
| C2 | "P=L1正样本共享率严格 > 0.08097285816257246（高于R2/R3）" | P > 0.08097285816257246 |
| C3 | "N=L1负样本共享率严格 < 0.008752316503966327（低于R2/R3）" | N < 0.008752316503966327 |
| C4 | "C=前三码碰撞率严格 < 0.1511367795989751（低于R2/R3）" | C < 0.1511367795989751 |
| C5 | "P/N用#277原57,439 validation行、最后历史商品与真实target、合法均匀负总体精确概率；C=1−unique(first3)/Nitems" | the frozen #277 evaluator on `valid.parquet` (57,439 rows) |
| C6 | "至少10轮新的、有效的、3000 epoch正式Stage2机制迭代（R01–R10）；#287七轮不计入。已达标且证据完整可提前关闭" | ≥10 real rounds unless all pass earlier |
| C7 | "保持曲率只选样、距离排序唯一新增监督的边界，不擅自新加全局均衡或碰撞loss" | mechanism boundary |
| C8 | "保持768维商品语义输入、32维瓶颈、三层各256码、欧氏选码、原残差量化/Sinkhorn/重建/commitment/codebook和SID导出协议" | frozen protocol |
| C9 | "训练样本仅来自训练图；验证行为只用于评估及预先规定的候选筛选；不访问test进行调参" | data boundary |
| C10 | "相同合法硬件协议、seed42、原global batch4096，完整正式候选3000 epochs。单卡/四卡差异如实记录" | recipe |
| C11 | "保存每轮DIAGNOSIS/HYPOTHESES/CONTRACT、样本/资产manifest、代码与必要测试、持久命令/退出码/日志、真实梯度/参数更新、checkpoint/SID/导出身份、逐轮指标和最终PASS表、冻结#277预测（仅诊断）、性能、VERDICT、infra零漂移和提交推送" | per-round deliverables |
| C12 | Each round: "提交结果、判定三个阈值、定位当前失败的具体原因（区分假设与已验证事实），提出下一轮如何针对原因进行干预及预计影响P/N/C，立即继续下一轮" | iteration closed loop |
| C13 | "本Issue严禁启动Stage3/T5、新test或产生新真实Recall" | no Stage3 |
| C14 | "先用#277统一评估器复核实际 Stage3 对应的checkpoint导出SID身份" — re-verify R2/R3 | evaluator identity check |

Frozen thresholds re-verified in #287 (to the last digit): R2 `9f6bf673…` P=0.0783613920855168,
N=0.008752316503966327, C=0.1511367795989751; R3 `9d9f3d68…` P=0.08097285816257246,
N=0.00901661164455637, C=0.1543498596819457.

## This round's (R01) scope

Mechanism files: a new Stage2 candidate tree from the frozen baseline (only the mechanism module and
its tests change). No public infra, no `assets/` rebuild, no TIGER retraining, no Stage3.

## R01 plan (evidence-driven, from #287's measured frontier)

The #287 frontier (own checkpoint exports): the best C (0.150933, passing) and a good N (0.009392)
came from the **L1-codeword** supervision with the #274 outer pairs (round 1); the best P (0.088198)
came from the **same-anchor** construction *but* supervised on the L1 codeword only, whose C
(0.158376) lost the L2/L3 shaping that the final hard-`q` supervision had (R2's C 0.1511).
R01 therefore tests the single-factor change **supervision position: the same-anchor pairs ranked on
the final hard `q`** (all three levels) instead of the L1 codeword alone. Expected: P keeps the
same-anchor gain, N and C approach R2's values (the deeper levels regain their shaping role). The
diagnosis below (R01 DIAGNOSIS.md) decomposes the #287 exports into bucket occupancy / in-bucket
behavioural purity / anchor distribution / L2-L3 collisions before the run.
