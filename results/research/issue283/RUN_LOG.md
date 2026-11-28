# Issue #283 — RUN_LOG

- Issue: #283 "Analysis-only: joint triplet SID assignment patterns conditioned on directed Graphlet and Orbit"
- Body sha256: `87367c48106f705043db17003c9f8aae29f1f2b510dd976356e00d71bf412e0a` (2601 chars, frozen 2026-11-28)
- Title/state: OPEN at freeze; `issue283.json` is the raw `gh issue view` record.
- **INFRA_MODE = FROZEN.** The issue states explicitly: "单 Agent,analysis-only,INFRA_MODE=FROZEN。复用已有253份 SID、
  129个有标签 SID、#277固定5折、训练有向图与全量三节点Graphlet/Orbit资产;不修改公共infra或原资产,不重训
  TIGER/RQ-VAE/T5,不启动Stage2/Stage3、不运行新test。允许研究专属只读分析脚本与派生数据。" No BUILD
  authorization sentence exists in the body.
- **Training required? NO.** Quote: "不重训TIGER/RQ-VAE/T5,不启动Stage2/Stage3、不运行新test" and
  "任何情况下不启动Stage3". The acceptance metrics are all recomputable from existing artifacts (253 SIDs,
  the frozen triplet asset, the frozen 5 folds) — §4 analysis-only branch.
- Analysis unit: independent SID (`sid_sha256`); near-duplicate label clusters = SIDs whose label-source
  signature is identical (2 clusters of size 2 found; bootstrap resamples the 127 clusters).

## Frozen acceptance checklist (C1..C12, verbatim requirements condensed)

| C | requirement (issue §) | criterion | threshold | data | wording |
|---|---|---|---|---|---|
| C1 | §1 five mutually exclusive partitions ALL/AM/MB/AB/NONE at cumulative prefix k=1,2,3; single-level equality as diagnostic; role order, direction, graphlet automorphism; AM/MB merged where outer roles are exchangeable; LCP joint with feasible combos; hand example with equal pairwise rates but different joint | probabilities, invariants, hand counts | sum=1 exact; refinement-only; rename invariance exact; hand example exact | 253 item_sids; visible_forward | "完整枚举五种互斥模式…逐项验证分区概率和为1、跨层只能细化不能合并、码字全局重命名不变" |
| C2 | §2 full legal triplet manifest (actual count), real directed-edge classification, induced/non-induced conventions, Orbit roles, full coverage, no undirected-wedge-as-chain, two populations, validation edges not written back, multi-intermediate pre-frozen weights, #282 negative-sample audit, deterministic exact stratification, permutation null + degree-stratified expectation | manifest/coverage audit + null model | count = actual; every row classified; 0 undirected wedges mislabelled; null verified within MC | assets/shards (13), assets/train, valid.parquet | "核对全量合法三元组 manifest…复用现有资产/分类器,不能将无向wedge自动当有向链" |
| C3 | §3 per-SID per-structure per-level 5-pattern distribution, LCP joint, cross-layer transitions and conditionals, centre macro + row micro; comparison vs pairwise marginals (only ALL is non-marginal); entropy/conditional information; lift vs matched null; matched-marginal cases; high/low recall distributions, support, CIs, counterexamples; #282 matching + L2/collision/L3 controls; SID-group bootstrap; #274 R2 diagnostic; #277 residual test; control 6 metrics/degree/popularity/support; FDR | descriptive tables + matched contrasts | all statistics present; every claimed contrast has support + CI; FDR applied | 253 SIDs + 129 labels + folds | "报告联合模式中无法由边际概率唯一决定的信息…" |
| C4 | §4 #277 corrected Ridge6 (0.002972/0.667), same 129/folds, strict 120/9, 13 high/6 strong; inner-fold-only ≤2 pre-frozen joint features; inner standardization/alpha/selection; original 6 + #282 types as comparison; primary: paired SID-group bootstrap CI lower >0, secondaries not worse; report low/high bias, strict subset, 0.065 NOT_EVALUABLE; freeze only if pass; else keep #277; no Stage3 | OOF + paired bootstrap | CI lower > 0 **and** secondaries not worse | SID_TABLE + FOLDS + JOIN table | "主判据:原−新 OOF MAE 的SID组配对bootstrap95% CI下界>0…" |
| C5 | §5 CONTRACT/HYPOTHESES/DIAGNOSIS; hand examples + tests; full per-SID/structure/level stats + manifest; coverage/negative/null audit; matched comparison/counterexamples; OOF/CI/ablation/residual; persistent commands/logs/exit codes; REPORT/VERDICT; infra zero drift; commit/push; final answers | deliverables | all files present; logs persisted; infra drift 0 | repo | "可复现分析完成即可按证据关闭,允许负结果" |

## Mechanism modification scope

Analysis-only: **no** candidate GNN/quantizer/loss changes; no training. Files created under
`results/research/issue283/` (scripts, docs, derived JSON). Public infra (`tools/`, `assets/`, stage2/3
entrances) only called, never modified. The derived row table and stats are research-local artifacts.

## Verification plan (impact-driven)

- New scripts + tests are the only code; the reference hand examples are the deterministic tests.
- No historical candidate test is invalidated (no mechanism code changed) → only the new tests run.
- The frozen inputs (graph, shards, SID table, folds) are verified by sha256 against their manifests.
