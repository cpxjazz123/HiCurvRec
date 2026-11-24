# Issue254 frozen diagnostic contract

Frozen 2026-11-24 on recovery, before inspecting paired effects. The initial interrupted scaffold was not a frozen contract; its field-name failure is recorded in RUN_LOG.md. No model/asset changes or training are allowed.

## Identity and population

- Primary inputs: the two immutable formal generations and checkpoint-owned SID exports listed in EVIDENCE.json, original predictions and `_test_shards`, frozen test parquet, existing asset manifest, and R2 ledger entries 100/101. Trainer-owned alternative SID tables are never authoritative.
- Verify SID/checkpoint/export/binding SHA against each test_final and R2 ledger. Validate every bound source/asset that is used. Compare Stage3 configuration after excluding only explicit generation paths and vocabulary fields that are catalogue-derived; record every difference. Same trainer SHA, split, 150 epochs, seed42, search beam20, cutoffs5/10, catalogue item IDs and exclusion/decoding rules are required.
- Dumps contain no explicit user/history ID. Reconstruct identity using the hash-bound original split and verified immutable sharding rule `global_row = rank + local_row_index * world_size`. Every prediction target item and canonical target SID must match that original record and its checkpoint-owned catalogue. Derive sample SHA256 from `(user, full history, seen_history, target)`; join arms on `(user, sample_sha256, target)` after validation, never on row number alone. Record reconstruction rather than claiming that raw user IDs were dumped. Reject duplicate row/key/user IDs, missing/extra samples, target/SID mismatch, altered split or shard-rank disagreement. Unhit targets remain present samples.
- N=57,439 paired samples. One row per user is verified; repeated target items remain independent contexts with distinct identities. Full sequences and IDs stay in ignored shared-storage tables, never public commits.

## Pairing and ranks

Ranks in raw dumps are zero-based positions in ten persisted **selected items**. Search beam width20 does not imply persisted ranks11–20. `hit_k(r)=1[r is not null and 0<=r<k]`, k in {5,10}; `NDCG_k=hit_k/log2(r+2)`. Recompute ranks by item lookup in candidate sequences, not from dumped rank fields; separately compare shard ranks/accumulators and published metrics (absolute tolerance1e-12).

Groups: both hit; only252 hit=regression; only253 hit=improvement; neither hit. Signed delta=253−252. Global delta Recall=(improvements−regressions)/N. An item's global contribution uses this same **global N**, not its own frequency. Item conditional hit rate uses its own target-sample count. Macro delta averages those item-conditional differences, not global contributions. Sums of per-item counts, deltas and NDCG contributions must equal population totals within1e-12.

Display ranks are one-based `r+1`; absent is represented by lower-bound11 with `rank_censored=true` and display `>10`, never exact11 or infinity. Capped rank differences use min(display rank,11) and are labelled censored scores, not exact displacement.

## Candidate and SID diagnostics

Preserve top10 item IDs, canonical and generated SIDs, and nearest in-order target-sharing candidate per arm. Prefix lengths within one catalogue compare only the first three codes. Fourth token is a collision-resolution suffix, not a fourth geometry level. Validate exact four-token decode, legitimate catalogue membership, item uniqueness and exclusion of all seen-history items. Source correction at intake: the actual dataset/collator has no target exemption; repeated seen targets are excluded too. Cross-model integer code equality is meaningless: compare within-arm prefix-neighbour relations/collision partitions only. Label saved-output branch coverage (no L1/L2/L3-sharing candidate vs covered branch but target absent); do not infer T5 probability, prefix pruning, causal ranking failure or internal beam trace from final ranks. Probabilities and pruned beam trajectories were not saved and remain NA.

## Frozen structure and mechanism joins

All item joins use explicit catalogue/global item IDs after confirming item_ids order (24,587). Reuse registered visible directed/undirected graph, complete 1/2-hop layout, full28,391,948 stored target triples, item_summary, exact three-node family vectors and sampled orbit/four-node records by SHA. No graph or target table is rebuilt. Per-item measures: in/out/reciprocal degree, degree-normalized reciprocal ratio, induced neighbour density, one/two-hop sizes, full curvature count/sign fractions/mean/variance/quantiles, exact family counts/distributions and sampled-role coverage. Empty/invalid triples have an explicit mask and missing values, not numerical zeros. Sampled orbit/four-node counts are never called exhaustive.

Gate/attention may be read-only recomputed on CPU from the exact source-bound checkpoints with eval/no_grad and unchanged parameters; GPU jobs are not launched by analysis. Prefer existing source-bound q/z/reconstructions and full per-item curvature results after validating hashes. Original five/three expert weights retained; compare sum(negative), zero, sum(positive), Shannon gate entropy, one/two-hop/null attention. Reconstruction MSE is mean over768 dimensions; quantization MSE over32 dimensions. Common geometry comparison uses `common_fixed3`, equal-weight{-1,0,+1}, the archived252 fixed scale0.08433422446250916 and identical full target/reference scope. Never compare native fixed5 MAE to fixed3 MAE. Missing values retained. Hard q must agree with first-three SID-selected codebook sum; no across-arm q/z coordinate-distance claim. Prefix-neighbour retention is within each arm on the same frozen edges.

## Statistics (post-hoc observational)

Primary endpoint: paired hit@10 delta. Exact two-sided McNemar conditional binomial test on discordant pairs, plus paired user bootstrap percentile95% CI. Fixed statistics seed254, 10,000 bootstrap replicates. Item-cluster bootstrap resamples target IDs and all their user rows, ratio numerator/denominator; item macro bootstrap resamples conditional item effects. No model seeds or training.

Predeclare strata: target test frequency (1,2–4,5–9,>=10), undirected degree (0–4,5–19,20–99,>=100), full history length (1–4,5–9,10–19,>=20), baseline raw rank (0–4,5–9,not in10), and curvature (missing; negative-only; positive-only; mixed-sign; zero/other). Report counts, discordant counts, user effect,95% CI, exact p, Holm correction across all reported subgroup tests including both cutoffs. Baseline-rank strata condition on an outcome and are descriptive, not independent causal tests. Report controls (both/neither/improve/regress), user-weighted vs item-macro, degree/history/frequency conditional association with robust item-cluster uncertainty when fitting adjusted models. Exploratory subgrouping/correlations explicitly labelled. No significance-selected recommendation or presumed causal factor.

## Verification and independent paths

| Module | Scientific definition / implementation source | Independent check and choice |
|---|---|---|
| Recall/NDCG | Frozen Stage3 single-relevant-item rank definition; no new paper claim | Candidate membership/discount lookup table versus production rank predicates; reference does not import contract or trainer |
| Paired change | 2x2 matched binary contingency table; exact binomial null | scipy binomtest and hand counts; contribution conservation by separate CSV reduction |
| Bootstrap | Matched sample/target-cluster empirical resampling | Deterministic hand-degenerate cases, sign reversal and seed reproducibility; no re-fitting model |
| Geometry | Existing #252/#253 GraphMoRE A.1 proxy and frozen evaluation_reference.py | Reuse full source/hash-bound evaluation, common_fixed3 only; not intrinsic sectional curvature |
| Graph/orbits | Registered visible split and canonical directed motifs | Hand graph relations, explicit masks, sampled coverage; no graph re-estimation |
| Gate/attention | Each archived attention/structure forward, no new architecture | Full/partition CPU forward equivalence, normalization/mask checks and unchanged checkpoint SHA |

Known-answer tests cover group arithmetic, sign/denominator, null/truncation, repeated targets, duplicate/missing identities, prefix relabeling invariance and masks. At least five designated numeric/structural/routing mutations must fail their assigned value assertions, not merely crash. MODULE_CHECKS.json records measured analysis outputs and original-data checks; gradients/optimizer updates/new DDP are NA because this issue explicitly forbids training or mechanism changes. Previously completed training receipts are provenance, not newly run tests.

All test-derived tables remain `results/research/issue254/tables/`, purpose=evaluation, outside TrainingView and assets/. MANIFEST.json stores schema, row counts, source/input/output SHA and reload proof. Report hypotheses as verified output facts / observed associations / unidentifiable; one proposed falsifiable next experiment, not launched. Both arms changed expert count, supervision metric, performance implementation and training trajectories, precluding single-factor causal attribution.
