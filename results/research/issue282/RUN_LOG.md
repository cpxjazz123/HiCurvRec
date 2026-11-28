# Issue #282 — RUN_LOG (frozen acceptance list)

| item | value |
|---|---|
| issue | #282 directed Graphlet/Orbit-conditioned L1 behavioural retention and Recall prediction |
| body sha256 | `2fd015a62aff559b0837f1e90a02f55221422ebca936f5e12b495644f8ac0182` |
| pinned inputs | #276 `SID_TABLE.json` `4c927914…`, `FOLDS.json` `1a8fb81d…`; #277 `PREDICTOR.json` `ec339f4a…`, `AUDIT_277.json` (corrected baseline 0.002972/0.667); #278–#281 records |
| INFRA_MODE | **FROZEN** — issue text: "单Agent；analysis-only，INFRA_MODE=FROZEN；复用#276–281已有SID/训练图/全量3、4节点Graphlet/Orbit及曲率资产，不建新公共infra、不改原资产、不重训TIGER/RQ-VAE/T5、不启动Stage2/3或新test" |
| training needed | **NO** — issue text: "仅允许专属离线分析…无论结果如何本轮不训练Stage3" |

## Frozen acceptance list

| C | requirement |
|---|---|
| C1 | Reuse the 129 labeled SIDs, 253 computable SIDs, fixed folds and the 120/9 identity split; new public labels only as separate records; validation rows (a = last history item, b = target) and legal negatives as in #277; verify the 92.4 % unseen share; training graph only; only offline analysis |
| C2 | Pair-level directed context vector from the training graph: chain a→m→b, reverse chain, shared-successor merge a→m←b, shared-predecessor fork a←m→b, closed directed triangle only for observed edges (feed-forward/feedback/mutual), with a/b Orbit roles and induced/non-induced statements; multi-label types, overlaps, support counts, normalisation by centre size; no closed triangle on unseen edges (a potential closure is an open-structure candidate, not an observed edge) |
| C3 | Full 3-node support counts preferred; 4-node graphlets only where the existing asset certifies pair-level roles, otherwise secondary/excluded; node-level Orbit totals never masquerade as pair-level structure; multi-label strata and pre-registered weights; support/overlap/seen-unseen coverage, zero-support/unclassifiable, direction and degree distributions; matching by edge existence, degrees, distance, support |
| C4 | Pre-registered main types (chain / merge / fork; closed only on seen edges), main contrasts and minimum support; other subdivisions and curvature exploratory; curvature only under the same structure/direction with sufficient coverage |
| C5 | Per SID and type t: P_t (L1 equality among validation positives of type t), N_t on matched legal negatives of the same type/direction/degree (and the #277 uniform-negative reference as a separate denominator), ER_t = P_t/N_t, P_t−N_t, lift; fixed reproducible matched-negative sample with the sampling error stated; anchor-clustered CI; L1+L2 secondary; role-specific codeword sharing (chain adjacent/endpoints, fork/merge centre-periphery) |
| C6 | Key contrast: SIDs with similar overall L1 P/N/ER (frozen tolerance) compared on their type retention and Recall; #274 R2 only a residual diagnostic; per-type high/low and seen/unseen comparisons controlling overall fidelity, frequency and degree; full per-SID type table, support/overlap and manifest |
| C7 | Prediction increment: 1–2 low-redundancy type contrasts (e.g. merge−chain or degree-matched merge/fork ER difference) chosen inside training folds only; the #277 corrected baseline 0.002972/0.667 recomputed and used (the #281 baseline is not mixed in); same folds, leak-free scaler |
| C8 | Cross-fitted type-vs-residual association with group bootstrap and FDR; OOF MAE/RMSE/Spearman/Kendall/reversals, >0.06 TP/FP/FN, low/high bias; strict identity subset separately; 0.065 NOT_EVALUABLE; matched SID pairs are not independent samples; no causal claim |
| C9 | Success: the pre-registered incremental model's SID-group bootstrap 95 % CI lower bound for `MAE(A) − MAE(new)` > 0 and Spearman/Kendall/high-recall misclassification not worse; descriptive type differences may exist without predictive increment; if ineffective, state whether coverage, collinearity, insensitivity or no increment |
| C10 | Freeze a new version only if the criterion passes (formula, features, parameters, hashes, domain, uncertainty, old/new predictions for truly unknown SIDs); otherwise keep #277; no Stage3 in any case |
| C11 | Deliverables: CONTRACT/HYPOTHESES/DIAGNOSIS, pair-level type/Orbit definitions with hand examples, support/overlap/matching audit, 253-SID type fidelity table, similar-overall SID contrasts and counterexamples, full OOF/CI/ablations/residual analysis, code/tests, persisted logs/manifest, REPORT/VERDICT, infra zero drift, commit+push; final answers: with overall L1 fidelity close, do chain/fork/merge/closed retention differences correlate with Recall; which types deserve the next mechanism; is there a reproducible predictive increment |
