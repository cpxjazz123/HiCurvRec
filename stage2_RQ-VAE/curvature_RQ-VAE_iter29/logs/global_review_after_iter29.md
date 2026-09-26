# Global Review After iter29

## Trigger and scope

This review is triggered by S12's newly judged independent repeated-neutral/unpromoted FCCR-1 result criterion. It is **not** triggered by the formal three-clean-iteration count: iter26 and iter29 are the two completed clean fixed-mapping observations; iter27 aborted before Stage2/Stage3; iter25 has no required protocol manifest and used learnable/cyclic curvature; iter28 used CAO-1/SREMA with cyclic/learnable curvature and is not a clean FCCR-1 test. The protocol/baseline inconsistency that triggered the prior review after iter28 was already addressed there; it is not newly discovered in this review. Neither trigger determination proves failure of the FCCR-1 family.

The active contract remains FCCR-1: fixed, non-trainable, time-invariant, closed-form curvature from behavior branching and raw residual medians. Alternative bounded mappings are allowed as one-factor experiments. No learnable/scheduled curvature, curvature regularization, optimizer/Sinkhorn/auxiliary-loss mechanism, mechanism stacking, or Stage1/Stage3 change is authorized here.

## Protocol-compatible evidence only

Iter29's S01 lock identifies iter26 as its **sole direct control**. Both exact primary Stage3 `test_final.json` records have `n_eval=57439`; iter29's protocol manifest explicitly labels iter18 `HISTORICAL_NONCOMPARABLE`. Iter18 is excluded from direct ranking and no iter29-to-iter18 score delta is asserted.

| Run | FCCR-1 mapping | R@5 | R@10 | NDCG@5 | NDCG@10 | n_eval |
|---|---|---:|---:|---:|---:|---:|
| iter26 | First clean fixed mapping; direct control | 0.03788366789115409 | 0.057017009349048554 | 0.025169911931406087 | 0.03133359761524377 | 57439 |
| iter29 | Second clean bounded alternative fixed mapping | 0.03953759640662268 | 0.05921064085377531 | 0.026252776900288842 | 0.03257647953179143 | 57439 |

Primary records: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`; `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`; S01 protocol lock: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`.

Observed iter29-minus-iter26 point differences: R@5 `+0.0016539285154685904`, R@10 `+0.002193631504726755`, NDCG@5 `+0.0010828649688827546`, NDCG@10 `+0.0012428819165476584`. The R@10 difference is within the prior review's approximate ~0.003 historical noise reference, which has not been independently estimated. With one run per mapping there is no run-to-run variance estimate. These point observations establish neither a robust/causal gain nor an established negative result.

## Cleanly tested vs confounded or invalid

- **Cleanly tested:** iter26 is one valid test of its registered fixed mapping. Iter29 is a second valid test of its bounded alternative mapping. S08 MVG and S10 integrity evidence support that iter29 curvature remained fixed and measurably affected quantization/assignments, and the Stage2 exports passed integrity. These are implementation/activation findings, not proof of downstream efficacy.
- **Not clean FCCR-1 evidence:** iter25 lacks its protocol manifest and used learnable/cyclic curvature; iter27 aborted before Stage2/Stage3; iter28 tested CAO-1/SREMA while retaining cyclic/learnable curvature. None establishes that the fixed closed-form family fails.
- **Provenance and execution caveats:** S03 accepted historical method/value provenance at medium confidence, not checkpoint-level reproduction or historical byte identity. Preserve that boundary. Disclosed nonfatal NCCL/c10d warnings and the `unknown_variant` optional-metadata logging discrepancy are caveats; S12 did not find evidence that they invalidate the completed Stage3 result. Stage2 geometry metrics remain descriptive, not gates.

The S12 classification remains `MECHANISM_STATUS=ACTIVE_NEUTRAL` and `PROMOTION_STATUS=PROMOTION_FAIL`. The observed positive delta is not upgraded to causal benefit, and the two observations do not establish a negative result for all FCCR-1 mappings.

## Strongest evidence-supported direction

Register a **separate, prospective, one-factor matched-seed replication** of the two already tested fixed mappings. This is more informative now than a third unreplicated mapping: it directly tests whether the only observed mapping difference is reproducible or compatible with run variation, before selecting another equation. It preserves the active contract and introduces no new mechanism. An immediate contract transition would exceed the evidence: two noisy single-run mapping observations do not establish family failure, and other mechanisms remain forbidden/deferred unless separately adjudicated between iterations.

Make the replication a finite three-pair block using new predeclared seeds **43, 44, and 45**. Each pair contains both arms under the same Stage2 seed and the same Stage3 seed; the sole conceptual difference within each pair is the fixed-curvature mapping. For every pair, hold the Stage1 embeddings and item identities, Stage0 data/splits and ordering, warm-start checkpoint, source/configuration other than the mapping, Stage2 settings, Stage3 code/configuration and runtime settings, evaluation set, and metric computation identical. Carry forward the S01-locked Stage1/data identities and the same immutable warm start (recorded SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`); lock the exact identities and resolved use again in the replication's own S01/preflight gates. Use Stage2 100,000 steps, 3 layers × 256 codes, and identical Stage3 settings for both arms: 150 epochs, beam 20, and `n_eval=57439`, with one pinned Stage3 code/configuration in both arms. The Stage2/Stage3 seed for each arm is respectively 43, 44, or 45.

The arms are exactly the existing mappings, not newly invented mechanisms:

- **iter26 mapping:** `s_l=log1p(B_l)/log1p(m_l_raw/min(m_raw))`; `z_l=(s_l-mean(s))/(std(s)+1e-12)`; `c_l=clip(0.5*exp(0.2*z_l), 0.05, 1.5)`. Registered outputs: `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`.
- **iter29 mapping:** `c_l=0.05+1.45*0.5*(B_l/(B_l+2.0)+m_l_raw/(m_l_raw+0.1))`. Registered outputs: `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.

Use the same approved historical formula inputs in both arms, in L0/L1/L2 order: `B=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`; `raw_residual_medians=[1.0, 0.10941, 0.09331]`. Preserve S03's medium-confidence limitation: this is historical method/value provenance, not checkpoint-level reproduction or input-byte identity. Re-establish the raw-input provenance gate for the new experiment and never substitute normalized layer scales.

Report each arm's R@10 against the inclusive user target `>=0.065`, each within-seed paired difference, and the observed paired spread/uncertainty. The three pairs entail six full Stage2/Stage3 pipelines, a material but proportionate cost for resolving the immediate mapping-versus-run-variation ambiguity; the existing pipeline has already completed these stages. Three pairs remain a modest sample, do not establish precise tails or eliminate stochasticity/mapping-by-seed interaction, and cannot promise the target will be reached. Retain the seed-42 pair as prior evidence; pool it only if the replication's S01 lock confirms the necessary identity and protocol compatibility. If a protocol/provenance/one-factor hard gate fails, do not run a confounded experiment; use the established independent 2+1 process for an autonomous, contract-valid replan.

## Paused directions

Defer a third standalone mapping until the matched comparison better characterizes the present uncertainty. Keep learnable/scheduled curvature, curvature regularization, curvature-conditioned optimizer changes, new Sinkhorn or behavior-loss mechanisms, manifold replacement, mechanism stacking, and Stage1/Stage3 changes paused/out of scope under FCCR-1. Stage2 SID/Gini/collision/entropy proxies cannot select the direction or replace Stage3. Any future between-iteration contract transition must be proposed and adjudicated separately; this review does not authorize one.

## Unresolved core hypothesis

It remains unresolved whether changing only the fixed closed-form mapping from behavior branching and raw residual magnitude can reproducibly improve quantization representations and downstream recommendation performance enough to meet the target. The current runs show that the mappings execute and are active; one score per mapping, with a small point difference inside an unestimated noise reference, does not resolve mapping efficacy or its variance.

## Target, status, and autonomous continuation

The inclusive target is `test_recall@10 >= 0.065`. Iter29's exact result is `0.05921064085377531`, short by `0.005789359146224693`; promotion failed. Replication is selected to improve the evidence for the next autonomous direction decision, not because it is expected or guaranteed to meet the target.

`USER_INPUT_REQUIRED=NO`

`AUTONOMOUS_NEXT_ACTION=After this S14 review is pushed to GitHub main, local and remote main hashes match, and the deliberation gate passes with phase=GLOBAL_REVIEW, register a separate iter30 matched-seed FCCR-1 replication of the iter26 and iter29 mappings with predeclared pairs 43, 44, and 45; complete its independent 2+1 protocol and contract gates before any Stage2/Stage3 run.`

S13 recorded iter29 closure at `57b4a594d92435fd74fa4bba7c7c1439a33eb102` with matching local `main` and permitted GitHub `origin/main` hashes. This S14 report and its Judge decision are new canonical outputs and are not covered by that earlier hash. **This review must itself be pushed to GitHub `main`, and the resulting local/remote `main` hashes must be verified identical; the required `phase=GLOBAL_REVIEW` deliberation-gate pass is also required before any next iteration starts.** No Stage2/Stage3 job, code change, Git operation, or contract transition is performed by this review.
