ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S14_GLOBAL_REVIEW/round_1/source_packet.md
STAGE_ID=S14_GLOBAL_REVIEW

# Iter29 Global Review — independent direction proposal

## Trigger and current state

The **formal three-clean-protocol-valid-iteration trigger is not met**. There are two completed clean FCCR-1 fixed-mapping observations in the current evidence set: iter26 and iter29. Iter27 aborted before Stage2/Stage3 and does not count. Iter25 lacks its required protocol manifest and used learnable/cyclic curvature; iter28 tested CAO-1/SREMA with cyclic/learnable curvature, not FCCR-1. Neither adds a clean FCCR-1 iteration.

The **independent repeated-neutral/unpromoted trigger is met**. S12 and the source packet identify iter29 as the second clean FCCR-1 fixed-mapping observation following the prior S14-authorized bounded alternative mapping, and it remains unpromoted with an observed result inside the approximate historical noise scale. This permits a direction review; it does not establish family-level failure. The protocol/baseline inconsistency trigger was already handled by the iter28 review and is not a newly discovered trigger here.

Current status: iter29 is `ACTIVE_NEUTRAL` and `PROMOTION_FAIL`. Its registered fixed mapping passed activation checks and affected quantization/assignments; activation is not efficacy. Its completed Stage3 score is below the inclusive `test_recall@10 >= 0.065` objective by 0.005789359146224693. The shortfall is a promotion outcome, not proof that either mapping or FCCR-1 as a family failed.

## Direct evidence only

Iter29's S01 lock makes iter26 its sole protocol-compatible direct control. Both exact `test_final.json` records have `n_eval=57439`:

| Run | Fixed curvature | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| iter26 | `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]` | 0.03788366789115409 | 0.057017009349048554 | 0.025169911931406087 | 0.03133359761524377 |
| iter29 | `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` | 0.03953759640662268 | 0.05921064085377531 | 0.026252776900288842 | 0.03257647953179143 |

Observed iter29-minus-iter26 point differences are R@5 `+0.0016539285154685904`, R@10 `+0.002193631504726755`, NDCG@5 `+0.0010828649688827546`, and NDCG@10 `+0.0012428819165476584`. The R@10 point difference is smaller than the previously cited approximate ~0.003 historical noise band, which has not been independently estimated. With one run per mapping, there is no run-to-run variance estimate. These observations support near-parity/neutral classification, not a significant, robust, or causal gain and not a reliable negative result.

Iter18 is explicitly `HISTORICAL_NONCOMPARABLE` in the iter29 protocol manifest and is excluded from iter29 direct ranking and all direct-outcome comparisons in this review. Equal `n_eval` does not establish protocol compatibility.

## Activation, efficacy, promotion

- **Activation:** S08 MVG and S10 integrity records support that the iter29 closed-form mapping was fixed/invariant and measurably affected the quantization computation/assignments; Stage2 completed and exports passed integrity. These establish an active mechanism, not a recommendation benefit. Stage2 geometry/SID metrics remain descriptive and are not gates.
- **Efficacy:** The two protocol-compatible single-run observations do not resolve whether the tested mapping change causes a downstream benefit. Iter29's positive point deltas are small relative to the approximate, non-independent noise reference; neither mapping's efficacy direction is established.
- **Promotion:** Iter29 `R@10=0.05921064085377531` misses the inclusive target `0.065`; promotion fails irrespective of the neutral mechanism classification.

Preserve S03's medium-confidence limitation: historical method/value provenance was accepted, but checkpoint-level reproduction and historical byte identity were not established. Also preserve the disclosed optional-metadata mismatch (`RQVAE_VARIANT` is correct in the wrapper; train-start logging says `unknown_variant`) and nonfatal NCCL/c10d warnings. The Stage3 run completed with exit 0; the records do not justify treating these caveats as a demonstrated model/data identity failure or claiming warning-free execution.

## Direction decision

**Recommend an autonomously registered, matched-seed replication of the iter26-versus-iter29 mapping contrast as the next direction, under FCCR-1.** Do not begin another mapping search yet. The strongest immediate uncertainty is whether the observed small positive delta is reproducible or ordinary run variation; another unreplicated mapping would add a new point estimate while leaving that key ambiguity intact. The two active mappings show the mechanism can execute, but neither has demonstrated promotion and no between-run variance is available. Replication has higher near-term information value than interpreting the point delta or selecting another mapping from it.

Make the replication a genuine one-factor comparison: register two conditions that differ **only** in the already specified fixed mapping (iter26 mapping versus iter29 mapping). Use the same newly selected seed policy in both arms, and match data/Stage1 identities, parent implementation and warm start, Stage2 steps and all settings, Stage3 code/configuration and settings, evaluation set and metric computation. Keep the mapped curvature precomputed, closed-form, bounded, immutable, non-trainable, and time-invariant in both arms. Treat the two-arm run as a separately registered replication experiment (not as a silent modification/restart of iter29); record the seed and exact protocol identities before execution and adjudicate the protocol and one-factor design through the required stages. Do not pool unmatched historical scores as if they were a paired replicate or call the result significant from one new pair. If operationally feasible within the registered design, use multiple matched seed pairs and report the per-pair differences and spread; do not add other mechanism changes to achieve the target.

This choice deliberately prioritizes resolving uncertainty over claiming quick target progress. A replication that remains neutral would justify an explicit next between-iteration review/contract decision on whether to keep searching bounded FCCR-1 mappings or transition to a separately adjudicated research contract; it would not retroactively prove family failure. A repeatable improvement would establish a stronger efficacy signal but still must independently meet the promotion threshold to promote.

## Alternatives, paused directions, and unresolved hypothesis

- **Further alternative fixed mappings:** allowed as single-factor FCCR-1 experiments, but defer another one until the present contrast's uncertainty is better characterized. A third mapping result without replication risks selecting on noise and still cannot distinguish mapping effect from run variation.
- **Contract transition:** do not silently change FCCR-1 now. If later evidence justifies one, explicitly propose and adjudicate a separate between-iteration transition before implementing any new family/mechanism. The present two observations alone do not establish that a contract change is necessary.
- **Paused/deferred:** learnable or scheduled curvature, curvature regularization, curvature-conditioned optimizer, Sinkhorn or behavior-loss changes, manifold replacement, mechanism stacking, and Stage1/Stage3 changes remain paused/outside FCCR-1 absent a separately adjudicated transition. No Stage2 proxy is a reason to gate or choose among directions.

**Unresolved hypothesis:** whether changing only the fixed closed-form mapping from behavior branching and raw residual magnitude can reproducibly improve quantization representations and downstream recommendation performance enough to meet `R@10 >= 0.065`. The present observations establish implementation activation but leave mapping efficacy and its variance unresolved.

## Assumptions, risks, and self-rejection

Assumptions: the iter26 and iter29 S01 lock establishes adequate direct protocol compatibility for their already reported comparison; new replication arms can be made identical in every registered respect except the mapping; and seed variation is a meaningful source of uncertainty to estimate. The source packet notes S03 provenance is method/value-level only, so the replication must retain that caveat rather than claim checkpoint-level historical reproduction.

Risks: a matched seed pair may not estimate variance broadly; rerunning conditions can introduce runtime nondeterminism; a changed seed is a protocol change relative to existing single runs; and a replication focused on efficacy characterization may not itself reach the promotion target. Mitigate by preregistering the pair, keeping seed and all protocol choices matched within pair, recording exact inputs/checkpoints/configurations, and reporting uncertainty honestly. Do not compare against iter18 as a substitute control.

I would reject this replication recommendation if primary protocol records show iter26 and iter29 differ in an uncontrolled factor beyond the registered mapping, if the proposed pair cannot keep all non-mapping factors identical, or if activation/validity records are overturned by direct evidence. In those cases, do not infer an efficacy result from the current contrast: first adjudicate the protocol/provenance issue or choose a fully specified, independently adjudicated FCCR-1 experiment. I would also reject using this recommendation to justify an unadjudicated contract transition, to claim a significant gain, or to stop because SID proxies look unfavorable.

## Primary evidence consulted

- Shared candidate packet: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S14_GLOBAL_REVIEW/round_1/source_packet.md`.
- Skill rules: §2.7, §2.8, §2.10, §3, §4, §13–§16, §18, and §21 of `skill://curvature-rqvae-iter`.
- Iter29 direct protocol lock: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`; direct control protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md`.
- Exact direct Stage3 records: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json` and `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`.
- S12 primary adjudications and result interpretation: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/judge.md`, `round_2/judge.md`, `logs/stage3_outcome_iter29.md`, `logs/failure_attribution_iter29.md`, and `logs/gate_decision_iter29.md`.
- Prior S14 direction/trigger disposition: `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md` and `logs/global_review_after_iter28.md`.
