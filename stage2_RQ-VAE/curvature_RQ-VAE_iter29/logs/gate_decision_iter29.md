# Iter29 Gate Decision

## Decision

`PROMOTION_STATUS=PROMOTION_FAIL`
`MECHANISM_STATUS=ACTIVE_NEUTRAL`
`STAGE3_RESULT_VALIDITY=VALID_COMPLETED_SINGLE_RUN`

The user-defined promotion threshold is inclusive: `test_recall@10 >= 0.065`. Iter29's exact `test_recall@10` is `0.05921064085377531`, a shortfall of `0.005789359146224693`. The observed result does not pass promotion. Promotion failure is separate from mechanism status: S08/S10 show the FCCR-1 mapping was active and affected computation, while one Stage3 run per condition does not resolve a causal efficacy direction.

## Direct comparison

S01 specifies iter26 as the sole direct control. Its exact `test_final.json` and iter29's exact `test_final.json` both report `n_eval=57439`. Iter29 minus iter26: R@5 `+0.0016539285154685904`; R@10 `+0.002193631504726755`; NDCG@5 `+0.0010828649688827546`; NDCG@10 `+0.0012428819165476584`. These are observed point differences only. The R@10 difference is below the approximate ~`0.003` historical noise band cited by the prior Global Review, which was not independently estimated; no run-to-run variance is estimated from one run per condition. Record near-parity/neutral, not a causal gain or robust improvement. Iter18 is `HISTORICAL_NONCOMPARABLE`; it is not a direct comparator or ranking reference here.

## S14 and autonomous next action

The independent repeated-neutral/negative-results trigger in skill §15 is newly met: iter29 is the second clean FCCR-1 fixed-mapping observation after the prior S14's one-more-bounded-mapping direction and adds a second active but unpromoted/near-parity result. This warrants direction selection only and does not establish FCCR-1 family failure. The formal three-clean-protocol-valid-iteration trigger is not met. The protocol/baseline inconsistency trigger was already adjudicated in the iter28 review and is not a new discovery.

`AUTONOMOUS_NEXT_ACTION=Start S14_GLOBAL_REVIEW using independent Agent A, Agent B, and Judge C deliberation before selecting any iter30 mechanism. Use protocol-compatible iter26 and iter29 evidence for direct comparison; keep iter18 HISTORICAL_NONCOMPARABLE. Do not rerun or restart iter29. The review must autonomously select the next between-iteration direction.`

## Preserved caveats

S03 is a medium-confidence PASS for historical method/value provenance only, not checkpoint-level reproduction or byte identity. The approved wrapper sets `RQVAE_VARIANT` correctly while the train-start logger reports `unknown_variant`; this is optional-metadata mismatch, with no evidence of changed model/data identity. Preserve the nonfatal NCCL/c10d startup warnings: the run completed with exit 0, but was not warning-free. Stage2 geometry remains descriptive, not a promotion gate.
