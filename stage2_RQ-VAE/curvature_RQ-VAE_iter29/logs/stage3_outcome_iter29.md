# Iter29 Stage3 Outcome

## Result identity and protocol

- Protocol: `FCCR-1_iter29_bounded_alternative_closed_form_vs_iter26`, locked in `logs/protocol_manifest_iter29.md`.
- Candidate `test_final.json`: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`.
- Best checkpoint: `results/stage3_T5Train/curvature_RQ-VAE_iter29/ckpt/Amazon_2023_Instruments/Sep-27-2026_06-11-34/HG_Rec_best.pth`.
- Completed evaluation: `n_eval=57439`; one Stage3 run, epoch 150, completed and recorded the final test result.
- Sole direct control: iter26 `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json` (`n_eval=57439`). Iter18 is `HISTORICAL_NONCOMPARABLE`; it is not used as a direct comparator.

## Exact metrics and observed deltas

| Metric | Iter29 | Iter26 direct control | Iter29 minus iter26 |
|---|---:|---:|---:|
| `test_recall@5` | 0.03953759640662268 | 0.03788366789115409 | +0.0016539285154685904 |
| `test_recall@10` | 0.05921064085377531 | 0.057017009349048554 | +0.002193631504726755 |
| `test_ndcg@5` | 0.026252776900288842 | 0.025169911931406087 | +0.0010828649688827546 |
| `test_ndcg@10` | 0.03257647953179143 | 0.03133359761524377 | +0.0012428819165476584 |

The observed deltas are positive point differences, not established causal gains. There is one run per condition. The prior iter28 Global Review cites an approximate historical R@10 noise band of about `0.003`, not independently estimated; the observed `+0.002193631504726755` is smaller than that approximate band. Treat this comparison as near-parity/neutral, not as a robust improvement or family-level negative.

## Mechanism and promotion classification

- `MECHANISM_STATUS=ACTIVE_NEUTRAL`
- `PROMOTION_STATUS=PROMOTION_FAIL`
- The S02-registered FCCR-1 bounded rational/additive fixed-curvature mapping was active: `x=B/(B+2)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, `c=0.05+1.45*u`, with fixed `c=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. S08 reports MVG PASS, fixed-curvature invariance and a measurable counterfactual activation; S10 records completed Stage2 and `STAGE2_OUTPUT_INTEGRITY_PASS`. These establish execution/activation, not recommendation efficacy. Stage2 geometry metrics remain descriptive, not gates.
- User success threshold is inclusive `test_recall@10 >= 0.065`. Iter29's `test_recall@10=0.05921064085377531` misses by exactly `0.005789359146224693`, so promotion fails independently of the mechanism classification.
- S03 passed only historical method/value provenance at medium confidence. It did not establish checkpoint-level reproduction or historical byte identity; retain that limitation.

## Validity caveats

The approved wrapper sets `RQVAE_VARIANT=iter29_bounded_rational_additive_mapping`, while the training-metrics `train_start` event reports `variant=unknown_variant` because the logger reads `config.get('variant')`. This is a disclosed optional-metadata mismatch; the recorded code path, expected short-root paths, Stage2 SID path, test record and checkpoint identity agree, and available evidence does not show a model/data identity failure. The launcher records nonfatal NCCL initialization warnings and a c10d barrier device-context warning. Training, test and supervisor completed with exit 0; preserve the warnings without treating them as silent-clean startup or as evidence of invalid output.

## S14 trigger and next action

S14 is newly triggered under the independent repeated-neutral/negative-results criterion: iter29 is the second clean FCCR-1 fixed-mapping observation after the prior Global Review's bounded-mapping direction, and it adds another active but unpromoted/near-parity outcome. This is a trigger for direction review, not proof of family-level failure. The formal three-clean-protocol-valid-iteration count is not met. The prior protocol/baseline inconsistency trigger was already adjudicated after iter28 and is not newly counted. Proceed to S14_GLOBAL_REVIEW via independent A/B/Judge deliberation; use protocol-compatible iter26/iter29 evidence for direct comparison and retain iter18 as `HISTORICAL_NONCOMPARABLE`. Do not rerun or restart iter29.
