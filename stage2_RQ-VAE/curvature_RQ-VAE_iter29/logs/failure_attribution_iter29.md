# Iter29 Failure Attribution

## Classification

`MECHANISM_STATUS=ACTIVE_NEUTRAL`
`PROMOTION_STATUS=PROMOTION_FAIL`
`STAGE3_RESULT_VALIDITY=VALID_COMPLETED_SINGLE_RUN`

The term “failure attribution” here records why promotion was not achieved; it does not classify the registered FCCR-1 mapping as inactive or invalid. S01 locks iter26 as the sole direct control. The exact candidate test record is `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, with `n_eval=57439`, `test_recall@5=0.03953759640662268`, `test_recall@10=0.05921064085377531`, `test_ndcg@5=0.026252776900288842`, and `test_ndcg@10=0.03257647953179143`. Iter26's exact result is `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, also with `n_eval=57439`; its corresponding metrics are `0.03788366789115409`, `0.057017009349048554`, `0.025169911931406087`, and `0.03133359761524377`. All four observed iter29-minus-iter26 deltas are positive: R@5 `+0.0016539285154685904`, R@10 `+0.002193631504726755`, NDCG@5 `+0.0010828649688827546`, and NDCG@10 `+0.0012428819165476584`.

The inclusive user threshold is `test_recall@10 >= 0.065`. Iter29 is short by `0.005789359146224693`; promotion therefore fails. Its R@10 point difference is smaller than the prior review's approximate ~`0.003` historical noise band, which was not independently estimated. There is only one run per condition and no measured run-to-run variance. The data support an active, near-parity/neutral outcome; they do not establish a causal gain, robust positive effect, or reliable negative effect. Do not infer a family-level FCCR-1 failure from this score.

## Verified mechanism activation versus unresolved efficacy

The registered iter29 mapping is the bounded closed-form mapping `x=B/(B+2)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, `c=0.05+1.45*u`, producing fixed `c=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. S08 records MVG PASS, fixed-curvature invariance and a measurable candidate/control counterfactual (`candidate quantize loss=2.10162615776062`, control `0.8013777732849121`; assignment fraction changed by `0.4062500298023224`). S10 records the 100,000-step Stage2 run, aligned SID export and `STAGE2_OUTPUT_INTEGRITY_PASS`. This establishes that the registered mechanism was implemented and affected computation; it does not establish downstream recommendation benefit. Stage2 Gini, collision, entropy and related geometry remain descriptive, not gates.

S03 was approved only for historical method/value provenance, at medium confidence, not checkpoint-level reproduction or historical byte identity. Keep this as an input provenance/replay limitation; it is not evidence by itself that the approved iter29 mechanism was inactive or invalid.

## Runtime and identity caveats

The approved wrapper sets `RQVAE_VARIANT=iter29_bounded_rational_additive_mapping`, but `training_metrics.jsonl` records `variant=unknown_variant`; `train_HG-Rec.py` logs `config.get('variant', 'unknown_variant')`. The event otherwise records the expected code path, iter29 log/checkpoint roots, Stage2 SID path and `world_size=4`; the final test names the matching checkpoint. This is a disclosed optional-metadata mismatch. Available evidence does not show that the variant label controls model, data, or evaluation identity, so do not recast it as an assumed identity failure.

The launcher records four startup `NCCL WARN lib wrapper not initialized` messages and a c10d barrier device-context warning. Preserve these as operational warnings. The single supervisor completed with exit 0, ranks 0–3 reached epoch 150 and test completed; the packet's scan reported no fatal traceback/runtime/NCCL error/OOM matches. These observed facts mean the warnings did not prevent this run's completion, not that startup was warning-free or all distributed-runtime risk was absent.

## S14 disposition

S14 is triggered now by the independent repeated-neutral/negative-results criterion: iter29 supplies a second clean FCCR-1 mapping observation after the prior iter28 review recommended one bounded alternative mapping, and this second observation is also unpromoted and near-parity. This is only a direction-selection trigger; it is not a claim that the FCCR-1 family has failed. The formal three-clean-protocol-valid-iteration trigger remains unmet. The protocol/baseline inconsistency trigger was already addressed by the prior review and is not newly discovered here. For direct evidence use only iter26 and iter29; keep iter18 `HISTORICAL_NONCOMPARABLE` and do not rank it against iter29.
