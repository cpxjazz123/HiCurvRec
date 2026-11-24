# R01 post-result R1/R2 questions

R01 is a valid completed3000/150-epoch candidate, not an implementation-failure rerun. Exact SID export, complete final test and independent reduction passed; R@10=0.05273420498267727 fails the issue's strict0.065 threshold. R2/102 preserves the complete generation. Same-checkpoint recommendation validation is still running. No next mechanism is registered here.

## Evidence, counterexamples and alternatives

The four preregistered Stage2 forecasts hold individually: noncollapse, more triples, better ordinary reconstruction and larger validation direct-minus-control without increasing control. This does not establish the reason for recommendation failure. Full training-graph macro curvature improves but micro/sign-balanced metrics worsen; gate means strongly favor Euclidean space. These are associations, not isolated causes.

The final-weight rank0 replay has maximum weighted C/O norm ratio10.4584; sharedKey and Query0 have negative original-dot-combined raw-gradient alignment. However, R01 promised an initialization-only global norm budget. A local final batch is neither a four-rank averaged gradient nor the optimizer's AdamW direction. Global cancellation, sampling variability, changing geometry scales, proxy weighting, gate saturation and insufficient behavioral information remain alternatives. The root cause is unknown.

## Minimum train-only discriminating diagnostic, frozen before execution

Question: does the observed final local imbalance remain after averaging actual VJP vectors across all four ranks?

Use the exact final R01 checkpoint680439b57bfdbec622a5a109b0d589ab21566f71236fc007eb19d59b6108fc0c and its immutable executed source, original losses and frozen coefficient. Reuse existing train.parquet, embeddings and graph/curvature assets without creating or rebuilding any asset. Four ranks each replay the first1024-item batch of DistributedSampler(seed42,epoch2999), with the trainer's StructuralSampler.draw(step2999*len(loader),rank,world4,track=False). Fresh diagnostic RNG is recorded; this is not a reconstruction of the formal final RNG/optimizer state.

Compute each rank's actual original and curvature VJPs, then SUM/world on the vectors before norms. For each of the same17 disjoint coupled groups record global norms, weighted ratio, cosine and `<gO,gO+lambda*gC>/||gO||²`. The first-step budget tolerance remains relative2e-6; a final ratio>1+2e-6 is a witness that the initialization-only bound is not a final-time bound. A negative normalized dot witnesses a raw Euclidean original-ascent direction on this measured global batch, not an AdamW finite-step loss increase or Recall causality. Zero norms stay explicitly undefined where appropriate; no zero-weight fallback or recalibration is performed.

Coverage is one real global4096-item batch, not all3000epochs or the full training population. Counterevidence is essential: if averaging removes the local violations, the local replay is insufficient support for a global-imbalance explanation. Regardless of outcome, retain the result and combine it with complete validation-only paired diagnostics before another hypothesis. No recommendation-test error rows or structure-test labels enter this decision.

This is read-only current-mechanism diagnosis: no optimizer, parameter update, formal training, SID rewrite, extra seed, ablation or threshold change. Launch only through the sanctioned four-card launcher, after the current validation GPU consumer exits. Archive the inactive R01 live tree before any new-mechanism design/copy/implementation.
