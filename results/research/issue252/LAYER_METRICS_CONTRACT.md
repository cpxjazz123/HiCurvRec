# Issue252 layered SID diagnostics (evaluation only)

Frozen before implementation. No additional training, model selection, sampled target asset or Stage3 score input. Existing Stage0 train/valid, items and actual exported SID are the only inputs; invalid global IDs fail closed.

For the uniformly weighted global catalogue of N items, define prefix P_k and H_k = -sum_p (n_p/N)log2(n_p/N). L3 refinement gain is H_3-H_2 = H(C3|C1,C2), in bits. Three-code collision is 1-|P_3|/N; the fourth export counter never enters any prefix or entropy. Report occupied codes at each level and distinct P_2/P_3.

Recommendation-validation L1 positive rate uses each valid row's final history item as anchor and target as positive. Empty histories are excluded with a count. For the negative diagnostic, use the exact expectation over uniformly selected catalogue items outside seen_history ∪ history ∪ {anchor,target}; no random bank is created and no future test data is read. For each row and level, negative prefix probability = (catalogue prefix count - excluded prefix count)/(N-|excluded|). Mean those probabilities over usable rows; do not call fractional expected mass an observed negative-hit count. Hard ER is positive-hit rate / negative-hit probability, or explicit NA if zero/undefined. This is a declared exact uniform-negative diagnostic, not the old fixed ten-negative sample.

The removed legacy Conditional G3 implementation has no surviving defining formula. Do not invent its value: persist `conditional_g3` as MISSING with this reason. Separately report two explicitly named quantities, without claiming legacy equivalence: log2[(positive survival L3|L2)/(negative survival L3|L2)] in bits, and the signed difference between those survival probabilities. Undefined denominators/zero ratios are explicit NA, never filled with0.

Known catalogue: (000),(001),(010),(100) has3 distinct two-prefixes,4 triples, collision0, refinement0.5bits. For validation anchor0/positive1, excluded{0,1}, the two legal negatives2/3 give L1 negative probability0.5, positive rate1, ER2; both L2/L3 negative probabilities0, so conditional log enrichment is NA. A fully excluded catalogue is excluded, not divided by zero.

Record actual SID path/SHA256, train/valid paths/SHA256, item counts, excluded rows, estimator scope, evaluator source SHA256, arm and output path. Persist once at results/stage2_RQ-VAE/<ARM>/stage2_metrics/<sid_sha256>.json before reading candidate Stage3 scores. Existing records are immutable; an identical recomputation may reuse them, inconsistent content must fail.
