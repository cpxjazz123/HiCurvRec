ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=logs/deliberation/S14_GLOBAL_REVIEW/round_1/source_packet.md
STAGE_ID=S14_GLOBAL_REVIEW

# Trigger validity

The Global Review trigger fires. Iter27 was explicitly aborted and is excluded. The review is warranted by the iter25/26/28 completed Stage3 outcomes plus, independently, the skill's trigger for a core hypothesis that still lacks a clean test. These outcomes must not be conflated as three FCCR-1 tests: iter26 is the direct FCCR-1 fixed-curvature test; iter25 used learnable curvature and cyclic scheduling; iter28 tested SREMA under the inherited iter18 cyclic/learnable curvature contract. Only iter26 directly probes the required fixed mapping.

# Evidence and protocol classification

**Iter18 canonical baseline.** Its exact `test_final.json` reports `n_eval=57439`, `test_recall@10=0.05988962203380978`, R@5 `0.040460314420515675`, NDCG@5 `0.02697174940916111`, and NDCG@10 `0.03323035190128754`.

**Iter26 — protocol-compatible FCCR-1 evidence (with stated manifest as protocol record).** Its mechanism contract identifies FCCR-1, closed-form curvature, non-trainable/non-time-varying values, no cyclic schedule or curvature regularization, and inputs `behavior_branching` plus `raw_residual_median`; fixed values are `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`. The hypothesis provides the numeric inputs and formula. Its failure attribution reports invariant buffers, checks at steps 0/25k/50k/100k, model updates and a nonzero behavior gradient, so the registered mapping was implemented and active rather than an invalid/inactive test. Protocol manifest records same Stage2 seed 42, 100,000 steps, 3×256 quantizers, Stage3 seed 42, 150 epochs, beam 20, and `n_eval=57439`, with iter18 as canonical baseline. The manifest's Stage3 commit (`0052f4b...`) differs from iter28's (`bdcbbf9...`), but direct `git diff --exit-code` of `stage3_T5Train/train_HG-Rec.py` at those commits returned no differences; iter28's and iter18's checked-in Stage3 gin configs also match (the read snapshots have the same content hash). The iter26 manifest asserts the comparable protocol. Its exact result is `n_eval=57439`, R@10 `0.057017009349048554`, R@5 `0.03788366789115409`, NDCG@5 `0.025169911931406087`, NDCG@10 `0.03133359761524377`. Delta from iter18 R@10 is `-0.002872612684761226`. It is below the user's `>=0.065` target and below baseline. This is an observed single-run outcome, not proof of a family-level causal failure.

**Iter25 — HISTORICAL_NONCOMPARABLE for direct ranking / not a clean FCCR-1 test.** Its exact outcome is R@10 `0.05882762582914048`, `n_eval=57439`. Its hypothesis explicitly retains cyclic `c_l(t)` and a learnable `c_layer_scale`, and its failure attribution calls it a prior-mapping experiment rather than causal isolation. The source packet points to a protocol manifest, but that file is absent at the specified path; same `n_eval` does not repair the missing protocol verification. Do not count it as an FCCR-1 clean test or use its result to revise FCCR-1.

**Iter28 — protocol-compatible to iter18 for Stage3 outcome, but not FCCR-1 evidence.** The iter28 manifest declares the iter18 data, Stage3 configuration, seed/epochs/beam/evaluation protocol and identifies SREMA as its one Stage2 change; declared code commits differ but the Stage3 trainer source comparison showed no diff, and the iter28/iter18 gin snapshots match. The exact result is R@10 `0.05649471613363742`, `n_eval=57439`, delta `-0.00339490590017236` vs iter18. Its attribution reports direct SREMA activation (every codeword updated; finite, unsaturated trajectory) and explicitly names active-negative/promotion-fail. This is evidence against this SREMA configuration under the stated inherited regime, not against FCCR-1 or all codebook/manifold methods. Its contract says `CAO-1`, `iter18_cyclic_learnable_layer_scale`, and a new SREMA mechanism; it violates FCCR-1 if misrepresented as an FCCR-1 experiment.

**Iter27** is aborted (no Stage3 result), not a clean negative and not part of the completed-iteration count.

# Proposed Global Review content

## Cleanly tested vs confounded / invalid

- Cleanly tested within the active contract: iter26's registered raw-residual + branching fixed-curvature mapping, based on its FCCR-1 contract, implementation checks and recorded protocol. It was active and produced a completed Stage3 result, but missed the adoption threshold and scored below canonical iter18 in one seed. Its exact output tests this particular mapping, not every allowed fixed closed-form mapping or the entire branching/raw-residual hypothesis.
- Confounded with respect to FCCR-1: iter25 (cyclic, learnable curvature plus a changed prior; not isolated FCCR-1) and iter28 (SREMA under cyclic/learnable curvature). Both completed their declared Stage3 runs; neither can be counted as a clean fixed-curvature test. Iter27 aborted and contributes no outcome.
- No evidence here demonstrates that the iter26 run is contract-invalid. Its Stage2 proxy differences are descriptive only and cannot establish or negate Stage3 performance.

## Observed facts vs causal interpretation

Observed: iter26's fixed values were registered as non-trainable, reported invariant at checked steps, and the completed same-manifest Stage3 outcome was 0.0570170 (n=57,439). Observed: this is 0.0028726 below iter18's 0.0598896 and 0.0079830 below the user's target. Observed: iter25 and iter28 are also below target, but use distinct mechanism specifications. Inference, low confidence: this particular fixed mapping did not promote under this seed/protocol. A single run, without measured repeat-seed variance, does not establish a robust negative causal effect or falsify the broader FCCR-1 scientific hypothesis. Do not adopt the iter28 outcome's stated approximate noise-band conclusion as direct proof of iteration-independent noise; no repeated-seed distribution is supplied here.

## Strongest evidence-supported next direction under FCCR-1

Continue only with a newly registered, single-factor FCCR-1 experiment that tests a distinct, bounded alternative closed-form mapping of the same behavior-branching and raw-residual-median inputs to three precomputed, immutable curvatures. Specify the full equation, actual numeric substitutions, and falsification checks before implementation; keep Stage1, Stage2/Stage3 protocol, and all other mechanisms fixed to the canonical parent/baseline. This directly tests whether the negative result is specific to iter26's mapping rather than treating historical cyclic/learnable or SREMA results as adjudication of fixed curvature. Do not tune iter26's constants post hoc within its iteration.

## Paused directions

Pause iter25-style learnable/cyclic per-layer curvature priors, curvature schedule/regularization changes, SREMA or other codebook optimizer replacements, Stage1/Stage3 changes, and optimizer-/Sinkhorn-/behavior-loss additions as explanations for FCCR-1. They are outside the active contract or add a separate mechanism; the SREMA result does not authorize importing any into an FCCR-1 iteration. A contract change, if later evidence warrants one, must be canonically adjudicated between iterations.

## Unresolved core hypothesis

Whether behavior branching together with raw residual magnitude, mapped to fixed closed-form curvature, improves useful quantization structure and thereby Stage3 recall remains unresolved. Iter26 is a clean test of one mapping and failed promotion, but it does not establish a result for alternative allowed mappings; iter25 and iter28 are not substitutes for those tests.

# Assumptions

1. Iter26's protocol manifest is an accurate record of its executed Stage3 seed, epoch count, beam/evaluation settings, and its compatibility declaration; the available exact Stage3 result and source comparison corroborate the relevant evaluation/trainer facts, but the iter26-local gin file was unavailable at the path checked.
2. The registered iter26 contract, hypothesis, and failure-attribution checks accurately describe the executed fixed-curvature implementation. I found no primary evidence here contradicting those claims.
3. Comparisons use only the exact result files identified by the packet. A matching `n_eval` alone is not taken as proof of compatibility.
4. The user's `>=0.065` task threshold controls promotion for this review, irrespective of historical documents using strict `>0.065`.

# Risks and self-rejection conditions

- Protocol inference risk: if a direct audit of iter26's actual Stage3 launch/configuration contradicts its manifest (e.g. seed, beam, data, epochs, or evaluator differ), reclassify iter26-vs-iter18 as `HISTORICAL_NONCOMPARABLE`; retain the result as an absolute reported score, not a direct rank. In that case the packet does not provide a clean, protocol-compatible FCCR-1 outcome.
- Attribution risk: if direct checkpoint/runtime evidence shows the fixed values varied, were trainable, or the mapping inputs were not raw residual medians, withdraw the claim that iter26 cleanly tested FCCR-1 and classify accordingly.
- Uncertainty risk: one seed cannot resolve stochastic variance or demonstrate causality; do not call this mapping family-wide failure or promise the next mapping meets target.
- Scope risk: the recommendation is invalid if its proposed follow-up changes curvature away from fixed closed form, uses normalized scale in place of raw residual, changes parent/protocol or adds another mechanism, or fails to record preregistered actual values before training. Under any of these conditions, reject this recommendation and return to contract-compliant replanning.

# Concrete recommendation

Autonomously register a new one-factor iteration under FCCR-1 to test one explicitly specified bounded alternative mapping from the same branching and raw-residual inputs to fixed curvature, with the numeric values preregistered before implementation. Use the canonical protocol and baseline unchanged, complete the matched Stage3 evaluation, and treat Stage2 geometry only as descriptive. Judge the result against the user's `R@10 >= 0.065` target and report uncertainty; do not revive cyclic/learnable curvature, SREMA, or any other deferred mechanism inside that experiment.
