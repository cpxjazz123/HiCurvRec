ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S02_HYPOTHESIS/round_1/source_packet.md
STAGE_ID=S02_HYPOTHESIS

# Iter30 S02_HYPOTHESIS — Agent A proposal

## A. Research question

Across the three predeclared matched seeds 43, 44, and 45, does replacing only the iter26 fixed-curvature mapping with the already-tested iter29 fixed-curvature mapping increase mean Stage3 `test_recall@10` (paired `d_s=R29_s-R26_s`, with mean greater than zero), while each valid final result is separately classified against the inclusive user target `test_recall@10 >= 0.065`?

## B. Exact mappings, constants, and one-factor contrast

All layer-indexed vectors use `[L0,L1,L2]` order and both arms consume the same registered `B_l` and explicitly named `m_l_raw` inputs. These are the only two registered maps; no constants are tuned and no third map is proposed.

**Fresh paired control — iter26 map.** Let `m_min=min_l(m_l_raw)`, `s_l=log1p(B_l)/log1p(m_l_raw/m_min)`, and use the population standard deviation over the three layer values in `z_l=(s_l-mean(s))/(std(s)+1e-12)`. Then

```text
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
```

The constants are `c_base=0.5`, `alpha=0.2`, clip lower bound `0.05`, upper bound `1.5`; there is no added map-specific constant. The clip is part of the existing registered control equation.

**Candidate — iter29 map.** For each layer,

```text
x_l = B_l / (B_l + B_ref)
y_l = m_l_raw / (m_l_raw + m_ref)
u_l = (x_l + y_l) / 2
c_l = c_min + (c_max - c_min) * u_l
```

with exactly `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; equivalently `c_l=0.05+1.45*u_l`. No cross-layer normalization, extra clipping, or retuning is proposed.

**One-factor boundary.** Within each seed pair, only the closed-form fixed mapping/vector differs. Preserve identical input identities and order, immutable iter8 warm-start and its verified loading, all other Stage2 mechanisms/settings and Stage1 inputs, and unchanged Stage3 code/settings/evaluation. The protocol specifies six complete serial Stage2→Stage3 runs: both maps at each of seeds 43, 44, and 45, using the same Stage2 and Stage3 seed within each pair. Do not omit/replace a seed or arm, stop early, or pool historical seed 42 into the primary estimate. Stage2 descriptive metrics are not admission gates.

## C. Actual numeric substitutions and provenance qualification

The registered layer-ordered substitution vectors are:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
m_raw = [1.0, 0.10941, 0.09331]  # raw_residual_medians; PROVISIONAL_PENDING_S03
```

The residual vector is explicitly provisional pending independent S03 provenance adjudication. It is the historical `raw_residual_medians` vector, **not** normalized layer scales `[0.001, 0.932889, 1.0]`. Iter26 historically consumed the ambiguous JSON key `residual_norm`; the equal-valued explicit raw field elsewhere in the record does not certify that old alias's semantics. Any new use must name/read the explicit raw-residual field. The values below reproduce the registered decimal calculations, not historic byte identity, checkpoint-level reproduction, or S03 approval.

**Control substitution.** `m_min=0.09331`; applying the control formulas gives:

```text
s = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]
z = [1.03129493309937, 0.3223997103378134, -1.3536946434371857]
fixed_curvature = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
```

The first three arrays are the registered `s`, standardized `z`, and final clipped outputs in layer order; all three final values lie inside the clip bounds.

**Candidate substitution.** Applying the rational features and affine bounded map gives:

```text
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

The `B` and `m_raw` substitutions and candidate intermediates/output are recorded in the canonical packet and independently reflected in iter29's `compute_closed_form_curvature.py` constants/equations and `computed_behavior_branching.json`. Iter26's primary `mechanism_manifest_iter26.md` records its equations, `s`, `z`, and output, while its mapping script reads the historical alias `residual_norm`. These records support the arithmetic and historical mapping definition, not resolution of S03's raw-residual provenance gate.

## D. Direct effects, fixed-curvature invariants, and counterfactual

Both mappings are FCCR-1: compute each arm's three curvatures before Stage2; store/use them as fixed, non-trainable values. For each arm, directly verify that actual values equal its registered vector within the implementation's declared floating-point tolerance, are finite and supported, have no trainable curvature parameter/optimizer membership/curvature gradient, and are unchanged across training steps, optimizer updates, and train/eval mode. They must not depend on batch activations, step, or mutable optimizer state. This experiment does not add a schedule, curvature regularizer, curvature-conditioned optimizer, or auxiliary loss.

The preregistered same-checkpoint/same-batch counterfactual holds checkpoint, batch, mode, and all non-curvature settings fixed, evaluates the same curvature-consuming Stage2 path under each registered fixed vector, and records the actual layer curvatures plus the resulting per-example/per-layer curvature-dependent distances and nearest-code assignments (or the existing direct assignment/loss signal if that is the path's exact consumer). Report a concrete numerical difference, e.g. max absolute distance change and changed-assignment fraction against the common number of assignments, and compare against deterministic repeat/tolerance noise. A reproducible nonzero difference beyond numerical tolerance is required to show direct activation; identical distance/assignment outputs within tolerance falsify activation for the tested consumer. This counterfactual establishes only a direct mapping-mediated computational effect, not retrieval benefit.

The fixed vectors themselves are an immediate direct effect: candidate-minus-control curvature is `[0.7515595785009063, 0.2014808741174853, 0.2625879974275077]` from the registered decimals. This does not by itself prove a downstream consumer or retrieval change.

## E. Downstream rationale and registered paired estimand

**Inference, not established fact:** the candidate's mapped curvature is higher than control at each layer; because the unchanged quantization path consumes curvature, this may change distances, assignments, and residual allocation. If those geometry/quantization changes preserve useful item distinctions for the unchanged Stage3 recommender, retrieval could improve. The chain from the historical branching/residual structure through fixed geometry and quantization to Stage3 benefit remains an inference; neither the historical single-seed difference nor the proposed counterfactual demonstrates that it will occur.

For each `s ∈ {43,44,45}`, define `R26_s` and `R29_s` as final `test_recall@10` from the corresponding protocol-valid control and candidate runs and `d_s=R29_s-R26_s`. Primary estimand is `mean_delta=(d_43+d_44+d_45)/3`. Complete all six runs and audit their final test records before reporting it. Report all three signed pair differences; mean, sample SD (`ddof=1`), SE, and paired range; optionally a df=2 paired-t interval explicitly as highly imprecise descriptive uncertainty. Also report each arm's three-run mean/spread and each individual run's target classification. The seed-42 historical values—iter26 `0.057017009349048554`, iter29 `0.05921064085377531`, both `n_eval=57439`—are context only and are not a prospective pair, variance estimate, or input to the primary estimate.

Classify each protocol-valid final result against the user's inclusive criterion independently: `test_recall@10 >= 0.065` meets the target, including equality; lower values do not. Target attainment and the paired mapping effect are distinct: success on one does not imply success on the other. No target-based or trend-based early stop, run omission, or seed-42 pooling is permitted.

## F. Falsification and decision boundaries

This proposal registers exactly one scientific hypothesis: under the locked protocol, the iter29 map yields a positive three-seed mean paired R@10 contrast relative to fresh iter26-map controls (`mean_delta > 0`). A complete, protocol-valid estimate `mean_delta <= 0` falsifies that directional retrieval hypothesis for this two-map comparison. Report mixed signs and small effects with the paired descriptive uncertainty; three pairs do not justify a precision or noise claim beyond those descriptive summaries. The historical `+0.002193631504726755` seed-42 difference is not evidence establishing this prospective hypothesis.

Separately, implementation/contract falsifiers are: wrong recomputed vector/intermediate, nonfinite or unsupported curvature, any curvature trainability or time/update/mode variation, or failure to demonstrate a beyond-tolerance direct counterfactual effect. These invalidate mapping implementation/activation, not by themselves the retrieval hypothesis or FCCR-1 family. A change to any non-mapping factor violates the one-factor interpretation. S03 failure to establish the declared raw-residual semantics/provenance blocks approval/propagation and any later execution; it is not a pass and must not be repaired by substituting normalized layer scales. Regardless of paired effect, classify every valid result against the inclusive `0.065` threshold; a below-threshold score alone does not establish mechanism inactivity or family failure.

**Assumptions.** The two map equations, shared decimal inputs, seed block, and prospective paired analysis in the source packet/S01 are the locked design; the iter29 calculation script and iter26 mechanism record correctly document the historically tested mappings. The counterfactual can be measured at an existing curvature-consuming signal without changing the mechanism.

**Evidence.** Canonical S00/S01 records authorize only these two mappings and new matched seeds 43/44/45; protocol lock names iter29 as source parent but iter26 as fresh paired control. Iter29 calculation script and JSON document candidate constants/intermediates; iter26 mechanism manifest and mapping script document its standardization/exponential mapping. Historical raw-residual provenance remains medium-confidence and unresolved at S03; normalized scales are distinct and forbidden as a substitute.

**Risks.** The core input's historical residual semantics/byte provenance are unresolved; iter26's `residual_norm` alias is ambiguous. The three-seed design gives limited precision and cannot ensure target attainment or establish a tiny effect as meaningful. A direct geometric/assignment change may not translate to Stage3 retrieval. Any protocol mismatch, failed warm-start verification, or missing pair would invalidate the intended paired inference.

**Self-rejection conditions.** I would reject this proposal as non-executable/unsupported if primary adjudication shows either equation/vector is not the approved previously tested mapping, the common inputs cannot be semantically supported as specified, the registered counterfactual signal is unavailable without changing another factor, or the locked seed/protocol comparison cannot be executed as six distinct complete runs. In particular, S03 provenance failure blocks onward mapping approval/MVG/Stage2; this artifact does not pre-empt that decision.

## Execution boundary

This is an independent S02 candidate only. It does not claim S03 passed, approve provenance, authorize code changes or Stage2/Stage3 launches, or prove provenance, activation, causal effect, or target success. All remaining S03 and downstream deliberation, verification, per-run gradient/input/hash/path/warm-start gates, and execution authorizations remain intact; S03 failure blocks subsequent mapping approval/MVG/Stage2.
