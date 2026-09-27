# Iter30 FCCR-1 Hypothesis — S02 Canonical

```text
STAGE_ID=S02_HYPOTHESIS
ROUND=1
STATUS=CANONICAL_S02_HYPOTHESIS
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
PARENT_ITER=iter29 (implementation/source-lineage parent; not the prospective paired control)
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
CANONICAL_BASELINE_ITER=iter26
EXPERIMENT_TYPE=single_factor; prospective matched-seed comparison of two existing FCCR-1 mappings
RAW_RESIDUAL_PROVENANCE=PROVISIONAL_PENDING_S03
EXECUTION_AUTHORIZATION=NO
```

## A. Research question

Across the three predeclared matched seeds 43, 44, and 45, does the existing iter29 fixed-curvature mapping produce a positive mean paired Stage3 `test_recall@10` difference versus a fresh iter26-mapping control, with every protocol-valid final result independently classified against the user's inclusive `test_recall@10 >= 0.065` target?

## B. Exact two-arm equations, constants, and one-factor boundary

Every vector is in `[L0,L1,L2]` order. Both arms use the same registered behavior-branching vector `B_l` and the explicit `raw_residual_medians` field as `m_l_raw`, subject to the S03 gate in §C. These are the only two allowed mappings; no constants are tuned and no third mapping is introduced.

**Fresh paired control — iter26 mapping.** Let `m_min=min_l(m_l_raw)`. The historical implementation uses NumPy population standard deviation (`std`, `ddof=0`):

```text
s_l = log1p(B_l) / log1p(m_l_raw / m_min)
z_l = (s_l - mean(s)) / (std_population(s) + 1e-12)
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
```

Constants: `c_base=0.5`, `alpha=0.2`, standardization epsilon `1e-12`, lower clip `0.05`, upper clip `1.5`.

**Candidate — iter29 mapping.**

```text
x_l = B_l / (B_l + B_ref)
y_l = m_l_raw / (m_l_raw + m_ref)
u_l = (x_l + y_l) / 2
c_l = c_min + (c_max - c_min) * u_l
```

Constants: `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; equivalently `c_l=0.05+1.45*u_l`. There is no cross-layer normalization or extra clipping in this candidate.

**One-factor rule.** Iter29 is the implementation/source-lineage parent, but each prospective seed pair compares a fresh iter26-map control with the iter29-map candidate. Within each pair, the only conceptual contrast is which of these two fixed mappings supplies the three curvature values. Keep the input identities/order, Stage1 and Stage0 data, immutable iter8 warm-start, model/code other than the registered mapping, all Stage2 losses/optimizer/Sinkhorn/data/runtime settings, Stage3 trainer/config/runtime, evaluation split, and metric computation fixed. In particular, do not add/change a behavior or Sinkhorn rule, optimizer, loss, manifold, Stage1, or Stage3 behavior. The existing curvature-mediated effective Sinkhorn epsilon is part of the unchanged quantizer and is not a new mechanism. Stage2 SID/Gini/collision/entropy metrics remain descriptive, not admission gates.

The S01 lock requires six complete, distinct Stage2→Stage3 pipelines, run serially without overwriting destinations:

| Pair seed (Stage2=Stage3) | Arm | Stage2 root | Stage3 root |
|---:|---|---|---|
| 43 | iter26 control (`iter26_mapping_seed43`) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| 43 | iter29 candidate (`iter29_mapping_seed43`) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| 44 | iter26 control (`iter26_mapping_seed44`) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| 44 | iter29 candidate (`iter29_mapping_seed44`) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| 45 | iter26 control (`iter26_mapping_seed45`) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| 45 | iter29 candidate (`iter29_mapping_seed45`) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

Locked Stage2 settings include 100,000 global steps; three quantizer layers with 256 codes each; input/hidden/embedding dimensions `768` / `[512,256,128]` / `32`; commitment weight `1.0`; batch 640 per GPU, four ranks; configured Sinkhorn `sk_eps=0.05` and 3 iterations; AdamW `lr=1e-3`, weight decay `1e-4`; existing behavior-loss weight `0.20`, temperature `0.07`, curvature regularization weight `0`, and all other existing settings. Every actual Stage2 invocation separately requires the root-mandated one-checkpoint/one-batch gradient-path check. Fixed curvature itself must not receive gradients.

Stage3 remains unchanged: matching seed within each pair; current trainer SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`; configured maximum 150 epochs with existing `NO_EVAL=True` / train-loss patience behavior retained; final test required (`SKIP_TEST=False`); beam size 20, top K `[5,10]`, expected `n_eval=57439`, batch 4096, inference batch 1024, four ranks and serial port 50201. Complete all six registered pipelines under their unchanged normal protocol; do not stop, omit, replace, or overwrite a run because of target scores, paired trends, or Stage2 descriptive metrics. Rehash locked inputs and warm-start and verify actual resolved consumers/use before applicable invocations as required by S01; the S01 hashes are check-time records, not future-use proof.

## C. Actual numeric substitution and provenance status

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual = [1.0, 0.10941, 0.09331]  # PROVISIONAL_PENDING_S03
```

The residual vector is the recorded `raw_residual_medians` value, not the distinct normalized layer scales `[0.001, 0.932889, 1.0]`; those normalized values MUST NOT be substituted. Iter30's future mapping input must be read only from the explicit `raw_residual_medians` field. Iter26's historical calculator read the ambiguous legacy key `residual_norm`; the equal-valued alias in its JSON does not prove that key's raw-residual semantics. The numeric substitution below verifies decimal arithmetic only: it is not a current remeasurement, checkpoint-level replay, historic byte-identity proof, or S03 approval. S03 provenance adjudication remains unresolved and is a hard gate before mapping approval/MVG/Stage2.

**Control — iter26 mapping intermediate terms:**

```text
m_min = 0.09331
s = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]
z = [1.03129493309937, 0.3223997103378134, -1.3536946434371857]
fixed_curvature = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
```

**Candidate — iter29 mapping intermediate terms:**

```text
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Candidate-minus-control curvature from these registered decimals is `[0.7515595785009063, 0.2014808741174853, 0.2625879974275077]`. Each output is finite and inside the registered `[0.05,1.50]` interval. These decimals are the required numeric substitutions, not evidence that the unresolved raw-residual source semantics have passed S03.

## D. Direct effects, fixed invariants, and counterfactual

For each arm, compute its three values once before Stage2 and store/use them as fixed, non-trainable curvature. Directly verify the arm's values against its registered vector within the implementation's declared formula tolerance; verify finiteness and support; confirm the curvature buffers are not parameters, receive no gradients, and are not in optimizer parameter groups; and verify values remain invariant across steps (including the registered 0 / 25k / 50k / 100k checks), optimizer updates, and train/eval mode. Curvature cannot depend on batch activations, mutable optimizer state, or time. No cyclic/scheduled curvature, curvature regularization, new curvature-conditioned optimizer, or new curvature-conditioned auxiliary loss is allowed.

The primary activation counterfactual is a same-checkpoint, same-batch, no-training comparison of the existing **L0** quantizer path. Use identical encoder output, codebook weights, batch, mode, device/dtype, and all non-curvature state in both arm evaluations; substitute only the registered control versus candidate L0 curvature. In `modules/quantize.py::Quantize.forward`, capture `D26` and `D29`, the actual per-example-by-code Poincaré distance tensors after the existing distance-flattening fallback (if taken) and immediately before `_center_distance_for_constraint` and Sinkhorn. Define the one primary direct signal:

```text
M_distance = max_ij(abs(D29[i,j] - D26[i,j]))
```

This is measured in the returned Poincaré-distance units and uses the tensor that enters the unchanged assignment path. Repeat each unchanged-arm computation on the same checkpoint/batch/device/dtype and record the maximum within-arm repeat difference for this same statistic as the deterministic numerical-repeat tolerance. Count direct activation only when `M_distance` exceeds that repeat tolerance; report the tensor dtype, device, both arm values, and repeat tolerance. Do not impose an unsupported absolute `1e-6` activation threshold. The quantizer's existing `distances.max()-distances.min() <= 1e-6` check selects its float64 Euclidean fallback; that is a different statistic and branch condition, not an activation threshold for `M_distance`. The actual changed-assignment fraction may be logged as a secondary diagnostic, not substituted for the registered primary signal. No counterfactual has yet been run and no activation result is claimed.

## E. Downstream rationale and paired estimand

**Downstream rationale — inference, not established fact:** the registered branching/residual structure is transformed by two different fixed maps into the above distinct curvature vectors. The unchanged quantizer consumes fixed curvature in Poincaré distance computation and curvature-scaled effective epsilon, which may alter assignment probabilities/IDs and residual allocation. If resulting representations retain useful item distinctions for the unchanged Stage3 recommender, retrieval could change or improve. Each link from geometry/assignment change to Stage3 benefit is an inference to be tested, not a measured result or expected target success. Stage2 SID statistics cannot establish downstream benefit.

For each new matched seed `s ∈ {43,44,45}`, let `R26_s` and `R29_s` be the protocol-valid final `test_recall@10` values from its respective fresh control and candidate runs. Define:

```text
d_s = R29_s - R26_s
mean_delta = (d_43 + d_44 + d_45) / 3
sample_sd = sqrt(sum_s((d_s - mean_delta)^2) / 2)   # ddof=1, three pairs
SE = sample_sd / sqrt(3)
paired_range = [min(d_43,d_44,d_45), max(d_43,d_44,d_45)]
```

Report all six final result paths and run metrics (including actual `n_eval`), all three signed pair differences, their mean/sample SD/SE/range, per-arm three-run mean/spread, and each individual run's target classification. A df=2 paired-t interval may be shown only as highly imprecise descriptive uncertainty. Finish all three pairs and all six complete runs before reporting the primary estimate. Do not pool historical seed 42, use it as a fourth pair, or replace any prospective pair.

The exact historical Stage3 records read for context are:

| Historical run | `n_eval` | R@5 | R@10 | NDCG@5 | NDCG@10 | Inclusive target status |
|---|---:|---:|---:|---:|---:|---|
| iter26 seed 42, canonical historical baseline | 57439 | 0.03788366789115409 | 0.057017009349048554 | 0.025169911931406087 | 0.03133359761524377 | below 0.065 |
| iter29 seed 42, historical candidate context | 57439 | 0.03953759640662268 | 0.05921064085377531 | 0.026252776900288842 | 0.03257647953179143 | below 0.065 |

Their historical point difference is `+0.002193631504726755`; one observation per mapping is not a prospective effect or a run-to-run variance estimate. The per-result user criterion is inclusive: `test_recall@10 >= 0.065` meets the target, including equality; a lower value does not. Target attainment is separate from the paired mapping effect and does not authorize stopping or dropping any run. This inclusive user criterion controls over lower-priority strict `>0.065` wording in repository/skill text.

## F. Falsification, risks, assumptions, and execution gates

**Scientific hypothesis and falsifier.** The single directional hypothesis is `mean_delta > 0` for the iter29 mapping versus fresh iter26-map controls under the locked protocol. Once all three protocol-valid pairs finish, `mean_delta <= 0` falsifies this directional retrieval hypothesis for this two-map comparison. A positive point estimate alone, particularly with three pairs, does not establish a precise or robust effect; report mixed signs and descriptive uncertainty without using seed 42 as additional evidence.

**Implementation/contract/protocol falsifiers.** Failure to reproduce either equation/intermediate/fixed vector within the declared formula tolerance; nonfinite/unsupported curvature; any curvature gradient, trainability, optimizer membership, or time/update/train-eval variation; failure of the registered same-checkpoint L0 counterfactual to exceed its deterministic repeat tolerance; or a within-pair difference in any non-mapping factor invalidates the applicable mapping implementation, activation, or protocol claim. Such failures are not by themselves a negative Stage3 result or evidence against all FCCR-1 mappings. S03 failure to establish raw-residual semantics/provenance blocks propagation and all later approval/execution; it is not repaired by substituting normalized scales. Missing, invalid, or overwritten members of the six-run block make the registered paired estimate unavailable; do not report a partial block as the primary estimate.

**Risks and assumptions.** The raw-residual vector has unresolved historical semantic/source and byte-replay limitations at this stage; numeric agreement cannot resolve them. The explicit two maps, common values, seeds, protocol, and roots are those approved by S14/S00/S01; the historical scripts/manifests accurately record their equations, while the iter26 legacy-key ambiguity remains. Three prospective pairs have limited precision and may not distinguish a small effect from run variability. A measurable quantizer change may not improve Stage3 retrieval, and neither historical result meets the target. Protocol/input/warm-start mismatch or a missing pair invalidates paired inference.

**Execution boundary and next action.** This artifact registers the hypothesis only. It does not claim S03 passed, direct activation, Stage2/Stage3 results, target success, implementation approval, or permission to edit implementation or launch GPU work. The concrete next stage is `S03_PROVENANCE`: independently adjudicate the declared raw-residual semantic/source/value provenance. If S03 fails, stop before mapping approval/MVG/Stage2 and do not substitute normalized values. If it passes, proceed only through the remaining separately adjudicated contract, one-factor, implementation, preflight/MVG, per-run input/hash/path/warm-start/gradient checks, and later launch authorizations. No Stage2 or Stage3 execution is authorized by S02.

## Primary evidence used

- Iter30 S00 canonical: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`; S01 canonical and decision: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md` and `logs/deliberation/S01_PROTOCOL_LOCK/round_1/judge.md`.
- Iter26 mapping primary records: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/compute_closed_form_curvature.py::closed_form_curvatures`, `scripts/computed_behavior_branching.json`, and `logs/hypothesis_iter26.md`.
- Iter29 mapping primary records: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/compute_closed_form_curvature.py::compute_closed_form_curvature`, `scripts/computed_behavior_branching.json`, and `logs/mechanism_manifest_iter29.md` (historical provenance limits and raw/normalized distinction).
- Quantizer consumer: iter26/iter29 `modules/quantize.py::Quantize.forward` and iter29 `modules/hyperbolic.py::_poincare_distance_t` / `_sinkhorn_algorithm`; the L0 Poincaré distance tensor feeds the unchanged centering/Sinkhorn/ID path.
- Historical Stage3 results: the exact iter26 and iter29 `test_final.json` paths listed in §E; S01 locks the prospective seeds, settings, roots, and inclusive threshold.
