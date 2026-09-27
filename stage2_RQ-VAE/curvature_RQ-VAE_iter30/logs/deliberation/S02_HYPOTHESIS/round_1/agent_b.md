ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S02_HYPOTHESIS/round_1/source_packet.md
STAGE_ID=S02_HYPOTHESIS

# Iter30 S02 — Agent B independent candidate

## Assumptions

- I take the Judge-approved iter30 S00 source snapshot, S01 protocol manifest, and S01 judge decision as locked; the primary sources below support the equations and recorded inputs but do not supersede those locks.
- The only prospective contrast is between the existing iter26 and iter29 FCCR-1 mappings, applied to common layer-ordered inputs. I assume no result exists yet for prospective seeds 43–45.
- The vector called `m_l_raw` below is the explicit historical `raw_residual_medians` field. Its semantic provenance remains **provisional pending independent S03 adjudication**; neither its label nor an equation substitution certifies provenance.

## Evidence

- The S02 packet registers both exact maps, decimal inputs, outputs, seeds, paths, complete-block rule, and unresolved S03 state. S00 likewise records the same existing mappings and specifically prohibits replacing raw residuals with normalized scales.
- Iter30 S01 protocol manifest (`stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md`, §§ historical results, mappings, and paired analysis) identifies iter29 as source-lineage parent, but iter26 as the fresh same-seed control; it locks seeds 43/44/45, six complete pipelines, exclusion of historical seed 42 from the primary estimate, and `test_recall@10 >= 0.065` as the inclusive per-result user threshold. `EXECUTION_AUTHORIZATION=NO` and subsequent gates remain outstanding.
- Primary iter26 records support the control equation and values: `logs/hypothesis_iter26.md` and `scripts/computed_behavior_branching.json`. The JSON contains both `raw_residual_medians=[1.0,0.10941,0.09331]` and the legacy `residual_norm` alias with equal values; the alias does not establish raw semantics. The distinct normalized scale vector is `[0.001,0.932889,1.0]` and is not an allowed substitute.
- Primary iter29 sources support the candidate equation, constants, and arithmetic: `logs/hypothesis_iter29.md`, `scripts/compute_closed_form_curvature.py` (registered constants and `compute_closed_form_curvature`), and `logs/mechanism_manifest_iter29.md`. The latter explicitly characterizes historic raw-residual evidence as method/value provenance with reproducibility limitations, not checkpoint-level reproduction or historic byte identity.
- The two historical seed-42 `test_final.json` values recorded in S00/S01 are iter26 `0.057017009349048554` and iter29 `0.05921064085377531`, both `n_eval=57439`. Their observed difference `+0.002193631504726755` is historical context only: one pair cannot estimate prospective effect/noise and is not pooled.

## A. Research question

Across the three newly matched seeds 43, 44, and 45, does the existing iter29 FCCR-1 mapping yield a positive mean within-seed `test_recall@10` difference relative to a freshly run iter26-mapping control, with all other registered factors held fixed?

## B. Exact two-arm equations, constants, and one-factor contrast

Layer order throughout is `[L0,L1,L2]`; shared inputs are `B_l` and the explicit `m_l_raw` field. Both arms compute curvature once before Stage2 and use it as fixed, non-trainable, time-invariant curvature. Only the mapping from these same inputs to curvature differs.

**Control arm — iter26 mapping** (`m_min = min(m_raw)`):

```text
s_l = log1p(B_l) / log1p(m_l_raw / m_min)
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
```

Constants: `c_base=0.5`, `alpha=0.2`, `c_min=0.05`, `c_max=1.5`, standard deviation as in the registered computation, and epsilon `1e-12`.

**Candidate arm — iter29 mapping**:

```text
x_l = B_l / (B_l + B_ref)
y_l = m_l_raw / (m_l_raw + m_ref)
u_l = (x_l + y_l) / 2
c_l = c_min + (c_max - c_min) * u_l
```

Constants: `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; equivalently `c_l=0.05+1.45*u_l`. There is no cross-layer normalization in this candidate map.

**Single factor:** mapping identity only. Preserve the same registered inputs/input identities and use, common warm-start, Stage1/Stage0, seed within each pair, Stage2 architecture/training/loss/optimizer/Sinkhorn/data/runtime, Stage3 trainer/settings/evaluation, and all other mechanisms. No third mapping, retuning, mechanism stacking, or Stage1/Stage3 change. The existing curvature-consuming computation may respond to mapped curvature but is not an additional mechanism change.

## C. Actual numeric substitution (raw-residual status provisional)

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual = [1.0, 0.10941, 0.09331]  # PROVISIONAL_PENDING_S03
```

The vector is the explicitly recorded historical `raw_residual_medians`, not normalized layer scales. Numbers below reproduce the registered decimal arithmetic; they do not prove raw-residual provenance, reproduce the historical checkpoint computation, or establish historic input-byte identity.

**Control intermediate terms and fixed curvature:**

```text
m_min = 0.09331
s = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]
z = [1.03129493309937, 0.3223997103378134, -1.3536946434371857]
fixed_curvature = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
```

**Candidate intermediate terms and fixed curvature:**

```text
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Do not substitute `[0.001,0.932889,1.0]` for `raw_residual`; S03 has not passed in iter30. If S03 fails, this raw-residual mapping cannot propagate and later gates/execution remain blocked.

## D. Direct effects, invariants, and measurable counterfactual

Direct registered effects are deterministic fixed vectors above. In each arm, its own vector must be finite, equal its listed preregistered values within implementation-declared floating-point tolerance, non-trainable (not in optimizer groups and receiving no gradients), invariant across training steps and optimizer updates, and identical in train and eval mode. The vectors must be computed before training and cannot depend on batch activations, mutable state, or time. No schedule, curvature regularization, new curvature-conditioned optimizer, or new curvature-conditioned auxiliary loss is introduced.

Before any full run, the later applicable contract/MVG gates must compare both fixed vectors and test activation using the same checkpoint and batch with all non-curvature inputs/state held identical. Predeclare the direct signal as the mean absolute difference in the curvature-consuming geometry-distance tensor between candidate and control; require a finite difference `> 1e-6` to count as measurable activation. Record the tensor definition, units, and numerical tolerance in the applicable verification artifact. A zero/sub-tolerance difference means this mapping contrast has not demonstrated the registered direct effect; it is not a Stage3 outcome. Model loss/gradient health is separately required, while curvature itself must not receive gradients.

## E. Downstream rationale (inference, not established fact)

The recorded branching values are much larger at L0 than L1/L2; the provisionally recorded residual values are also larger at L0. The two approved maps transform these common inputs differently and yield different fixed-curvature vectors. **Inference:** because the unchanged quantization/geometry path consumes fixed curvature, that contrast could alter distances, assignments, or residual allocation; altered representations could in turn affect the unchanged Stage3 retrieval metric. This is a rationale for measurement, not evidence that the candidate improves retrieval or will reach the user threshold. Stage2 SID quality statistics remain descriptive, not gates or substitutes for Stage3.

The paired estimand is, for each `s ∈ {43,44,45}`, `d_s = R29_s - R26_s`, where each R is that arm's protocol-valid final `test_recall@10`; primary point estimate is `mean_delta=(d_43+d_44+d_45)/3`. Report all six results and all signed pair differences, the mean, sample SD across the three differences, `SE=sample_sd/sqrt(3)`, and min–max paired range; any df=2 interval is optional and explicitly very imprecise/descriptive. Report per-arm three-run means/spread and per-result target counts separately. These three pairs provide descriptive uncertainty only, not a precise variance estimate. Complete all six runs and all three pairs before primary-estimate reporting; no early stop, omission, substitution, or seed-42 pooling. Show the seed-42 historical pair only separately as context.

For every completed, protocol-valid result classify the user's inclusive target independently: `test_recall@10 >= 0.065` meets it, including equality; below it does not. This target classification is separate from whether the mapping-effect hypothesis is supported. A below-target result alone is not evidence of mechanism inactivity or failure of FCCR-1 generally.

## F. Falsification and classification

The single scientific hypothesis is that the three-seed mean paired effect `mean_delta` is positive. It is falsified for this mapping under the locked valid protocol if the completed primary estimate is `mean_delta <= 0`; a positive point estimate is not by itself proof of a robust effect given only three pairs and descriptive uncertainty. Do not use historical seed 42 to alter that decision or estimator.

Separately, implementation/protocol falsifiers include: a recomputed equation failing to reproduce either listed vector within declared numerical tolerance; non-finite/out-of-range curvature; incorrect mapping/constants/input vector/order; curvature trainable, changing over steps/updates/train-eval mode, or receiving gradients; failed measurable counterfactual activation; or any non-mapping factor differing within a pair. These indicate contract, implementation, activation, provenance, or protocol invalidity as applicable, not a retrieval-mechanism negative. S03 provenance failure blocks downstream approval/execution; it is not repaired by the numeric substitution.

No S03 pass, launch authorization, target success, prospective retrieval effect, or historical raw-residual checkpoint reproduction is claimed here. Preserve the six-run block, no-quality-gate/no-early-stop policy, and every later authorization gate.

## Risks

- The historical raw-residual semantic/source evidence is medium-confidence and lacks checkpoint-level replay and historic byte identity. The registered numbers alone cannot resolve that risk; S03 is a hard prerequisite.
- Stochasticity remains despite matched seeds and settings; three paired observations have limited precision. A small positive mean may be near run noise.
- A geometry counterfactual can establish direct activation only, not improved quantization or retrieval. Stage2 descriptive metrics do not establish the downstream rationale.
- The historical seed-42 difference is not a prospective treatment effect and cannot be pooled or used as an extra pair.

## Self-rejection conditions

I would reject this proposal as non-propagatable if primary evidence or S03 shows the explicit residual input is not the registered raw residual semantic/value, if either exact mapping or numeric substitution cannot be reproduced from the primary record, or if the proposed contrast cannot be implemented as mapping-only under the locked protocol. I would reject the scientific improvement hypothesis if all three valid prospective pairs complete and `mean_delta <= 0`. I would not relabel a failed provenance/contract/activation/protocol gate as a scientific negative, substitute normalized scales, omit any seed, infer success from seed 42, stop early, or treat this candidate as launch permission.
