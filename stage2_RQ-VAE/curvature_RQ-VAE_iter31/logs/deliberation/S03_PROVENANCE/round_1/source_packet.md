# Iter31 S03 Semantic / Provenance — frozen source packet

```text
STAGE_ID=S03_PROVENANCE
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Canonical inputs and scope

Use only Judge-approved S00–S02 outputs as active instructions:

- `logs/source_snapshot_iter31.md` (S00 `MERGE_AB`)
- `logs/protocol_manifest_iter31.md` (S01 `MERGE_AB`)
- `logs/hypothesis_iter31.md` (S02 `MERGE_AB`)
- corresponding S00/S01/S02 Judge reports.

S02 registers a candidate HRA-inspired common-reference Step6 operation and requires that provenance be traced before any contract or implementation decision. S03 audits semantic definitions, data provenance, layer order, transformations, numerical domains, and evidence limits. It does not choose a machine contract, modify code, recompute/recalibrate missing historic measurements, decide one-factor scope, change a checker, or authorize training. S04 must decide the contract transition and preflight path; S05/S06/S07/S08 and root §6 checks remain later.

S01 locks Iter29 as parent condition and sole direct comparator. The user-requested HRA mechanism is not part of the currently active FCCR-1 mapping-only contract until an explicit later adjudication. Keep the one-seed-42 protocol, current strict target `test_R@10 > 0.065`, and output/source path rules.

## Canonical HRA formula to audit

For each layer `l ∈ {0,1,2}`:

```text
e_l   = tangent-coordinate selected quantizer embedding at layer l
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
h     = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
z     = log0^{c0}(h) ∈ T_0(B_{c0})
```

`z` replaces only the Iter29 Euclidean Step6 tangent sum and is consumed by the existing decoder. Audit every symbol and verify that the actual source data flow agrees. Never apply `log0` to tangent `e_l` directly. The unclipped identity `log0^{c_l}(exp0^{c_l}(e_l))=e_l` and therefore `q_l^0=exp0^{c0}(e_l)` is only valid in the non-saturated origin-map domain; Iter29 helpers project exponential outputs and clamp the logarithm's atanh argument. Do not assume the identity for actual values without measuring clipping.

## Required provenance table and evidence

Independently produce a canonical table for **every** symbol/data input, including semantic meaning, primary source, producer/consumer, layer order, raw recorded value or runtime origin, transformation, final formula input, expected range/domain, confidence, and missing replay/hash evidence. At minimum audit:

1. `e_l`: `Quantize.forward` output, tangent-coordinate codebook embedding selected from the trainable codebook; training forward value uses `x + (embedding - x).detach()` STE and evaluation value is selected embedding. Trace `x` to the per-layer encoder/residual input and distinguish embedding parameter from ball point. Numeric `e_l` values are model/batch dependent and not a preregistered constant; do not fabricate them.
2. `c_l` and reference `c0`: layer-ordered fixed curvature values `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. Verify the vector, order, computation/consumer path, fixed/nontrainable status, and how Step4 and Step6 use it. Do not select, tune, or refresh values in S03.
3. `q_l`: actual Poincaré-ball codeword formed by `exp0^{c_l}(e_l)`; distinguish it from the returned tangent `e_l` and from quantized IDs. Confirm norm/radius validity and projection behavior from the implemented helper.
4. `q_l^0`: radial origin transfer from source curvature `c_l` to reference `c0`; confirm source and target curvature arguments, the actual exp/log implementation and clipping/clamping semantics. This pointwise transfer has no general Möbius-sum homomorphism guarantee.
5. `h`: right-nested, ascending-layer `L0,L1,L2` Möbius composition in the single reference geometry `c0`. Audit non-associativity/order, broadcast semantics, and ball-domain expectation.
6. `z`: final `log0^{c0}(h)` tangent-coordinate decoder input. Confirm the existing decoder receives a batch×embedding tensor and reconstruction loss remains at c0. Do not call it an exact residual reconstruction.
7. Curvature-source formula inputs retained from Iter29: `B_l=behavior_branching` and `m_l^{raw}=raw_residual_medians`, with exact layer order, semantics and transformations below. The HRA equation consumes the resulting fixed `c_l`; it does not introduce new `B/m` values or compute a new curvature mapping.

## Historical fixed-curvature derivation and limits to preserve

Primary Iter29 sources:

- `scripts/computed_behavior_branching.json`
- `scripts/compute_closed_form_curvature.py`
- `curvature_RQ-VAE.py`, especially `_load_closed_form_curvatures` lines 305–382
- `modules/quantize.py`, especially fixed-curvature buffer/getter and `forward`
- `modules/rqvae.py`, especially constructor lines 108–158, residual Step4/Step5 lines 246–280, Step6 lines 319–326, `forward` lines 354–365
- `modules/hyperbolic.py`, especially `_expmap0_t`, `_logmap0_t`, `_transport_between_t`, and `_mobius_add_t` lines 9–38 and 66–73
- Iter29 `logs/mechanism_manifest_iter29.md`; Iter10 residual calibration source/log; Iter12 branching producer source as identified in the JSON.

The preserved Iter29 input JSON records:

```text
branching B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians m = [1.0, 0.10941, 0.09331]
B_ref=2.0; m_ref=0.1; c_min=0.05; c_max=1.5
x_l = B_l/(B_l+2.0)
y_l = m_l_raw/(m_l_raw+0.1)
u_l = (x_l+y_l)/2
c_l = 0.05 + 1.45*u_l
c = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Verify that the exact explicit JSON key `raw_residual_medians` is consumed; it is not interchangeable with normalized layer scales or a legacy key named `residual_norm`. Iter29 source requires exact registered input vectors and recomputes the mapping before training; the S03 audit must not rerun any historical writer or alter the file. Preserve exact decimals and `[L0,L1,L2]` order.

Provenance must be stated at its actual confidence, not inflated:

- The branching values are historical outputs from a producer that used the three-column Iter8 raw SID table and Stage0 train parquet, with conditional entropies `H(T0|source)`, `H(T1|source,T0)`, `H(T2|source,T0,T1)` and branching `exp(H)`. The producer metadata records 396,958 rows, 339,519 behavior pairs, 24,587 items, and 3 layers. Current Iter8 raw SID file is absent and has no historical digest, so branching cannot now be independently recomputed from those SID bytes.
- The raw residual medians are historical per-layer medians of Euclidean residual-vector L2 norms across Stage1 items, based on the recorded calibration procedure at source label Iter1/step 100,000. The historical calibration checkpoint is absent and its run log contains no checkpoint/embedding/SID hash; the original forward pass cannot now be replayed or checked against checkpoint bytes. Current Stage1/Stage0/Iter8 checkpoint hashes are locked in S01 but only prove current file identities, not historical consumption.
- Iter29's prior S03 passed **historical method/value provenance with reproducibility limitations**: branching high confidence in recorded semantics/values, raw residual medians medium confidence, no checkpoint-level replay, independent recomputation, or historical byte-identity proof. S03 must preserve this limit; do not call the vector newly measured or fully reproduced.
- The normalized layer-scale vector `[0.001,0.932889,1.0]` is not the raw median vector and is not used in the Iter29 curvature formula. Explicitly keep `raw_residual_median != normalized_layer_scale != learnable_c_layer_scale`.

## Candidate questions / failure rules

Agent A and B must independently determine whether every registered formula symbol has sufficient semantic and historical provenance to pass with explicit limitations, or whether a hard provenance failure remains. Distinguish (a) missing historic replay from (b) an unknown/ambiguous formula input. The HRA Step6 operation has no new external data input; its runtime `e_l`/`c_l`/geometry path must be traceable. Do not overstate the old `B/m` sources. A valid S03 may pass only historical method/value provenance with limitations and name every consequence for later contract/preflight gates.

If any symbol is undefined, wrong-domain (e.g. applying log0 to tangent `e_l`), wrong-layer-ordered, or relies on an unsupported raw/normalized substitution, state `FAIL` and recommend the required replan/abort outcome rather than a fallback. A numerically invalid/clipped HRA operation is an S07/S08 issue unless source semantics alone demonstrate it cannot be defined. Do not claim current model activation or performance from provenance records.

## Candidate report and canonical output requirements

Both candidates must return a detailed semantic/provenance report with:

- `ROLE=AGENT_A` or `ROLE=AGENT_B`, independence declaration, packet path, stage/round;
- provenance verdict and confidence for `e_l`, `c_l`, `q_l`, `q_l^0`, `h`, `z`, `B_l`, and `m_l^{raw}`;
- exact ordered values, source/producer paths, transformation and consumer wiring;
- separation of direct code facts, historical method/value evidence, missing replay, and inference;
- explicit non-use of normalized residual scales and no historic-replay claim;
- projection/log-clamp and nonhomomorphism limitations, no exact telescope claim;
- any hard gaps, downstream constraints, and `USER_INPUT_REQUIRED=NO`;
- no code, contract, checker, training, Stage2/Stage3 authorization, tests, formatting, lint, or suite.

Judge C must verify candidate claims against primary files and produce `logs/mechanism_manifest_iter31.md` plus the S03 Judge report. The manifest must include the provenance table and confidence/limitation boundaries. It is not an HRA contract; S04 decides the machine-readable contract and checker compatibility. No code or training authorization at S03.