# Iter31 Mechanism Manifest — S03 Provenance Canonical

```text
STAGE_ID=S03_PROVENANCE
ROUND=1
VERDICT=MERGE_AB
STATUS=PASS_HISTORICAL_METHOD_VALUE_PROVENANCE_WITH_REPRODUCIBILITY_LIMITATIONS
OVERALL_CONFIDENCE=MEDIUM
LAYER_ORDER=[L0,L1,L2]
USER_INPUT_REQUIRED=NO
NO_CODE_AUTHORIZATION=TRUE
NO_CONTRACT_OR_CHECKER_AUTHORIZATION=TRUE
NO_TRAINING_AUTHORIZATION=TRUE
```

## Scope and decision boundary

S03 verifies semantic definitions, inherited historical input provenance, exact registered arithmetic, and the proposed S02 HRA-inspired Step6 dataflow. The allowed decision is limited to **historical method/value provenance with explicit reproducibility limitations** for the Iter29 curvature inputs and to source-level semantics of the S02 proposal. It is not a machine contract, a contract transition, an implementation audit, a runtime test, or execution authorization. Active FCCR-1 remains unchanged unless and until an authorized later adjudication changes it. S04 must explicitly decide contract and preflight compatibility.

Primary evidence inspected: Iter29 `scripts/computed_behavior_branching.json`, `scripts/compute_closed_form_curvature.py`, `_load_closed_form_curvatures` in `curvature_RQ-VAE.py`, `modules/quantize.py`, `modules/rqvae.py`, `modules/hyperbolic.py`, Iter10 calibration implementation and contemporaneous log, and Iter12 branching producer. Also reviewed canonical Iter31 S00/S01/S02 artifacts, the frozen S03 packet, and both completed candidate reports. Direct checks found the current Iter8 checkpoint in its recorded output directory, did not find the Iter8 `sids_raw.npy` at the recorded location, and did not find the historical calibration checkpoint at its recorded location. No SHA-256 was computed here; no current Stage1/Stage0 path availability or historical byte identity is asserted by this Judge.

## Registered HRA-inspired operation and literal semantics

Canonical S02 registers, for each layer `l ∈ {0,1,2}`:

```text
e_l   = tangent-coordinate selected quantizer embedding at layer l
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
h     = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
z     = log0^{c0}(h) ∈ T_0(B_{c0})
c0    = c_0 = 1.3660953164241916
```

`z` is proposed to replace only Iter29's Euclidean Step6 sum of tangent embeddings and to feed the unchanged decoder. The reconstructed output and existing reconstruction loss remain on the existing layer-0 curvature path. The exact composition is right-nested, with ascending leaves `L0,L1,L2`; Möbius addition is not generally associative, so reassociation or layer reordering changes the registered expression.

The type distinction is essential: Iter29 `Quantize.forward` returns selected codebook embeddings in tangent coordinates (with the training straight-through value `x + (embedding - x).detach()`; evaluation returns the embedding). These are not semantic IDs or Poincaré points. Iter29's current Step4 constructs ball-valued quantities by applying origin exponentials to tangent residual/embedding and uses Möbius subtraction; Step5 transports residuals from `c_l` to `c_(l+1)`. Its current Step6 sums the returned tangent embeddings in Euclidean coordinates. This audited source is not evidence that the proposed HRA Step6 has been implemented or executed.

`_expmap0_t` implements the origin exponential with `tanh(sqrt(c)||u||)/(sqrt(c)||u||)` and projects outputs to at most `(1-1e-6)/sqrt(c)` by default. `_logmap0_t` clamps the `atanh` argument to at most `1-1e-5`. Accordingly, `log0^{c_l}(exp0^{c_l}(e_l))=e_l` and the resulting pointwise identity `q_l^0=exp0^{c0}(e_l)` hold only in the valid, unclipped domain; they are not assumed for actual values. The Möbius helper broadcasts curvature over batch dimensions but does not itself project its output to the ball. Runtime finiteness/domain validity and projection/clamp incidence are unmeasured and remain later preflight/MVG concerns.

The radial transfer is not claimed to preserve Möbius addition. No exact telescope, heterogeneous-curvature residual inversion, or residual reconstruction is claimed: the paper's exact HRA cancellation uses paired residual subtraction and reverse-nested aggregation in one shared curvature, whereas Iter29 has heterogeneous fixed curvatures and explicit cross-layer residual transport. No runtime activation, gradient, output, clipping frequency, or recommendation-performance evidence is inferred from this source audit.

## Provenance table

| Symbol / registered input | Semantic meaning and primary source | Producer / consumer and layer order | Recorded/raw value; transformation; final formula role | Expected domain / confidence / limitations |
|---|---|---|---|---|
| `e_l` | Selected quantizer embedding in tangent coordinates; `modules/quantize.py::Quantize.forward` returns `embeddings_out`. | `modules/rqvae.py::get_semantic_ids`: L0 input is encoder output; subsequent layer inputs follow Step4 residual update and Step5 transport. Proposed S02 Step6 reads embeddings in L0,L1,L2 order. | Runtime/model/batch-dependent; no fixed registered decimal. Training forward value is selected embedding through STE; evaluation value is selected embedding. This tangent tensor is input to `exp0^{c_l}`. | Batch×embedding-dimension tangent-coordinate value; not an ID or ball point. HIGH confidence in audited source semantics; no runtime `e_l` values were sampled and no Iter31 HRA execution is established. |
| `c_l` | Inherited FCCR-1 fixed curvature at each RQ layer. | Iter29 loader reads explicit vectors and recomputes formula; `RqVae.__init__` passes each ordered scalar into the corresponding `Quantize`; `Quantize` stores `_fixed_c` as persistent registered buffer and fixed branch of `get_c()` reads it. Step4/5 and proposed HRA use layer-specific values. | Canonical decimal vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` `[L0,L1,L2]`. Formula intermediates and substitution below. Runtime buffer uses float32 tensor representation, so its bit-level value is not the same claim as exact preservation of the JSON decimal string. | Positive finite per-layer values in source-enforced `[0.05,1.5]`. HIGH confidence for recorded vector, loader validation, and Iter29 fixed-buffer source path; this is not a fresh Iter31 optimizer/runtime immutability check. |
| `c0` | Shared reference curvature is an alias of layer-0 curvature, `c0 := c_0`; not a new estimate or independent formula input. | S02 uses c0 as target curvature in each `q_l^0` transfer, curvature for both nested Möbius additions, and final log; existing Iter29 reconstruction path consumes layer-0 curvature. | `1.3660953164241916`, first element of the same fixed vector. | Positive finite scalar; HIGH confidence as registered alias and source-level consumer semantics. No separate measurement or refresh. |
| `q_l` | Actual layer-curvature Poincaré-ball codeword formed from tangent `e_l`, not the tangent embedding or token ID. | S02 `q_l=exp0^{c_l}(e_l)`; helper is Iter29 `modules/hyperbolic.py::_expmap0_t`. | Runtime value derived from `e_l` and its `c_l`, not preregistered. Origin exponential followed by ball projection. | Intended `B_{c_l}`, projected to radius no greater than `(1-1e-6)/sqrt(c_l)` with default helper epsilon. HIGH confidence in definition/helper semantics; actual norm, finiteness, and whether projection acts are unmeasured. |
| `q_l^0` | Radial origin transfer of the source-curvature ball point to the common reference ball. | S02: `exp0^{c0}(log0^{c_l}(q_l))`; `log0` source curvature is `c_l`, `exp0` target curvature is `c0`. Preserve L0,L1,L2. | Runtime transformation is source `log0` with clamp, then target `exp0` with projection. Only when unclipped, `log0^{c_l}(exp0^{c_l}(e_l))=e_l`, so pointwise `q_l^0=exp0^{c0}(e_l)`. | Input expected in `B_{c_l}`, output intended in `B_{c0}`. HIGH confidence in registered argument semantics; numerical identity/domain and clipping incidence not verified. No homomorphism implication. |
| `h` | Common-reference ball aggregation with exact S02 right-nested tree. | `q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)`, first inner L1⊕L2 then L0⊕inner; both use c0. Iter29 `modules/hyperbolic.py::_mobius_add_t` supports broadcast curvature. | Derived runtime value; no registered numeric value. Möbius operation is not generally associative. Helper returns rational Möbius expression and does not itself project result. | Inputs intended in `B_{c0}` and output expected finite/in-domain; actual output norm/denominator and ball validity unmeasured. HIGH confidence in equation/order and helper semantics only. |
| `z` | Decoder-facing tangent-coordinate aggregate. | S02 `z=log0^{c0}(h)`; replaces only Iter29 Step6's Euclidean tangent sum. Iter29 `modules/rqvae.py::decode` accepts rank-2 batch×embedding input; reconstruction loss uses c0. | No registered numeric value; derived from runtime `h` by final c0 origin log. | Tangent-coordinate batch×embedding-dimension value, not itself ball-bounded. HIGH confidence in intended source-level interface; no HRA value, finite check, activation, or gradient observed. Not an exact residual reconstruction. |
| `B_l` / `behavior_branching` | Effective branching `exp(H_l)`, not literal unique-child count. `H_0=H(T0|source)`, `H_1=H(T1|source,T0)`, `H_2=H(T2|source,T0,T1)` with `source=history[-1]`; natural-log entropies are context-observation weighted. | Iter12 `scripts/compute_behavior_branching.py` reads Iter8 three-column `sids_raw.npy` plus Stage0 train parquet `history`/`target`, counts ordered target tokens, computes conditional entropies and `exp(entropy)`. The Iter8 checkpoint path is metadata only; producer does not load it. Iter29 saved JSON records the output. | `B=[19.324911558712664,1.4605688962651735,1.0148104414712726]` `[L0,L1,L2]`; recorded `H=[2.9613950179410296,0.37882601480060585,0.014701837881298745]`; formula consumes saved B directly as `x_l=B_l/(B_l+2.0)`. | Finite nonnegative effective branching; HIGH confidence in saved semantics and historical values, not historical byte-level replay. Current Iter8 raw SID path was directly observed absent; no historic SID digest is recorded. The current Iter8 checkpoint's presence does not supply/recreate the missing SID table or prove its historical use. |
| `m_l^{raw}` / exact JSON key `raw_residual_medians` | Historical per-layer median of Euclidean residual-vector L2 norms over Stage1 items, norm over embedding dimension then median over items. | Iter10 `scripts/calibrate_residual_scales.py::_measure_layer_norms` loads checkpoint and Stage1 embedding, eval-forwards in chunks, takes `result.residuals.norm(dim=1)` then `median(dim=1)`; Iter10 log records source label Iter1, step 100000, values, and separate normalized values. Iter29 loader explicitly requires this key and rejects a missing key; no fallback. | `m_raw=[1.0,0.10941,0.09331]` `[L0,L1,L2]`; formula consumes these raw values as `y_l=m_l_raw/(m_l_raw+0.1)`. | Finite strictly positive raw magnitudes. MEDIUM confidence in historical method/value provenance; historical calibration checkpoint is absent at the recorded path, its log has no checkpoint/embedding/SID hash, and historical forward cannot be replayed or verified against checkpoint bytes. This is not newly measured Iter31 data. |

## Exact inherited formula, arithmetic and input-key discipline

Iter29 `scripts/compute_closed_form_curvature.py` defines `B_REF=2.0`, `M_REF=0.1`, `C_MIN=0.05`, `C_MAX=1.50`, requires ordered length-three branching and raw-residual vectors, requires nonnegative branching and strictly positive raw medians, then computes:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + (1.50 - 0.05) * u_l
```

Independent arithmetic from the exact JSON decimals and registered constants gives:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
m_l_raw = [1.0, 0.10941, 0.09331]
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.472264114617378, 0.40965228087632294]
c = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

These values match the recorded Iter29 JSON `x_l`, `y_l`, `u_l`, and `closed_form_c_l`; this is arithmetic reproduction from saved decimals, **not historical source-data replay, checkpoint reproduction, or a new measurement**. Iter29 `_load_closed_form_curvatures` explicitly requires both `branching` and `raw_residual_medians`, validates their registered vectors and mapping constants, recomputes intermediate vectors, and compares the JSON and contract curvature outputs. The input actually accepted by the loader is the exact key `raw_residual_medians`.

Keep the three distinct quantities explicitly separated:

```text
raw_residual_median != normalized_layer_scale != learnable_c_layer_scale
```

The normalized historical vector is `[0.001,0.932889,1.0]`; Iter10 computes it separately by logarithmic normalization with a `1e-3` floor, and Iter12 uses its similarly labeled `RESIDUAL_NORMS` only in separate side calculations, not to produce conditional entropies/branching. The ambiguous key `residual_norm` is not a valid alias, fallback, or substitute. Any future input lacking explicit `raw_residual_medians` fails; there is no normalized fallback.

## Replay, availability, and confidence limits

- The Iter29 JSON records Iter12 branching producer, Iter8 SID path, Stage0 parquet path, row/pair/item/layer counts, definitions, and values. Iter12 primary code confirms the reported conditional token-count/entropy/exp(H) semantics. The raw SID file is currently absent at the referenced location and no historical digest is recorded; branching cannot now be independently recomputed from those SID bytes. Iter12's Iter8 checkpoint path is metadata, not a loaded producer input.
- Iter10 primary code and its contemporaneous log establish the documented measurement procedure and recorded paired medians/normalized scales. The historical baseline checkpoint named in the log/JSON is absent at the recorded path; the log records no checkpoint, embedding, or SID digest. Its historic forward pass cannot now be replayed or checked against checkpoint bytes.
- A current Iter8 checkpoint is present, but it does not recover missing raw SIDs and does not establish historical consumption. No SHA-256 was computed in this S03 audit. Hashes recorded in S01 or old manifests, if any, are not promoted here as fresh current-byte checks or proof of historical use.
- Confidence is HIGH for recorded branching semantics and values and for the audited source formula/geometry semantics; MEDIUM for historical raw-residual method/value provenance; MEDIUM overall. Runtime `e_l`, `q_l`, `q_l^0`, `h`, and `z`, projection/clamp incidence, valid ball composition, direct Step6 intervention, gradient behavior, and performance remain unobserved.

## S04 and later-stage constraints

1. S04 must explicitly adjudicate the proposed HRA Step6 aggregation against active FCCR-1's mapping-only scope and determine whether any authorized between-iteration contract transition exists. This manifest is not that transition and does not define a machine-readable contract.
2. S04/S07 must resolve compatibility with the FCCR-1-specific preflight path through adjudication; neither bypass nor checker modification is authorized by S03.
3. Preserve exact source/target curvature arguments, tangent-versus-ball types, fixed inherited vector, no-fallback explicit raw input key, right-nested L0/L1/L2 order, and the stated projection/clamp/domain limits in any later contract. Actual values, runtime validity, clipping, direct intervention, gradients, and fixed-buffer/optimizer invariance require their later authorized gates.
4. Stage3 route conflict noted in S01 remains unresolved; this S03 finding does not authorize Stage2 or Stage3. Root `CLAUDE.md` §6 gradient-path checks and all subsequent adjudications remain mandatory before any training.
5. The iteration remains performance-seeking. No sweep, replication, root-cause study, code/source/test/format change, training, or performance claim is part of S03. If S03 passes, S04 is the next action.

## FCCR preflight semantic vocabulary

- **Behavior branching** refers to the existing saved `branching` input vector and its recorded conditional-entropy provenance.
- **Raw residual** refers to the historical per-layer raw residual median under the exact JSON key `raw_residual_medians`; it remains distinct from normalized quantities.
- **Normalized layer scale** refers to the separately recorded log-normalized diagnostic vector. It is not the raw residual input, an alias, or a permitted fallback for `raw_residual_medians`.

This vocabulary makes the existing S03 distinctions explicit for the unchanged FCCR preflight. It changes no recorded value, provenance limitation, formula, contract field, or mechanism.
