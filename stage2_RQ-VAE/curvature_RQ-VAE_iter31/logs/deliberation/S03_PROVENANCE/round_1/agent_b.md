ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S03_PROVENANCE/round_1/source_packet.md
STAGE_ID=S03_PROVENANCE
ROUND=1

## Overall verdict

**PROVENANCE_VERDICT=PASS_WITH_LIMITATIONS — historical method/value provenance for the inherited Iter29 curvature vector; semantic/source tracing for the registered HRA equation is adequate, but actual runtime activation, clipping incidence, and output values are not established here.**
**CONFIDENCE=MEDIUM overall.** The fixed vector arithmetic and code semantics are directly inspectable. Historical raw residual provenance is medium confidence; branching semantics and recorded values are high confidence, with historical replay limits. This is not current checkpoint-level reproduction, a new measurement, or proof that the proposed HRA operation has run or is active.

No code, contract, checker, training, test, formatter, linter, suite, or Stage2/Stage3 authorization is given. No historical writer, calibration, replay, or model execution was run. No repository file was modified.

## Exact values and arithmetic reproduction

The explicit Iter29 JSON key `raw_residual_medians`, and only that key, records layer order `[L0,L1,L2]`:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
B_ref=2.0; m_ref=0.1; c_min=0.05; c_max=1.5
x_l = B_l/(B_l+2.0)
y_l = m_l_raw/(m_l_raw+0.1)
u_l = (x_l+y_l)/2
c_l = 0.05 + 1.45*u_l
```

Direct scalar substitution gives:

```text
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
c = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

This is **arithmetic reproduction from the registered decimal inputs/constants**, consistent with the Iter29 JSON and closed-form function. It is **not historical measurement reproduction**: the historical calibration forward pass and historical branching input bytes cannot be replayed from currently available source data. The Iter31 HRA hypothesis inherits this unchanged vector; it defines `c0` as `c_l` at layer 0, i.e. `1.3660953164241916`. No remapping or recalibration is proposed.

The normalized historical layer-scale vector `[0.001, 0.932889, 1.0]` is distinct from the raw medians `[1.0, 0.10941, 0.09331]`. The old producer uses normalized scales in its separate capacity/branch-side calculations, not to compute `branching`. The calibration code separately computes normalized logarithmic scales from raw medians. Neither this normalized vector nor any field named `residual_norm` is an allowable substitute or fallback. Preserve:

```text
raw_residual_median != normalized_layer_scale != learnable_c_layer_scale
```

Use only `raw_residual_medians`; absence is a failure, not a reason to fall back.

## Symbol verdicts

| Symbol | Provenance verdict / confidence | Definition, source, transformation and layer order | Consumer and expected domain |
|---|---|---|---|
| `e_l` | **PASS, HIGH** for code semantics; no historical numeric value claimed | `Quantize.forward` receives the current layer residual `x` (the first is the encoder's latent; subsequent residuals follow Step4 then Step5). It selects IDs using curved distance/Sinkhorn, looks up `embedding(ids)`, and returns that selected codebook embedding as `embeddings`. In training the returned value is `x + (embedding - x).detach()` (straight-through value); in eval it is the selected embedding itself. Thus `e_l` is an embedding-dimension tangent-coordinate tensor, not a ball point or semantic ID. Values depend on model/batch and are not registered constants. | Batch×embed_dim tangent coordinates; HRA must map it with `exp0(c_l, e_l)` before any ball logarithm. No actual `e_l` values or runtime forward were observed. |
| `c_l` | **PASS, HIGH** for exact vector/fixed-buffer path | Ordered fixed values `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`. Iter29 loader reads and checks exact `branching` and `raw_residual_medians`, recomputes FCCR-1 and checks the JSON intermediates/contract. `RqVae` passes each ordered scalar into its layer's `Quantize`; `Quantize` stores `_fixed_c` as persistent float32 registered buffer and `get_c()` returns that buffer cast to embedding dtype. It is not an `nn.Parameter`; the fixed-curvature route has no trainable curvature scale. | Positive finite scalar curvature per layer, in source-enforced interval `[0.05,1.5]`. Consumers include quantizer distances/loss, residual Step4, transport Step5, behavior-distance code, and the S02 HRA source/transfer/Möbius geometry. Exact bitwise equality after a future model construction/runtime was not tested in S03. |
| `c0` | **PASS, HIGH** as reference alias | `c0 := c_0`, the first entry of the inherited fixed vector, `1.3660953164241916`; not a new value or independent curvature source. | Single reference curvature for every transfer target, every Möbius operation in `h`, and final `log0`. Positive finite; same declared curvature range. |
| `q_l` | **PASS, HIGH** for registered meaning; runtime norm unmeasured | Defined as `exp0^{c_l}(e_l)` using the layer-specific `c_l`. This is a Poincaré-ball point, distinct from both tangent `e_l` and quantizer ID. Helper `_expmap0_t` computes `tanh(sqrt(c)||e||)/(sqrt(c)||e||) * e`, then projects to the ball. | Expected `B_{c_l}` with norm no greater than the implementation's projected radius `(1-1e-6)/sqrt(c_l)` (default epsilon). Actual values, finite status, and projection incidence depend on real model inputs and were not observed. |
| `q_l^0` | **PASS, HIGH** for equation/argument order; runtime transfer unmeasured | Defined in S02 as `exp0^{c0}(log0^{c_l}(q_l))`: log at the source layer curvature, then exp at common reference `c0`. The helper log clamps its atanh argument to at most `1-1e-5`; exp projects at the target curvature. In the valid unclipped origin-map domain `log0^{c_l}(exp0^{c_l}(e_l))=e_l`, yielding the pointwise identity `q_l^0=exp0^{c0}(e_l)`. Do not assume this identity when projection/clamping changes the input. | Expected `B_{c0}` after the reference exp/projection. It is not evidence for a map preserving sums or Möbius addition. Actual clipping and domain validity remain later preflight/MVG matters. |
| `h` | **PASS, HIGH** for literal order/operator; runtime ball validity unmeasured | `q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)`, explicitly right-nested and ascending `L0,L1,L2` order. All additions use one `c0`; the implementation supports curvature tensor broadcasting. Möbius addition is not generally associative; reordering/reassociation is not semantics-preserving. The helper formula returns the rational Möbius expression but does not itself project the result back into the ball. | Intended `B_{c0}` composition of reference-ball inputs, with finite denominator/output required. Source argument order/broadcast path is inspectable, but actual numerical ball validity must be verified on runtime values later. |
| `z` | **PASS, HIGH** for registered consumer semantics; runtime value unmeasured | `log0^{c0}(h)` is the final tangent-coordinate output. S02 specifies it replaces only Iter29 Step6's Euclidean `e_0+e_1+e_2`; existing decoder and reconstruction-loss path are otherwise unchanged. Existing decoder checks rank 2 and embedding dimension, and reconstruction loss is evaluated using layer-0 curvature. | Batch×embed_dim tangent-coordinate tensor consumed by decoder. No actual value, finiteness, clipping, gradient, or activation measurement was made. Do not characterize it as exact residual reconstruction. |
| `B_l` | **PASS, HIGH** for recorded historical semantics/value; replay limitation applies | `behavior_branching` is the effective factor `exp(H_l)`, not a literal unique-child count. `[L0,L1,L2]` definitions are `H(T0|source)`, `H(T1|source,T0)`, `H(T2|source,T0,T1)`, where `source=history[-1]`; producer aggregates natural-log conditional entropies weighted by context observations and exponentiates. Exact recorded vector above; entropy record `[2.9613950179410296,0.37882601480060585,0.014701837881298745]`. Producer is Iter12 `scripts/compute_behavior_branching.py`, using Iter8 3-column raw SID tokens and Stage0 train parquet. Producer's reference to Iter8 checkpoint is metadata; producer does not load it. | Finite nonnegative effective branching, in source code's expected range `B>=0` (values are positive). Formula uses same-layer ordered entries. Recorded method/value are supported, but absent historical SID bytes prevent independent recomputation. |
| `m_l^{raw}` | **PASS WITH LIMITATION, MEDIUM** for historical method/value | Explicit `raw_residual_medians` vector `[1.0,0.10941,0.09331]`, `[L0,L1,L2]`. Iter10 calibration method loads a checkpoint and Stage1 embedding, obtains per-layer residual vectors, takes Euclidean norm over embedding dimensions for each item, then PyTorch median over items. Log records source label `iter1`, step `100000`, raw medians, and the distinct normalized vector. Historical checkpoint bytes cannot be replayed. | Finite strictly positive raw magnitudes; formula uses the explicit raw JSON key only. It is not normalized scale, trainable scale, or a newly measured Iter31 quantity. |

## Fixed buffer and tangent-versus-ball verification

Code path directly inspected: Iter29 `Quantize.__init__` creates `_fixed_c` with `register_buffer(..., persistent=True)` when `fixed_curvature` is supplied; the registered vector is passed by layer index from `RqVae` constructor. `get_c()` returns `_fixed_c.to(dtype=embedding.weight.dtype)` and the fixed route never enters a learned `c_layer_scale` branch. The exact loader requires and checks the explicit `raw_residual_medians` key, checks registered vectors and mapping identity/constants, recomputes intermediate values, and compares the contract's final vector. This establishes the **source-level fixed/non-parameter path**. It does not constitute a fresh optimizer-group audit, runtime invariance test, or proof for a future Iter31 patch.

In the actual Iter29 forward path, `e_l` is the tangent embedding returned by quantization; Step6 currently sums those tangent embeddings. HRA's registered proposal instead converts each `e_l` to a ball point before taking any logarithm, transfers with source `c_l` and target `c0`, combines only reference-geometry ball points, and applies final `log0` to `h`. Applying `log0` directly to `e_l` would be a semantic/domain error. S02 is a proposed operation, not evidence that this modified path exists or has executed.

## Replay, presence, and hash limits observed now

Current availability checks were direct read/glob observations during this audit:

- Present: Iter29 `computed_behavior_branching.json` and primary source files; current `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`; current Stage1 `sentence_t5.npy`; current Stage0 `train.parquet`.
- Absent at the recorded expected locations: Iter8 `sids_raw.npy`; historical baseline checkpoint `results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth` named by the calibration log.
- No SHA-256 was computed during this audit. S01 or historical manifests may record hashes, but they are not used here as proof of current bytes or historical consumption. The presence of a current Iter8 checkpoint does not recover missing raw SID bytes and does not show that checkpoint was loaded by the branching producer.
- Iter8 raw SID export has no recorded digest in the JSON; thus historical branching cannot now be independently replayed from the required SID bytes. The calibration log supplies no checkpoint, embedding, or SID digest, and the checkpoint is absent; therefore its historical forward pass cannot now be replayed or verified against checkpoint bytes. Current Stage1/Stage0/Iter8 checkpoint presence alone cannot establish historic byte identity.

Accordingly the data provenance claim is limited to documented historical method and saved values. Do not call the curvature vector newly measured, independently reproduced from historical source files, or hash-verified historically.

## Errors and ambiguity boundaries

1. **Key ambiguity:** `residual_norm` is unsafe/ambiguous. Iter12 assigns it the normalized `[0.001,0.932889,1.0]` vector; later JSON records can use the label differently. Use exactly `raw_residual_medians`, never `residual_norm` and never a fallback.
2. **Historical data absence:** missing Iter8 raw SID table and missing calibration checkpoint preclude the two historical replays described above; this is missing replay, not ambiguity in the saved canonical input values or their declared formula roles.
3. **Ball-domain evidence gap:** the exp/log helper's projection/clamp behavior is known from source, but actual HRA inputs and clipping incidence are not measured. The Möbius helper does not project outputs, so runtime ball/finite checks remain necessary in later gates.
4. **No telescope/homomorphism:** the pointwise unclipped radial coordinate identity does not imply transfer preserves Möbius addition, does not invert heterogeneous-curvature residual transport, and does not establish any exact telescope. No such claim is made.
5. **No activation/performance inference:** static provenance says nothing about the magnitude of the HRA intervention, gradients, model activation, or retrieval effect.

## S04 constraints / preflight handoff

- S04 must explicitly adjudicate the HRA Step6 proposal's relationship to the active FCCR-1 mapping-only contract, including whether an authorized between-iteration contract transition exists. S03 does not create one.
- S04 must resolve the known FCCR-1-specific contract/preflight compatibility for this HRA change. The existing FCCR-1 contract path cannot be silently bypassed or treated as proof of HRA compliance; no checker or contract change is authorized by this report.
- Any later contract must preserve the equation's exact order and tangent/ball types, fixed inherited curvature vector, correct source/target curvature arguments, finite/domain expectations, projection/log-clamp semantics, and explicit input-key discipline. S03 does not define that machine contract or authorize implementation.
- Actual projection/clamp incidence, Möbius output validity, direct intervention, gradient health, buffer immutability over steps, and optimizer exclusion are later preflight/MVG concerns. Root §6's training gradient-path checks remain mandatory before any Stage2 run if one is ever authorized.
- S01's Stage3 routing conflict remains unresolved at this stage; it does not change the current audit verdict or grant Stage3 authorization.

USER_INPUT_REQUIRED=NO
NO_CODE_AUTHORIZATION=TRUE
NO_TRAINING_AUTHORIZATION=TRUE