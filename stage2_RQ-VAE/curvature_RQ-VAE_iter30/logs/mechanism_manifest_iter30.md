# Iter30 Mechanism Manifest — S03_PROVENANCE Canonical

```text
STAGE_ID=S03_PROVENANCE
ROUND=1
STATUS=PASS_LIMITED_TO_HISTORICAL_METHOD_VALUE_PROVENANCE
CONTRACT=FCCR-1
LAYER_ORDER=[L0,L1,L2] == [0,1,2] == [T0,T1,T2]
EXECUTION_AUTHORIZATION=NO
```

## Scope and decision boundary

This audit approves only the two registered inputs' historical semantic, method, and recorded-value provenance. It does not assert present-day recalculation, replay of either historical pipeline, checkpoint-level reproduction, historical input-byte identity, or successful downstream implementation. No mapping is recomputed, changed, or newly selected. No historical writer was rerun, no calibration was performed, and no GPU or training work was launched.

## Canonical formula-input provenance

| Symbol / canonical key | Exact semantic meaning | Source / producer and producing record | Exact raw recorded value `[L0,L1,L2]` | Transformations and actual S02 input | Expected range / sanity bound | Confidence and identity/replay limitations |
|---|---|---|---|---|---|---|
| `B_l` / `behavior_branching` (JSON key `branching`) | Effective branching factor `B_l=exp(H_l)`, not a literal distinct-child count. `H_0=H(T0\|source)`, `H_1=H(T1\|source,T0)`, `H_2=H(T2\|source,T0,T1)`. `source` is the final item of a nonempty training history; `T0,T1,T2` are the ordered raw SID tokens of the target item. Within each context, entropy uses natural logarithms; context entropies are averaged weighted by observation counts. | Direct producer: `stage2_RQ-VAE/curvature_RQ-VAE_iter12/scripts/compute_behavior_branching.py`; it reads the hardcoded iter8 `sids_raw.npy` and Stage0 `train.parquet` history/target columns. The preserved value record is `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`; `source.script` names the iter12 producer and its metadata records the iter8 SID path, train path, layer definitions, and counts. The JSON records `entropy=[2.9613950179410296,0.37882601480060585,0.014701837881298745]` and the corresponding branching vector. | `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` | Per-context natural-log entropy → observation-count-weighted conditional entropy → exponentiation `B_l=exp(H_l)`. No subsequent transformation of `B_l` occurs before either S02 mapping; this exact vector is the actual S02 `behavior_branching` input. Iter12's `RESIDUAL_NORMS=[0.001,0.932889,1.0]` is used only for separate `raw_capacity` and `branch_mid` side fields; it does not enter entropy or `branching`. | Finite and `B_l >= 1` (entropy nonnegative); these recorded values satisfy that bound. Layer/token order is L0/T0, L1/T1, L2/T2. | **HIGH** confidence in recorded method, semantics, order, and output values from producer logic and preserved entropy/branching record. **Not** historic-byte/replay confidence: the referenced iter8 raw SID file is currently absent and its historical hash was not recorded. Current S01 hashes for current Stage0/Stage1/warm-start files attest only their bytes at that check and do not establish what the historic run consumed. The JSON's source metadata attributes branching to iter12; it does not attribute residual calibration to iter12. |
| `m_l^{raw}` / explicit `raw_residual_medians` | For each RQ layer, median over Stage1 items of the Euclidean L2 norm of that item's residual vector, measured by `result.residuals.norm(dim=1)` where residuals are documented `(n_layers, embed_dim, batch)`. Thus norm is over embedding dimension per layer/item, then `median(dim=1)` is taken over items after concatenating item chunks. This is a raw residual magnitude, not a normalized layer scale and not a trainable curvature parameter. | Measurement method: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py::_measure_layer_norms`; it loads a named baseline checkpoint and Stage1 embeddings, runs `get_semantic_ids` in chunks, and records the checkpoint step. Contemporaneous paired execution record: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log`, reporting `step=100000`, both vectors, and checkpoint/embedding paths. The iter26 preserved JSON explicitly records `raw_residual_medians`; its `source.script` metadata names the separate iter12 branching producer, so the raw field is a separately sourced calibration record, not an iter12-produced measurement. The log names baseline checkpoint `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth` and Stage1 embedding `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`. | `[1.0, 0.10941, 0.09331]` | The raw per-layer medians are recorded unchanged as `raw_residual_medians`; this exact vector is the actual S02 `m_l_raw` input to both locked mappings. A separate historical log normalization computes `s_max=max(m)`, `s_min=min(m)`, then `max(log(s_max/m_l)/log(s_max/s_min),1e-3)`, yielding `[0.001,0.932889,1.0]`. That is a distinct historical normalized-layer-scale output, not an S02 input, alias, or fallback. There is no transformation from the raw vector to the S02 input beyond selecting the explicit raw field. | Each median must be finite and strictly positive. Values are ordered by the three RQ layers and satisfy positivity; normalized scales are bounded `[1e-3,1]` for this unequal-positive-median case. | **MEDIUM** confidence in historical method/value provenance from primary measurement code, contemporaneous paired raw/normalized output log, and preserved explicit JSON field. The named calibration checkpoint is currently absent; no historical checkpoint or embedding digest is recorded. The referenced historical checkpoint cannot be replayed/byte-verified. The original iter8 `sids_raw.npy` is separately absent and unhashed, so branching cannot be replayed either. Current iter8 checkpoint and S01 current-file hashes do not prove historic input bytes or historic runtime use. |

The audited quantities are distinct and must remain so:

```text
raw_residual_median != normalized_layer_scale != learnable_c_layer_scale
```

`learnable_c_layer_scale` denotes a model parameter concept; it is neither the raw measurement nor its normalized historical transform. FCCR-1 requires the first quantity as the residual input and fixed, non-trainable final curvature.

## Recorded values and S02 boundary

The exact registered common input vectors, in `[L0,L1,L2]` order, are:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
normalized_layer_scales = [0.001, 0.932889, 1.0]  # distinct; prohibited as substitute
```

These are the actual values registered by S02 for its two already-approved candidate mappings. This S03 decision preserves them and their mapping definitions; it neither recalculates the mapped curvatures nor grants approval to any later stage.

## Producer attribution and legacy-key hazard

The iter26 JSON is an aggregate historical record: its `source.script` points to the iter12 behavior-branching producer, while the explicit raw-residual field is separately supported by iter10 calibration code/log and is not thereby attributable to iter12. In iter12, the key `residual_norm` refers to normalized `[0.001,0.932889,1.0]` side-field inputs. The preserved iter26 JSON happens to contain a `residual_norm` field numerically equal to its explicit `raw_residual_medians` value, but equality in that artifact does not establish a stable meaning for the alias. The iter26 `compute_closed_form_curvature.py` consumer reads `payload["residual_norm"]`, so its historical consumer path has this alias ambiguity.

Every future iter30 consumer MUST read the explicit `raw_residual_medians` key and MUST fail closed if it is absent. It MUST NOT fall back to `residual_norm`, `RESIDUAL_NORMS`, normalized scales, a different checkpoint, or an inferred/recomputed value. Do not rerun the historical iter12/iter26 writer to reconstruct or refresh this input: the iter12 writer's payload serializes its normalized `RESIDUAL_NORMS` under `residual_norm` and does not provide the explicit raw residual field; the required historical raw SID source is absent. Preserve the explicit saved raw field and its separate provenance.

## §7 gate decision and limitations

**PASS, limited to historical method/value provenance.** Direct primary evidence supports both registered inputs' semantic definitions, producer/method lineage, layer order, exact recorded values, transformations, and sanity bounds as required by skill §7. The residual input is medium-confidence rather than replay-verified; the missing historical calibration checkpoint and raw SID table and absence of historic digests do not permit a claim of historical byte identity or replay. This PASS does not erase that limitation and does not claim provenance stronger than the recorded method/value chain. The current S01 hashes establish current check-time file identity only, not historical consumption.

No failure condition in §7 is demonstrated: the raw median is not replaced by normalized scale; source/method lineage is bounded and identified; the recorded raw value and actual registered input agree; and the unsafe legacy alias is explicitly rejected for future consumption. If future implementation cannot consume/validate the explicit raw field, that is a later blocking failure; it is not permission to substitute another quantity here.

## Evidence reviewed

- Iter30 S03 canonical source packet, S00 snapshot, S01 protocol manifest/Judge, and S02 hypothesis/Judge, treated as scope and registration records rather than sole primary proof.
- Primary iter10 calibration implementation and contemporaneous paired log.
- Primary iter12 branching producer and preserved iter26 JSON output/metadata.
- Primary iter26 closed-form consumer, inspected for the legacy `residual_norm` lookup; iter26 hypothesis/manifest only as historical corroboration of its registered distinction.
- Current path checks: named calibration checkpoint `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth` and referenced `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/sids_raw.npy` are absent. Historical hashes are not recorded. Current S01 hashes are check-time identities only.
- Iter29 S03 was consulted only as subordinate historical context; this decision rests on the primary evidence listed above and does not rely on its verdict.

## Propagation limit

`AUTONOMOUS_NEXT_ACTION=Advance only to S04_CONTRACT using this canonical manifest and the unchanged S02 inputs/mappings. Continue through separately adjudicated gates; no implementation, MVG, mapping approval beyond the S02 registrations, Stage2/Stage3 execution, recalibration, replay, or GPU work is authorized by S03.`
