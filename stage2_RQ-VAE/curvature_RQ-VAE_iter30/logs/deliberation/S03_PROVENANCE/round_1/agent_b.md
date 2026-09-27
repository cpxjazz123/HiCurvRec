ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S03_PROVENANCE/round_1/source_packet.md
STAGE_ID=S03_PROVENANCE

# Iter30 S03 Provenance Proposal — Agent B

```text
ROUND=1
PROPOSED_STATUS=PASS_HISTORICAL_METHOD_VALUE_PROVENANCE_ONLY
CONTRACT=FCCR-1
LAYER_ORDER=[L0,L1,L2] == [T0,T1,T2] == [0,1,2]
EXECUTION_AUTHORIZATION=NO
```

## Decision boundary

I find direct primary evidence sufficient to support the semantics, historical methods, and recorded values of both S02 inputs, but only as **historical method/value provenance**. This is not a present-day recalculation, a replay of either historic producer, proof that the historic checkpoint/SID bytes were the bytes consumed, or approval of later S03 stages. No historical writer was rerun; no calibration, mapping, source, or canonical artifact was changed. The S02 values and both registered mappings remain untouched.

## Proposed complete iter30 manifest

### Formula input ledger

| Symbol / canonical key | Exact semantic definition | Primary producer / producing run and recorded source lineage | Raw recorded value, `[L0,L1,L2]` | Transformations and value chain | Actual S02 formula input | Expected range / sanity bound | Evidence strength and limitations |
|---|---|---|---|---|---|---|---|
| `B_l` / `behavior_branching` | Per-layer effective branching factor `B_l=exp(H_l)`; it is not a literal count of unique children. `H0=H(T0|source)`, `H1=H(T1|source,T0)`, `H2=H(T2|source,T0,T1)`. `source` is the final item in a nonempty interaction history, and `T0,T1,T2` are the target item's ordered raw SID tokens. Within each context, entropy is natural-log entropy over next-token frequencies; context entropies are averaged weighted by observation count. | Primary implementation: `stage2_RQ-VAE/curvature_RQ-VAE_iter12/scripts/compute_behavior_branching.py`. It reads the iter8 raw SID table and Stage0 `train.parquet` history/target columns. Preserved result: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`, whose `source.script` explicitly names the iter12 producer and records source paths, counts, and the three entropy definitions. Its source metadata lists iter8 checkpoint, iter8 raw SIDs, and train parquet; the audited calculation itself loads raw SIDs and train parquet, not the checkpoint. | Recorded entropies: `[2.9613950179410296, 0.37882601480060585, 0.014701837881298745]`. Recorded branching: `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`. | Iter12 groups observations by the progressively conditioned context, computes observation-weighted conditional entropies with natural logarithms, then applies `np.exp(entropy)`. There is no transformation by `RESIDUAL_NORMS` in the entropy or branching path. `RESIDUAL_NORMS=[0.001,0.932889,1.0]` is used downstream only for separate `raw_capacity` and `branch_mid` side fields. The saved branching vector is therefore the direct `exp(H)` result in token-layer order. | `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` | Each conditional entropy must be finite and nonnegative; consequently `B_l` must be finite and at least 1. Values are strictly positive with L0 largest here, consistent with the saved entropy ordering. Expected order is three entries `[L0,L1,L2]`, not a sorted vector. | **HIGH** for recorded semantic definition, layer order, and values: producer algorithm plus paired entropy/branching artifact agree. The currently referenced iter8 `sids_raw.npy` is absent, so the historical branching calculation cannot now be independently recomputed; no historic SID hash was recorded. Current identity checks of available inputs at S01 are only check-time hashes and cannot establish which bytes historic execution used. |
| `m_l^{raw}` / `raw_residual_medians` | For each RQ layer, the median across Stage1 item embeddings of the Euclidean norm of that layer's residual vector. The residual norm is taken over embedding dimension for each item; the median is then taken over items. It is a raw magnitude, not a relative/normalized layer factor and not a trainable curvature parameter. | Measurement implementation: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py`, `_measure_layer_norms` and `_normalize`. Contemporaneous record: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log`; identifies baseline checkpoint `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth`, Stage1 embedding `.../stage1_GeneEmbedding/output/sentence_t5.npy`, and `step=100000`. Historical run is identified in the script payload as the iter1 calibration (`source: scripts/calibrate_residual_scales.py:2026-09-24 iter1`). Preserved iter26 result artifact separately carries the explicit `raw_residual_medians` field. | Raw median output recorded by the contemporaneous calibration log and iter1 hypothesis: `[1.0, 0.10941, 0.09331]`. | The script loads the named baseline checkpoint, requires model weights, sets the checkpoint's recorded global step, switches to eval mode, processes Stage1 embeddings in chunks of 1024, obtains `result.residuals`, computes `norm(dim=1)` over embedding dimension, concatenates item chunks, and takes `median(dim=1)` over items. Those medians are the raw measurement. Separately, it computes `s_max=max(medians)`, `s_min=min(medians)`, then `max(log(s_max/m_l)/log(s_max/s_min), 1e-3)` as normalized scales. For the logged medians, those separate scales are `[0.001,0.932889,1.0]`; the `0.001` at the maximum raw median is the epsilon floor for the zero log ratio. This normalized transformation is not part of the raw-median input path. | `[1.0, 0.10941, 0.09331]` | Residual norms and their medians must be finite and nonnegative; the registered values are strictly positive, as required by the S02 maps' positive denominator/ratio operations. Exactly three entries in `[L0,L1,L2]` order; no normalization, clipping, sorting, or replacement is applied before S02. | **MEDIUM** for historical method/value provenance: executable measurement code, contemporaneous log pairing raw medians with distinct normalized values, iter1 semantic record, and iter26 JSON's explicit raw key agree. The historic baseline checkpoint and iter8 raw SID file are absent at their historical paths; historical SHA256 values for the checkpoint, embedding, or SID inputs were not recorded. Therefore the original calibration cannot be replayed or bound to historic bytes. S01 hashes of present Stage1/Stage0 files and the present iter8 checkpoint are check-time identity only, not proof of historic use. |

### Semantic distinctions and safe consumer key

```text
raw_residual_median != normalized_layer_scale != learnable_c_layer_scale
```

- `raw_residual_medians=[1.0,0.10941,0.09331]` is the S02 residual input.
- `normalized_layer_scales=[0.001,0.932889,1.0]` is a separate historical log transform used by the older cyclic-curvature configuration. It is **not** the S02 residual input and must never be substituted.
- `learnable_c_layer_scale` denotes a model parameter concept, not either historical measurement vector; FCCR-1 curvature is fixed and non-trainable.

The iter26 JSON includes both `raw_residual_medians` and `residual_norm`, with numerically equal vectors. The alias is unsafe: iter12's producer uses the key name `residual_norm` for normalized scales `[0.001,0.932889,1.0]`, while iter26's closed-form consumer reads `payload["residual_norm"]`. Equal values in this particular iter26 JSON do not define the alias's semantics or remove the ambiguity. Every new iter30 consumer must require and use the explicit `raw_residual_medians` key; it must fail closed if missing and must not fall back to `residual_norm`, normalized scales, or inferred values.

### Historical attribution, presence, and identity limits

- The iter26 JSON's `source.script` attributes its behavior result to iter12. Its recorded iter8 paths, training path, sample counts, and definition accompany the output. The raw residual key is an additional historical measurement field; the JSON does not establish that iter12 produced that field. Do not conflate the branching producer with the separate iter10/iter1 calibration lineage.
- The iter10 log establishes that a calibration run reported step 100000 and both raw medians and normalized scales, alongside the checkpoint and embedding paths. Its lack of a checkpoint digest means this is not historic byte identity or replay evidence.
- At this audit, the named historic baseline checkpoint and the referenced iter8 `sids_raw.npy` are absent. The iter8 checkpoint itself is present, but its current hash (as recorded at S01) does not reconstruct or identify the missing historical SID export. S01-recorded hashes for current Stage1 embedding and Stage0 train parquet establish their identity only at the S01 check; they cannot prove those bytes were consumed in the historic measurement/branching runs.
- Iter26's preserved JSON values are recorded historical output, not proof that current inputs regenerate them. This proposal does not rerun, recalibrate, replay, or attest historic byte identity.
- The iter26 legacy calculator reads `residual_norm` and writes its output back to the same `computed_behavior_branching.json`. Iter26's behavior producer is not a safe regeneration path for S02 inputs: its payload contains neither `raw_residual_medians` nor a residual vector under the legacy `residual_norm` key, and its code reads the absent iter8 raw SIDs. Rerunning it would replace the preserved JSON with a payload lacking the explicit raw-residual field. Do not rerun it to reconstruct/refresh values; any future approved consumer must preserve the explicit key and fail if absent.

## S02 input registration and scope

The exact common input values remain:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
normalized_layer_scales = [0.001, 0.932889, 1.0]  # distinct; prohibited as substitute
```

These inputs feed only the two already locked S02 maps: iter26 control and iter29 candidate. This proposal does not choose, calculate, alter, or retune either mapping/output. The layer order and values remain exactly as registered. S00/S01/S02 canonical locks are preserved; the S02 raw-residual status is pending this adjudication and is not pre-approved by the existence of those records.

## Proposed verdict against active hard gate

**PROPOSED VERDICT: PASS, LIMITED TO HISTORICAL METHOD/VALUE PROVENANCE.**

The active skill §7 gate asks for semantics, provenance, recorded raw value, transformation, actual formula input, and sanity range; direct primary code/log/output evidence supports those historical method/value elements for each vector. It does not require successful historic byte replay as a substitute gate, and the evidence must not be represented as such. Branching has high confidence; raw residual has medium confidence because source code and paired calibration output support the method/value, while the exact historical inputs/checkpoint cannot be verified or replayed. Consequently, the PASS is narrow and qualified, not a claim of current recalculation, historic runtime byte identity, or general reproducibility.

If Judge C requires checkpoint-level replay or historic input-byte identity as an additional hard-gate criterion, this evidence does not satisfy it; record that narrower failure explicitly rather than manufacturing a replacement. No normalized input or alternate checkpoint/value is authorized.

## Assumptions

1. The iter10 calibration code and paired contemporaneous log accurately record the historical procedure and its output; the iter1 hypothesis corroborates the raw-versus-normalized interpretation but is not sole proof.
2. The preserved iter26 JSON accurately records the behavior producer attribution and saved values. Its source metadata names iter12 for branching, not for the separate residual calibration.
3. `[L0,L1,L2]` corresponds to the three ordered RQ layers / `[T0,T1,T2]` tokens, as shown by the producer's indexed operations and three-layer artifacts.
4. S02's explicit `raw_residual_medians` key is the canonical residual input contract; ambiguous legacy aliases are not accepted.
5. No conclusion about current producer replay, input byte identity at historic execution time, or mapping correctness is implied by the provenance verdict.

## Evidence reviewed

- Primary iter12 behavior producer, including hardcoded sources, entropy aggregation, `exp(entropy)`, and independent normalized side-field use: `stage2_RQ-VAE/curvature_RQ-VAE_iter12/scripts/compute_behavior_branching.py`.
- Preserved behavior/residual record and explicit metadata/values: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`.
- Primary residual measurement and normalization implementation: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py`.
- Paired contemporaneous calibration report of `step=100000`, raw medians and separate layer norms, and named historic inputs: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log`.
- Historical semantic distinction: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/hypothesis_iter1.md`.
- Directly reviewed iter26 legacy consumer and its `residual_norm` lookup: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/compute_closed_form_curvature.py`.
- Current canonical locks consulted: iter30 S00 `logs/source_snapshot_iter30.md`; S01 `logs/protocol_manifest_iter30.md` and S01 Judge; S02 `logs/hypothesis_iter30.md` and S02 Judge. S00/S01/S02 are scope/constraint records, not substitutes for the primary historical evidence above.
- Historical iter26 manifest and iter29 S03 canonical record consulted only as subordinate cross-checks, not as sole proof.
- Current path-presence check: historic baseline checkpoint and iter8 raw SID file absent; iter8 checkpoint present. S01's present-file hashes remain check-time-only evidence.

## Risks

- The recorded values could not be regenerated from historic source inputs because the named baseline checkpoint and raw SID export are missing; absent historical hashes prevent byte-level attribution.
- The legacy `residual_norm` name has conflicting historical uses; a consumer using it could silently map normalized scales as raw residuals even when another artifact previously stored raw values under the alias.
- The JSON combines values with distinct producers. Misattributing residual provenance to the iter12 branching writer would overstate evidence.
- Medium confidence in residual provenance is materially weaker than a replay; the limitation must remain attached to every propagation of these values.
- A fresh run using current files might produce values, but that would be a new measurement and is not permitted as a way to repair this historical audit.

## Self-rejection conditions

I reject my proposed PASS and would require a provenance FAIL/block if primary evidence disproves the logged median method, fails to pair the residual values with the raw rather than normalized output, shows a layer-order mismatch, or shows that S02 consumed anything other than the explicit raw key/value vector. I would also reject PASS if the alleged producer/output record were found to be only an unsupported later declaration rather than contemporaneous/primary evidence. Missing historic bytes mean I cannot claim replay or historic byte identity; I will not treat that limitation as resolved by current hashes or by another iteration's Judge conclusion.

## Autonomous next action

Submit this proposal for Judge C adjudication against direct primary sources. If Judge C accepts the limited historical-method/value PASS, propagate only the adjudicated canonical manifest, preserve confidence levels and all identity/replay limits, and require subsequent iter30 consumers to use `raw_residual_medians` explicitly with fail-closed behavior. Continue only through separately adjudicated later gates; this proposal authorizes no implementation, recalibration, MVG, Stage2, or Stage3. If Judge C finds evidence insufficient, block mapping approval/MVG/Stage2 and stop this provenance path without substituting normalized scales or inferring replacement inputs.
