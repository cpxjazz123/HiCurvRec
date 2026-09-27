ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S03_PROVENANCE/round_1/source_packet.md
STAGE_ID=S03_PROVENANCE

# Agent A — Iter30 S03 Provenance Proposal

## Proposed canonical artifact: complete draft manifest

# Iter30 Mechanism Manifest — S03 candidate (Agent A; not canonical)

```text
STAGE_ID=S03_PROVENANCE
ROUND=1
STATUS=PROPOSED_PASS_LIMITED_TO_HISTORICAL_METHOD_AND_VALUE_PROVENANCE
EXECUTION_AUTHORIZATION=NO
INPUT_LAYER_ORDER=[L0,L1,L2]
```

This proposal audits the provenance and semantics of the shared values registered by the accepted S02 hypothesis. The values below are historical recorded method/value evidence, not a current recalculation or checkpoint replay. No mapping output is recalculated or changed here.

### Formula inputs

| Symbol / canonical key | Exact semantic meaning | Primary source and producing run | Raw recorded value `[L0,L1,L2]` | Transformations and actual S02 formula input | Expected range / sanity bound | Evidence strength and limitations |
|---|---|---|---|---|---|---|
| `B_l` / `behavior_branching` (`branching` in JSON) | Effective behavior-conditioned branching factor for the next token at RQ layer `l`: `B_l = exp(H_l)`. `H_0=H(T0 | source)`, `H_1=H(T1 | source,T0)`, and `H_2=H(T2 | source,T0,T1)`. `source` is the last item in a nonempty user history; each target item's three ordered raw SID tokens provide `T0,T1,T2`. Each layer's conditional entropy is the observation-count-weighted mean of natural-log context entropies over valid behavior pairs. | Producer: `stage2_RQ-VAE/curvature_RQ-VAE_iter12/scripts/compute_behavior_branching.py`; recorded output: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`. The script reads iter8 `sids_raw.npy` and Stage0 `train.parquet`; JSON metadata says 396,958 train rows, 339,519 behavior pairs, 24,587 items, 3 layers. It declares an iter8 checkpoint path in metadata, but the inspected producer code does **not** load/use that checkpoint in the calculation. | Entropy (intermediate): `[2.9613950179410296, 0.37882601480060585, 0.014701837881298745]` nats. Recorded branching: `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`. | The producer counts next-token frequencies by the conditional contexts above, computes weighted conditional entropy using natural logarithms, then exponentiates each entropy. These recorded `branching` values are the actual S02 `B_l` formula inputs, unchanged and in L0/L1/L2 order. `RESIDUAL_NORMS=[0.001,0.932889,1.0]` is used only downstream in the producer's separate `raw_capacity` and `branch_mid` calculations; it does not enter entropy or `branching`. | `H_l` finite and nonnegative; therefore `B_l >= 1` and finite. The recorded values satisfy this: L0 is high branching; L1/L2 are near one. | **Medium/high for historical method and recorded values:** directly inspected producer implementation plus preserved JSON values, entropy, definitions and source metadata agree. **Not replay/byte identity:** current iter8 `sids_raw.npy` is absent at the recorded path, and no historical hash for it is recorded. The iter8 checkpoint is currently present and was hash-checked as a warm-start under S01, but that does not establish historical bytes or use; in fact the inspected branching calculation reads SIDs, not the checkpoint. Current Stage0 train and Stage1 embedding identities were S01 hash-checked at check time, but do not prove the historical data bytes/runtime. The JSON's `source.script` attribution is direct metadata, corroborated by its schema and matching producer logic; it is an aggregate artifact, not an execution transcript. |
| `m_l_raw` / explicit `raw_residual_medians` | Per-layer median across embedding dimensions of each item's residual-vector Euclidean norm after the baseline RQ-VAE's `get_semantic_ids` inference. The layer residual tensor is documented as `(n_layers, embed_dim, batch)`; `norm(dim=1)` gives one magnitude per layer and item; the median is taken over all items independently by layer. This is a raw residual magnitude, not a layer-normalized factor or curvature parameter. | Measurement implementation: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py`; contemporaneous output record: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log` (`step=100000`); explicit preserved field: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json::raw_residual_medians`. Historical calibration run was iter1 / baseline calibration, not iter10 training's own checkpoint. Script hardcodes `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth` and obtains the Stage1 embedding path from `curvature_config.ITEM_EMB_NPY`. | `[1.0, 0.10941, 0.09331]` (recorded raw medians). | The script checks checkpoint and embedding files, loads checkpoint model state strictly modulo `strict=False` but rejects any missing/unexpected keys, sets the saved global step, evaluates the model without gradients, processes embeddings in chunks of 1024, calculates residual norms over embedding dimensions, concatenates item observations, checks finite values, and takes `median(dim=1)` over items. The contemporaneous log reports `step=100000` and these medians. These exact recorded medians are the S02 `m_l_raw` inputs, with no transformation before entering either registered S02 mapping. The calibration script separately maps the medians to `layer_norms` using `max(log(s_max/m_l)/log(s_max/s_min), 1e-3)`, yielding `[0.001,0.932889,1.0]`; that vector is not the raw median and is not used as `m_l_raw`. | Each residual norm and its median must be finite and nonnegative; the positive measured medians are plausible magnitude scales. Layer order is the model's three quantizer/residual layers, L0 through L2. The recorded values are monotone decreasing `[1.0,0.10941,0.09331]`; no additional expected-ratio or normalized-range constraint is implied. | **Medium for historical method/value provenance:** primary measurement code specifies the quantity and operation; contemporaneous calibration log gives checkpoint step and exact raw/normalized pairs; iter1 semantic record independently repeats both vectors and their distinction; iter26 JSON carries a separate explicit `raw_residual_medians` field with the exact raw values. **Not checkpoint replay or historic bytes:** the named historical calibration checkpoint is absent at its recorded path now; its SHA256 was not recorded. The historical iter8 SID file is also absent, but it is a branching producer input, not the calibration script's residual-calibration input. The Stage1 embedding path is present now and was S01 hash-checked at check time, but no historical checksum/runtime trace binds it to calibration. The iter10 log is contemporaneous evidence of emitted values, not proof of the historical input bytes. |

### Three distinct quantities (must remain distinct)

```text
raw_residual_median = [1.0, 0.10941, 0.09331]
normalized_layer_scale = [0.001, 0.932889, 1.0]
learnable_c_layer_scale = a different trainable parameter concept; not this input
```

`raw_residual_median != normalized_layer_scale != learnable_c_layer_scale`. Under FCCR-1, S02 uses only the first vector as `m_l_raw`; do not substitute the normalized vector or any learnable scale. The normalized transform is historical context, not an alternate accepted input.

### Historical source/key and consumer caveats

The iter26 JSON combines branching and residual records: its `source.script` names the iter12 branching producer, while the residual measurement lineage is separately supported by the iter10 calibration implementation/log and iter1/iter26 semantics records. Do not imply iter12 computed the raw residual medians. The JSON's `residual_norm` legacy field happens to equal `raw_residual_medians` in this artifact, but that equality alone does not define the alias. The iter12 producer uses a field named `residual_norm` for the **normalized** `[0.001,0.932889,1.0]` vector in its `raw_capacity`/`branch_mid` side calculations. The iter26 closed-form calculator also reads `payload["residual_norm"]`; therefore the historical consumer's semantic reliance on that ambiguous key is not proven by the JSON equality. Iter26 hypothesis/manifest document its intended raw-residual meaning but do not remove the consumer-key ambiguity. S02 explicitly requires all new iter30 consumers to read `raw_residual_medians` only and fail closed if absent; no fallback to `residual_norm` and no normalized-scale substitution.

### Provenance classification and gate

- `behavior_branching`: supported historical method/value provenance, with no current replay or historic byte-identity claim; confidence **MEDIUM/HIGH** for method/value, limited by missing historical SID input/hash.
- `raw_residual_medians`: supported historical method/value provenance; confidence **MEDIUM**, not independently replayed and not historically byte-bound to checkpoint/embedding inputs.
- **Proposed §7 decision: PASS, limited strictly to historical method/value provenance.** Direct primary measurement code and contemporaneous output log support what was measured and its recorded values; the JSON's explicit key and older semantic records corroborate the raw/normalized distinction. This is not a finding that all historical input bytes/runtime use are recoverable. It does not establish checkpoint-level reproduction, current recalculation, historical-byte identity, or S03 adjudication by Judge C. If the Judge interprets §7 as requiring those stronger forms of identity rather than supported historical method/value provenance, the residual input must be classified FAIL and all later approval/MVG/Stage2 blocked; there is no fallback.

## Assumptions

1. The preserved iter26 JSON is the historical value artifact corresponding to the iter12 producer's recorded output, as its `source.script` and internally coherent entropy/branching fields specify; this does not assert that the file was regenerated in the current audit.
2. The iter10 calibration log is the contemporaneous output of the inspected iter10 calibration script, and the script's `medians` label refers to the exact `median(dim=1)` output it constructs.
3. Layer indices 0, 1, 2 map in order to FCCR-1's L0, L1, L2, consistent with the three-column raw-SID array, model layers, JSON values and S02 registration.
4. Historical provenance means supported measurement method and recorded-value lineage. It does not entail reproducing the old run from historic checkpoint/input bytes.

## Evidence inspected

- Canonical packet: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S03_PROVENANCE/round_1/source_packet.md`.
- Iter30 S00 canonical snapshot; S01 protocol manifest and Judge record; S02 source packet, accepted hypothesis and Judge record. These all retain the S03 hard gate and explicitly provisional raw-residual status; they are controlling status/context, not sole evidence for this proposal.
- Primary iter12 producer: `scripts/compute_behavior_branching.py`, including path constants, entropy/conditional-entropy functions, token-context accumulation, branching exponentiation, residual-norm side calculations, and output metadata.
- Primary preserved value artifact: iter26 `scripts/computed_behavior_branching.json` (entropy, branching, explicit raw-residual field, legacy alias, source metadata, counts and layer definitions).
- Primary iter10 residual code: `scripts/calibrate_residual_scales.py` (checkpoint/embedding configuration, residual norm over embedding dimension, item median, separate logarithmic normalization).
- Contemporaneous iter10 calibration log: `logs/calibrate_residual_scales.log` (`step=100000`, exact raw medians, separate layer norms, checkpoint and embedding paths).
- Iter1 semantics record: `logs/hypothesis_iter1.md` (raw medians vs normalized factors).
- Iter26 hypothesis, mechanism manifest and closed-form consumer. The consumer uses `residual_norm`; its ambiguity is preserved rather than treated as proof.
- Current availability spot-check: the historical calibration checkpoint path and recorded iter8 raw SID path both return file-not-found now. The current iter8 warm-start checkpoint, Stage1 embedding, and Stage0 train parquet are present; S01 records check-time hashes for the warm-start, embedding and train parquet (and other locked inputs). These current/check-time facts do not establish historical bytes or past runtime use. No historical SHA256 for the absent calibration checkpoint or raw SIDs is recorded in the inspected canonicals.

## Risks

- Missing historical calibration checkpoint and missing historical iter8 raw SID table prevent a faithful checkpoint/input replay and prevent establishing historic byte identity; no historic hashes were recorded.
- The contemporaneous calibration log confirms values but is not an execution trace binding exact historical input bytes or proving current reproducibility.
- The iter26 JSON is a joint artifact with branching-oriented source metadata; residual provenance must not be attributed to the iter12 script.
- The duplicated `residual_norm` key is semantically hazardous. Numeric equality in the iter26 artifact is not enough to license a future consumer to use it.
- A stronger interpretation of the active provenance hard gate could require exact historic input identity; under that reading raw residual fails and the S02 mappings cannot proceed. The Judge must resolve this against the packet's express separation of method/value provenance from byte identity.

## Self-rejection conditions

Reject this proposal / change the proposed limited PASS to FAIL if direct evidence establishes any of the following: the iter10 logged medians were not emitted by the described residual-median method; the stored explicit raw field cannot be attributed to those recorded medians; layer ordering is inconsistent; a required raw measurement source/value is materially unknown rather than limited only in historical bytes/replay; or active skill §7 is interpreted to require historic checkpoint/input byte identity for provenance validity. Under any such failure, do not substitute normalized scales, another checkpoint, or a newly inferred value; stop before subsequent mapping approval, MVG or Stage2.

## Proposed verdict

`PROPOSED_VERDICT=PASS_LIMITED_HISTORICAL_METHOD_VALUE_PROVENANCE` — separately medium-confidence for `raw_residual_medians` and medium/high for `behavior_branching`. This verdict is a candidate recommendation only, not S03 canonical approval. It does not claim current replay or historic byte identity and does not clear any later gate.

## Autonomous next action

Judge C should adjudicate both inputs against these primary files and active FCCR-1 §7. If it accepts the limited historical method/value basis, materialize a canonical manifest that preserves the confidence limits and explicit-key requirement, then advance only to the next separately authorized stage; otherwise record `PROVENANCE_INVALID`/FAIL and block all subsequent mapping approval, MVG and Stage2. In either outcome, no recalibration, normalized substitution, code change, replay, or Stage2/Stage3 execution is authorized by this proposal.
