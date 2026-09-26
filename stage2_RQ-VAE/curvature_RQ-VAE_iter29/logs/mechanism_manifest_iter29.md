# Iter29 Mechanism Manifest — S03_PROVENANCE Canonical

```text
STAGE_ID=S03_PROVENANCE
ROUND=1
STATUS=PASS_HISTORICAL_METHOD_VALUE_PROVENANCE_WITH_REPRODUCIBILITY_LIMITATIONS
CONTRACT=FCCR-1
LAYER_ORDER=[0,1,2] == [T0,T1,T2]
```

## Decision scope and boundary

S03 approves the two exact S02 formula inputs as historically documented method/value provenance. This is **not** a current checkpoint-level reproduction, does not certify historic file-byte identity, and is not a new measurement. The registered decimal values and layer order are unchanged. No checkpoint replay, SID re-export, recalibration, or mapping alteration was performed. S03 does not re-validate or change the S02 equation; it authorizes only the stated inputs to proceed.

## Canonical formula inputs

| Symbol / canonical key | Exact semantic meaning | Primary source and producing run | Raw recorded value, layer order `[L0,L1,L2]` | Transformation into the actual S02 formula input | Actual formula input | Expected range / sanity bound |
|---|---|---|---|---|---|---|
| `B_l` / `behavior_branching` | Effective branching factor `exp(H_l)`, not a literal count of unique children. `H_0=H(T0\|source)`, `H_1=H(T1\|source,T0)`, and `H_2=H(T2\|source,T0,T1)`. Each conditional entropy is natural-log entropy averaged across contexts weighted by their observation counts. `source` is the final item in each nonempty interaction history; `T0,T1,T2` are the target item's three ordered raw SID tokens. | Historical behavior output is `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`; its `source.script` explicitly identifies `stage2_RQ-VAE/curvature_RQ-VAE_iter12/scripts/compute_behavior_branching.py`, with source paths for iter8 raw SIDs and Stage0 `train.parquet`. The audited producer groups target tokens by `source`, `(source,T0)`, `(source,T0,T1)`, computes observation-weighted conditional entropies, and sets `branching=np.exp(entropy)` (`iter12` script lines 41–62, 65–104; same conditional-grouping/`exp` mechanism is present in the preserved iter26 script lines 64–103). The JSON records 396,958 train rows, 339,519 nonempty behavior pairs, 24,587 SID rows/items, and 3 layers. | `H=[2.9613950179410296, 0.37882601480060585, 0.014701837881298745]` | `B_l=exp(H_l)`; no residual normalization or residual calibration enters this transformation. The JSON records `branching` directly. | `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` | For finite, nonnegative conditional entropy: finite `B_l >= 1`. Three entries must align with token levels/layers `[L0,L1,L2]`. |
| `m_l^{raw}` / `raw_residual_medians` | For each RQ layer, the PyTorch median over all Stage1 item embeddings of the Euclidean norm of that layer's residual vector, measured by running the historical baseline RQ-VAE checkpoint at its recorded global step. The per-item residual norm is taken over embedding dimension; the median is taken across items. | Historical measurement method: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py` (`_measure_layer_norms`, lines 54–111). The contemporaneous execution record is `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log`; it reports source label `iter1`, checkpoint step `100000`, the baseline checkpoint path and Stage1 embedding path. `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/hypothesis_iter1.md` records the same distinction. Iter26's manifest/hypothesis/one-factor record and its JSON explicit `raw_residual_medians` field are corroborating records, not the sole semantic evidence. | `medians=[1.0, 0.10941, 0.09331]` | **No normalization or other transform** is applied for S02: these raw median magnitudes are the formula inputs. The historical script's separate `_normalize` result is not used. | `[1.0, 0.10941, 0.09331]` | Finite and positive for this registered S02 map; residual norms are nonnegative. Values must remain in layer order `[L0,L1,L2]`. |

The approved substitution is exactly:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```

## Primary-source detail and residual-key distinction

### Behavior branching

The behavior producer loads the three-column iter8 raw SID table and Stage0 training `history`/`target` columns. For each nonempty history it selects `source=history[-1]`, reads the target's `(T0,T1,T2)`, updates the three progressively conditioned next-token count groups, computes observation-weighted natural-log conditional entropies, then exponentiates them. The saved entropy and branching vectors have three values in this same layer/token order.

The iter12 script also defines `RESIDUAL_NORMS=[0.001,0.932889,1.0]`. In that script this normalized vector is used only later in the separate `raw_capacity` and `branch_mid` side-field calculations; it is not used by the conditional entropy calculation or by `branching=exp(entropy)`. Those side fields are not S02 inputs. The iter12 JSON labels that normalized vector `residual_norm`. Iter26's JSON separately carries `raw_residual_medians=[1.0,0.10941,0.09331]` and a legacy `residual_norm` alias with the same raw values, while retaining the historical behavior output. The legacy field name is therefore not a safe semantic contract and MUST NOT determine S06's residual input.

The iter8 checkpoint path appears in the behavior JSON's source metadata, but the audited branching producer computes from `sids_raw.npy` plus `train.parquet`; it does not load the checkpoint during the branching calculation. Do not treat the checkpoint's current hash as proof of the historical SID bytes.

The preserved iter26 behavior writer has an additional S06 hazard: it sets `OUTPUT_PATH` to this same `computed_behavior_branching.json`, but its `payload` does not contain `raw_residual_medians` and its `write_text` replaces the file. Rerunning it would therefore discard the explicit residual provenance key; it also reads the currently absent iter8 `sids_raw.npy`. S06 MUST NOT rerun this writer as a way to regenerate the registered inputs. It must preserve and explicitly validate the canonical saved `raw_residual_medians` field, and any producer/write path touching this JSON must retain that field and fail if it is missing. Do not treat the current iter26 writer as a reproducible source for the raw SID data.


### Raw residual medians versus normalized layer scales

The calibration script requires a checkpoint and embedding file, loads the checkpoint model weights, sets its recorded step, switches to evaluation mode, and processes the Stage1 embedding matrix in chunks of 1024. `result.residuals` is documented as `(n_layers, embed_dim, batch)`; `norm(dim=1)` produces per-layer/per-item Euclidean norms, chunks are concatenated along the item dimension, and `stacked.median(dim=1).values` produces the three raw medians.

The same script separately defines the historical logarithmic normalization:

```text
s_max = max(medians)
s_min = min(medians)
normalized_l = max(log(s_max / medians[l]) / log(s_max / s_min), 1e-3)
```

The paired contemporaneous log records both outputs from that same calibration run:

```text
raw medians       = [1.0, 0.10941, 0.09331]
normalized scales = [0.001, 0.932889, 1.0]
```

The first normalized value is the `1e-3` floor applied to the zero logarithmic value at the maximum median. Thus the three quantities are distinct:

```text
raw_residual_median != normalized_layer_scale != learnable_c_layer_scale
```

`learnable_c_layer_scale` is a separate model-parameter concept and is not either historical input vector. For iter29, only the raw medians are the residual formula input. The normalized vector is not an alias, fallback, or permissible replacement.

## Historical source paths and identity/replay limits

- The calibration log names checkpoint `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth`, Stage1 embeddings `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`, and `step=100000`. The named historical baseline checkpoint is absent from the current results inventory (`stat` reports no such file). The log contains no checkpoint SHA-256, embedding SHA-256, or historical SID hash. The calibration code and contemporaneous paired values establish method/value provenance, but the original forward pass cannot now be replayed or checked against checkpoint bytes.
- The behavior JSON names iter8 `sids_raw.npy` at `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/sids_raw.npy`; that raw SID file is absent at the referenced path in the current inventory. The JSON preserves its result, paths, dimensions, row/pair counts, and definition, but it does not record a hash of the SID bytes; the branching vector cannot now be recomputed from that missing file.
- The current iter8 checkpoint is present and its SHA-256 matches S01 (`189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`). The current Stage0 train parquet and Stage1 embedding also match S01's current-file hashes (`80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` and `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`, respectively). These checks verify present locked files only. They do not prove those exact bytes were consumed by the historical behavior or calibration runs, do not replace the missing iter8 raw SID export or calibration checkpoint, and do not establish historic file-byte identity.
- The saved iter26 JSON's explicit `raw_residual_medians` field plus the calibration method, paired run-log labels, and iter1 hypothesis corroborate the exact raw median vector. Iter26's mechanism manifest, hypothesis, and one-factor record corroborate how that vector was registered and distinguish it from the earlier normalized vector. Those later declarations are not treated as sole primary proof.

Accordingly, the audit approves **historical method/value provenance with reproducibility limitations only**. It does not assert checkpoint-level reproduction, independent recomputation, or historical hash identity.

## Registered mapping and one-factor preservation

The canonical S02 mapping remains unchanged:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
```

S02 records `fixed_curvature=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` as arithmetic from these registered decimal inputs and constants, not as checkpoint-level reproduction. This S03 decision does not select, recompute, or retune those outputs. Relative to the iter26 direct control, the iter29 experiment remains the single registered mapping change; both input vectors, their order, and all other protocol/mechanism conditions stay fixed.

## Mandatory S06 consumer-key hygiene

S06 MUST consume the explicit JSON key `raw_residual_medians` for `m_l^{raw}`. It MUST NOT consume the ambiguous legacy key `residual_norm`, even when that field currently happens to contain equal numeric values. If `raw_residual_medians` is absent, S06 must reject/fail the input; **no fallback** to `residual_norm`, normalized `layer_norms`, or any other field is permitted. Preserve the exact values `[1.0,0.10941,0.09331]`, `[L0,L1,L2]` order, S02 formula, and its constants unchanged. Do not rerun the iter26 behavior writer to reconstruct inputs: it writes to the same JSON but omits `raw_residual_medians`, and its required iter8 raw SID file is absent. Any S06 writer touching this JSON must preserve and validate the explicit raw key. No recalibration from a different checkpoint, normalization, substitution, or retuning is authorized in iter29.

## Confidence and propagation boundary

- `behavior_branching`: **HIGH confidence** in the recorded semantics, layer order, and historical values from producer logic plus saved entropy/branching output; current replay is blocked by the absent raw SID export and no historical SID hash.
- `raw_residual_medians`: **MEDIUM confidence** in historical method/value provenance from the calibration implementation plus the contemporaneous paired raw/normalized log values and historical hypothesis; original-checkpoint replay and historical byte identity are unavailable.
- Overall S03: **PASS, limited to historical method/value provenance**. Only this judge-approved manifest and `judge.md` may propagate. The stated limits remain part of the canonical record.
