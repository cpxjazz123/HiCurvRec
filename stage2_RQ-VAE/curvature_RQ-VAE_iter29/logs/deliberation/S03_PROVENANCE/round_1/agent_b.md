ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S03_PROVENANCE/round_1/source_packet.md
STAGE_ID=S03_PROVENANCE

# S03 Provenance Audit — Agent B (noncanonical)

## Verdict

**OVERALL S03 HARD-GATE: PASS**, limited strictly to historical method/value provenance for the two registered inputs. The evidence supports the stated semantics and layer-ordered values sufficiently to distinguish raw residual medians from normalized layer scales. It does **not** reproduce the historical residual measurement from its checkpoint, establish historical input-byte identity, or certify a current checkpoint-level recalculation. S03 passing this limited provenance gate permits these recorded inputs to proceed for adjudication; it does not convert the evidence into a reproduction claim.

The registered iter29 values remain `behavior_branching=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and `raw_residual_median=[1.0, 0.10941, 0.09331]`, aligned in layer order `[0,1,2]`. These are evidence-backed historical values, not newly measured ones. I make no change to or independent selection of the S02 mapping.

## Formula-input semantic and provenance table

| symbol/meaning | primary code/script | source iteration/run | raw value | transformation | actual formula input | expected range |
|---|---|---|---|---|---|---|
| `B_l` / `behavior_branching`: effective branching `exp(H_l)` for successive RQ token levels; `H_0=H(T0\|source)`, `H_1=H(T1\|source,T0)`, `H_2=H(T2\|source,T0,T1)` | `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/compute_behavior_branching.py` computes the conditional entropies from iter8 `sids_raw.npy` and Stage0 `train.parquet`, then uses `branching=np.exp(entropy)`. Its JSON output is `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`. | Historical behavior computation recorded as iter26; source data are iter8 raw SIDs and Stage0 train interactions. JSON reports 396,958 rows, 339,519 nonempty behavior pairs, 24,587 items, and 3 layers. | `H=[2.9613950179410296, 0.37882601480060585, 0.014701837881298745]` (natural-log conditional entropy). | `exp(H_l)`; no additional normalization is part of the recorded `branching` input. | `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` | Finite `B_l >= 1`; with at most 256 next-token categories, `B_l <= 256`. Values decrease with the recorded conditional entropies and are consistent with three 256-entry token layers. |
| `m_l^raw` / `raw_residual_median`: median, over Stage1 item embeddings, of the per-item norm of the residual vector at RQ layer `l` when run through the historical promoted baseline RQ-VAE checkpoint | Historical measurement method is preserved in `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py`: load model/checkpoint, call `get_semantic_ids` over embedding chunks, take `result.residuals.norm(dim=1)`, concatenate per layer across items, then `median(dim=1)`. `_normalize` separately maps those medians to normalized layer factors. `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log` records the outputs. | Iter1 calibration run, as identified by the script and its embedded source label; logged checkpoint step `100000`, canonical baseline checkpoint path `results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth`, and Stage1 `sentence_t5.npy`. Later recorded in iter26 JSON as `raw_residual_medians`. | `[1.0, 0.10941, 0.09331]` | For the declared raw input: **identity** (no normalization). A different output of the same calibration script applies logarithmic normalization using `s_max`, `s_min`, and epsilon floor `1e-3`, yielding `[0.001, 0.932889, 1.0]`; that is a distinct derived quantity and is not this input. | `[1.0, 0.10941, 0.09331]` | Finite and strictly positive residual magnitudes in the embedding/model units; layerwise median norm, not constrained to `[0,1]`. |

The layer alignment follows the three ordered residual rows / RQ token levels in the measurement code and the three ordered conditional entropies/token levels in the branching computation. The logs and JSON carry three entries in this order. They do not provide a new paired per-item crosswalk, and none is needed to identify the vector's layer order.

## Independent support for behavior branching

The branching value has a traceable method and source record. The iter26 script explicitly loads `ITER8_RAW_SIDS`, reads `history` and `target` from `TRAIN_PARQUET`, indexes the target's three raw tokens, and groups next-token counts by `source`, then `(source,t0)`, then `(source,t0,t1)`. It computes observation-weighted natural-log conditional entropy for each level and exponentiates each entropy. The output JSON records the same three entropy values, branching values, source paths, dimensions, and counts. The behavior script does **not** compute residual medians; that input comes from the separate historical calibration evidence.

S01's current locked identities corroborate the Stage0 train parquet and iter8 checkpoint identities: their present SHA-256 values match S01 (`80597e…88815` for `train.parquet`; `189e0a…56c3b` for `rqvae_best.pth`). The current Stage1 embedding also matches the S01 hash (`6490c7…77fb`). The explicitly referenced iter8 `sids_raw.npy` is absent at its current path, so it could not be hash-checked or rerun here. Further, the historical branching JSON did not record a hash for the raw SID bytes; matching current checkpoint/parquet identities therefore cannot establish byte identity of historical inputs. The branching vector is **directly supported as a historical computed-and-recorded result**, but is not independently recomputed in this audit.

## Residual distinction and source quality

There is independent, concrete evidence that the two residual vectors mean different things:

1. The calibration code measures raw per-layer residual norms and takes their medians (`median(dim=1)`), returning those medians before any transform.
2. The same script's separate `_normalize` function computes log-scaled factors and floors them at `1e-3`.
3. The contemporaneous calibration log labels the separate outputs `medians=[1.0, 0.10941, 0.09331]` and `layer_norms=[0.001, 0.932889, 1.0]` on the same recorded run. Iter1's hypothesis records the same distinction.
4. Iter26's data JSON stores `[1.0,0.10941,0.09331]` under the explicit key `raw_residual_medians`; its legacy `residual_norm` alias has the same vector. Iter26's manifest, hypothesis, and one-factor record corroborate the substitution and contrast it with the earlier normalized vector. These later declarations are corroboration, not sole provenance.

Thus equality of the legacy `residual_norm` alias and raw medians in iter26 is not the basis for interpreting the quantity: the original calibration method, separate transform, and paired run-log labels establish the semantic distinction. The evidence supports historical method/value provenance; it does not prove that the residual values can now be regenerated from the original checkpoint.

## Missing-checkpoint and reproducibility limits

The calibration log names the baseline checkpoint used for the iter1 measurement, but the packet states that this referenced checkpoint is not present in the current `results/stage2_RQ-VAE/**/rqvae_best.pth` inventory. It cannot be loaded to reproduce the residual forward pass and median calculation. The current iter8 checkpoint does exist and matches S01's warm-start SHA, but it is a different checkpoint and cannot replace the historical calibration checkpoint. No replacement measurement was made.

Accordingly, this is **not checkpoint-level reproduction** and **not proof of historical input-byte identity** (the historical calibration log did not record checkpoint, embedding, or SID hashes). Present-day S01 hashes establish the identity of the currently locked files only; they do not retroactively identify the historical bytes. These limits reduce reproducibility confidence but, given the preserved calculation code plus contemporaneous logged raw and separately normalized outputs, do not erase historical method/value provenance.

## Required S06 hygiene

S06 must read the explicit JSON key `raw_residual_medians` for the formula input, not the ambiguous legacy `residual_norm` key. This is a naming/wiring repair only: preserve the exact values `[1.0, 0.10941, 0.09331]`, layer order, S02 equation, and registered constants unchanged. Do not substitute `[0.001, 0.932889, 1.0]`, recompute/recalibrate from another checkpoint, or otherwise alter these inputs within iter29. The historical normalized factors are not a valid alias for the raw-residual-median input.

## Confidence and propagation boundary

- **Behavior branching:** high confidence in recorded semantic definition, method, and values; no current SID-byte hash/recomputation.
- **Raw residual medians:** medium confidence in historical method/value provenance; no original-checkpoint replay or historical byte identity.
- **Overall S03:** PASS for historical method/value provenance only. This is noncanonical Agent B evidence pending Judge C; only the judge-approved canonical artifact may propagate. Do not describe this as checkpoint-level reproduction or claim identities beyond those actually verified.
