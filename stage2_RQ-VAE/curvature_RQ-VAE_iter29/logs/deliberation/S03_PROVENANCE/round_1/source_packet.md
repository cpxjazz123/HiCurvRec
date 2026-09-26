# S03 Provenance Source Packet — iter29

STAGE_ID=S03_PROVENANCE
ROUND=1

## Canonical context

- Active contract: FCCR-1. Canonical S00, S01 and S02 artifacts are `logs/source_snapshot_iter29.md`, `logs/protocol_manifest_iter29.md`, and `logs/hypothesis_iter29.md`; S02 Judge accepted candidate A. Do not read/use the rejected candidate as active instructions; its existence is audit context only.
- Iter26 is the sole direct control, with all protocol settings held fixed except the accepted alternative mapping. The S02 residual values remain `PROVISIONAL_PENDING_S03` and may not propagate to implementation/execution without this stage's explicit pass.
- Approved S02 mapping needs both `behavior_branching` and `raw_residual_median`. The fixed constants are `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`. This stage audits the semantics, source, layer alignment, transformation, and provenance of its two inputs; it does not change or select the mapping.

## Exact provisionally recorded values

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_median = [1.0, 0.10941, 0.09331]   # PROVISIONAL_PENDING_S03
normalized_layer_scale = [0.001, 0.932889, 1.0] # a distinct transform, NOT the raw input
```

## Primary repository evidence to audit independently

1. Iter26 `scripts/computed_behavior_branching.json` has the branching vector and both `raw_residual_medians` and the legacy `residual_norm` alias. Its `source` block identifies the behavior script but does not itself fully describe the residual calibration source.
2. Iter26 `scripts/compute_behavior_branching.py` computes `H(T0|source)`, `H(T1|source,T0)`, `H(T2|source,T0,T1)` from iter8 raw SIDs and Stage0 `train.parquet`, then `branching=exp(entropy)`. It does **not** compute the raw residual medians. Its source metadata records the iter8 checkpoint/SIDs path, parquet, 24,587 items, 3 layers, 396,958 rows and 339,519 nonempty behavior pairs.
3. Iter26 `scripts/compute_closed_form_curvature.py` currently reads `payload["residual_norm"]`, an ambiguous legacy key; iter26 JSON records that alias equal to `raw_residual_medians`. Identify the naming risk and state the required S06 repair to consume the semantically explicit `raw_residual_medians` key without changing input values or the scientific mapping.
4. Iter26 `logs/mechanism_manifest_iter26.md`, `hypothesis_iter26.md`, and `one_factor_diff_iter26.md` declare `[1.0,0.10941,0.09331]` as raw residual medians, distinguish them from the normalized `[0.001,0.932889,1.0]` factors, and attribute them to iter1 residual calibration. Treat these declarations as claims to corroborate, not as sole proof.
5. The preserved historical copy `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py` documents the measurement method: load the promoted baseline checkpoint, use the Stage1 item embedding matrix, compute `result.residuals.norm(dim=1)` for every item in chunks, concatenate `[n_layers, n_items]`, and take `median(dim=1)`. It separately transforms medians to `layer_norms` via logarithmic normalization and epsilon clamp. The script's embedded source label says it was the iter1 calibration script.
6. `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log` records execution with the canonical baseline checkpoint path `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth`, the Stage1 `sentence_t5.npy` path, `step=100000`, raw `medians=[1.0,0.10941,0.09331]`, and distinct normalized `layer_norms=[0.001,0.932889,1.0]`. `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/hypothesis_iter1.md` records the same distinction.
7. The referenced canonical baseline checkpoint path is not currently present in the `results/stage2_RQ-VAE/**/rqvae_best.pth` inventory. The prior run log and source code thus preserve method/value provenance but do not permit current checkpoint-level recomputation; candidate workers must decide whether that limits but still establishes provenance, or makes it insufficient under the hard gate.
8. For branching, verify the current iter8 raw-SID path/checkpoint and train parquet identity against S01. For residual calibration, verify the Stage1 path and current S01 embedding hash. Do not imply that current file hashes prove the bytes used in a historical run when the historical run did not record them.

## Candidate task

Agent A and Agent B independently perform a hard-gate semantic/provenance audit of **both** formula inputs. For each, state exact meaning, source code/script, source iteration/run, raw and transformed values, layer order/alignment, expected range, confidence, and reproducibility limitations. Explicitly decide whether the raw-residual evidence genuinely distinguishes medians from normalized layer scales, whether the missing reference checkpoint makes provenance `PASS` with a reproducibility caveat or `FAIL`, and whether the branching vector is directly supported. Give a concrete overall S03 `PASS` or `FAIL`; if failing, state why the registered iter29 hypothesis cannot safely propagate, and do not suggest silently substituting/recomputing with a different checkpoint or changing input values in the same iteration. If passing, specify the narrow S06 key/metadata repair needed so runtime code consumes the declared raw input explicitly.

Candidates write only `logs/deliberation/S03_PROVENANCE/round_1/agent_a.md` or `agent_b.md`, begin with the mandated role, independence declaration, source packet path, and stage id; use primary code/logs and distinguish fact from inference. No source changes, formula retuning, GPU, training, builds, tests, or formatters. S03 is a hard gate: no S04 or implementation until Judge C approves a canonical pass. If S03 fails, the orchestrator must stop this iteration and follow the skill's autonomous abort-and-close path; do not ask the user.