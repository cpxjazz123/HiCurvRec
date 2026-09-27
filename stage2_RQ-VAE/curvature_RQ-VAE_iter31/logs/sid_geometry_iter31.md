# Iter31 Stage2 SID and geometry analysis — S10 canonical

## Scope and evidence basis

This report adjudicates S10 for registered HRA-STEP6-1 under the inherited FCCR-1 contract. Agent A and Agent B both declare independence, identify `STAGE_ID=S10_STAGE2_ANALYSIS`, and cite `logs/deliberation/S10_STAGE2_ANALYSIS/round_1/source_packet.md`. Their recorded SHA-256 values are identical: `86d6666ca25d433865c7788d1aab5265f0393106d21c23d26594149758c01688`; the frozen packet copy in the Iter31 tree hashes to the same value. Primary evidence reviewed includes the two contracts, S02 hypothesis, S05 one-factor decision, S08 MVG record, S09 conditional execution decision and completion record, terminal trainer/export log, registered SID metric implementation, and the actual Stage2 results.

## 1. FCCR-1 fixed-curvature contract compliance

**Observed.** `logs/mechanism_contract_iter31.json` registers FCCR-1 closed-form, non-trainable and time-invariant curvature, with no cyclic schedule, curvature regularization, curvature-conditioned optimizer, or new curvature-conditioned auxiliary loss. Its fixed vector is `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. The active HRA Step6 contract explicitly inherits this unchanged FCCR-1 sub-contract.

The S08 MVG record reports the closed-form and live values, non-trainable fixed buffers excluded from optimizer parameters, and invariance snapshots at training steps 0, 25,000, 50,000, and 100,000 (also evaluation snapshots). At step 100,000 the final training log records `c_live=[1.366095, 0.734783, 0.643996]` and `c_closed=[1.366095, 0.734783, 0.643996]`, matching within `1e-6`; the logged curvature regularization is zero. S08 records `MVG PASS` and finite nonzero intended model gradients. The terminal record shows successful completion at global step 100,000, and the completion record reports supervisor exit status 0 with no restarts.

**Assessment.** The inspected evidence supports fixed FCCR-1 values and invariance through step 100,000, and a completed, non-aborted Stage2 run. This is contract/execution evidence, not evidence of recommendation benefit.

## 2. Registered HRA Step6 direct effect versus Euclidean

**Observed.** In the registered S08 same-batch comparison, the HRA operation and Euclidean tangent sum use the same quantized embeddings and 640 ordered source/target pairs. Both outputs are finite with shape `[640,32]`. The recorded maximum absolute difference is `0.07198049873113632`, L2 difference `2.018216848373413`, and relative L2 difference `0.08106279373168945`. The S08 record says no direct-effect threshold was preregistered (`report_only`).

**Interpretation.** This establishes a nonzero direct decoder-facing Step6 output difference on that registered batch. It does not show that the operation improves reconstruction, SID quality, or recommendation performance, nor does one batch establish a population-wide effect. It is not a Stage3 result.

## 3. Stage2 SID observations and output wiring

**Observed artifact facts.** The final artifacts are under the Iter31 results root:

- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/rqvae_best.pth`: checkpoint; completion evidence reports `global_step=100000` (CPU inspection), keys `global_step/model/optimizer`, and 14 model tensors.
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/sids_raw.npy`: `int32`, shape `[24587,3]`, raw code values in `[0,255]`.
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy`: `int64`, shape `[24587,4]`, extension values in `[768,778]`.
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`: 24,587 ordered dense string keys (`"0"` through `"24586"`), four values per row.

The completion record's artifact hashes match the frozen packet. Its CPU artifact audit reports exact equality between the JSON row matrix and exported NPY, exact equality between raw NPY and the exported NPY's first three columns, and uniqueness of all 24,587 exported four-token rows. The final export log records `SID_WIRING_PASS`.

**Registered descriptive metrics.** The registered implementation is `modules/sid_quality.py::evaluate_sid_quality`; `should_early_stop` returns `(False, "")`. The final trainer log and candidate CPU recomputations report the following raw three-token SID statistics:

- `n_items=24587`; `n_unique_full=22441`; largest tuple multiplicity `11`.
- `full_gini=0.08242568913963332`.
- `per_layer_gini=[0.17365233024565827, 0.25000365411599623, 0.29055099620734537]`.
- `l01_unique_pairs=13052`; `H(L1|L0)=5.343793126431163` bits.
- All 256 codes are occupied in each layer; observed per-code occupancy-count ranges are L0 `[29,197]`, L1 `[4,289]`, and L2 `[1,284]`.

**Collision terminology correction, independently checked from raw tuples.** Counting raw SID tuple frequencies directly gives 22,441 distinct tuple keys among 24,587 rows, 1,598 tuple keys with multiplicity greater than one, and 3,744 item rows belonging to those colliding keys. Therefore:

- **Duplicate-excess-row rate**: `(N - number_of_unique_tuple_keys) / N = 2146/24587 = 0.08728189693740594`. This counts rows beyond the first representative of each unique tuple.
- **Collision-participation fraction**: `(number of rows whose tuple key has multiplicity > 1) / N = 3744/24587 = 0.15227559279293937`. This counts every item in any colliding tuple group.

The frozen packet calls `0.0872818969` a “collision-item fraction”; that label is imprecise. It is the duplicate-excess-row rate, not the fraction of items participating in a collision. The packet remains unchanged. This naming correction does not invalidate the run, change any registered metric, or alter any gate. Agent A correctly reported 2,146 as a duplicate-row fraction; its report does not calculate the separate participation fraction. Agent B correctly distinguishes both descriptive rates. Neither rate is a registered gate.

All Gini, occupancy, unique-tuple, collision, conditional-entropy, and SID-export observations are descriptive only. The registered implementation defines no Stage2 SID gate, and the repository Stage2 policy states these metrics do not gate progression.

## 4. Preregistered prediction and causal interpretation limits

The S02 registered prediction concerns a protocol-valid Stage3 result: Iter31 `test_R@10` improving over Iter29's `0.05921064085377531` and exceeding the strict `0.065` target. Neither the observed Stage2 SID statistics nor the S08 direct-output difference establishes either prediction. Stage2 statistics are not a causal decomposition, quality score, or proxy-based promotion rule; this single run also does not estimate run-to-run uncertainty.

**Disposition.** The evidence supports a completed, numerically valid, contract-valid, non-aborted Stage2 candidate with consistent SID output wiring and the registered S08 direct effect observed on its specified batch. No Stage2 metric gate applies. No Stage3 outcome is known, and S10 makes no promotion decision. Stage3 remains unauthorized unless and until the separate S11 Judge approves its wiring/evaluation stage. The valid non-aborted candidate proceeds to that separately adjudicated S11 stage; this report itself authorizes no Stage3 launch.
