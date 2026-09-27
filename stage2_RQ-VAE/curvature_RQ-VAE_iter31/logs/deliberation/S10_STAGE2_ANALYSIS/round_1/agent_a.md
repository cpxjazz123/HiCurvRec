ROLE=AGENT_A
STAGE_ID=S10_STAGE2_ANALYSIS
ROUND=1
SOURCE_PACKET=logs/deliberation/S10_STAGE2_ANALYSIS/round_1/source_packet.md
SOURCE_PACKET_SHA256=86d6666ca25d433865c7788d1aab5265f0393106d21c23d26594149758c01688
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
RECOMMENDATION=ANALYSIS_PASS
CONFIDENCE=HIGH for independently checked artifact identity, shape/wiring, and registered contract evidence; MEDIUM for any downstream causal interpretation (which is not established in S10).

# S10 Iter31 Stage2 analysis

## Evidence register and audit method

The frozen packet hash above matches the project copy at `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S10_STAGE2_ANALYSIS/round_1/source_packet.md`. I independently read the canonical S02 hypothesis, S04 HRA/FCCR contracts, S05 one-factor boundary, S08 MVG record, S09 execution/completion records, final trainer log, and the registered SID metric implementation. I loaded the actual NPY/JSON files using CPU NumPy/JSON, computed SHA-256 from the files, recomputed the registered descriptive metrics with the same counting/Gini/entropy formulas in `modules/sid_quality.py::evaluate_sid_quality`, and loaded the checkpoint with CPU `torch.load(map_location='cpu')`. No GPU, training rerun, Stage3 action, model/source/result edit, or project-wide check was used.

Primary evidence and independently checked SHA-256 values:

- Frozen packet: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S10_STAGE2_ANALYSIS/round_1/source_packet.md` — `86d6666ca25d433865c7788d1aab5265f0393106d21c23d26594149758c01688`.
- Registered hypothesis: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/hypothesis_iter31.md` — `3ec1230e02736476fd4d13aa56e1b30a29ac3313f5aa44ab39a955e6b7099ea4`.
- HRA Step6 contract: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/hra_step6_contract_iter31.json` — `3e7394e91ed3f31ec4d2753e6d634c6e99590e18e5e4da98d975df0c2ae5cf17`.
- FCCR-1 contract: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/mechanism_contract_iter31.json` — `0ea83b1f55916d02fa3eed97e02d4c39f46d7229a0afa76d5fe1262d96459b32`.
- Final run log: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/train_migrated.log` — `6a730ae334ad7731240045fbda015ad616a19f9f9d8ec52682cb998588347303`.
- S08 direct-effect / contract evidence: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/mvg_check_iter31.log` — `49318b59588ce9339e2173956733bf7b699fac4d5d113b3921bb77e7e8b59031`.
- Actual checkpoint: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/rqvae_best.pth` — `d07953be968ea02cb8a6183935264c2af6315ea5195bd90c40764f39279961b7`.
- Actual raw 3-token SID: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/sids_raw.npy` — `bc5db3c9cc28e490cec8ecef0ec14ad135df027b8841d61399cafffde95cfaa5`.
- Actual exported 4-token SID: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy` — `398c13bd7439d87df01e197dbdf447162c5a32e1149be11d11a16c389166f138`.
- Actual full item mapping: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json` — `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`.

## 1. FCCR-1 contract compliance and actual run validity

**Observed facts.** The registered machine contract declares `contract_version=FCCR-1`, closed-form curvature, non-trainable and non-time-varying curvature, no cyclic schedule or curvature regularization, and final values `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. The HRA contract explicitly inherits and preserves that FCCR sub-contract. Its registered sole model-behavior change is the Step6 replacement: for ordered L0/L1/L2 tangent-coordinate embeddings, map to each layer's ball, radially transfer to `c0`, compute `h=q0^0 ⊕_c0 (q1^0 ⊕_c0 q2^0)`, then `z=log0^c0(h)` for the unchanged decoder; it does not claim exact paper HRA telescope or permit reassociation/reordering.

The final log reports the closed-form and live curvature values matching at step 0, 10,000, 25,000, 50,000, and 100,000; each recorded invariant check states it matches within `1e-6`. The live vector is consistently displayed as `[1.366095, 0.734783, 0.643996]`. The log records `curv_reg=0.000000`; S08 additionally reports fixed-curvature buffers excluded from optimizer parameters, `requires_grad=False`, and invariant snapshots at train steps 0/25k/50k/100k (plus eval checks). S08's registered gradient evidence shows finite nonzero gradients on intended encoder/codebook/decoder parameters and `MVG PASS`. The final Stage2 run reached global step 100,000, took 1157.0 s, saved the final checkpoint, exported the SIDs, and logged successful completion. The completion record reports supervisor exit status 0, zero restarts, and no Stage3 process. The run log's final line is the 100,000-step completion record.

CPU inspection of the actual checkpoint independently found `global_step=100000`, keys exactly `global_step`, `model`, `optimizer`, and 14 tensors in `model`. Its hash matches the packet/completion evidence. The actual product hashes also match the frozen packet and completion record.

**Assessment.** The evidence supports FCCR-1 numerical invariance and a completed, non-aborted Stage2 run. No contradiction was found between the declared fixed values, recorded checkpoints, and the final checkpoint step. This is contract compliance evidence, not proof of recommendation performance.

## 2. Registered HRA Step6 direct effect versus Euclidean (S08 same-batch evidence)

**Observed facts.** S08's `MVG PASS` record compares the registered HRA operation with the Euclidean tangent sum using the same quantized embeddings and the same 640 ordered source/target pairs. Both outputs are finite and shape `[640,32]`. Recorded HRA and Euclidean norms are `25.257963180541992` and `24.896957397460938`; maximum absolute difference is `0.07198049873113632`, L2 difference is `2.018216848373413`, and relative L2 difference is `0.08106279373168945`. The record marks the direct-effect threshold as “none preregistered; report_only”; common-ball maximum `sqrt(c)||x||` is `0.855758547782898`, inner/common ball domains are valid, and projection-incidence diagnostics are present. S08 also records intended nonzero gradients and FCCR fixed-curvature/optimizer checks.

**Interpretation.** This is direct evidence that the registered Step6 changes decoder-facing outputs on that same-input batch, so the operation is not numerically identical to the Euclidean sum there. It is not evidence of a Stage3 gain, nor does a single counterfactual batch quantify downstream quality. The report-only difference has no extra hard threshold.

## 3. Independent Stage2 SID observations and artifact wiring audit

**Actual file checks (CPU).** `sids_raw.npy` is `int32 [24587,3]`; observed per-column minimums are `[0,0,0]` and maximums `[255,255,255]`. `sids_for_hgrec.npy` is `int64 [24587,4]`. The JSON contains 24,587 rows with width 4 and sequential string keys from `0` through `24586`. Converting the JSON rows to an integer matrix yields exact equality with the exported NPY; the exported first three columns equal the raw NPY exactly. Every exported four-token row is unique. The extension column ranges from 768 to 778. Thus raw SID → four-token export → JSON row wiring and collision disambiguation are consistent for these artifacts.

**Recomputed canonical descriptive metrics.** I recomputed the `evaluate_sid_quality` implementation from `modules/sid_quality.py` on the actual raw array: `n_items=24587`; `n_unique_full=22441`; `full_gini=0.08242568913963332`; per-layer Gini `[0.17365233024565827, 0.25000365411599623, 0.29055099620734537]`; `l01_unique_pairs=13052`; `H(L1|L0)=5.343793126431163` bits. There are 2,146 repeated 3-token rows, giving duplicate-row fraction `2146/24587 = 0.08728189693740594`; the largest tuple multiplicity is 11. Each layer uses all 256 codes. These independent recomputations agree with the final trainer log's rounded values and packet values. `should_early_stop` in the registered implementation returns `(False, "")` unconditionally.

**Assessment / any mismatch.** No mismatch was found across the NPY/JSON mappings, raw-to-export prefix, actual file hashes, checkpoint step, packet claims, and rounded trainer metrics. All SID occupancy, collision, Gini, uniqueness, and entropy metrics are descriptive only. No Stage2 SID metric is a hard gate or promotion decision.

## 4. Causal interpretation limits and disposition

The registered S02 hypothesis predicted a protocol-valid Iter31 Stage3 `test_R@10` exceeding the Iter29 reference `0.05921064085377531` and strict target `0.065`. Neither the SID metrics nor S08's direct-output difference establishes either prediction. Stage2 shows a valid completed candidate with the registered operation's direct effect observed in S08 and a consistent SID export; it does not estimate the causal effect on recommendation quality, robustness across runs, or whether the downstream target is met. The Stage3 outcome remains unknown here.

`RECOMMENDATION=ANALYSIS_PASS` means only that S10's evidence audit found a numerically valid, contract-valid, non-aborted Stage2 result without an identified artifact-wiring contradiction. Stage3 evaluation remains separately adjudicated in S11; a numerically valid, contract-valid, non-aborted Stage2 candidate proceeds to that stage under the research workflow. **S10 does not authorize Stage3 or any S11 action**, and makes no promotion decision. No Stage2 metric gate, quality threshold, or claim of Stage3 benefit from a proxy is applied.

PARALLEL_GROUP_1=Independent frozen-packet A/B Stage2 analyses; CPU artifact/hash inspection and primary log/contract review were independent read-only evidence tasks.
PARALLEL_GROUP_2=No dependent execution stage was performed by Agent A.
SERIAL_DEPENDENCIES=No Stage2/Stage3 execution dependency was invoked; S11 remains separately adjudicated.
