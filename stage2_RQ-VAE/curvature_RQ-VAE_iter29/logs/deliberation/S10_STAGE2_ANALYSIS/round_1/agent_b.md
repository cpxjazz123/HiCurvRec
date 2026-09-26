ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=logs/deliberation/S10_STAGE2_ANALYSIS/round_1/source_packet.md
STAGE_ID=S10_STAGE2_ANALYSIS

# S10 Stage2/SID analysis — Agent B

## Method

I read the canonical S10 source packet and primary S09/S08 evidence, then inspected `modules/sid_quality.py:90-144` for the exact metric definitions. I independently loaded the final `sids_raw.npy` and recomputed metrics from its rows: full-SID tuple occupancy counts and `_gini_from_counts`; each token column's occupancy Gini; number of distinct 3-token tuples; distinct `(L0,L1)` pairs; and the weighted base-2 conditional entropy `-sum((count/N) * log2(count/count_of_L0))`. The available Eval Python lacked PyTorch, so importing the whole `sid_quality` module (which imports torch) failed; I used its inspected NumPy/Counter definitions directly without modifying project code.

I also independently loaded the 4-token NPY and JSON, compared every raw 3-token row to the first three columns of the 4-token NPY, compared every JSON row/key to its indexed NPY row, counted collision extensions, checked full 4-token row uniqueness and raw token bounds, and loaded the final checkpoint with the project's Python 3.9/PyTorch runtime to inspect its step and fixed-curvature buffers.

## Computed SID facts

Canonical final raw file: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/sids_raw.npy`.

- Raw shape/dtype: `(24587, 3)`, `int32`; observed token range `[0,255]`.
- Full 3-token tuple Gini: `0.0877262220772868` (log rounds to `0.0877`).
- Per-layer Gini, L0/L1/L2: `[0.17491728987879773, 0.3076309698722902, 0.3610293021100741]` (log rounds to `[0.1749, 0.3076, 0.3610]`).
- Distinct 3-token tuples: `22301 / 24587`.
- Distinct `(L0,L1)` pairs: `12960`.
- `H(L1|L0)`: `5.30124157008481` bits (log rounds to `5.3012`).

These independently recomputed values agree with the final worker-log record at `logs/train_migrated.log:263-264` and the output summary at `logs/stage2_output_integrity_iter29.log:20` within the log's four-decimal rounding. They are descriptive statistics, not thresholds or a Stage2 quality gate.

## Collision-extension and export alignment

The exporter implementation in `curvature_RQ-VAE.py:428-443, 465-479` groups identical 3-token tuples in input order, preserves their three-token prefix, and assigns the fourth token as `sum(codebook_sizes) + occurrence`; with three 256-entry codebooks, the default extension is 768. It also asserts that extended 4-token rows are unique. My independent artifact checks found:

- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`: shape `(24587,4)`, `int64`.
- Its first three columns equal the complete raw NPY exactly, row for row.
- All `24587` four-token rows are unique; all `2286` repeated-prefix excess rows have extension IDs greater than the default 768. Observed extension values are exactly `768..780` (13 IDs); 1645 distinct 3-token prefixes have multiple extension values.
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`: 24587 keys, exactly string indices `0..24586`; every JSON value matches the same-index 4-token NPY row. This is stronger than the exporter's in-code three-row sample check.
- The worker log records `[export] SID_WIRING_PASS` at `logs/train_migrated.log:265-268`; the independent integrity record also reports complete row-by-row JSON/NPY equality and freshness (`logs/stage2_output_integrity_iter29.log:24-32`).

## Run, contract, and provenance evidence

- The supervised run record states a single authorized invocation, `restarts=0`, `exit=0`, and 17m16s uptime. Its port-50200 TCP readiness check **timed out after 120 seconds while the process remained running**; this was a readiness timeout, not a worker failure or nonzero exit. Worker evidence subsequently showed DDP rank `0/4`, ongoing progress to step 100000, and completion. Do not erase the warning, but distinguish it from the actual completed run (`logs/stage2_output_integrity_iter29.log:3-10,17`; final worker markers at `logs/train_migrated.log:260-269`).
- The worker log confirms 11 tensors loaded from the registered iter8 warm-start checkpoint (`logs/train_migrated.log:16`). It records the candidate closed-form vector and invariance markers at steps 0, 25000, 50000, and 100000 (`logs/train_migrated.log:15,19-20,84,141-142,260-261`).
- Final checkpoint `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/rqvae_best.pth` independently loaded successfully: `global_step=100000`; its `model` state has `layers.0._fixed_c`, `layers.1._fixed_c`, `layers.2._fixed_c` values `[1.3660953044891357, 0.7347829937934875, 0.6439958214759827]`, and no `c_layer_scale` keys. These are float32 checkpoint representations of the locked vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. This supports fixed-curvature checkpoint consistency; full implementation/activation evidence is in the adjudicated S08 MVG, not inferred from the SID statistics.
- `logs/stage2_preflight_iter29.log:15-20` records final prelaunch SHA-256 values for the configured Stage1 embedding, item-ID sidecar, Stage0 train parquet, and iter8 warm-start checkpoint, each matching canonical S01 as recorded. This is separate and stronger immediate-prelaunch identity evidence than S08 had; it does not retroactively fill S08's gaps.
- Keep the S08 limits explicit: `logs/mvg_check_iter29.log:7-11` says Stage1/Stage0 files were **not rehashed for S08**, and no batch fingerprint was emitted. S09's final prelaunch rehash does support input-byte identity at launch, but neither S08 nor these hashes establish a fingerprint for the actual MVG batch. The S08 counterfactual/MVG remains implementation-and-activation evidence only, not Stage2 or downstream effectiveness evidence (`logs/deliberation/S08_MVG/round_2/judge.md:17-25`).
- `logs/stage2_output_integrity_iter29.log:24-33` records output hashes/freshness, successful integrity smoke, no Stage3 launch, and no source-tree model/array artifacts. Worker log has final checkpoint/raw SID/export and completion markers; the run record reports no traceback, runtime, floating-point, or CUDA error markers.

## Risks and interpretation limits

The port-readiness timeout is operational evidence worth retaining, but the supervisor-observed worker completion and exit 0 plus final outputs do not support classifying this as a failed Stage2 execution. Input hashes establish the named files' prelaunch byte identities relative to S01; they do not independently fingerprint the selected S08 batch. The training log's periodic SID reports are logged descriptive measures; only the final raw SID array above was independently recomputed here. Gini, unique counts, collision statistics, entropy, and S08 quantizer counterfactual values must not be used as gates or as proxies for recall. No Stage3 evaluation has run and no recall/effectiveness conclusion follows.

## S10 verdict recommendation

**Recommend S10 PASS for Stage2 execution/contract/export readiness.** The available direct run and checkpoint evidence supports completion at the locked 100000 steps with fixed-curvature snapshots; independent recomputation agrees with the final worker metrics, and the 3-token source, 4-token export, and full JSON index are aligned and covered. Preserve the documented 120-second readiness timeout and S08 provenance/batch-fingerprint limitations. No Stage2 proxy is a gate under the active policy.

**Next authorized action:** proceed to the independent S11 Stage3 wiring/evaluation audit and adjudication. Stage3 is the single authorized next pipeline stage after S10 adjudication; do not launch Stage3 in this S10 candidate task. No recall or effectiveness claim is made.