# Failure attribution — iter18

## Decision

**Selected outcome: `TRUE_MECHANISM_FAIL`, qualified as this run's downstream target miss—not proof of a causal mechanism failure.** The complete Stage3 run used the registered iter18 SID file and missed strict `test_recall@10 > 0.065`. Evidence below does not isolate the optimizer change's causal contribution. `training_metrics.jsonl` records `variant="unknown_variant"`; its `code_path` does identify the iter18 SID JSON, so the mechanism label itself remains a metadata caveat.

## Evidence and attribution checks

### Implementation — pass; `IMPLEMENTATION_FAIL` not supported

`mvg_check_iter18.log` reports `MVG PASS` on one iter8 checkpoint and one `(640, 768)` batch with 640 active pairs. The fixed per-layer AdamW `β₂` values were `[0.998991, 0.990604, 0.990001]`; all registered loss components retained gradient paths, and all three quantizer embeddings had nonzero gradients. Five-step relative parameter updates were `[0.02443, 0.23058, 0.25270]`. Stage2 reached global step 100,000, saved its final checkpoint, and exported four-token SIDs with `SID_WIRING_PASS`. The Stage2 training log had no `Traceback`, `ERROR`, `FAIL`, `NaN`, or `Inf` matches. These checks provide no evidenced implementation failure.

**Provenance caveat:** iter18 lacks a local `layer_norms.json`; the existing fallback used iter1 residual scales `[0.001, 0.932889, 1.0]`. This was recorded before Stage2 training and is not evidence that the implementation failed.

### Activation — registered direct effects pass; `ACTIVATION_FAIL` not supported

The registered mechanism checks in `mvg_check_iter18.log` show:

- Initial curvature `[0.050085, 0.244323, 0.273815]` produced normalized scales `[0.001, 0.932889, 0.9999]` and fixed `β₂` `[0.998991, 0.990604, 0.990001]`.
- Required loss-component gradient paths were present; quantizer gradients were nonzero at all three layers.
- Five-step relative updates were nonzero on every layer: `[0.02443, 0.23058, 0.25270]`.
- Same-seed 200-step ON/OFF losses were `2.47846961` and `2.46642065` (`|Δ|=0.01205`), above the registered `1e-6` threshold.
- The cyclic-curvature checks at steps 0, 50,000, and 100,000 were finite and returned to their initial values at the cycle boundary.

These checks establish the registered direct effects, not that they should improve downstream recall.

### Geometry — descriptive metrics only; `GEOMETRY_FAIL` not supported

The Stage2 summary reports `full_gini=0.0642`, layer Ginis `[0.1896, 0.2173, 0.2015]`, 22,951 unique three-token SIDs, 14,812 unique `(L0,L1)` pairs, `H(L1|L0)=5.5907` bits, and 1,636 three-token collision rows. These values describe the exported SIDs; they neither gate Stage3 nor establish downstream causality. `hitrate@50=0.9966` is a descriptive log field and was not used as a gate.

### Pipeline — completed on the registered input; `PIPELINE_FAIL` not supported, with metadata caveats

The Stage3 launch audit records the iter18 `item_sids.json`, four-GPU launcher, port 50201, and required NCCL environment settings. The first `training_metrics.jsonl` event records `world_size=4` and `code_path=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter18/item_sids.json`; the launcher reports `n_digit=4`, beam 20, and rank IDs 0–3. The run completed all 150 epochs and produced the final test event for `n_eval=57439`.

Two warnings remain documented: the metrics event's `variant` is `unknown_variant` although the wrapper sets `iter18_initial_curvature_beta2`, and the launcher reports NCCL wrapper initialization warnings plus a `destroy_process_group()` warning at exit. The final evaluation exists, and a targeted search of `HG_Rec.log` and `_stage3_launcher.log` found no `Traceback`, `RuntimeError`, `ERROR`, or `FAIL` markers. These caveats do not evidence wrong SID input or a failed evaluation.

### Observed downstream outcome

`test_final.json` reports `test_recall@10=0.05988962203380978`, `n_eval=57439`; the strict target shortfall is `0.005110377966190224`. The score is higher than iter11 by `0.00012186841692925915`, iter16 by `0.0030118908755375207`, and iter17 by `0.0008182593708107727`. These are observed comparisons, not causal estimates.

## Alternative attribution classes

- **`IMPLEMENTATION_FAIL` — not selected:** MVG passed; Stage2 completed and exported the expected four-token SID data.
- **`ACTIVATION_FAIL` — not selected:** all registered direct effects passed their thresholds.
- **`GEOMETRY_FAIL` — not selected:** geometry values are descriptive and are not a Stage2 gate; no causal link is established.
- **`PIPELINE_FAIL` — not selected:** the run used the iter18 SID path and completed the final test. The unknown variant label and cleanup warnings remain caveats, not demonstrated score corruption.
- **`TRUE_MECHANISM_FAIL` — selected with qualification:** factual finding is that this completed run missed the fixed downstream target. It does **not** prove that P18B caused the score, has no effect, or that the target is unreachable.

## Evidence sources

- `logs/mvg_check_iter18.log`; `logs/train_migrated.log`; `logs/sid_geometry_iter18.md`; `logs/hypothesis_iter18.md`
- `scripts/run_stage3_iter18.py`; `logs/stage3_launch_iter18.log`
- `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/{training_metrics.jsonl,HG_Rec.log,test_final.json}`
- `logs/stage3_outcome_iter18.md`
