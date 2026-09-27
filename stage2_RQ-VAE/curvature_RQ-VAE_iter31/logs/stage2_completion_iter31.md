# Iter31 Stage2 completion and artifact verification

`STAGE_ID=S09_STAGE2_EXECUTION`
`RUN_STATUS=COMPLETED`
`SUPERVISOR=Iter31Stage2`
`SUPERVISOR_EXIT_STATUS=0`
`SUPERVISOR_RESTARTS=0`
`STAGE3_AUTHORIZATION=NO`

## Training contract

`logs/train_migrated.log` records `[train] done at global_step=100000, total time=1157.0s`; the supervised launcher exited 0. Four DDP ranks ran with seed 42, per-rank batch 640/global batch 2,560, and no compile. The final warm start loaded 11 tensors from the pinned Iter8 checkpoint; no random-init fallback warning occurred. Step0 reported zero stale artifacts cleaned.

FCCR-1 remained invariant at the final step: `c_live=[1.366095, 0.734783, 0.643996]`, equal to the closed-form target within `1e-6`. The final checkpoint save reports `step=100000`. A CPU-only `torch.load` through the prescribed Python 3.9 environment independently confirmed checkpoint `global_step=100000`, keys `global_step/model/optimizer`, and 14 model tensors.

The final exporter reported `SID_WIRING_PASS`. The log contains no `Traceback`, `RuntimeError`, `ERROR`, `Stage3 export FAIL`, or warm-start-missing warning. Both logs are retained: `logs/train_run.log` (empty outer redirection target, as the script writes subprocess output to the inner log) and `logs/train_migrated.log` (52.8 KiB trainer/torchrun output).

## Final SID outputs

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/rqvae_best.pth` | 13,778,293 | `d07953be968ea02cb8a6183935264c2af6315ea5195bd90c40764f39279961b7` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/sids_raw.npy` | 295,172 | `bc5db3c9cc28e490cec8ecef0ec14ad135df027b8841d61399cafffde95cfaa5` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy` | 786,912 | `398c13bd7439d87df01e197dbdf447162c5a32e1149be11d11a16c389166f138` |
| `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json` | 1,258,304 | `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7` |

CPU artifact audit passed: raw SIDs are `int32 [24587, 3]`, final SIDs are `int64 [24587, 4]`, JSON has exactly ordered keys `0..24586` with width 4, the full JSON row matrix equals the exported NPY, and the first three final columns equal raw SIDs. Raw codes are within `[0,255]`; raw unique rows are 22,441; all 24,587 four-token rows are unique; collision-extension values span `[768,778]`.

The final trainer metrics are descriptive only (no gate): `full_gini=0.0824`, per-layer Gini `[0.1737, 0.2500, 0.2906]`, 22,441/24,587 unique three-token rows, `l01_pairs=13052`, `H_l1_given_l0=5.3438`. No early stop or SID-quality threshold was used.

All model/SID products are under the Iter31 `results/stage2_RQ-VAE` short-name root. The source iteration contains no `.pth`, `.npy`, or `item_sids.json` products. No Stage3 process was launched; Stage3 still requires its separate S11 adjudication.
