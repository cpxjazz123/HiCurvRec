# Issue #269 Stage2 report

Run `stage2_bba78cf7c160467da1c8ae7de9b6e26f`, arm `TIGER_calibrated_kappa_rq`,
single-card A100-80GB (owner authorisation), global batch 4096, seed 42, 3000 epochs,
port 50202. Source snapshot
`results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/src`,
checkpoint sha256 `896e69b7…`, SID sha256 `a89110b8…`.

## Training

| item | value |
|---|---|
| epochs | 3000 (no early stop) |
| wall time | `1213.8 s` |
| mean epoch | `0.392 s` (median of all 3000 epochs) |
| final loss | `1.335e-4` |
| final recon | `9.537e-5` (#265 parent final `1.23e-4`) |
| final curvature estimate | `1.546` (graph target RMS `2.0004`) |
| final gradient norm | `2.74e-4` |
| final LR | 0 |
| peak allocated | `610,671,104 B` |
| catalogue 3-code collision | `0.1567` (min after epoch 1000: `0.1509`) |
| sampler | 6 updates/epoch, 648 centres/epoch, 14,013 unique references, 24,554 aux items |

Curvature calibration on the first real batch: `lambda=5.8836e-6`,
encoder `||g_orig||=1.024e-4`, `||g_curv||=6.190`, weighted ratio `0.3555`;
codebook ratios `0.706 / 1.000 / 0.453`. `tangent_scale=0.082563`,
`target_scale=2.000375`.

## Exported SID (final, post export gate)

`item_sids.json` sha256 `a89110b86168eb09621e5f9cfea6f6668253d6105e9f2a6647234b195efc725f`.
Independent recomputation (`analysis/SID_RECHECK.json`):

| metric | value |
|---|---|
| items | 24,587 |
| distinct L1 | 256 |
| distinct L2 prefixes | 6,246 |
| distinct L3 triples | 20,744 |
| L3 collision | `0.156302` |
| H(L1) / H(triple) | `6.7703 / 14.2065` bits |
| L3 refinement gain | `2.8495` bits (parent `1.6114`) |
| L1 positive hit / negative mass / hard ER | `0.08360 / 0.01461 / 5.722` (parent `0.06241 / 0.00721 / 8.656`) |
| conditional log enrichment | `1.6368` bits (parent `1.5353`) |

Trainer SID (`item_sids.trainer.json`, sha256 `f32ee975…`) differed on 10 triples
(collision `0.156709`); the export gate re-derived the table from the checkpoint.

## Geometry diagnostics

Hard q vs frozen `Z_ref` under the trained per-triple κ and `S*`
(20,000 rows, independent references; `analysis/OWN_KAPPA.json`,
`analysis/A1_ZREF.json`):

| space | Pearson | Spearman | sign acc | MAE | mean J |
|---|---|---|---|---|---|
| `Z_ref` | `0.0927` | `0.0838` | `0.548` | `1.698` | — |
| hard q | `0.3336` | `0.3531` | `0.684` | `1.686` | `0.2883` |
| hard z | `0.3475` | `0.3678` | `0.6926` | `1.685` | `0.2883` |

Independent fixed Euclidean / equal-weight three-expert, full 28,391,948 rows
(`analysis/CURVATURE_FULL_TIGER_calibrated_kappa_rq_{Q,Z}.json`):

| space | Pearson | Spearman | sign acc | MAE | RMSE |
|---|---|---|---|---|---|
| q | `0.0272` | `-0.1110` | `0.3632` | `2.0168` | `2.4698` |
| z | `0.0245` | `-0.1189` | `0.3620` | `2.0151` | `2.4692` |
| #265 parent q | `-0.0331` | `-0.0934` | `0.3323` | `2.2460` | — |

Graphlet/AF3 fidelity on the canonical graph (`analysis/LOCAL_STRUCTURE.json`): L1
triangle-leaf vs open-wedge-leaf code sharing `0.08945` vs `0.02736` (`3.27x`); frozen
TIGER `0.0990` vs `0.0240`; AF3 strata per degree/triangle bucket are in the file.

## Cost

- Offline fit: `115.3 s`, 325 MB cache (κ, J0, Jfit), 211-point grid.
- Stage2: 20.2 min wall on one card; no four-card comparable run exists for this arm.
- SID quality gate: collision `0.1563 < 0.40`, quant_error `0.00875` (below the 0.035
  diagnostic band), effective rank `4.33 < 8.0`.
