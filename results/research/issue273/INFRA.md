# Issue #273 — infra inventory (FROZEN)

Baseline snapshot: `INFRA_SNAPSHOT_START.json` (61 files, schema `genrec.issue259.infra_snapshot/1`).
Recompute at the two remaining checkpoints (before the formal dispatch and before archiving) with
`tools/iteration/integrity.py::snapshot` + `compare`; the change list must be empty.

## Reused assets (read-only, hash-frozen)

| asset | path | sha256 / identity |
|---|---|---|
| frozen per-triple kappa_t cache | `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/` | manifest `4b1d777bee24c0743239076a937d6a38bfc220c542908ccac53ce5b501b2fcbe`; files `kappa.npy`, `j_fit.npy`, `j_zero.npy` |
| frozen local evaluator (ruler) | `results/research/issue271/local_geometry.py` | `6260b8bb89bcd9b1f359a0bdc36652c2ce82a8d1f5e7c79efbec885b61b3727a`; scale `0.08433422446250916` |
| SID metric collector | `results/research/issue271/collect_metrics.py` | `390f655cb97dd38daaa49675d189898dd191cc7715203b48e8d51a5ad3389d9f` |
| graph/triplet assets | `assets/manifest.json` + `assets/train/`, `assets/shards/` | manifest `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b` |
| #269 baseline SID | `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/item_sids.json` | `a89110b86168eb09621e5f9cfea6f6668253d6105e9f2a6647234b195efc725f` |
| #269 baseline checkpoint | `results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/rqvae_best.pth` | `896e69b7bfc701a4fbd7413190cf8aadf0334b43d935ef3cbf66a73aae3c7410` |
| #271 geometry (baseline metrics) | `results/research/issue271/assets/geometry/*.json` + `*.npz` | MAE 1.8935063111536612 (`TIGER_global_j_rq`), sign/Spearman per arm |
| common-ruler J inventory | `results/research/issue272/ruler_inventory.json` | #269 full J 0.338335882630658, 1-hop 0.760281, 2-hop 0.264863 |
| frozen TIGER baseline | `results/stage2_RQ-VAE/TIGER_RQ-VAE/` | MAE 7.309217352939646, Spearman 0.015850898303185326, sign 0.4324533490974272, L2 7,374, rank 11.5, J 0.739635 |

## Policy

- **No new public infrastructure, no new public assets, no modification of any file above.** The
  issue-273 additions are candidate-tree code, candidate tests and research evidence under
  `results/research/issue273/`.
- The issue-273 analysis script (within-centre difference RMSE, within-centre ordering, variance
  ratio, slope) is a **research analysis** that reuses the frozen evaluator's assets, scale and
  algorithm; it does not modify `results/research/issue271/local_geometry.py` or any public
  evaluator. It must reproduce the frozen evaluator's MAE/Spearman/sign exactly on the frozen
  baselines before its new statistics are trusted (self-check).
- The Stage3 entry, the Skill, `CLAUDE.md`, `tools/iteration/`, `tools/research/stage2/`,
  `tools/research/verification/` and `assets/manifest.json` are hash-frozen for the whole issue.
- The #272 formal run in flight (`stage2_0a2be6e306e145fe91b3d7be08a37c69`) uses its own immutable
  source snapshot; the #273 tree is created only after that tree is archived (§1.0a, top level keeps
  one active mechanism).
