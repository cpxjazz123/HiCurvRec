# S10 Source Packet — iter28 Stage2 Analysis

Stage2 completed 100,000 steps and produced `sids_raw.npy` (24587, 3) and `sids_for_hgrec.npy` (24587, 4).  Iter28 SIDs consumed by Stage3 (which ran successfully to completion).

Primary evidence:
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/out/rqvae/instruments/rqvae_best.pth`
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/out/rqvae/instruments/sids_raw.npy`
- `logs/sid_geometry_iter28.md`
- `logs/mvg_check_iter28.log`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/train_migrated.log`

Canonical baseline: iter18's `results/stage2_RQ-VAE/curvature_RQ-VAE_iter18/out/rqvae/instruments/sids_raw.npy` (same protocol).