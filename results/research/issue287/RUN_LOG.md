# Issue #287 — RUN_LOG

- Issue: #287 "Stage2-only: curvature-guided pair ranking with improved L1 positive retention, negative separation and SID collision"
- Body sha256: `952b1ec1d522d6c0411a94eca44c56e5e36c8eb14401203c7f89ce39c71b8234` (frozen 2026-11-28; the exact value is recorded in `issue287.json`).
- **INFRA_MODE = FROZEN** (issue: "INFRA_MODE=FROZEN;复用已有全量训练图、三元组/曲率/Graphlet资产、#274候选实现、#277评估器与原训练入口。不修改公共infra,不重建公共资产").
- **Training required: YES** — "完整正式候选3000 epochs", Stage2 only ("本Issue严禁启动Stage3/T5、新test或产生新真实Recall").
- New candidate tree: `stage2_RQ-VAE/TIGER_l1_pairrank_rq` (created from the frozen baseline
  `stage2_RQ-VAE/baseline/TIGER_RQ-VAE`; the #274 pair-ranking implementation is reused as the issue
  instructs, with the single main change of this round, see `CONTRACT.md`).
- §1.0a: the three remaining top-level mechanism trees (`TIGER_pairrank_rq`,
  `TIGER_pairrank_cover_rq`, `TIGER_pairrank_enc_rq`) were archived to `stage2_RQ-VAE/_archive/`
  before the new candidate was created; `baseline/` and `_archive/` stayed in place, no historical
  evidence was rewritten.
- Hardware: one A100-80GB visible (`nvidia-smi`), so the single-card branch runs with the frozen
  global batch 4096 (§19d0b); the four-card target of ~0.4 s/epoch is recorded as the reference and
  the single-card epoch time is reported separately.

## Frozen acceptance checklist (C1..C5)

| C | issue § | requirement | criterion |
|---|---|---|---|
| C1 | §1 | re-verify the checkpoint-exported SID identities with the #277 evaluator (R2 `9f6bf673…` P=0.0783613920855168 N=0.008752316503966327 C=0.1511367795989751; R3 `9d9f3d68…` P=0.08097285816257246 N=0.00901661164455637 C=0.1543498596819457); the candidate's exported SID must satisfy **P > 0.08097285816257246**, **N < 0.008752316503966327**, **C < 0.1511367795989751** simultaneously; P/N on the 57,439 #277 validation rows, C = 1 − unique(first3)/Nitems; no substitution of the #274 pair hit rates | all three strict inequalities on one and the same checkpoint export |
| C2 | §2 | curvature only for sample selection (never in the loss value, weight, margin, distance or code selection); real training-graph curvature proxy; fixed margin m > 0; L_geo = mean relu(d_pos − d_neg + m); no new curvature/global-distance/collision/auxiliary loss; L1 distance must be the **actually selected L1 codeword** through the legal codebook path; #274 code/semantics/selection audited; curvature's incremental information over closed/open labels, direct edges, common neighbours and degrees quantified; #274 failure localised (distance ranking → L1 hard selection) | every mechanism property asserted by a test or a measurement; the curvature's extra information reported, redundancy reported honestly |
| C3 | §3 | low-cost real-batch diagnostics first (curvature screening discrimination, whether the ranking acts on the L1 selection, encoder/three-codebook gradient norms and directions, post-step code changes, margin scale, reconstruction gradient budget); STE/codebook routing, flips, indices, curvature-invariance, overlap, NaN/collapse guards, real hard forward; then the full 3000 epochs; one main factor per round; P/N/C trajectories, occupancy, Hard ER, gradients, time and sample coverage recorded; the SID is re-exported from the checkpoint | the diagnostics exist before the formal run; the formal run is 3000 epochs at seed 42 and global batch 4096 |
| C4 | §4 | no Stage3/T5/new test/new real Recall; at least one complete 3000-epoch candidate must satisfy the three thresholds on one real checkpoint export; the best metrics of different candidates must not be stitched; delivery with DIAGNOSIS/HYPOTHESES/CONTRACT, sample/asset manifest, code and tests, persistent commands/exit codes/logs, real gradients/parameter updates, checkpoint/SID/export identity, per-round metrics and the final PASS table, the frozen #277 predictions (diagnostic), performance, VERDICT, infra drift 0, commit+push; no faked four-piece Stage3 archive | the PASS table names one candidate and one SID |
| C5 | §5 | the curvature-selection contribution still needs independent evidence; the real recommendation effect is validated only by a future Stage3 | stated as a limitation, not a claim |
