## Gate and calibration result (step 1-2)

**INFRA_MODE = FROZEN.** No public infra or asset was modified. Issue body sha256
`407618e5…` (2635 chars) frozen in `RUN_LOG.md`; acceptance C1–C15 extracted verbatim.
Training is required by step 3–4 (Stage2 3000 epochs + Stage3 150 epochs), so the round
is not analysis-only.

**Global-scale check.** The weighted least-squares optimum of
`J(c) = Σ(c·a − 1)²`, `a = ||Z_i−Z_j||/(edge_scale·g)`, reproduces #268's
`S* = 0.06828367045803868` from the closed form `c* = Σ(t1+a_bc)/Σ(t2+a_bc²)`:
`scale_reproduced = 0.06828365095335823`, relative error `2.86e-7`
(`results/research/issue269/analysis/GATE.json`). No hard-coded value is used in the fit;
`S*` is frozen in `CONTRACT.md`.

**Pre-registered identifiability gate** (fit on reference set A seed 268, evaluate on the
independent set B seed 269, 2048 rows, 211-point grid `[-0.05, 1.0]` step 0.005):

| criterion | value | pass |
|---|---|---|
| mean held-out J improvement vs one global κ | `0.048371` (4.84 %) | yes |
| centre bootstrap 95 % CI | `[0.041905, 0.055416]` | yes (lower > 0) |
| reference-perturbation Spearman | `0.946566` | yes (≥ 0.5) |
| boundary fraction A / B | `0.1543` / `0.2773` | reported |

Held-out J: κ=0 `0.181664`, one global κ `0.182597`, per-triple κ `0.173216`. A free
per-row scale still reaches `0.148579`, so the fitted κ is a partial local-shape correction,
not a complete geometric model (`H269-3`).

**Full cache.** 28,391,948 rows refitted at `S*` (115.3 s, 325 MB,
`results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/`): mean J
`0.178983 → 0.168399` (5.91 % mean, 2.97 % median), boundary `17.7 %`, ties `2.5 %`,
κ mean `0.0288`, std `0.0642`, positive `60.7 %`. Contrast #268's 99.78 % floor.

**Mechanism bug fixed (#268's NaN/floor stall).** The curvature term's encoder gradient
was NaN: `project` put `inf` in the unselected `torch.where` branch for `k ≥ 0`, and
`sqrt(0)` had an infinite derivative. Fixed with a finite radius placeholder and an exact
forward/floored backward `_SafeSqrt`. The real 60-epoch single-card probe at batch 4096
passes (59 s); `assignment_floor` failure was downstream of the same NaN. Candidate suite
passes (121 tests).

**Stage2 dispatch.** One candidate (`TIGER_calibrated_kappa_rq`, per-triple κ at `S*`;
the gate passed, so the κ=0 degenerate arm is not trained). Run
`stage2_bba78cf7c160467da1c8ae7de9b6e26f`, single-card branch (owner authorisation),
port 50202, batch 4096, seed 42, 3000 epochs. Formal training is live.

Evidence: `results/research/issue269/{RUN_LOG,CONTRACT,DIAGNOSIS,HYPOTHESES,INFRA}.md/json`,
`analysis/GATE.json`, `analysis/PER_ROW.npz`.
