## Stage3 result and closing verdict

**Stage3** `TIGER_calibrated_kappa_rq`, single-card A100 (owner authorisation), 150 epochs,
seed 42, beam 20, frozen protocol, 4870.9 s.

| metric | value |
|---|---|
| test R@10 | `0.045230592455` |
| test R@5 | `0.030049269660` |
| test NDCG@10 | `0.024823080495` |
| frozen TIGER R@10 | `0.05520639286895663` |
| #265 parent R@10 | `0.052594926791900976` |
| target | `> 0.065` |

Independent replay (`tools/iteration/replay.py`): `PASS`.

**Verdict: negative recommendation result.** R@10 `0.045231` is below the #265 parent
`0.052595` (`−0.007364`) and the canonical frozen TIGER `0.055206` (`−0.009976`); the `0.065`
target is not met. The five historical TIGER reruns span `0.040199`–`0.055537`, so the baseline
itself moves by more than a thousandth; the loss vs the parent is still larger than that band.
The mechanism improved the continuous A.1 ranking (hard-q Spearman 0.353 vs Z_ref 0.084) but
compressed the discrete SID (6,246 vs 10,918 L2 prefixes) and lost recommendation. Recorded as
a negative result, not tuned further.

**Answers to the four issue questions:** see `results/research/issue269/VERDICT.md`.

**Delivered:** `results/research/issue269/` (RUN_LOG, CONTRACT, DIAGNOSIS, HYPOTHESES, INFRA,
STAGE2_METRICS, VERDICT, gate/fit/validation scripts, analysis JSON), the calibrated κ cache
under `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/`, the candidate
tree `stage2_RQ-VAE/TIGER_calibrated_kappa_rq/`, Stage2 run
`stage2_bba78cf7c160467da1c8ae7de9b6e26f`. R2 `r2:genrec/111/`: `stage2_source.tar.gz`
`b6fc4901…`, `item_sids.json` `a89110b8…`, `rqvae_best.pth` `896e69b7…`, `recall.json`
`78e68902…`, `archive_binding.json` `4e51a695…`; ledger updated (`results/research/issue139/r2_ledger.json`).

**Commit:** `80466a12` (mechanism + Stage2 evidence) and the verdict commit below.

**Closure:** every acceptance item C1–C15 has a current PASS; no unrun item is reported as a
pass; public infra hashes are unchanged (`INFRA_SNAPSHOT_END.json`).
