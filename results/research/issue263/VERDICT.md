# Issue #263 verdict — frozen topology GNN tokens

## Decision

**Completed with a negative recommendation result.** The pretraining/freeze path
was implemented and ran end to end, but the frozen token mechanism did not beat
the frozen comparison: Stage3 `test_R@10 = 0.0518985`, below both the parent R01
reference (0.052734) and the 0.065 target. This is a valid scientific result, not
a protocol failure; no retry, seed search, or test-guided adjustment was made.

## Evidence

| item | result | path / evidence |
|---|---|---|
| Pretraining | 1500 steps, lambda 1.278927 | `results/research/issue263/PRETRAIN.json` |
| Pretrain held-out CE / top1 | own 3.374/0.1106, dgdv 1.778/0.3433, curve 1.514/0.6280 | same |
| Token cache | 24587×4×32; max reload diff 6.62e-6 at tolerance 1e-4 | `PRETRAIN.json`, `token_manifest.json` |
| Stage2 | 3000 epochs, final collision 0.060764 after export | `_runs/.../result/logs/training_metrics.jsonl` |
| Export gate | VERIFIED; checkpoint reproduces SID; 220 live items rewritten | `export_manifest.json`, `archive_binding.json` |
| Layer record | collision 0.060723; L1 Hard ER 8.568; L1 positive 0.0639 | `stage2_metrics/a4d7…json` |
| Stage3 | 150 epochs, n_eval 57439, R@10 0.0518985, R@5 0.0342450, NDCG@10 0.0282147 | `logs/.../test_final.json` |
| R2 | four pieces uploaded | `r2:genrec/108/`, `r2_ledger.json` |
| Performance | Stage2 3000 epochs mean 0.3197 s; no online GNN | `results/research/issue263/PERFORMANCE.json` |

Comparison: frozen TIGER `test_R@10=0.055537`; prior mixed-F0 `0.057278`.
Issue target `>0.065` was not met. The Stage3 test is exploratory because all
historical arms have read it, but that does not change the negative sign.

## Hypotheses

- **H1 (end-to-end entanglement prevented stable structure): INCONCLUSIVE.** The
  frozen token path trained, but the experiment does not isolate whether the
  remaining performance loss is token quality, gate usage, or quantization.
- **H2 (explicit multi-role/curvature supervision improves tokens beyond
  degree/frequency): SUPPORTED for the structural probe, NOT for recommendation.**
  Held-out token probes beat degree/majority baselines (own 0.1106 vs 0.0854;
  dgdv 0.3433 vs 0.2968; curve 0.6280 vs 0.2765), but Stage3 did not improve.
- **H3 (freezing reduces online GNN cost without making SID unusable):
  SUPPORTED.** No online GNN forward/optimizer; Stage2 3000 epochs at 0.3197 s
  mean; collision 0.0607 < 0.40.
- **N1 (parent GNN already sufficient; freezing cannot improve): INCONCLUSIVE /
  consistent with the observed negative Stage3.** No single-factor causal claim is
  made; this was one frozen run.

## Acceptance checklist

| ID | Status | Note |
|---|---|---|
| C1 independent pretraining | PASS | `PRETRAIN.json`, one run |
| C2 80/20 item-ID split, no valid/test rows | PASS | fit 19669 / eval 4918; scope recorded |
| C3 TRAIN-only orbit/DGDV/curvature targets, mask | PASS | labels tests and `HYPOTHESES.json` |
| C4 learned distributions, gradients/parameters update | PASS | held-out metrics; calibration gradients nonzero |
| C5 held-out evaluation | PASS | `PRETRAIN.json` metrics |
| C6 freeze/cache/reload | PASS within stated tolerance | max diff 6.62e-6; contract text said zero, hardware gave non-bitwise GPU replay; recorded as residual |
| C7 Stage2 reads frozen tokens only | PASS | `FrozenTokenAttention`, zero GNN calls |
| C8 end-to-end result and cost | PASS, negative | Stage2 + Stage3 + performance; startup overhead is tracked separately in #264 |
| C9 no online GNN in Stage2 | PASS | no `EgoStore`/`SubgraphTokens` in mechanism path |
| C10 calibration gradient not truncated | PASS | `test_calibration.py`; no `no_grad` on calibration |
| C11 final checkpoint SID + collision | PASS | export VERIFIED; collision 0.060723 |
| C12 only mechanism changed | PASS | framework constants/launcher unchanged; infra hashes in `INFRA.md` |

## Limits

- Stage3 `test_R@10` does not reach the target; the negative result is final for
  this frozen single run.
- Tile-cache tolerance is not bitwise zero because GPU sparse scatter is not
  deterministic; the issue's frozen tolerance (1e-4) is used and the max is
  recorded. CPU/device replay is not used to claim exact bitwise identity.
- The test split has been read by prior arms; the result is exploratory, not a
  fresh unbiased selection signal.

## R2

`r2:genrec/108/` contains the 112 files staged by
`results/research/issue139/upload_stage3.py` (Stage2 source archive, SID,
checkpoint, recall block). Ledger entry is `results/research/issue139/r2_ledger.json`.

## Closing conditions

Production code implemented: yes. Necessary tests executed: yes. Required
training completed: yes (Stage2 3000 and Stage3 150). Infra boundary: FROZEN;
public framework files unchanged except generated outputs. Remaining executable
tasks for this issue: none after the close comment and push.
