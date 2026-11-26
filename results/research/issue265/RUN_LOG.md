# Issue #265 — Euclidean selection + topology-adaptive mixed-curvature supervision

## Frozen identity

- Issue: #265, `Euclidean residual quantization with topology-adaptive mixed-curvature supervision on hard q`
- Live body sha256: `748eb1b2d6450e40adea96b71b9889d692f70d1708f1e2d753398a580286e797` (`issue_live.json`).
- **INFRA_MODE=FROZEN**. Issue has no authorization to build/repair public infra; the only changes
  allowed are the candidate mechanism and its tests/diagnostics. #264's Stage3 startup fix is reused
  as existing infrastructure, not modified.
- Training: **required** (Stage2 3000 epochs and Stage3 150 epochs are the acceptance criteria), but
  only after the diagnosis/contract below and the normal machine checks. No new GNN pretraining.

## Acceptance checklist

| ID | Criterion | Data / threshold | Evidence |
|---|---|---|---|
| C1 | Inherit #263 frozen GNN/tokens; GNN forward/backward/optimizer calls = 0 | token sha `4051...`; #263 token cache | token manifest + runtime counters |
| C2 | Competing explanations diagnosed before mechanism | mixed-vs-Euclidean selection hard codes/margins; fixed-q/fixed-gate curvature probes; alternative “no behavior info” | `DIAGNOSIS.md` |
| C3 | Frozen math contract: Euclidean selection, hard q, local gate weights, scale, targets, gradient route | formulas and shapes in CONTRACT | `CONTRACT.md` |
| C4 | Independent reference + known-answer + mutation tests | TDD; 3-node cases; ≥5 mutations | `tests/` |
| C5 | Real connection: gate does not select codes; structural loss reaches encoder/gate/codebook | actual training loss/gradients | `MODULE_CHECKS.json` |
| C6 | Formal Stage2 3000 epochs, seed42, fixed batch, no early stop | collision < 0.40; final checkpoint | run workspace |
| C7 | SID export/checkpoint consistency + §63 layer metrics | exact re-export; metrics file | stage2_metrics |
| C8 | Formal Stage3 150 epochs, beam20, frozen protocol; test R@10 | test_R@10 threshold 0.065 reported, not chosen by test | test_final.json |
| C9 | Fixed equal-weight + Euclidean independent curvature evaluation; raw training objective separate | MAE/RMSE/Pearson/Spearman both calibers | evaluation report |
| C10 | Performance includes periodic work; 0.8 s hard ceiling, target ≤0.3197 s | real four-card run | performance stream |
| C11 | R2 four-piece archive plus frozen token/GNN dependency | r2 ledger | upload log |
| C12 | No public infra/data/SID/split/hyperparameter/GNN/Stage3 changes | source diff + hashes | `INFRA.md` |

## Initial scope

Candidate only: Euclidean selection, local mixed-curvature supervision on hard q, attention/gate
routing, candidate tests/diagnostics. No new Infra, no new assets, no online GNN.

## Candidate/diagnosis state

- Old top-level #263 mechanism archived to `_archive/TIGER_frozen_topology_gnn_rq_issue263_final`;
  new candidate `TIGER_euclid_local_curve_supervision_rq` copied from the immutable #263 run snapshot.
- Selection diagnostic: frozen #263 checkpoint, 24,587 items, mixed vs Euclidean hard SID differs on
  **44 items (0.17896%)**; hashes in `DIAGNOSIS_SELECTION.json`. This directly supports the
  selection-insensitivity explanation; it does not establish behavior sufficiency.
- Candidate implementation: Euclidean cost in `assignment_cost` (gate ignored); local curvature loss
  on final hard q with a single center gate vector (mean of three layer gates); fixed equal-weight
  evaluation remains separate. Tests: new `test_issue265.py` 3/3 and adapted mechanism tests 8/8.
- Entry check PASS: one epoch, collision 0.050718; formal run `stage2_4891fa8df3e843839de770e52e156892`
  started with 3000 epochs. Performance observed around epoch 170: ~0.25 s/epoch.

## Formal Stage2

- Run `stage2_4891fa8df3e843839de770e52e156892`, 3000 epochs, four ranks, seed42, no early stop.
- Final collision **0.04120063448163669** (<0.40); wall **813.357 s**; mean epoch time
  about **0.271 s**; final checkpoint/export written under the immutable run.
- Export gate: `VERIFIED`, checkpoint reproduces SID; 216 live-export items rewritten from checkpoint.
- §63 layer record: collision **0.0412413064**; L1 positive **0.062414**; L1 negative
  **0.0072105**; L1 Hard ER **8.6560**; L2 prefixes **10918**; conditional G3 MISSING.
- Stage3 started with the frozen 150-epoch/beam20 protocol; SID quality collision 0.0412,
  quant error 0.0060 (warning), effective rank 3.049 (warning only, hard gate collision).

## Formal Stage3 / final

- Stage3 `TIGER_euclid_local_curve_supervision_rq`, 150 epochs, beam20, seed42,
  n_eval 57439: **test_R@10 0.0525949268**, R@5 0.0340883372,
  NDCG@10 0.0282504920. This is +0.000696 vs #263 but below frozen TIGER
  (0.055537) and the 0.065 target; recorded as a negative result.
- Fixed equal-weight/Euclidean probe on one frozen training sample: fixed loss
  1.649104, local trainable loss 1.581097 (`FIXED_EVAL.json`); full all-triple
  replay is NOT_CLAIMED.
- Candidate GPU suite: 130 passed, 1 module-budget assertion; after adapting the
  gate/query expectation the module-budget test passes with GPU. New/adapted
  focused tests pass.
- R2 four pieces uploaded: **`r2:genrec/109/`** (113 files, 23 recall keys).

## Verification closure

- Final candidate GPU suite: **131 passed, 0 failed** (`logs/candidate_suite_gpu_final.log`).
- Fixed equal-weight probe: fixed loss 1.649104, local 1.581097 on one frozen
  training sample (`FIXED_EVAL.json`); full all-triple replay is not claimed.
- R2: `r2:genrec/109/` (113 files, 23 recall keys).
- Stage3: R@10 **0.0525949268**, R@5 0.0340883372, NDCG@10 0.0282504920,
  n_eval 57439; below 0.065 and frozen TIGER, recorded as negative.
