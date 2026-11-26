# Issue #263 acceptance checklist

| ID | Criterion | Evidence | Status |
|---|---|---|---|
| C1 | One independent GNN pretraining run; no RQ-VAE/joint stage | `PRETRAIN.json`, `RUN_LOG.md` | PASS |
| C2 | Frozen 20% item-ID split (4918 items); no validation/test rows | `PRETRAIN.json` split/ids | PASS |
| C3 | TRAIN-only own-orbit 376, DGDV 30, 12-bin curvature targets; mask/no imputation | `HYPOTHESES.json`, `pretrain_gnn.py`; `test_labels_tokens.py` | PASS |
| C4 | GNN learns role/curvature distributions; parameters actually update | `PRETRAIN.json` gradients/loss; `test_labels_tokens.py` | PASS |
| C5 | Eval metrics on held-out items, not just train loss | `PRETRAIN.json` held-out metrics | PASS |
| C6 | Freeze + one global pass; cache 24587×4×32; reload/re-export check | `PRETRAIN.json`, `token_manifest.json` | PASS (GPU tolerance 1e-4, max diff 6.62e-6) |
| C7 | Stage2 reads only frozen tokens; no GNN forward/backward/optimizer | `FrozenTokenAttention`, smoke/candidate tests | PASS |
| C8 | Real Stage2/Stage3 result and cost record | `PERFORMANCE.md`, `PERFORMANCE.json` | PENDING final |
| C9 | No GNN in Stage2 path; dummy/shuffled token is out of scope | `test_frozen_tokens_issue263.py`, source audit | PASS |
| C10 | No `no_grad` truncation of calibration; gradient reaches encoder | `test_calibration.py`; formal smoke | PASS |
| C11 | Final checkpoint reproduces SID exactly; collision < 0.4 | Stage2 export + checkpoint re-export | PENDING |
| C12 | Only this mechanism changed; no unexpected code changes | source diff vs R01/R2-102 | PENDING review |

Failures and negative results are reported in `VERDICT.md`; no threshold is silently discarded.
