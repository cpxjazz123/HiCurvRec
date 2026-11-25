### R05 formal Stage2 finished — 3000 epochs, gate passed, preregistered signature falsified

| item | value |
|---|---|
| run / arm | `issue256_r05_formal` / `TIGER_role_conditioned_l2_rq` |
| epochs | 3000 / 3000, four ranks, global batch 4096, seed 42, no early stop |
| wall | 2005.5 s (33m25s) training; exit 0 |
| checkpoint | `rqvae_best.pth` sha256 `f93da1a7466d62f07c6482c64cf1a8efdd9dd1fa7ccdf7891be98d6d37bc7f26` |
| SID (trainer) | `item_sids.json` sha256 `dc206dba7fbdadc28bbdedb582c7c755e412d70bdfec930b380a2bd5ea9d5ba6` |
| legal re-export | from the saved checkpoint: **0 differing items**, SID unchanged, `export_manifest.json` + `archive_binding.json` written |
| three-code collision | **0.055273** (< 0.40 gate) |
| quantisation error / effective rank | 0.007134 (warning: below 0.035 band) / 4.333 (< 8.0) |

**Preregistered signature (frozen in R05 §3), measured on the exported SID:**

| prediction | threshold | measured | verdict |
|---|---|---|---|
| distinct two-code prefixes | ≤ 6000 | **9609** | not met |
| failure line | ≥ 7000 | **9609** | **failure line met — mechanism falsified** |
| absolute L1 prefix carry | 0.0496–0.0696 | **0.059959** | level 1 unchanged |
| H(L3 | L1 L2) | reference only | 1.8516 bits | recorded |
| distinct triples | reference only | 23228 of 24587 | recorded |
| validation Recall@10 | ≥ 0.0583 | **NOT_AVAILABLE** | FAULT-1 (no identity-parameterised validator) |

So the R05 change (role mixture moved from L3 to L2) **did not coarsen level-2 branching**: it opened more distinct
second codes than R04 (9609 vs 8781), while level 1 stayed exactly where the contract expected. This is a valid
negative Stage2 result, not an implementation failure: the same SID is reproduced exactly from the checkpoint, the
module/gradient/switch/mutation checks passed, and the run followed the frozen recipe.

Evidence: `results/research/issue258/FINAL_STAGE2.json` (sha256 `ae182c24…`),
`SIGNATURE.json` (sha256 `ee81c86d…`), `SID_QUALITY.json`, layered metrics
`results/stage2_RQ-VAE/TIGER_role_conditioned_l2_rq/stage2_metrics/dc206dba….json`.

Per the issue, the one frozen Stage3 150-epoch test now runs on this SID; it is reported as exploratory.
