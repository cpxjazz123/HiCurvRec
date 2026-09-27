# Iter31 S01 Protocol Lock — frozen source packet

```text
STAGE_ID=S01_PROTOCOL_LOCK
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Scope and authority

Choose Iter31's parent, one historical direct comparator, and fixed single-run Stage2/Stage3 protocol using primary records. Do not select an HRA equation, change code, authorize Stage2/Stage3, or settle the HRA/FCCR-1 contract transition here. No matched-seed replication, sweep, or root-cause iteration. The current strict adoption criterion is `test_R@10 > 0.065` per root `CLAUDE.md` §2; Iter29's older protocol used inclusive `>=0.065` and does not supersede the current rule.

S00 is canonically adjudicated `MERGE_AB`: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/source_snapshot_iter31.md` and its Judge at `logs/deliberation/S00_SOURCE_TRUTH/round_1/judge.md`. Carry forward, without resolving in S01: (a) Iter29's source path and exact Stage3 outcome; (b) Iter30 was explicitly cancelled by the user and has no scientific result; (c) the old Iter29 S14 three-pair replication recommendation is superseded by cancellation and the current no-replication skill rule; (d) the HRA direction and its contract transition remain proposals; (e) the FCCR-1 preflight is hardcoded and must be handled at S04/S07; (f) `e_l` is a tangent coordinate and any paper-style `q_l` must be defined as a ball point; and (g) no exact telescope is established for Iter29's heterogeneous transported residual path.

The user requests a forward HRA-only aggregation direction for Stage2. No d-HSTE, loss/optimizer/Sinkhorn/codebook/curvature changes, Stage1 change, or Stage3 model/evaluation change. S01 records protocol constraints only; subsequent adjudicated stages must decide whether/how that direction is made executable. Stage2 quality statistics stay descriptive and cannot gate training or stop it early. Before actual Stage2 training, root §6 requires the specified nonzero-gradient-path checks.

## Candidate parent and baseline evidence (not preselected)

The independent A/B reports must decide Iter31's parent and the single protocol-compatible historical comparator; the candidates must not inherit Iter29's own comparator by default. In particular, Iter29 used Iter26 as its comparator, but that does not automatically make Iter26 the Iter31 comparator. Assess the most recent active Iter29 result as a possible direct parent/comparator, and compare any alternative against primary protocol and result records. Iter18 is already classified `HISTORICAL_NONCOMPARABLE` and must not be promoted to a direct comparator.

Iter29 primary records:

- `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md` records Amazon 2023 Instruments / 2026-09 Refactor; Stage2 seed 42, 100,000 global steps, three layers, codebook size 256, and warm start `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` with SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`; Stage3 seed 42, 150 epochs, beam 20, `n_eval=57439`.
- The same manifest records Stage0 train/valid/test/items SHA-256 values `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`, `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`, `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`, `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3`; Stage1 embedding SHA-256 `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`; item-ID sidecar SHA-256 `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
- Iter29 exact final result path: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`. Its exact metrics are R@10 `0.05921064085377531`, R@5 `0.03953759640662268`, NDCG@5 `0.026252776900288842`, NDCG@10 `0.03257647953179143`, `n_eval=57439`. It misses the current strict target by `0.005789359146224693`. These are Iter29 observations, not Iter31 results.
- Iter29 Stage2 output identity records: `item_sids.json` SHA-256 `58665e08a97e122f47a77ed3616b67efdeb8eadba8253771a1498cb5e57487ff`; `out/rqvae/instruments/rqvae_best.pth` SHA-256 `3a464a8de3ba3cec8dab0223b0705abb6e485567f19ddc1bf02db70a49df35bb`; actual Stage3 input `dataset/Instruments/sids_for_hgrec.npy` SHA-256 `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1`. The Stage2 source tree contains no model/SID exports.
- Iter29 Stage3 artifact hashes recorded in its closure audit: `_stage3_launcher.log` `fe9a49a8d7a4f46ffc0a91d20cf6a4768696bc54b778774714231f029f283806`; `test_final.json` `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`; `training_metrics.jsonl` `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`; best Stage3 checkpoint `70facbde7de7961bbe71e23be6f4f951f30e696b959d04be66cdf7d32dab7e94`.
- Iter29 outcome disclosure: the wrapper supplied `RQVAE_VARIANT=iter29_bounded_rational_additive_mapping`, but `training_metrics.jsonl` reports `unknown_variant` because the logger reads `config.get('variant')`; available evidence does not show model/data identity failure. Launcher logs recorded nonfatal NCCL initialization and c10d barrier device-context warnings; run/test/supervisor completed with exit 0. Preserve both caveats; do not describe startup as warning-free.
- Version/provenance caveat: Iter29's protocol manifest names source/root commit `612a5a41dfe524205b6afa46370ad0d0ce377882`; its closure records code/results commit `57b4a594d92435fd74fa4bba7c7c1439a33eb102`, followed by S14 review commit `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6`. Its Stage3 launcher smoke records final `train_HG-Rec.py` SHA-256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb` and wrapper SHA-256 `1cab47ff035e69aecf3aa5a2e1cfe7777c114a60b753b5e935453d90271b2ec7`; the smoke itself launched no training child. Keep the historical manifest/closure distinction explicit and do not claim a runtime file hash that the run did not record. Lock the current Iter31 Stage3 source and runner identities before execution.

Iter26 records to evaluate if useful as an alternative:

- Its exact test result, cited by Iter29's protocol and outcome, is `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, R@10 `0.057017009349048554`, `n_eval=57439`.
- Iter29's global review reports Iter29-minus-Iter26 R@10 `+0.002193631504726755`, within an approximate historical noise reference of about `0.003` that was not independently estimated. This pair is one observation per condition, not a causal estimate or run-variance estimate.
- Directly verify Iter26's exact protocol and current-source compatibility before proposing it. Do not use Iter18 for direct ranking.

## Protocol obligations to adjudicate and lock

A canonical S01 decision must declare exactly one parent, one direct comparator and exact result path/metric, and one fixed protocol. If Iter29 is selected, lock the current Iter31 parent root commit separately from Iter29's historical artifact/run commits. Recheck current Stage0/Stage1/warm-start file identities and actual resolved inputs before Stage2 launch; declarations/hashes in a prior manifest alone do not prove runtime consumption. Preserve Stage2 seed 42, 100,000 steps, 3×256 quantization, and the immutable warm-start checkpoint unless primary evidence or a higher-priority rule establishes an incompatible protocol; do not launch a second baseline or replicate.

Preserve the source/result separation and required paths: Iter31 Stage2 artifacts only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; Iter31 Stage3 artifacts only under `results/stage3_T5Train/curvature_RQ-VAE_iter31/`; iteration source under `stage2_RQ-VAE/curvature_RQ-VAE_iter31/` contains code/configs/scripts/logs and allowed input copies only, not `.pth`, `.npy`, or `item_sids.json`. Stage3 paths use the short directory `curvature_RQ-VAE_iter31`, while any `RQVAE_VARIANT` keeps the full descriptive mechanism name. All paths/settings remain hard-coded in source; no CLI/environment overrides.

Stage2 SID statistics are descriptive only; complete all 100,000 steps if training is authorized. Stage3 comparison uses the locked Instruments test set with `n_eval=57439`, same Stage3 seed 42, 150 epochs, and beam 20; lock current Stage3 code/runtime settings and evaluation identity before launch. The sole success condition is completed protocol-valid `test_R@10 > 0.065`.

## Candidate reporting / primary references

Independent Agent A and Agent B: make separate source-backed recommendations for parent, comparator and protocol; explain compatibility, exact score and limitations; identify any irreducible provenance gap. Do not decide the mechanism equation, amend FCCR-1, modify code, or authorize training. Judge C adjudicates; only its canonical manifest may proceed.

Primary files: `CLAUDE.md` §§0–2, 6, 8–13; `skill://curvature-rqvae-iter`; S00 Judge and canonical snapshot; Iter29 `logs/protocol_manifest_iter29.md`, `logs/stage3_outcome_iter29.md`, `logs/git_closure_iter29.md`, `logs/stage2_output_integrity_iter29.log`, `logs/stage3_launcher_smoke_iter29.log`, exact Stage2/Stage3 result files and hashes; Iter29 `logs/global_review_after_iter29.md` and its S14 Judge; Iter26 `logs/protocol_manifest_iter26.md` and exact `test_final.json`; Iter18 only to verify its historical-noncomparable status.