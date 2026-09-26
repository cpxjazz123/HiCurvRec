# S01 Protocol Packet — iter29

STAGE_ID=S01_PROTOCOL_LOCK
ROUND=1
OBJECTIVE=Independently reconstruct the Stage2/Stage3 protocol and baseline from primary repository files. Decide which result is the canonical direct comparator for a single-factor FCCR-1 alternative mapping controlled against iter26; mark uncertain historical comparisons HISTORICAL_NONCOMPARABLE. Do not choose the mapping.

## Canonical source inputs
- S00 approved snapshot: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/source_snapshot_iter29.md`.
- Current project rules: `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md`.
- Current research contract and protocol requirements: `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/SKILL.md` §§3–5, 17–19.
- S14 Judge decision/action: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md` and canonical `logs/global_review_after_iter28.md`.

## Parent/control evidence
- Iter26 FCCR-1 contract: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/mechanism_contract_iter26.json`.
- Iter26 protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md`.
- Iter26 hypothesis/one-factor code context: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/hypothesis_iter26.md`, `logs/one_factor_diff_iter26.md`, `curvature_RQ-VAE.py`, `curvature_config.py`, `modules/quantize.py`, `modules/rqvae.py`.
- Iter26 exact outcome: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`.
- For this iteration, parent source tree is iter26; exact current main HEAD after S14 closure is `612a5a41dfe524205b6afa46370ad0d0ce377882`.

## Baseline candidates and limits
- Iter18 exact outcome: `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`.
- Iter26 manifest declares iter18 the canonical baseline and directly rankable; verify against primary run records. No `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/protocol_manifest_iter18.md` was found at the expected path.
- Iter18 bridge: `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/iteration_bridge.md` records its full 150-epoch Stage3 run, beam-20 final evaluation, `n_eval=57439`, and exact final metric; inspect related logs/source to reconstruct missing protocol fields.
- Iter28 protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/protocol_manifest_iter28.md`; it declares same Stage3 settings/iter18 comparison but is CAO-1 and not FCCR-1 evidence.
- Stage3 source at iter18 run: best available Git history shows whole-repository HEAD immediately before the run timestamp as `112042ceae844e08ffa0fd4cb7dc55afbad5aaf6`; iter26 manifest records Stage3 code commit `0052f4bff117e0aa16c9a1080be1014fbd417cc6`; iter28 records `bdcbbf97bd92ef6b5f0ab016cdfbae2b62c5d73f`. Verify the `stage3_T5Train` subtree across these commits and current HEAD; do not infer compatibility merely from matching metric counts.

## Current inputs and measured identities
- Stage1 embeddings: `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`, SHA-256 `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`.
- Stage1 item ID sidecar: `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json`, SHA-256 `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
- Stage0 parquet hashes (current): train `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`; valid `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`; test `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`; items `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3`.

## Registered protocol values to verify
- Dataset/version: Amazon_2023_Instruments / 2026-09 Refactor; current Stage1/Stage0 data paths as above.
- Parent: iter26; parent commit for this iteration: `612a5a41dfe524205b6afa46370ad0d0ce377882`.
- Stage2 from iter26 source: seed 42; 100000 global steps; 3 layers × 256 codes; fixed-curvature FCCR-1; no Stage2 quality gates.
- Stage3 from iter26 manifest: seed 42; 150 epochs; beam 20; `n_eval=57439`; unchanged trainer/config.
- User success threshold: `test_recall@10 >= 0.065` (user instruction controls task; record conflict with strict lower-priority `>0.065` text).
- S14 Judge requires one alternative bounded closed-form FCCR-1 mapping controlled against iter26, keeping all other settings and seed policy unchanged; this S01 artifact must reconcile whether iter18 remains a directly comparable canonical baseline or must be labeled HISTORICAL_NONCOMPARABLE.

## Candidate task
Each agent independently reads the exact protocol/results/source files, verifies every required field, explains the best baseline classification with evidence, and proposes a complete protocol manifest. Do not select a mechanism, implement code, write the canonical manifest, or read the other candidate. State assumptions, risks, and self-rejection conditions.
