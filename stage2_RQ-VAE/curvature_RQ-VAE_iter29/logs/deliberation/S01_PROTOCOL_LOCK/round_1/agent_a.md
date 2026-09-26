ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md
STAGE_ID=S01_PROTOCOL_LOCK

# Proposed iter29 Protocol Lock (candidate; not canonical)

## Proposed manifest values

```text
PROTOCOL_ID=FCCR-1_iter29_alternative_bounded_closed_form_vs_iter26
DATASET_VERSION=Amazon_2023_Instruments / 2026-09 Refactor
DATASET_SHAPE=5-core; 24,587 items; Stage3 final evaluation n_eval=57,439
STAGE1_EMBEDDING=/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy
STAGE1_EMBEDDING_SHA256=6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb
STAGE1_ITEM_ID_SIDECAR=/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json
STAGE1_ITEM_ID_SIDECAR_SHA256=3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30
STAGE0_TRAIN_SHA256=80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815
STAGE0_VALID_SHA256=48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9
STAGE0_TEST_SHA256=5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc
STAGE0_ITEMS_SHA256=6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3
PARENT_ITER=iter26
PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
PARENT_SOURCE_TREE=iter26 FCCR-1 fixed closed-form implementation and run configuration
EXPERIMENT_TYPE=single_factor
CANONICAL_BASELINE_ITER=iter26
CANONICAL_BASELINE_TEST_FINAL=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
CANONICAL_BASELINE_TEST_RECALL_AT_10=0.057017009349048554
STAGE2_SEED=42
STAGE2_MAX_GLOBAL_STEPS=100000
RQ_LAYERS=3
CODEBOOK_SIZE_PER_LAYER=256
STAGE2_PROTOCOL=iter26 settings, data, warm start, optimizer/loss/Sinkhorn and export wiring; Stage2 SID metrics descriptive only; no quality gate
STAGE3_CODE_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
STAGE3_SEED=42
STAGE3_EPOCHS=150
STAGE3_BEAM_SIZE=20
STAGE3_N_EVAL=57439
SUCCESS_CRITERION=User criterion test_recall@10 >= 0.065; no guarantee
```

The parent commit is the packet-identified current `main` HEAD after the S14 closure; it is the exact source state on which iter29 starts. The iter26 protocol manifest records its earlier Stage3 code commit (`0052f4bff117e0aa16c9a1080be1014fbd417cc6`). Direct subtree comparisons show no tracked `stage3_T5Train` changes from the iter18 run-era source commit `112042ceae844e08ffa0fd4cb7dc55afbad5aaf6` to that iter26 commit, from iter26 to iter28's `bdcbbf97bd92ef6b5f0ab016cdfbae2b62c5d73f`, or from iter28 to the packet's current HEAD. Use `612a5a41dfe524205b6afa46370ad0d0ce377882` as the locked Stage3 code commit so it names the code actually present for iter29; source-tree equality establishes continuity of the tracked Stage3 subtree, not equality of Stage2 mechanisms.

## Explicit baseline / iter18 comparability decision

**Canonical direct comparator: iter26 (not iter18).** The S14 action specifically makes iter26 the immediate parent and requires the single conceptual delta to be the alternative fixed, bounded closed-form mapping, holding Stage1, initialization/warm start, data, Stage2/Stage3 configurations, evaluation, and seed policy fixed. Iter26's exact outcome is `0.057017009349048554`, `n_eval=57439`, from the path above. This is the valid comparator for iter29's mapping delta, subject to reproducing those locked conditions.

**Iter18 classification: `HISTORICAL_NONCOMPARABLE` for direct ranking in this iter29 protocol.** Its exact record is `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`, reporting `test_recall@10=0.05988962203380978`, `n_eval=57439`. The value is real historical context, but equal `n_eval`, same Stage3 code subtree, and matching recorded Stage3 settings alone do not prove full Stage2 protocol compatibility. There is no iter18 protocol manifest at the expected path. Its bridge records cyclic/learnable curvature and its P18B fixed initial-curvature AdamW beta2 mechanism, whereas iter26 is fixed closed-form curvature with no cyclic schedule or curvature-conditioned optimizer. The bridge also identifies an iter1 residual-scale fallback due to missing local `layer_norms.json`. Thus iter18 is not the controlled single-factor baseline for this fixed-mapping experiment; do not call it the current best protocol-compatible comparator or report a direct causal delta against it. Iter26's older manifest's declaration of iter18 as directly rankable is not enough to override this independent full-evidence check. The exact iter18 score may be listed as historical context with the explicit label above.

## Protocol and one-factor lock

FCCR-1 remains controlling: the candidate is one bounded alternative closed-form mapping from the same behavior-branching and raw-residual-median inputs to three curvatures calculated before Stage2; all curvature values must be fixed, non-trainable, and time invariant. This protocol candidate does **not** select an equation, constants, or numeric curvature values. Those belong to the subsequent independent hypothesis/mechanism adjudication. No new optimizer, assignment/Sinkhorn behavior, auxiliary loss, Stage1 change, Stage3 change, or interaction/stacking is authorized.

Keep iter26's seed policy and all other experimental conditions: seed 42 in Stage2 and Stage3; three RQ layers, 256 codes/layer, 100,000 Stage2 global steps; Stage3 150 epochs, beam 20, `n_eval=57439`; same Stage1 files/IDs, Stage0 dataset, initialization/warm start and code/config protocol as iter26, except the registered mapping itself. Root `CLAUDE.md` controls unified result output paths and the absence of Stage2 hard gates; do not put Stage2 binaries in the iteration source tree. User success is `test_recall@10 >= 0.065`; success must be observed in protocol-valid Stage3 output, not inferred or guaranteed.

## Direct evidence supporting the lock

- `logs/source_snapshot_iter29.md` identifies iter26 as the most recent direct FCCR-1 run, gives its exact contract curvature `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`, raw inputs/branching and Stage3 output; it also directs independent verification of baseline compatibility.
- `logs/mechanism_contract_iter26.json` specifies FCCR-1, closed-form source, non-trainable/time-invariant curvature, no cyclic schedule/regularizer/new curvature optimizer or auxiliary loss, and formula inputs `behavior_branching` plus `raw_residual_median`.
- `logs/protocol_manifest_iter26.md` records Amazon_2023_Instruments / 2026-09 Refactor; sentence-T5 24,587×768 embeddings; Stage2 seed 42 / 100,000 steps / 3×256; Stage3 seed 42 / 150 epochs / beam 20 / 57,439 evals; and baseline iter18. Iter29 retains settings but designates immediate parent iter26 as comparator for a true single-factor mapping test.
- The exact iter26 `test_final.json` gives recall@5 `0.03788366789115409`, recall@10 `0.057017009349048554`, NDCG@5 `0.025169911931406087`, NDCG@10 `0.03133359761524377`, `n_eval=57439`.
- Independent `sha256sum` of the two Stage1 files reproduced both hashes listed above. Dataset hashes are those captured in the exact shared packet; they are to be locked unchanged.
- `logs/one_factor_diff_iter26.md` describes the prior experiment's isolated mechanism change and unchanged Stage1/Stage3. `logs/hypothesis_iter26.md` records actual formula inputs and immutable outputs. Iter26 failure attribution reports the fixed-curvature checks and activation passed; this supports a viable, executable parent protocol, not a guarantee of recall improvement.
- Iter18 `iteration_bridge.md` says Stage3 completed 150 epochs, beam-20 final evaluation, `n_eval=57439`, exact result `0.05988962203380978`; the exact result JSON confirms the metric and evaluation count. The bridge also documents its distinct cyclic-curvature/P18B setup and missing local residual-scale provenance.
- Stage3 run `HG_Rec.log` headers for both iter18 and iter26 show 150 epochs and the same logged T5 configuration. Source `git diff --stat` across the three specified commit pairs returned no tracked Stage3 subtree changes. These facts support downstream Stage3 continuity, but do not establish overall protocol identity for the Stage2-generated representations.
- `logs/global_review_after_iter28.md` and the S14 judge's canonical action select one bounded alternative FCCR-1 mapping controlled against iter26 and explicitly require baseline bookkeeping reconciliation. The global review distinguishes protocol compatibility from contract-valid FCCR-1 evidence and warns against assuming score differences are causal.

## Assumptions and limits

1. I assume the S14-referenced `612a5a41dfe524205b6afa46370ad0d0ce377882` is the packet's current `main` HEAD and is the source state for iter29; do not silently substitute a different commit.
2. Dataset hashes and dataset/version attribution are taken from the shared packet/S00 snapshot; this assignment verified the Stage1 embedding and sidecar hashes directly.
3. “Same settings as iter26” requires the later protocol audit to verify actual configuration, initialization/warm-start artifact identity, Stage2 inputs, and Stage3 wiring before launch; declarations alone do not guarantee execution compliance.
4. Same Stage3 tracked source and equal `n_eval` do not imply identical trained SIDs, vocabulary, or Stage2 protocol. Accordingly the iter18 result remains historical/noncomparable for direct ranking here.
5. Even an exact iter29-versus-iter26 protocol match compares separate trained runs; seed matching does not remove all stochasticity or establish mechanism causality beyond the registered controlled design.

## Risks

- Accidental retention of iter18 as the direct baseline would compare across a different curvature/optimizer regime and undermine the single-factor claim.
- Unchanged settings may drift in actual source or runtime (notably initialization, warm-start contents, dataset files, Stage3 SID path or code/config); verify before running.
- Iter26 scored below the user's threshold; reproducing it or changing the mapping may still miss `0.065`. The target is a criterion, not a prediction.
- Stage2 quality metrics are descriptive; using them to stop or select would violate root policy and bias Stage3 comparison.
- Baseline ambiguity must not be hidden by an unsupported historical direct delta. Preserve iter18 as historical evidence only, and report exact paths/labels.

## Self-rejection conditions

Reject this proposed lock or amend it through S01 adjudication if primary evidence shows any of the following: the parent source state is not the stated commit/tree; the actual Stage1 or Stage0 inputs differ from the locked hashes; iter26's stated Stage2/Stage3 settings, seed policy, initialization, or exact result provenance cannot be recovered; the claimed iter29-to-iter26 design cannot hold all non-mapping factors fixed; Stage3 source at the locked code commit differs materially from iter26; or the user threshold is superseded by higher-authority task instructions. If full primary evidence establishes iter18's complete protocol and direct comparability, revisit its historical classification without displacing iter26 as the one-factor parent/control absent a Judge-approved change. Do not self-select or implement a mapping at S01.

## Autonomous next action

Submit this candidate to Judge C for primary-evidence adjudication. If accepted/merged, materialize the canonical `logs/protocol_manifest_iter29.md` with iter26 as the controlled canonical baseline, preserve iter18 only as `HISTORICAL_NONCOMPARABLE`, and proceed to S02 to independently specify one bounded alternative mapping under FCCR-1. Before any Stage2 execution, require the later contract/preflight and deliberation approvals plus the mandated one-checkpoint/one-batch gradient-path verification; run Stage2/Stage3 only once after canonical approval. If the Stage1, parent, or locked-condition evidence fails a self-rejection criterion, resolve it from primary records before protocol propagation; do not launch an ungrounded comparison or ask the user to choose.
