# Issue255 execution log

Issue: [#255](https://github.com/cpxjazz123/HiCurvRec/issues/255) — Layer-wise three-curvature gating and cumulative sectional supervision with direction-aware ego tokens

Body SHA256: `2fa23b1ee2faa897155f2a12999b3cf4cd7c085a5fa0a5f8a99c53f54b1c361d`. The 13 original acceptance items are frozen verbatim in `ACCEPTANCE.json`.

## Training decision

**Training required.** Criterion 10 makes the acceptance metrics (three-code collision gate, Stage3 `test_R@10` on the same 57,439 population, layer-wise curvature MAE) producible only from new weights, so this issue must run exactly one formal four-card Stage2 (3000 epochs) and exactly one frozen Stage3 (150 epochs). No second arm, no extra baseline training, no multi-seed, no TIGER retraining, no early stop, no score-driven rerun.

## Mechanism identity (frozen before implementation)

| Item | Value |
|---|---|
| New candidate | `stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq`, an independent copy of the archived `stage2_RQ-VAE/_archive/TIGER_three_expert_attention_rq` (which is itself a copy of the frozen `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` plus the #253 three-expert mechanism) |
| Identity change | `ARM_NAME` constant only; every other source file starts byte-identical to the #253 generation |
| Preserved | 768-d semantic input, 32-d latent, three 256-word levels, #253's actual assignment protocol (L1/L2 floor-constrained nearest, L3 Sinkhorn), original reconstruction/commitment/codebook losses, four-card global batch 4096, seed 42, 3000/150 epochs, expert curvatures `{-1,0,+1}` only, end-to-end trainable GNN and gate |
| New inside the existing framework | direction-aware ego tokens (centre+all in-neighbours induced, centre+all out-neighbours induced) kept separate until expert attention; per-level three-curvature gating `w_i^(l)`; cumulative-prefix sectional supervision over `q_i^(l)=Σ_{j<=l} e_i,j` |
| Forbidden | behaviour-neighbourhood constraints, pair ranking, InfoNCE, edge prediction, extra behaviour supervision losses, PPR, four-node exhaustive graphlets, teacher/GNN pretraining, TIGER retraining, multi-seed, changing 768/32 or the SID protocol |

## Archive step (§1.0a)

Before the candidate was created, the only remaining top-level mechanism directory `stage2_RQ-VAE/TIGER_three_expert_attention_rq` was moved with `git mv` to `stage2_RQ-VAE/_archive/TIGER_three_expert_attention_rq`. `baseline/` and `_archive/` were left untouched. The top level now holds only `baseline/` and `_archive/` plus the new candidate, so exactly one active arm exists. No live training referenced the moved tree (no trainer process was running; `nvidia-smi` showed four idle L40S cards).

## Environment

- Four visible NVIDIA L40S (46 GB each), no trainer process, ports 50202/50201 free at intake.
- Interpreters: Stage2 `genrec_env_v2/bin/python3.9`; Stage0/1/3 `genrec_env/bin/python3.10`; MCP servers use system `python3`.

## Implementation notes (2026-11-25)

- New arm files, all inside `stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq/`: `model/layer_gate.py` (the per-level expert attention, the cumulative prefix and the level-loss aggregation), the inlined direction-layout loader and the four-token loop in `model/structure.py`, the layer gate in `model/structure_gate.py`, the per-level weights/prefixes/per-level curvature loss and the frozen level scales in `model/mechanism.py`, the level-scale freeze and the per-level metric stream in `train_rqvae.py`. `model/attention.py` was removed because the single-level gate it held is superseded; the four legacy test files that imported it were adapted to the layer-wise gate with the same intents and tolerances.
- `tests/test_layer_gate.py` (10 known-answer and reference-consistency tests), `tests/test_layer_mechanism.py` (the prefix identity, the earlier-prefix/later-codebook isolation, the per-level routing and the instance agreement), `tests/reference_layer_gate.py` (the independent numpy reference), and three new designated mutations in `tests/test_mutations.py` (12 mutations in total, documented in `results/research/issue255/MUTATIONS.md`).
- The first attempt to build the direction asset failed its own independent audit (24,587 mismatches): the edge pointers had been written per item instead of per node position and the audit compared the incoming direction against the forward edges. Both were fixed, the asset was rebuilt and the audit now reports 0 mismatches for both variants. The asset test's own hand-computed expectations were corrected twice against the actual (root-first) node order; no production expectation was relaxed.
- The contract gained one recorded clarification before the formal run: the cumulative prefix keeps the same-cost soft backward that #253's quantised path had, while its forward stays bitwise the hard codeword.
- The formal run is bound to an immutable snapshot under `results/stage2_RQ-VAE/_runs/issue255_formal/` and is launched only through `tools/research/stage2/launcher.py`.

## Acceptance

Each criterion's judge is an executable assertion or a hash-bound artefact recorded in `CHECK_1.md`/`CHECK_1.json` and `MANIFEST.json`; the scientific thresholds are the issue's own numbers (#253 `R@10 = 0.046431866850049616`, #252 `R@10 = 0.0492870697609638`, repository adoption `> 0.065`).

## Corrected generation authorization and pre-dispatch checks

The first run completed its budget but is `INVALID_IMPLEMENTATION`: its initial λ numerator used reconstruction-only instead of the full original RQ-VAE objective. The immutable old source, checkpoint, SID and logs remain preserved. This is not a score-driven retry. The owner explicitly selected `repair_rerun` in `OWNER_APPROVAL.json`, authorizing exactly one corrected same-arm generation, `issue255_formal_r1`, after complete current-source credentials. COMMENT_07/08 record authorization and the real-batch red→green calibration regression.

All GPU controllers must use `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`, not system `python3`: the launcher's visibility probe imports torch in the controller. A system-Python validation attempt refused before launching because torch was unavailable there; four L40S and the required interpreter's four CUDA devices were verified. The corrected validation command uses the project interpreter without visibility or protocol overrides (COMMENT_09).

The issue body was reread for `CHECK_3_ISSUE.json`; its SHA-256 remains unchanged. The historical #253 epoch3000 checkpoint and completed Stage3 population57,439 passed exact generation validation despite the appended asset registry; only the exact prior small config bytes come from the frozen Git revision, never old runtime asset replacements (COMMENT_10 and `HISTORICAL_CHECK.json`). Current-source complete test and actual four-rank credentials precede any corrected formal reservation.


## Shared-worktree commit preflight

The default commit guard refused the complete shared-worktree diff:1638paths/339,648lines, including an unrelated147,524-line diagnostic SID table and847archive paths. A prefix inventory identified188Issue255 paths before ignoring this issue's raw benchmark rank reports and diagnostic SID exports; those raw files remain locally and compact evidence retains their hashes. Unrelated files are not deleted, ignored or selected by this task.

For the selective draft only, the guard limits are deliberately raised to2000paths/400,000lines so it can inspect the pre-existing shared diff without changing other work. The draft must be restricted to journal-attributed Issue255 source, v10 metadata/manifest changes and its evidence; unrelated SID tables, old research outputs, binaries and other mechanism archives remain excluded. This exception changes tooling capacity, never training or scientific acceptance.

