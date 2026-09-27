# Iter31 S05 One-Factor Boundary — frozen source packet

```text
STAGE_ID=S05_ONE_FACTOR
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Canonical active inputs

Use only the Judge-approved S00–S04 artifacts and reports as active instructions:

- `logs/source_snapshot_iter31.md` and S00 `judge.md`
- `logs/protocol_manifest_iter31.md` and S01 `judge.md`
- `logs/hypothesis_iter31.md` and S02 `judge.md`
- `logs/mechanism_manifest_iter31.md` and S03 `judge.md`
- `logs/mechanism_contract_iter31.json`, `logs/hra_step6_contract_iter31.json`, and S04 `judge.md`

S04 explicitly activated the Iter31 HRA-inspired Step6 contract through a between-iteration transition from cancelled Iter30 while preserving FCCR-1 as an unchanged fixed-curvature substrate. It did not authorize source edits or training. Read the relevant Iter29 primary source yourself; do not treat candidate drafts or old summaries as active instructions.

## Locked parent and protocol

- `PARENT_ITER=iter29`, `PARENT_ROOT_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f` as reported by S00/S01 (S01 says it was not freshly queried); Iter29 historical source/run commits are separately documented in S01.
- Sole direct historical comparator: Iter29 exact `test_final.json` at `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`; `test_R@10=0.05921064085377531`, `n_eval=57439`.
- Iter31 protocol: seed 42, Stage2 max 100,000 steps, three 256-entry quantizers/layers, Iter8 checkpoint warm start, same Stage1/Stage0 identities subject to mandatory current-byte/consumer rechecks before use; Stage3 seed 42, max 150 epochs with existing train-loss patience 10, `NO_EVAL=True`, `SKIP_TEST=False`, beam 20, top-k `[5,10]`, n_eval 57,439. This is the one-run performance experiment; no seed replication, sweep, or control run.
- Strict adoption target remains `test_R@10 > 0.065`; it is not an S05 mechanism gate or Stage2 metric gate.

## Direct parent implementation facts

Primary Iter29 `modules/rqvae.py`:

- `get_semantic_ids` obtains three ordered quantized layer embeddings; the loop applies existing Step4 residual update and Step5 cross-layer transport, then stores layer embeddings in L0,L1,L2 order.
- `_step6_sum_embeddings` currently sums those tangent-coordinate embeddings in Euclidean space and returns a batch×embedding-dimension tensor.
- `forward` calls this Step6 output, passes it to the unchanged decoder, and computes the existing reconstruction loss at layer-0 curvature. All existing quantize/commitment/behavior losses and fixed-curvature residual geometry are separate.

Canonical S02/S04 specify exactly one conceptual delta: replace only the Step6 Euclidean tangent sum with

```text
e_l   = tangent-coordinate selected quantizer embedding
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
h     = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
z     = log0^{c0}(h)
c0    = c_0 = 1.3660953164241916
```

Keep exact ascending layer order and right nesting. Preserve conditional exp/log identity limits, helper projection/clamp semantics, inherited fixed-curvature values and provenance caveats, no-telescope/no-homomorphism/no-d-HSTE boundaries. No other mechanism is permitted.

## Required one-factor audit

Agent A and Agent B independently audit, against the same packet and direct Iter29 sources, whether the registered experiment is a clean single factor and enumerate **every expected file/symbol change** necessary to execute it:

1. Identify the sole scientific/model-code change and its exact parent symbol: Step6 aggregation in `modules/rqvae.py::_step6_sum_embeddings`, replacing Euclidean sum with the S02 operation.
2. List inherited components that must remain behaviorally identical: fixed FCCR-1 curvature source/vector/buffers, explicit `branching` + `raw_residual_medians` inputs and no substitution; encoder, quantizer/codebooks/assignments/STE; Step4 and Step5; every loss and weight; optimizer, Sinkhorn and schedules; seed, batch/step protocol and Stage1/Stage0 data; decoder/reconstruction path; Stage3 model/data/evaluator/configuration.
3. Separate non-conceptual path/iteration identity changes from scientific changes. Iter31 path constants must direct Stage2 products only to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; Stage3 result paths only to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/`. Source-tree `iter31` may contain code/config/scripts/logs and input copies only, never `.pth`, `.npy`, or `item_sids.json`. `MECHANISM_NAME` remains full `iter31_<descriptive>`; result directory stays the short `curvature_RQ-VAE_iter31`. `RQVAE_OUT_DIR` must be directly verified before startup. These relocations and contract filename/index updates are wiring, not additional scientific mechanisms, but every affected source/path must be enumerated.
4. Record new audit/checker artifacts required by S04 (`hra_step6_contract_iter31.json` and the separately adjudicated `scripts/preflight_hra_step6_iter31.py`) as verification support, not a model/loss change. The checker is not yet implemented; only S06 may approve it through its own canonical implementation plan and the orchestrator applies once.
5. Preserve S01’s unresolved Stage3 route conflict. Current `stage3_T5Train/train_HG-Rec.py` defaults to a generic SID/results path; Iter29 has a wrapper that overrides only SID/variant/result/launcher paths. S01 explicitly says neither direct route nor Iter29 wrapper is yet approved for Iter31. S05 must not silently resolve this conflict or call the Stage3 routing decision a model/evaluation change; S06/S11 must separately preserve the unchanged Stage3 model/evaluation and resolve the exact launch route before Stage3.
6. Decide if any mismatch, hidden mechanism, unexplained parent drift, or protocol incompatibility prevents a single-factor comparison. Distinguish evidence-backed mandatory wiring from an unapproved scope expansion. If one-factor isolation cannot be defended, block/replan or abort as skill requires; do not silently narrow the ask.

## Output and authorization boundary

Each candidate returns a report with `ROLE=AGENT_A` or `ROLE=AGENT_B`, independence declaration, packet path, exact changed/inherited components, one-factor verdict, evidence, unresolved issues (especially Stage3 routing), confidence, `USER_INPUT_REQUIRED=NO`, and no source edit/training authorization. Judge C verifies primary sources and writes `logs/one_factor_diff_iter31.md` and this stage's `judge.md`, selecting only an allowed verdict. If `MERGE_AB`, name the exact contribution from each.

S05 may establish the clean experimental boundary but cannot implement it, add a checker, modify Stage3, authorize any run, or infer model outcome. Only Judge-approved one-factor text may propagate to S06.
