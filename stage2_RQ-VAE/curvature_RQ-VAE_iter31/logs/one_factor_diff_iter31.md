# Iter31 One-Factor Diff — S05 Judge C canonical

```text
STAGE_ID=S05_ONE_FACTOR
ROUND=1
VERDICT=MERGE_AB
PARENT_ITER=iter29
PARENT_ROOT_COMMIT=21e0488dcad39b3fd277b071a12daec59b56ba2f (reported by S01; not freshly queried)
CANONICAL_BASELINE_ITER=iter29
EXPERIMENT_TYPE=single_factor (conditional on downstream implementation and protocol gates)
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
USER_INPUT_REQUIRED=NO
```

## Canonical decision and exact scientific delta

The one-factor boundary is **conditionally defensible**. The sole scientific/model-behavior change is the exact S02/S04 `HRA-STEP6-1` operation replacing only Iter29 `RqVae._step6_sum_embeddings`' Euclidean sum of layer tangent-coordinate quantizer outputs. In ascending layer order L0, L1, L2, with right nesting exactly as written:

```text
e_l   = selected tangent-coordinate quantizer embedding
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
h     = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
z     = log0^{c0}(h)
c0    = c_0 = 1.3660953164241916
```

This replaces `z_E = e_0 + e_1 + e_2` and returns the same decoder-facing tangent-coordinate batch×embedding-dimension interface. It is HRA-inspired only: no exact telescope, residual inversion/reconstruction, or universal cross-curvature Möbius homomorphism is claimed. The pointwise exp/log identity is conditional on the valid unclipped domain; existing helper projection/clamp semantics remain relevant and runtime incidence/effect is not established here.

## Direct evidence and inherited factors

Direct Iter29 source review supports this boundary:

- `stage2_RQ-VAE/curvature_RQ-VAE_iter29/modules/rqvae.py:319-326`: `_step6_sum_embeddings` sums `quantized.embeddings` over layers and transposes; it validates the rank/embedding width and finiteness.
- `rqvae.py:282-317`: the encoder/quantizer loop collects embeddings in L0,L1,L2 order after the existing Step4 and Step5 operations. `rqvae.py:246-267` is the existing residual update; `:269-280` is cross-layer curvature transport.
- `rqvae.py:354-365`: `forward` calls Step6, passes its output through the unchanged decoder and computes the existing reconstruction loss at layer-0 curvature. `rqvae.py:367-378` composes the existing quantization, behavior and zero-weight fixed-curvature regularization terms.
- `modules/quantize.py:68-73,98-111,165-248`: fixed curvature is a persistent `_fixed_c` buffer returned by `get_c`; assignments, curvature-scaled Sinkhorn epsilon, quantization loss, and training straight-through/eval embedding values are existing behavior. `modules/loss.py` retains the existing quantization/commitment and reconstruction losses.
- `modules/hyperbolic.py:9-28,66-73`: retain existing exp projection, log clamp and Möbius-add semantics; S05 does not authorize helper changes.
- `curvature_RQ-VAE.py:305-382` loads the explicitly named `branching` and `raw_residual_medians`, validates the registered FCCR arithmetic and exact ten-field contract. Iter31's `logs/mechanism_contract_iter31.json` retains FCCR-1 with fixed vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. `Quantize._fixed_c` / `get_c()` behavior is inherited.

Keep behaviorally identical: FCCR-1 fixed-curvature source, vector, buffers and explicit inputs; historical provenance limits (method/value provenance only; no replay/hash proof; no fresh measurement, normalized-scale substitution or fallback); encoder; quantizers/codebooks/assignments/STE; Step4 and Step5; all losses, weights and reductions; optimizer/training/schedule/Sinkhorn behavior; seed, batch/step protocol, Stage0/Stage1 inputs and warm-start identity; decoder and reconstruction consumer. Numerical values downstream may differ as a consequence of the sole Step6 intervention; do not compensate by changing these inherited factors. Preserve the current strict target `test_R@10 > 0.065`; it is not an S05 mechanism gate. No performance outcome or runtime activation is claimed.

## Wiring and audit artifacts — not additional mechanisms

These identity/path changes are non-scientific wiring, permitted only insofar as they preserve the same registered inputs, products and consumers:

- Iter31 source/output identity updates, including relevant `curvature_config.py` path constants and dependent artifact references, must keep Stage2 products under `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/` and Stage3 products under `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/`. The Iter31 source subtree is for code/configs/scripts/logs/allowed input copies only, not `.pth`, `.npy` or `item_sids.json`. Keep `MECHANISM_NAME=iter31_<descriptive>` and the short product directory `curvature_RQ-VAE_iter31`. Directly verify `RQVAE_OUT_DIR` before any eventual startup, as required by root rules.
- Iter31 `curvature_RQ-VAE.py` must refer to its own compatible FCCR contract (`logs/mechanism_contract_iter31.json`) and Iter31 identity/paths while preserving the exact loader semantics and ten-field FCCR schema. HRA fields remain in the separate canonical `logs/hra_step6_contract_iter31.json`; do not merge schemas or relax/bypass the FCCR checker.
- `scripts/preflight_hra_step6_iter31.py` is new audit tooling only, not a model/loss mechanism. Its implementation is **not authorized here**; S06 must adjudicate and approve it through the canonical implementation plan. S07 later audits/runs it independently from the unchanged FCCR preflight. The HRA contract and future checker logs are verification artifacts, not proof of runtime activation.

## Known expected loci and inventory limit

Known/expected future loci include `modules/rqvae.py::_step6_sum_embeddings` (sole scientific change); `curvature_config.py` and `curvature_RQ-VAE.py` (path/identity/contract-loader wiring); the existing Iter31 FCCR and HRA JSON contracts (preserve as canonical); and the new HRA preflight script only after S06 approval. A Stage3 Iter31 wrapper is conditional upon later route adjudication and is not selected by this diff. Iter31-local configs/scripts may need mechanical reference updates. This is a **known inventory from S05 evidence, not an exhaustive final source-file inventory**: S06 must inspect current Iter31 files and produce the exact approved file/symbol list before any edit. Unexplained non-mechanical changes or any change to a listed inherited factor invalidate this conditional one-factor finding and require re-adjudication.

## Stage3 route remains unresolved

`CURRENT_STAGE3_ROUTE_CONFLICT=UNRESOLVED`. Current `stage3_T5Train/train_HG-Rec.py` defaults to a generic SID path, variant-derived generic result paths and generic launcher identity/log (`:124-159,1175-1181`). Root `CLAUDE.md` §5 specifies direct trainer invocation; Iter29's `scripts/run_stage3_iter29.py:15-35` overrides SID, variant, result and launcher paths, but that precedent is not an Iter31 approval. S05 chooses neither route and must not silently bypass the conflict. S06/S11 must later adjudicate an exact route that consumes the Iter31 SID and writes to the mandated short Iter31 Stage3 root while preserving the Stage3 model, data, evaluator and evaluation protocol. No Stage3 model/evaluation change is part of this single factor, and S05 grants no Stage3 launch authorization.

## Hard conditions and next gate

S06 must independently adjudicate a complete implementation plan: only `_step6_sum_embeddings` changes model behavior; the exact registered operation/order/types are preserved; all inherited mechanisms remain fixed; contract/path/checker changes are listed separately; the checker is independently approved or declined; and route ambiguity remains open until its separate adjudication. Before any execution, current parent/input/warm-start/runtime identities, output paths, both contracts/checkers, required gradient and activation/domain/finiteness conditions, and all later stage gates remain mandatory. S03 historical provenance limits and no-telescope limits remain in force. S05 authorizes **no code/checker edit, test, formatting, runtime, training, Stage2 or Stage3 execution**.

```text
CONFIDENCE=HIGH for static source boundary and inherited-factor classification; MEDIUM for eventual execution/protocol viability (current identity rechecks, actual activation/domain behavior, and Stage3 route remain unresolved)
AUTONOMOUS_NEXT_ACTION=Proceed to S06_IMPLEMENTATION deliberation only; produce and adjudicate the complete minimal implementation/wiring/checker plan, with no implementation applied until that Judge decision and no execution authorization implied.
```
