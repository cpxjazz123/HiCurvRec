ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S00_SOURCE_TRUTH/round_1/source_packet.md
STAGE_ID=S00_SOURCE_TRUTH

# Iter31 S00 Source-Truth Candidate — Agent B

## Scope and evidence standard
This is an independent factual S00 extraction from the frozen shared source packet and its listed primary sources. I read root `CLAUDE.md`, the `curvature-rqvae-iter` skill, the cited Iter29 implementation/config/evidence, the Iter30 cancellation record, and the primary HRA paper. I did not inspect another Iter31 candidate. I made no edits, did not train or launch code, and did not select a protocol baseline or HRA equation. The arithmetic witness below is checked against the packet's stated values and the radial-map/Möbius operations defined in Iter29’s geometry source; it is not experimental evidence.

## Verified facts

### Iter30 status, user direction, and Git record

- The canonical Iter30 cancellation record states `STATUS=ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE`, `SCIENTIFIC_RESULT=NONE`, and `STAGE2_EXECUTED=NO`, `STAGE3_EXECUTED=NO`, `GPU_TRAINING_EXECUTED=NO` (`stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/iteration_cancel_iter30.md`, lines 1–18). It explicitly says this is not a finding of scientific infeasibility and that Iter30’s FCCR-1 mapping comparison is not reused or relabeled as HRA (same file, “Cancellation” and “New direction boundary”).
- The frozen Iter31 packet says the cancellation archive was pushed in commit `21e0488dcad39b3fd277b071a12daec59b56ba2f` to the sole allowed GitHub `origin/main`, with local and remote hashes equal (`source_packet.md`, “User direction and iteration boundary”). This is a packet-reported, time-qualified prior verification—not a fresh network query by this candidate. The Iter30 cancellation record itself describes the earlier state before that archive push: it records a GitHub-only pull from `ecd01e4` to `120c144`, with uncommitted work still present and no commit/push performed by that record (`iteration_cancel_iter30.md`, “Git state at cancellation”). Those statements refer to different moments and are not contradictory.
- The user-directed next subject is fresh Iter31 Curvature-Consistent Cross-Layer HRA, not an Iter30 continuation. The mapping proposal in the packet—transfer each codeword from `c_l` to `c0`, reverse-nested Möbius aggregation at `c0`, then `log0(c0,·)` before the existing decoder—is a proposal only. The packet explicitly leaves it pending Iter31 adjudication; it is not an approved equation or implementation authorization (`source_packet.md`, lines 15–16, 40–44).

### Controlling rules and current contract

- Root `CLAUDE.md` is the highest repository authority. Project scripts must hard-code paths/parameters and accept no CLI or environment overrides (§1). Development is restricted to `main`, `origin` must be the sole allowed GitHub URL `https://github.com/cpxjazz123/HiCurvRec.git`, worktrees and unauthorized force-pushes are prohibited, and a push requires local/remote `main` hash verification (§§8–9, 12). Stage2 outputs are restricted to `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; Stage3 artifacts to `results/stage3_T5Train/curvature_RQ-VAE_iter31/`; the Iter31 source subtree must not hold training binaries or exported SID products (§§0, 10–11).
- Stage2 SID quality statistics are descriptive, not gates; the root adoption objective is strict `test_R@10 > 0.065` (`CLAUDE.md` §2). Before any actual Stage2 training, the root rule requires one-checkpoint/one-batch gradient-path checks, including total loss and each applicable mechanism loss; a failure blocks training (§6). These are future constraints, not authorization to run.
- The skill mandates independent A/B work against the same packet, Judge C adjudication, and canonical-only propagation at every stage (§§2–2.8). It disallows sweeps, matched/multiple-seed replication, ablation-only work, and root-cause iterations; it requires a single protocol-locked seed and one forward performance-seeking structural mechanism (§§0, 2.9). Full Stage2/Stage3 execution occurs once after adjudication; direct infeasibility requires an abort rather than retuning (§§2.5, 2.10).
- The active research contract remains FCCR-1, fixed closed-form curvature, unless a higher-priority instruction or canonically adjudicated between-iteration transition changes it (skill §3; packet lines 18–24). FCCR-1 is not silently amended by the new user proposal. The packet explicitly flags HRA as outside FCCR-1’s mapping-only scope and says later adjudication must determine whether/how a new contract is registered (`source_packet.md`, lines 18–24). This candidate does not resolve that contract transition.

### HRA paper scope and applicability

- The primary source is Colombo and Ayoughi, *Geometry-Aware Hyperbolic Residual Quantization*, arXiv:2609.26342v1, dated 2026-09-22. Section 4.1 defines the HRA pairing as left Möbius residual subtraction, `r_i = (-q_i) ⊕_c r_{i-1}`, with reverse-nested aggregation, `q_1 ⊕_c (q_2 ⊕_c (... ⊕_c q_N))`. Its cancellation and exact-telescoping derivation uses a single shared curvature `c` throughout (§4.1, equations 7–9 in the paper).
- Section 4.2 separately introduces block-level discounted Hyperbolic STE (d-HSTE). The complete GHRQ method combines HRA and d-HSTE; they are distinct forward/backward changes. The Iter31 proposal requests aggregation only, not d-HSTE. The paper does not establish that d-HSTE is part of HRA itself (§§4, 4.2; packet lines 28–29, 43).
- Appendix A.6 uses `c=1` for paper hyperbolic models. The recommendation experiment uses Amazon Beauty and four residual layers (§5, Appendix A.6); this is not the HiCurvRec Instruments, three-layer setup. Table 3 reports Beauty results: full GHRQ `R@10=0.0604` and naive hyperbolic `R@10=0.0606`. These paper metrics are not HiCurvRec baselines or predictions.
- Section 6.6/Table 8 reports HRA-only ablation and describes it as the least faithful recommendation configuration; full HRA+d-HSTE is more faithful. This is paper-specific risk context. It is not direct performance evidence for Iter31’s proposed heterogeneous-curvature transfer, nor does it authorize adding d-HSTE.

### Iter29 implementation and measured record

- Iter29 configuration records `SEED=42`, `MAX_GLOBAL_STEPS=100000`, three layers, codebook size 256, and `MIDPOINT_LAYER_MASK=[False, False, False]` (`curvature_RQ-VAE_iter29/curvature_RQ-VAE.py`, lines 98–129). Its fixed per-layer curvature vector is `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` (`scripts/computed_behavior_branching.json`, `closed_form_c_l`; Iter29 source initialization/validation reads the registered closed-form vector).
- `Quantize.forward` returns the training straight-through embedding `x + (embedding - x).detach()` (with evaluation returning the selected embedding directly); the quantizer embedding is the tangent/codebook coordinate (`modules/quantize.py`, final return in `forward`). The residual update maps that embedding to a manifold codeword with `exp0(c_l, embedding)`, then all active Iter29 layers apply left Möbius subtraction `_mobius_add_t(-embedding_h, residual_h, curvature)` and map back with `log0` (`modules/rqvae.py`, `_step4_m2_residual`, lines 218–241). Thus the stated `q_l=exp0(c_l,e_l)` refers to the manifold codeword associated with tangent embedding `e_l`; it should not conflate the STE output with an already-ball-valued point.
- After each nonfinal layer, `_step5_transport` applies `_transport_between_t(residual, c_l, c_{l+1})` (`modules/rqvae.py`, lines 243–257; operation in `modules/hyperbolic.py`). This is explicit cross-layer transport, and the three residual stages do not operate at a common curvature.
- The existing `_step6_sum_embeddings` sums returned embeddings in Euclidean tangent coordinates, then the unchanged decoder consumes that sum (`modules/rqvae.py`, `_step6_sum_embeddings` and `forward`, lines 300–363). `forward` selects layer 0 curvature for reconstruction. `ReconstructionLoss` applies `exp0(c0,·)` to both decoder output and target and computes their Poincaré distance (`modules/loss.py`, `ReconstructionLoss.forward`).
- Iter29’s behavior artifact records `branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]`, `raw_residual_medians=[1.0,0.10941,0.09331]`, and the curvature vector above (`scripts/computed_behavior_branching.json`). Its embedded provenance limits explicitly state no historical checkpoint/SID replay, independent recomputation, or historical byte-identity proof for those sources. These are recorded values with stated provenance limits, not newly reproduced inputs in this S00.
- The Iter29 protocol manifest records seed 42, 100,000 Stage2 steps, 3×256, Stage3 seed 42, 150 epochs, beam 20, and `n_eval=57439` (`logs/protocol_manifest_iter29.md`). It names Iter26 as Iter29’s direct mapping-control baseline only; Iter31 must establish its own parent and protocol-compatible comparator in S01.
- The exact Iter29 `test_final.json` records `test_recall@10=0.05921064085377531`, `n_eval=57439` (and R@5 `0.03953759640662268`, NDCG@5 `0.026252776900288842`, NDCG@10 `0.03257647953179143`). This is an Iter29 outcome, not evidence for HRA and not a baseline selection for Iter31.

## Cross-curvature witness check and interpretation

The packet supplies `c1=0.7347829661951981`, `c0=1.3660953164241916`, `x=[0.30,0.10]`, and `y=[-0.08,0.25]`, and defines the radial transfer `T=exp0(c0, log0(c1,·))`. I checked its numerical distinction against the operations implemented in `modules/hyperbolic.py` (`_expmap0_t`, `_logmap0_t`, `_mobius_add_t)`:

```text
T(x ⊕_{c1} y)       = [0.2318117865, 0.3233175819]
T(x) ⊕_{c0} T(y)     = [0.2492224376, 0.3166295024]
L2 difference         = 0.01865103694
```

The coordinate vectors and difference agree with the packet witness to its displayed precision under those formulas. The inputs are an artificial point pair—not a checkpoint replay, model activation, data batch, training observation, Stage3 result, or performance estimate. The nonzero discrepancy is a counterexample to the claim that this radial curvature transfer universally preserves Möbius addition (a universal gyrogroup-homomorphism assumption). It does not itself test the proposed full three-layer aggregation.

The important structural mismatch is factual: the paper’s exact HRA proof assumes the same `c` for subtraction and reverse-nested aggregation at every layer, while Iter29 uses `c_l` values that differ and transports residuals between adjacent curvatures. Therefore, applying a common-reference aggregation only at Step 6 is **not proven to invert the actual heterogeneous-curvature residual cascade**. Exact telescoping for that cascade remains unproven. Do not state or imply that the paper’s shared-curvature guarantee transfers automatically to Iter29’s transported heterogeneous-curvature path.

## Conflicts, inferences, and unknowns

### Conflicts / boundaries

1. **Contract scope:** current FCCR-1 concerns the fixed curvature mapping; cross-layer HRA changes aggregation and is outside its mapping-only scope. The user’s direction requests a fresh Iter31, but does not itself create a canonically registered machine-readable contract. This remains for adjudication; no silent amendment is valid.
2. **Paper-to-project transfer:** paper HRA’s theorem concerns shared curvature. Iter29’s active layers have distinct fixed curvatures and cross-layer residual transport. The paper result cannot be cited as proof of exact cancellation for this cascade.
3. **Objective wording:** the shared packet reports the controlling project target as strict `test_R@10 > 0.065`, matching root `CLAUDE.md` §2 and the skill. Iter29’s older protocol text records an inclusive `>=0.065` target. Preserve the historical wording discrepancy; this S00 follows the currently controlling root/skill strict target and makes no claim about equality.
4. **Outcome separation:** Beauty paper R@10 values (0.0604 and 0.0606), Iter29’s Instruments result (0.05921064085377531), and the project threshold are distinct records under distinct contexts; none should be conflated.

### Inferences (explicitly limited)

- Since `T` is not a universal Möbius-addition homomorphism and the actual residual path also changes curvature with transport, no exact-telescoping conclusion for the proposed common-reference aggregation follows from the paper’s shared-curvature proof. This is a limitation of what has been established, not proof that the proposed mechanism cannot be useful.
- Any narrower performance claim about feeding reference-curvature-aggregated codes to the existing decoder would be a separate hypothesis requiring later adjudication. It is not selected here.

### Unknowns / later-stage questions

- Whether a valid between-iteration research-contract transition registers HRA, and whether the precise cross-curvature operation has coherent semantics with Iter29’s residual transport, remain unresolved at S00.
- Exact Iter31 parent and protocol-compatible baseline remain for S01; Iter29 is not automatically the baseline merely because it is the most recent completed iteration.
- A falsifiable hypothesis, semantic/provenance audit, machine-readable contract, and exact one-factor change remain for S02–S06. No source change or training follows from this S00 candidate.
- No fresh remote query was made; the Iter30 archive hash statement is attributed to the frozen packet’s prior verification. No Stage2 or Stage3 authorization exists at S00.

## Candidate source snapshot conclusion

1. Treat Iter30 as user-cancelled with no scientific outcome, not scientifically infeasible. The frozen packet records archive commit `21e0488dcad39b3fd277b071a12daec59b56ba2f` on allowed GitHub `origin/main` and matching local/remote hashes at that prior verification; do not present this as a fresh check.
2. Apply root project rules and the independent A/B/Judge, canonical-only skill workflow. FCCR-1 remains current absent a canonically adjudicated contract transition; HRA has not been registered merely by the user’s proposal.
3. State the paper’s shared-curvature HRA identity accurately and keep paper d-HSTE separate. The paper’s Beauty results and HRA-only ablation are contextual evidence only, not HiCurvRec Instruments outcomes.
4. Preserve Iter29’s exact residual, curvature, transport, summation, decoder, and result facts with the stated provenance qualifications. The packet’s numeric witness is a non-training arithmetic counterexample to a universal homomorphism claim.
5. Explicitly state that the cross-curvature transfer is not a universal Möbius homomorphism and exact telescoping remains unproven for the heterogeneous cascade. Do not select a protocol baseline, define the S02 mechanism equation, or authorize implementation/training.

**S00 boundary:** this candidate is not Judge C’s canonical snapshot and authorizes no Stage2 or Stage3 work.
