ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S00_SOURCE_TRUTH/round_1/source_packet.md
STAGE_ID=S00_SOURCE_TRUTH

# Iter31 S00 source-truth candidate — Agent A

## Evidence and scope
This is a factual S00 extraction from the frozen shared packet and its listed primary sources, with the root `CLAUDE.md` read directly. No code was edited and no tests, formatters, linters, training, or launches were run. No choice of protocol baseline or S02 mechanism equation is made here.

## Verified facts

### User direction, Iter30 status, and Git record
- The Iter30 cancellation record classifies the iteration as `ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE`, with `SCIENTIFIC_RESULT=NONE`, `STAGE2_EXECUTED=NO`, `STAGE3_EXECUTED=NO`, and `GPU_TRAINING_EXECUTED=NO`. It says the change is not a finding of scientific infeasibility and that Iter30’s FCCR-1 mapping comparison is neither reused nor relabeled as HRA. Source: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/iteration_cancel_iter30.md`, “Cancellation” and “Preserved evidence and state.”
- The user-requested Curvature-Consistent Cross-Layer HRA direction is still a proposal, not an adjudicated Iter31 hypothesis or implementation authorization. The recorded description maps each layer codeword from `c_l` to `c0` with `exp0(c0, log0(c_l, q_l))`, applies reverse-nested Möbius aggregation at `c0`, then `log0(c0, ·)` before the existing Euclidean decoder. The proposal excludes d-HSTE and other new components. Source: Iter30 cancellation record, “New direction boundary”; frozen S00 packet, “User direction and iteration boundary.”
- Git at cancellation: the record says the requested GitHub-only pull fast-forwarded `main` from `ecd01e4` to `120c144`, with `origin=https://github.com/cpxjazz123/HiCurvRec.git`; local uncommitted work remained, and that record performed no commit or push. Thus `120c144` is the hash recorded at the Iter30 cancellation boundary, not a claim about a later live checkout state. Source: Iter30 cancellation record, “Git state at cancellation.”
- Separately, Iter29’s closure audit records historical commit `57b4a594d92435fd74fa4bba7c7c1439a33eb102` for its closure and later S14 commit `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6`; the latter was pushed and local/remote hashes matched according to that audit. These historical closure hashes do not replace the later Iter30 cancellation record’s `120c144`. Source: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`, “Post-commit and remote verification” and “S14 Global Review synchronization.”

### Controlling rules
- Root `CLAUDE.md` is the controlling repository rules source. It requires work on `main`, only `origin=https://github.com/cpxjazz123/HiCurvRec.git`, no worktrees or unauthorized force-pushes, and local-vs-GitHub hash comparison after pushes (§§8–9, 12). These are policy, not a fresh live verification of current Git state.
- Project scripts hard-code parameters and paths; CLI arguments and environment overrides are disallowed (§1). Stage2 products belong under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/`, with no `.pth`, `.npy`, or `item_sids.json` products in the Iter source subtree (§§0, 10). Stage3 products belong under the corresponding `results/stage3_T5Train/curvature_RQ-VAE_iter<N>/` subtree (§11).
- Stage2 SID metrics are descriptive, not gates; the root policy says downstream Stage3 `test_R@10` is the adoption decision and states a strict goal `> 0.065` (§2). Before any actual Stage2 run, the required one-checkpoint/one-batch gradient-path checks apply (§6). No run is authorized at S00 by the packet.
- The active research workflow requires independent Agent A/Agent B work followed by Judge C and canonical-only propagation, one structural mechanism per iteration, and a single protocol-locked seed; the packet also prohibits sweeps, replications, ablation-only/reverse-control-only, and root-cause iteration. These workflow facts are packet-stated; this candidate does not make later-stage decisions.
- FCCR-1 fixed closed-form curvature is the current contract, and the user HRA proposal is outside its mapping-only scope. The packet explicitly leaves valid registration of a new contract to later adjudication, including S02/S04. FCCR-1 is not silently amended here.

### HRA paper scope and reported outcomes
- The primary paper is Colombo and Ayoughi, “Geometry-Aware Hyperbolic Residual Quantization,” arXiv:2609.26342v1, dated 2026-09-22. §4.1 pairs **left** Möbius residual subtraction `r_i = (-q_i) ⊕_c r_{i-1}` with reverse-nested aggregation `q_1 ⊕_c (q_2 ⊕_c (... ⊕_c q_N))`. The cancellation/telescoping argument uses the same curvature `c` for subtraction and aggregation throughout. Source: paper §4.1 and Appendix B.1–B.2.
- §4.2 separately introduces discounted Hyperbolic STE (d-HSTE), a block-level backward-gradient routing change. The paper’s full GHRQ-VAE combines HRA and d-HSTE; they are distinct modifications. The user-requested HRA-only scope does not itself include d-HSTE. Source: paper §§4.2 and 6.6; Appendix A.1.
- Appendix A.6 fixes paper hyperbolic models at `c=1`. The paper’s sequential recommendation experiment is Amazon Reviews 2014 Beauty, uses four residual layers and frozen MPNet embeddings (Appendix A.3). It is not the HiCurvRec Amazon 2023 Instruments, three-layer, per-layer-curvature setup.
- The paper’s Table 3 reports recommendation metrics on Beauty: full GHRQ-VAE `R@10=0.0604`, naive hyperbolic `R@10=0.0606`, and Euclidean `R@10=0.0530`. These are paper-reported outcomes, not HiCurvRec results or an Instruments baseline.
- §6.6/Table 8 reports HRA-only as the least faithful recommendation configuration (table’s recommendation residual-error entry `5.49`, vs full HRA+d-HSTE `0.03`; the table reports the HRA-only setup as the least faithful). This is paper-specific residual-reconstruction evidence, not a Stage3 prediction for the cross-curvature variant. It gives a reason not to conflate full-paper results with the requested HRA-only change.

### Iter29 implementation and recorded run
- `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py` sets `SEED=42`, `MAX_GLOBAL_STEPS=100_000`, and `MIDPOINT_LAYER_MASK=[False, False, False]` (around lines 99–124). Its fixed curvature loader recomputes/validates the layer values (around lines 305–307, 594–623). `curvature_config.py` points Iter29 outputs to the Iter29 `results/stage2_RQ-VAE` root.
- The verified per-layer curvatures are `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. Source: `scripts/computed_behavior_branching.json`, `closed_form_c_l`; implementation loads the fixed list in `curvature_RQ-VAE.py` and passes it to the model. `modules/rqvae.py` installs a fixed curvature per layer (`fixed_layer_curvatures`, around lines 135–157).
- Quantizer output embeddings are tangent-coordinate vectors with straight-through quantization behavior. The manifold codeword for residual subtraction is produced by applying `exp0(c_l, embedding)`; the all-false midpoint mask selects the intrinsic Möbius residual branch. In `modules/rqvae.py:246–267`, the update is `log0((-exp0(c_l,e_l)) ⊕_{c_l} exp0(c_l,residual))`. Each non-final layer then applies `_transport_between_t` from its curvature to the next (`:269–281`). The transport is a curvature-changing origin-radial map implemented in `modules/hyperbolic.py:31–39`.
- `_step6_sum_embeddings` in `modules/rqvae.py:319–326` sums embeddings in Euclidean tangent coordinates. `forward` passes the sum to the existing decoder, applies `l2norm`, and computes reconstruction loss with layer-0 curvature (`:355–365`). `modules/loss.py:10–23` maps both decoded output and target through `exp0(c0,·)` and measures squared Poincaré distance. Thus the current representation has layer-specific residual updates and transport but a Euclidean tangent sum at decoder input.
- The packet’s proposed codeword conversion `T_{c_l→c0}(q)=exp0(c0,log0(c_l,q))` is a radial curvature transfer. It is not a universal Möbius/gyrogroup homomorphism: the packet’s 2D cross-curvature witness explicitly reports unequal mapped-addition results (below). No theorem establishing homomorphism is present in the source set.
- The paper’s exact HRA proof cannot be directly imported as an exact inverse/telescoping proof for Iter29’s heterogeneous cascade: Iter29 uses different `c_l` across residual layers and transports residuals across curvature between layers, whereas paper §4.1’s cancellation law is applied at one common `c`. A Step6 replacement alone is therefore not established by the paper as inverting the actual Iter29 residual/transport trajectory. This is a source-backed limitation, not a finding that the proposal is impossible.
- Recorded Iter29 protocol: Stage2 seed `42`, `100000` steps, `3×256`; Stage3 seed `42`, `150` epochs, beam `20`, `n_eval=57439`. Source: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`.
- The exact recorded Iter29 Stage3 result is `test_recall@10=0.05921064085377531`, `n_eval=57439`, from `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`. This observed result is below the root strict target `>0.065`; it is not an outcome for Iter31.
- `computed_behavior_branching.json` records `branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]`, `raw_residual_medians=[1.0,0.10941,0.09331]`, and the curvature vector above. Its provenance caveats explicitly say no historical checkpoint replay/independent recomputation/byte-identity proof for the residual medians; branching used a historical raw SID export currently absent and lacks replay, independent recomputation, or historical byte-identity proof. These values are recorded artifact evidence, but their listed provenance limits prevent stronger reproduction claims.

## Cross-curvature arithmetic witness
The frozen packet gives `c1=0.7347829661951981`, `c0=1.3660953164241916`, `x=[0.30,0.10]`, and `y=[-0.08,0.25]`, and defines `T(z)=exp0(c0,log0(c1,z))`. Using the origin exp/log formulas and the Möbius-addition formula in `modules/hyperbolic.py` as the arithmetic definitions, the reported rounded values are:

- `T(x ⊕_{c1} y)=[0.2318117865,0.3233175819]`
- `T(x) ⊕_{c0} T(y)=[0.2492224376,0.3166295024]`
- Their L2 difference is `0.01865103694`.

These distinct values confirm that this transfer does not preserve Möbius addition generally; therefore it is not a universal Möbius homomorphism. This is an artificial two-point arithmetic witness evaluated from the listed formulas and Iter29 curvature values. It uses no model batch, checkpoint output, Stage2/Stage3 execution, or evaluation metric, and is **not training, model-data, or performance evidence**. Its scope is disproving the asserted universal algebraic preservation, not proving anything about the empirical efficacy of an HRA-like aggregation.

## Conflicts, inferences, and unknowns

### Conflicts / contract tension
1. The paper proves cancellation at a shared curvature; Iter29’s residual cascade is heterogeneous and explicitly transports residual tangent vectors from `c_l` to `c_{l+1}`. Treating the paper identity as an exact proof for a common-reference aggregation over this code path would exceed the cited proof’s assumptions.
2. The proposed map is a radial origin-tangent transfer, not a homomorphism, per the arithmetic witness. Therefore, mapping summands to `c0` does not by itself establish that mapping their layer-local Möbius composition equals composing the mapped terms at `c0`.
3. The current FCCR-1 contract is mapping-only; HRA changes aggregation and needs a separately adjudicated Iter31 contract transition. Neither the proposal nor S00 itself amends FCCR-1.
4. The user’s requested one-change scope excludes d-HSTE, while the paper’s full GHRQ combines HRA and d-HSTE and its HRA-only ablation is reported as least faithful for the recommendation residual metric. That paper result is a scope/performance caveat, not authorization to add d-HSTE and not a prediction for Instruments.

### Inferences (not established results)
- A narrower, falsifiable performance hypothesis could concern using a common-reference manifold aggregation at the decoder interface, but it must not claim exact inversion/telescoping of the complete heterogeneous residual cascade unless later mathematical analysis establishes that result.
- If the Stage3 result for Iter29 is used contextually, it is a single recorded result under Iter29’s own registered protocol; it cannot by itself establish a causal effect for a future aggregation change.

### Unknown / later-stage prerequisites
- Which parent and single-run comparison baseline are protocol-compatible for Iter31 remains open to S01; Iter29 is not automatically the baseline.
- Whether a precise HRA cross-curvature hypothesis is mathematically coherent and how its equation is defined remain for S02–S04. No mechanism equation or baseline is selected here.
- Exact equation inputs and tangent/codeword semantics must be traced/adjudicated before a machine-readable contract; the packet designates S03/S04 for this.
- No Iter31 code-change, Stage2, or Stage3 authorization exists at S00. The packet requires later canonical approvals; this candidate supplies facts only and makes no implementation or training recommendation.

## Source index
- Frozen packet: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S00_SOURCE_TRUTH/round_1/source_packet.md`.
- Project authority: `CLAUDE.md`, §§0–13.
- Iter30 status/Git boundary: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/iteration_cancel_iter30.md`.
- Iter29 implementation: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py`; `curvature_config.py`; `modules/rqvae.py`; `modules/quantize.py`; `modules/hyperbolic.py`; `modules/loss.py`.
- Iter29 numeric artifact/protocol/outcome: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/computed_behavior_branching.json`; `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md`; `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`.
- Iter29 historical Git closure: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`.
- HRA primary source: Colombo & Ayoughi, “Geometry-Aware Hyperbolic Residual Quantization,” arXiv:2609.26342v1, §§4.1–4.2, 6.2, 6.6, Tables 3/8, Appendix A.3/A.6 and B.1–B.2.
