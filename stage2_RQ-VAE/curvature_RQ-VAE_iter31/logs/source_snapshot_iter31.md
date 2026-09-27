# Iter31 S00 Source Snapshot — Judge C

```text
STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
VERDICT=MERGE_AB
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Authority, current policy, and iteration boundary

Root `CLAUDE.md` controls project rules, including Stage2/Stage3 output locations, the ban on CLI/environment overrides, Stage2 descriptive-only SID metrics, and the strict downstream adoption target `test_R@10 > 0.065` (`CLAUDE.md` §§0–2, 10–11). It also requires gradient-path checks before an actual Stage2 run (§6). No training, code change, or run is authorized by this S00 decision.

The current `curvature-rqvae-iter` skill requires independent A/B and Judge C stages, canonical-only propagation, no matched-seed or other replication iterations, and a forward performance-seeking structural mechanism (§§2, 2.9). FCCR-1 remains the active contract—fixed closed-form curvature—unless a between-iteration contract transition is canonically adjudicated (§§1, 3). HRA changes aggregation, outside FCCR-1's mapping-only scope; this S00 records that conflict and does not silently transition the contract or register an HRA equation.

Iter30 is explicitly `ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE`, with `SCIENTIFIC_RESULT=NONE` and no Stage2, Stage3, or GPU training (`stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/iteration_cancel_iter30.md`, lines 4–17). The cancellation record says it was not scientific infeasibility and that the Iter30 mapping comparison is not reused or relabeled as HRA (lines 16–27). Its Git state at cancellation was a pull to `120c144` with no commit/push performed by that record (lines 29–30); this is a time-bounded historical statement, not a fresh query of current GitHub state. Iter29's S13/S14 closure audit separately records its historical commit/push and local/remote hash checks ( `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`, lines 14–20).

Iter29 S14 recommended a three-pair matched-seed replication as its proposed next action (`logs/global_review_after_iter29.md`, lines 30–59; `logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md`, lines 24–39). That recommendation is historical and was overtaken by the explicit Iter30 cancellation; it is also forbidden by the current skill's no-replication policy. Do not revive or relaunch it. The fresh Iter31 HRA direction is a proposal, not yet an adjudicated hypothesis or implementation authorization (`iteration_cancel_iter30.md`, lines 26–27; Iter31 `source_packet.md`, lines 15–16).

There is a historical target-wording discrepancy: Iter29's protocol manifest records `test_recall@10 >= 0.065` (`logs/protocol_manifest_iter29.md`, lines 17, 70), while root `CLAUDE.md` §2 and the current skill require strict `test_R@10 > 0.065`. The higher-priority current rule governs Iter31; do not silently carry the older inclusive wording forward.

## Iter29 implementation facts verified against source

- Iter29 sets `SEED=42`, `MAX_GLOBAL_STEPS=100000`, `N_LAYERS=3`, `CODEBOOK_SIZE=256`, and `MIDPOINT_LAYER_MASK=[False, False, False]` (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py`, lines 99–130). Its recorded fixed curvatures are `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` (`scripts/computed_behavior_branching.json`, `closed_form_c_l`, lines 66–70); the fixed list is loaded and passed to the model (`curvature_RQ-VAE.py`, lines 594–624; `modules/rqvae.py`, lines 135–157).
- `Quantize.forward` returns tangent-coordinate codebook embeddings with training STE `x + (embedding - x).detach()` and the selected embedding directly in evaluation (`modules/quantize.py`, lines 237–248). In the residual update, that embedding is mapped to a ball-valued codeword `q_l=exp0(c_l,e_l)`; the residual is also mapped to the ball, left Möbius subtraction is applied, and the result is mapped back with `log0` (`modules/rqvae.py`, lines 246–267). Because every midpoint-mask value is false, all three layers use that intrinsic Möbius branch.
- After each nonfinal layer, `_step5_transport` applies `_transport_between_t` to the residual from `c_l` to `c_(l+1)` (`modules/rqvae.py`, lines 269–280; `modules/hyperbolic.py`, lines 31–38). Thus the actual residual cascade has distinct fixed curvatures and explicit between-curvature residual transport.
- Current `_step6_sum_embeddings` sums layer embeddings in Euclidean tangent coordinates (`modules/rqvae.py`, lines 319–326). `forward` passes this sum to the existing decoder, L2-normalizes the decoded output, and computes reconstruction loss using layer-0 curvature `c0` (`modules/rqvae.py`, lines 354–365). `ReconstructionLoss` maps both decoded output and target through `exp0(c0, ·)` and measures squared Poincaré distance (`modules/loss.py`, lines 11–24).
- The curvature input artifact also records `branching=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and raw residual medians `[1.0, 0.10941, 0.09331]` (`scripts/computed_behavior_branching.json`, lines 7–25). Its provenance limits are explicit: historical checkpoint/SID replay, independent recomputation, and byte-identity proof are unavailable. These are recorded historical values, not a fresh reproduction.

## Iter29 protocol and outcome context

Iter29's own protocol manifest names Iter26 as its direct control, and records Stage2 seed 42, 100,000 steps, 3×256 quantization, Stage3 seed 42, 150 epochs, beam 20, and `n_eval=57439` (`logs/protocol_manifest_iter29.md`, lines 7–17, 32–49). This historical Iter26 control does not automatically become Iter31's parent or baseline; S01 must independently establish a protocol-compatible parent and single-run comparator from primary records.

The exact Iter29 Stage3 result is `n_eval=57439`, R@10 `0.05921064085377531`, R@5 `0.03953759640662268`, NDCG@5 `0.026252776900288842`, and NDCG@10 `0.03257647953179143` (`results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, lines 1–8). This is a recorded Iter29 outcome only, below the current strict target; it is neither an Iter31 outcome nor a baseline selection.

The Iter29 global review treats Iter26 as its sole direct control, excludes Iter18 as `HISTORICAL_NONCOMPARABLE`, and characterizes the iter29-minus-iter26 R@10 point difference as +0.002193631504726755, within an approximate historical-noise reference that was not independently estimated (`logs/global_review_after_iter29.md`, lines 9–28). Those conclusions do not make the canceled Iter30 replication current policy.

## Primary HRA paper: scope and limitation

Colombo and Ayoughi, *Geometry-Aware Hyperbolic Residual Quantization*, arXiv:2609.26342v1, §4.1 (equations 7–9), defines HRA by pairing left residual subtraction `r_i=(-q_i) ⊕_c r_(i−1)` with reverse-nested aggregation `q_1 ⊕_c (q_2 ⊕_c (... ⊕_c q_N))`. Its cancellation/telescoping argument uses the same shared curvature `c` throughout. Section 4.2 treats discounted Hyperbolic STE (d-HSTE) as a distinct backward-pass change; the paper's full GHRQ method combines it with HRA, while the Iter31 proposal is HRA-only. The paper's recommendation experiment is Amazon Beauty with four layers and `c=1` for its hyperbolic models (Appendix A.3/A.6), not Iter29's Instruments three-layer heterogeneous-curvature setup. Table 3 reports Beauty R@10 of 0.0604 for full GHRQ and 0.0606 for naive hyperbolic; neither is an Iter31 baseline or performance prediction.

The HRA-only ablation is an explicit risk caveat, not direct evidence about the proposed Iter31 transfer: §6.6/Table 8 reports recommendation residual-reconstruction error 5.49 for HRA-only, compared with 0.03 for full HRA+d-HSTE. It does not authorize adding d-HSTE to the requested HRA-only proposal.

The packet's radial transfer is `T_(c1→c0)(z)=exp0(c0,log0(c1,z))`. Using the Iter29 geometry formulas (`modules/hyperbolic.py`, lines 17–28, 66–73), and `c1=0.7347829661951981`, `c0=1.3660953164241916`, `x=[0.30,0.10]`, `y=[-0.08,0.25]`, direct arithmetic gives:

```text
T(x ⊕_(c1) y)   = [0.2318117865, 0.3233175819]
T(x) ⊕_(c0) T(y) = [0.2492224376, 0.3166295024]
L2 difference     = 0.01865103694
```

This artificial two-point witness, independently recomputed from those operations, is a counterexample to universal Möbius-addition homomorphism for this radial transfer. It is not a model activation, checkpoint/data replay, training observation, or performance result. It does not prove the proposed aggregation ineffective. Because Iter29's residual path instead uses heterogeneous `c_l` values plus cross-layer residual transport, the paper's shared-curvature exact telescoping proof does not establish exact telescoping for that path. That remains unproven; do not claim or imply otherwise.

## Items intentionally unresolved for later stages

- **S01_PROTOCOL_LOCK:** choose Iter31's parent and protocol-compatible single-run comparator from exact primary protocol and test records. Iter29 is not automatically selected as baseline.
- **S02_HYPOTHESIS:** formulate one falsifiable, forward performance-seeking hypothesis and exact operation. It must not claim universal curvature-transfer homomorphism or exact inversion/telescoping of Iter29's heterogeneous transported residual cascade without a proof.
- **S02/S04 representation freeze:** `Quantize.forward` returns tangent-coordinate `e_l`; the residual update first maps it to the ball-valued codeword `q_l=exp0^(c_l)(e_l)` (`modules/quantize.py`, lines 237–248; `modules/rqvae.py`, lines 246–267). In exact origin-map algebra (within the valid, unclipped domain), `exp0^(c0)(log0^(c_l)(q_l)) = exp0^(c0)(e_l)`. S02/S04 must explicitly define `q_l` as the manifold point and freeze which equivalent representation is used; `log0` must not be applied directly to tangent `e_l`. This identity does not establish cross-layer telescoping.
- **S04_CONTRACT:** only register a machine-readable mechanism contract if the operation's semantics and invariants are coherent under the actual residual transport; adjudicate explicitly whether/how the HRA direction receives a between-iteration contract transition from active FCCR-1. No silent contract amendment.

- **Contract/preflight tooling constraint:** the skill-provided `.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py` hardcodes `EXPECTED["contract_version"]="FCCR-1"` (lines 18–27), requires exactly `formula_inputs=["behavior_branching","raw_residual_median"]` (lines 62–63), and reports `contract=FCCR-1` (lines 188–192). If S04 registers a different HRA contract, this preflight as written is incompatible; S04/S07 must resolve and adjudicate the contract/tooling path before any preflight or Stage2 authorization. This is an unresolved prerequisite, not approval to modify the checker or bypass it.

- S03 must trace tangent embeddings versus manifold codewords and all equation input provenance. S05/S06 must establish the one-factor boundary and exact implementation plan through 2+1 adjudication. No code change, Stage2, Stage3, baseline selection, or mechanism selection is authorized by S00.

## Source index

- Root authority: `CLAUDE.md`, §§0–2, 6, 8–13.
- Active policy: `skill://curvature-rqvae-iter`, §§1–3, 2.9, 2.11.
- Frozen packet and candidates: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S00_SOURCE_TRUTH/round_1/{source_packet.md,agent_a.md,agent_b.md}`.
- Iter30 cancellation: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/iteration_cancel_iter30.md`.
- Iter29 review and S14 Judge: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/global_review_after_iter29.md`; `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md`.
- Iter29 closure: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`.
- Iter29 implementation, configuration, protocol, input artifact, and exact result: paths cited inline above.
- Primary paper: https://arxiv.org/pdf/2609.26342, version 1, especially §§4.1–4.2, 6.6, Tables 3/8, Appendices A.3/A.6.
