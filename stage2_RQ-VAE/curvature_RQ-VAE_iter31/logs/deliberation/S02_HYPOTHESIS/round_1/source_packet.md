# Iter31 S02 HRA Hypothesis — frozen source packet

```text
STAGE_ID=S02_HYPOTHESIS
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Canonical inputs

Use only S00 and S01 canonical artifacts as active instructions:

- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/source_snapshot_iter31.md` (S00 `MERGE_AB`).
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/protocol_manifest_iter31.md` (S01 `MERGE_AB`).
- Their corresponding `logs/deliberation/S00_SOURCE_TRUTH/round_1/judge.md` and `logs/deliberation/S01_PROTOCOL_LOCK/round_1/judge.md`.

S01 locks Iter29 as Iter31 parent condition and sole direct historical comparator. Exact baseline is `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, `test_recall@10=0.05921064085377531`, `n_eval=57439`. Current strict target is `test_R@10 > 0.065`. No baseline rerun, replication, sweep, or root-cause iteration.

The user requested a fresh performance-seeking direction after explicitly cancelling Iter30: “Curvature-Consistent Cross-Layer Hyperbolic Residual Aggregation (HRA),” a Stage2 aggregation change only, not d-HSTE. This is still a proposal: S02 registers a falsifiable hypothesis only; S03/S04/S05/S06/S07/S08 are not passed by this packet. Do not write code, alter contract/checkers, or authorize Stage2/Stage3.

## Required S02 decision

Agent A and Agent B independently decide whether the requested HRA-only cross-curvature aggregation is a coherent and falsifiable forward performance-seeking hypothesis for Iter29's actual model. Assess the exact operation, coordinate semantics, HRA paper boundary, heterogeneous curvature/transport limitation, prior HRA-only risk, downstream hypothesis and falsification conditions. If the idea is viable, state a narrow hypothesis without claiming exact inversion/telescoping. If direct primary evidence establishes the proposal itself as mathematically/semantically infeasible without changing its registered meaning, recommend `ABORT_ITERATION`; do not silently replace it with d-HSTE, a new loss, or another mechanism. Judge C decides; only its canonical `logs/hypothesis_iter31.md` may propagate.

## User-proposed operation and code representation to assess

The user proposes: move each layer's codeword from its layer curvature `c_l` to reference curvature `c0` using the origin radial map `exp0(c0, log0(c_l, q_l))`; aggregate in reverse-nested Möbius order at `c0`; apply `log0(c0, ·)` before the existing Euclidean decoder. For three layers the proposed expression is:

```text
q_l^0 = exp0^{c0}( log0^{c_l}( q_l ) ),   l = 0,1,2
h     = q_0^0 ⊕_{c0} ( q_1^0 ⊕_{c0} q_2^0 )
z     = log0^{c0}(h)
```

Preserve right nesting and layer order L0,L1,L2; Möbius addition is not generally associative. The output `z` must have the same batch × embedding shape and tangent-coordinate role as Iter29's Euclidean Step6 sum because the existing decoder consumes that representation.

Crucial actual-code semantics (verify independently):

- `Quantize.forward` returns tangent-coordinate codebook embeddings `e_l`, with training STE `x + (embedding - x).detach()`; it does not return a stored Poincaré-ball point (`stage2_RQ-VAE/curvature_RQ-VAE_iter29/modules/quantize.py`, lines 237–248).
- Iter29 Step4 forms the ball-valued codeword as `q_l=exp0^{c_l}(e_l)` before left Möbius residual subtraction; Step5 transports the residual between successive, distinct fixed curvatures (`modules/rqvae.py`, lines 246–280).
- Therefore never apply `log0(c_l, ·)` directly to tangent `e_l`. To implement the user's expression, define `q_l=exp0^{c_l}(e_l)`. In the valid, unclipped origin-map domain, `exp0^{c0}(log0^{c_l}(q_l))=exp0^{c0}(e_l)`; actual helpers project/clamp, so do not claim this identity in saturated/clipped cases without checking.
- Current `_step6_sum_embeddings` is a Euclidean tangent sum. It receives `RqVaeOutput.embeddings` with shape `(n_layers, embed_dim, batch)` and returns `(batch, embed_dim)`; `forward` sends that result through the unchanged decoder and computes reconstruction loss using `c0` (`modules/rqvae.py`, lines 300–326 and 354–365).
- Layer curvatures remain fixed Iter29 values `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`; S03/S04 must verify their source and contract. S02 does not choose new curvature values.

## Mathematical/source limits that the hypothesis must respect

Primary paper: Colombo & Ayoughi, *Geometry-Aware Hyperbolic Residual Quantization*, arXiv:2609.26342v1, https://arxiv.org/pdf/2609.26342 (especially §4.1 equations 7–9, §4.2, §6.6/Table 8, Table 3, Appendix A.3/A.6).

- Paper §4.1's HRA pairs left residual subtraction `r_i=(-q_i)⊕_c r_(i−1)` with reverse-nested aggregation `q_1⊕_c(q_2⊕_c(...⊕_c q_N))`; its exact cancellation/telescope uses a single shared curvature `c` for residual and aggregation operations.
- Paper §4.2's d-HSTE is a separate backward-pass mechanism; the full GHRQ combines it with HRA. The user explicitly excludes d-HSTE; do not add it.
- Paper reports HRA-only as its least-faithful recommendation residual configuration (§6.6/Table 8; HRA-only error 5.49 versus 0.03 for full HRA+d-HSTE). Treat this as a risk prior, not a direct R@10 prediction for Instruments and not authority to scope-expand.
- Paper's recommendation experiments use Beauty, four layers, and `c=1` for hyperbolic models; Table 3 R@10 values do not provide direct Iter31 performance evidence.
- Iter29 uses heterogeneous fixed `c_l` and explicit cross-layer residual transport. The S00 canonical witness for `T_{c1→c0}(x)=exp0(c0,log0(c1,x))` gives different values for `T(x⊕_{c1}y)` and `T(x)⊕_{c0}T(y)` (L2 difference `0.01865103694`). This disproves a universal Möbius-addition homomorphism for that transfer; it is artificial arithmetic evidence, not model activation or performance.
- Consequently, the shared-curvature paper proof does **not** establish exact telescoping/inversion of Iter29's heterogeneous, transported residual cascade. A viable hypothesis may be narrower: common-reference, right-nested hyperbolic decoder aggregation of the per-layer codewords could improve representation consistency and downstream recommendation. Do not claim it reconstructs the residual exactly or that curvature transfer preserves all cross-layer sums.

## One-factor and contract boundaries

If S02 accepts the proposal, it may register only a Step6 aggregation intervention relative to Iter29: keep Iter29's per-layer fixed curvature values, residual subtraction, residual transport, quantizer/codebook, STE, quantization loss, reconstruction loss, behavior loss, optimizer, Stage1 assets, Stage3 model/evaluation and protocol unchanged. No d-HSTE, additional loss, optimizer/Sinkhorn change, learned/time-varying curvature, or second mechanism. The exact one-factor boundary and patch scope are later S05/S06 decisions.

HRA changes aggregation, outside FCCR-1's existing mapping-only scope. S02 must state this incompatibility accurately; S04 must explicitly adjudicate a between-iteration contract transition and a machine-readable contract. The FCCR-1 preflight is hardcoded; S00 records this as an unresolved S04/S07 prerequisite. S02 does not silently amend FCCR-1 or modify/bypass the preflight.

## Falsifiability and evidence interpretation

A candidate hypothesis must distinguish:

1. **Implementation invalidity/inactivity**: malformed/nonfinite ball points, wrong curvature/order/associativity, output shape mismatch, saturated/degenerate mapping, zero/nonfinite intended gradients, or no measurable direct Step6 intervention versus the Euclidean sum. Lightweight preflight/MVG evidence may disprove execution viability; do not spend full training if a registered mechanism is directly inactive/invalid under its locked definition.
2. **Downstream performance objective**: one authorized Iter31 Stage2+Stage3 run reaches a protocol-valid final test. Strict promotion criterion is `test_recall@10 > 0.065`; compare the exact result to Iter29 `0.05921064085377531` and report all metrics and `n_eval`. Missing the target is promotion failure, not by itself proof that HRA was inactive or mathematically invalid.
3. **Inference limit**: one seed-42 observation cannot establish robust causal effect or estimate run variance. It can falsify the registered target-level hypothesis for this protocol, but cannot prove broad HRA-family failure. No new replication/diagnostic iteration is allowed under the current skill.

## Candidate response requirements

Both candidates must independently return:

- one-sentence research question;
- exact falsifiable directional hypothesis and anticipated downstream metric versus Iter29, without claiming guaranteed success;
- exact right-nested common-reference equation and explicit tangent `e_l` / ball `q_l` semantics, including treatment of the unclipped identity;
- whether and why this is a distinct, forward performance-seeking structural mechanism rather than replication, ablation, or root-cause work;
- direct rationale and paper/code evidence, including HRA-only caveat and geometry limitation;
- exact intervention boundary and unchanged factors;
- implementation-validity and scientific falsification criteria, plus abort conditions;
- confidence and known risks; explicit `USER_INPUT_REQUIRED=NO` and no Stage2/Stage3 authorization.

Do not decide a new curvature mapping, baseline, d-HSTE, or extra parameter. Do not write a contract or code at S02.