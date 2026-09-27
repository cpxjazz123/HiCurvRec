# Iter31 HRA-Inspired Step6 Hypothesis — S02 canonical

## Research question

Can replacing only Iter29's Euclidean Step6 tangent sum with common-reference, right-nested hyperbolic aggregation produce a protocol-valid Instruments recommendation improvement over Iter29 and reach the strict `test_R@10 > 0.065` target?

## Registered equation and coordinates

For each layer `l ∈ {0,1,2}`, `e_l` is the tangent-coordinate quantizer embedding returned by `Quantize.forward` (including its STE value during training), not a Poincaré-ball point. Define the layer-curvature ball codeword and transfer it radially to the layer-0 reference curvature:

```text
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
```

Keep layer order L0,L1,L2 and the prescribed right nesting, then return a tangent-coordinate value for the existing decoder:

```text
h = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
 z = log0^{c0}(h)
```

The registered Step6 output `z` replaces only Iter29's `z_E=e_0+e_1+e_2`. The existing decoder and reconstruction-loss path (using `c0`) remain unchanged. Do not apply `log0` directly to tangent `e_l`; Möbius addition is not generally associative, so do not reassociate or reorder this expression.

In the valid, unclipped origin-map domain, `log0^{c_l}(exp0^{c_l}(e_l))=e_l`, hence `q_l^0=exp0^{c0}(e_l)`. This pointwise coordinate identity is not a cross-curvature sum-preservation law. Actual Iter29 helpers project exponential-map outputs to radius `(1-eps)/sqrt(c)` (`eps=1e-6` by default) and clamp the logarithm's `sqrt(c)||x||` argument to at most `1-1e-5`; therefore the identity is not assumed in clipped/saturated cases, which later preflight/MVG must measure.

This is an HRA-inspired common-reference aggregation hypothesis, not the paper's exact HRA residual telescope for Iter29. The paper's §4.1 cancellation pairs left residual subtraction and reverse-nested aggregation under one shared curvature. Iter29 instead has heterogeneous fixed `c_l` and explicit cross-layer residual transport. The radial transfer is not a universal Möbius-addition homomorphism (the canonical S00 witness has nonzero difference 0.01865103694), so no exact inversion, residual reconstruction, or telescope is claimed.

## Directional downstream prediction and rationale

Under the S01-locked seed-42 protocol, the single protocol-valid Iter31 run is hypothesized to improve `test_R@10` over Iter29's exact `0.05921064085377531` and reach the strict target `test_R@10 > 0.065` (more than `0.005789359146224693` above Iter29). This is a falsifiable prediction, not a guarantee. Report R@5, R@10, NDCG@5, NDCG@10, and `n_eval=57439` against the exact Iter29 record. A valid active run missing either directional improvement or the strict target falsifies that performance prediction for this protocol observation; missing the threshold is promotion failure, not by itself mechanism invalidity or inactivity. A protocol-invalid/missing result is not a valid test. One run cannot establish robust causal effect or estimate variance.

The structural rationale is that a common-reference curved composition may provide the unchanged decoder a more geometry-consistent combination of coarse-to-fine codes than flat tangent summation. The prior is risky: paper §6.6/Table 8 reports HRA-only recommendation residual reconstruction error 5.49 versus 0.03 for full HRA+d-HSTE. This is an adverse risk signal, not an Instruments R@10 estimate or evidence that this Step6-only operation is inactive. The paper's recommendation setup is Beauty, four layers, and shared `c=1`, unlike Iter29's three-layer Instruments model and heterogeneous curvatures. d-HSTE is explicitly excluded.

## Scope and unchanged factors

The only proposed conceptual intervention is Step6 aggregation. Preserve Iter29's fixed layer curvatures and values; Stage1 embeddings/data; residual subtraction and cross-layer residual transport; quantizer, codebooks, assignments and STE; all existing quantization, commitment, reconstruction and behavior losses; optimizer, Sinkhorn settings, schedule, seeds and protocol; decoder; Stage3 model, data, evaluator and evaluation protocol. No d-HSTE, added loss, curvature change/learning/schedule, optimizer or Sinkhorn change, codebook change, Stage1/Stage3 change, or second mechanism. S05 must later adjudicate the one-factor diff.

## Direct-effect, gradient, and numerical preconditions

Before any full run, later adjudicated preflight/MVG must establish correct tangent-to-ball semantics, source/target curvature use, exact layer order/right nesting, expected decoder input shape, and finite ball/intermediate/output values. The literal aggregation must exhibit a measurable nondegenerate direct Step6 output intervention versus the Euclidean sum on actual same-input model values; assess projection/log-clamp incidence and rule out destructive saturation. The existing training path must retain finite, nonzero intended gradients through the changed decoder-facing path and codebooks/encoder as appropriate. These are implementation/activation preconditions, not evidence of downstream benefit. If the registered map is invalid, nonfinite, effectively inactive, materially saturated, or gradient-broken, and fixing it requires changing the equation or another locked factor, abort before full training; an implementation repair that merely conforms to the registered equation is not a mechanism change.

## Contract and authorization boundary

FCCR-1 remains active and unchanged at S02; this aggregation intervention is outside its mapping-only scope. S04 must explicitly adjudicate any between-iteration contract transition and a compatible machine-readable contract/preflight path. The known FCCR-1-hardcoded preflight may not be bypassed or silently modified. Stage3 routing is also unresolved in S01 and requires later adjudication. This artifact authorizes no code, contract, checker, Stage2, or Stage3 change/run. No training is authorized now.

```text
CONFIDENCE=MEDIUM
USER_INPUT_REQUIRED=NO
NO_TRAINING_AUTHORIZATION=TRUE
```