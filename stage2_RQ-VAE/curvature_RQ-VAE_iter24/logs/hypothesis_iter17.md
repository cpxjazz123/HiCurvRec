# hypothesis_iter17 (Agent C)

## Mechanism

P17B changes only the Stage2 optimizer: retain the cyclic, learnable `c_l(t)` model and use AdamW normally, then scale each layer's adaptive parameter displacement by

`m_l(t) = [1 / (c_l(t) + 1e-3)] / M_ref`,

where `M_ref = mean_j(1 / (c_j(0) + 1e-3))` is computed once at initialization and held fixed. Implement this as per-layer AdamW learning rates `base_lr * m_l(t)`. Do not multiply raw gradients before AdamW, because Adam's moment normalization can cancel a scalar layerwise factor.

## Direct effects (Agent D must check every item)

- **DE-1:** each layer's optimizer group uses the specified `m_l(t)`; `c_l` and `m_l` remain finite and positive at steps 0, 50,000, and 100,000, with `m_l(0)` mean equal to 1 and at least one layer's midpoint multiplier differing from its step-0 multiplier by more than `1e-3`.
- **DE-2:** under the same 640-pair batch, each quantizer layer has a finite relative parameter update greater than `1e-7` after five ON steps.
- **DE-3:** same-seed, same-batch ON/OFF training for 200 steps yields `abs(L_on - L_off) > 1e-6`; the ON path uses `m_l(t)`, while OFF uses base AdamW learning rates.

## Proxy hypothesis (descriptive only; not a Stage2 gate)

- Relative to iter16's fixed-curvature fingerprint, cyclic c is restored and `H(L1|L0)` is at least 5.45; record L0/L1/L2 utilization, collisions, unique SIDs, entropy, and curvature schedule regardless of direction.
- The Stage3 hard target remains `test_R@10 > 0.065`; SID proxy metrics do not substitute for the full downstream run.

## Comparator

iter16: fixed `c_l=[0.663, 0.435, 0.433]`, unique `22827/24587`, `H(L1|L0)=5.4549`, `test_R@10=0.05688`. Compare optimizer effect against the same iter8 warm-start and unchanged Stage3 pipeline.

## Failure interpretation

If any DE item fails, classify as `IMPLEMENTATION_FAIL` or `ACTIVATION_FAIL`, repair the optimizer/checker and rerun MVG; do not start Stage2 GPU training. If DE passes but Stage3 misses 0.065, record the outcome and continue with a different gap-closing mechanism rather than treating repeated Stage2 outcomes as evidence that the target is unreachable.
