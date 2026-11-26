# Issue #265 frozen mechanism contract

Frozen before candidate implementation. Parent: #263 frozen GNN/token candidate; no GNN update.

## Selection

For residual `r_l` and shared codeword table `E_l`, the selection cost is exactly
`C_l = ||r_l - E_l||_2^2` (Euclidean squared), with no gate or curvature weight. Keep the parent
floor and last-level Sinkhorn rules, the original residual update, and the hard IDs. The final
`q_i = sum_l c_i,l` is the true sum of the three selected codewords; the forward path is hard and
the backward path uses the parent's proven STE/soft surrogate. The gate has no path into selection.

## Local mixed-curvature supervision

Frozen token -> trainable attention -> score -> softmax gives per-level expert weights
`w_{i,l,k}`. For the center of a curvature triple, aggregate the three levels by the pre-registered
equal mean: `w_{i,k} = (1/3) sum_l w_{i,l,k}`; no learned extra parameter. For a fixed center `i`
and any endpoint `j`, all references use that same center weight:
`D_i(j,m)^2 = sum_k w_{i,k} d_k(q_j,q_m)^2`,
where `d_k` is the existing fixed Euclidean/hyperbolic/spherical squared distance and the frozen
tangent scale. The distance is a local proxy, not a global Riemannian metric. The existing target
triples, reference populations, components, Huber/target scale and reduction are reused unchanged.

The structural loss uses the final hard `q`, not the per-level prefixes; target values and the
fixed evaluation remain unchanged. `L_total = L_RQVAE + lambda L_curvature`; `lambda` follows the
parent gradient-budget calibration, not a post-hoc Recall search.

## Gradient route

`q` must carry gradient to the encoder and codebooks through the parent STE; the gate must receive
gradient through `w_{i,k}` in the curvature term; the GNN/token cache is frozen and must have zero
forward/backward/optimizer participation. Codebook gradient routing is explicit and counted once.

## Fixed independent evaluation

Report a gate-independent, equal-weight three-expert curvature error/loss on the same frozen targets,
separately from the trainable local objective. A gate-only reduction in the training loss is not
accepted as structural improvement. Report L1/L2/L3 positive/negative prefix rates, ER, direct
behavior-edge and role-structure fidelity.

## Pass conditions

Euclidean selection is invariant to gate changes; structural loss produces nonzero encoder, gate and
codebook gradients; fixed evaluation shows a real representation/SID change without numeric failure.
Then run one formal Stage2 (3000, four cards, seed42) and one frozen Stage3 (150, beam20), with SID
collision <0.40 and the same §63 metrics. R@10 is reported once; a negative result is a valid result.

## Non-goals

No InfoNCE/behavior-edge/edge-prediction loss, no arbitrary codeword alignment, no online GNN, no
public infra changes, no test-guided mechanism selection, no second seed or retry.
