# Hyperbolic SID candidate proposals (Role M)

## Frozen protocol

These are Stage-2-only candidate mechanisms: keep the 32-dimensional latent, three 256-entry codebooks, three integer SID tokens, original data/splits, and the original Stage-2 and Stage-3 recipes unchanged. A candidate survives only if the same run/evaluation protocol strictly exceeds **all** of: Hard-L1 ER `16.38953942232631`, Stage-3 Recall@10 `0.06406796775709883`, and NDCG@10 `0.03530651701940677`. Hyperbolic auxiliary losses below do not replace the frozen reconstruction, quantization, or Stage-3 objectives.

Notation: use the 32-D Poincaré ball of curvature `-c`, with fixed `c>0`, radius `1/sqrt(c)`, origin `o=0`; use one declared `c` for the entire candidate (not a curvature sweep). Let `x` be the item latent, `p` the current residual-stage host, and `a,b` points in the ball. `Log^c_p(q)` and `Exp^c_p(v)` are the Riemannian log and exponential maps. All vector norms and inner products in a tangent space are with metric `g_p=λ_p² I`, `λ_p=2/(1-c||p||²)`. Clamp only for numerical safety, and report the clamped fraction.

## MECHANISM 1 — Signed geodesic-triangle-area assignment

**Operator and domain.** For stage `s`, form the target point `q=Exp^c_p(α Log^c_p(x))`, with fixed `α=1` (the stage target itself). Each of the 256 entries is a point `z_{s,k}` in the same Poincaré ball. Choose the code by minimizing

`S(k|x,p) = - A_c(p,q,z_{s,k}) + β d_c(q,z_{s,k})²`,

where `A_c` is the **oriented hyperbolic area** of the geodesic triangle `(p,q,z)`, with orientation fixed once by an ordered orthonormal 2-frame at `p` and the sign taken in the plane spanned by `Log_p(q), Log_p(z)`. In dimensions above two, define `A_c` as the signed area of that unique geodesic 2-simplex; degenerate/collinear triangles have area zero. `d_c` is the Poincaré geodesic distance, `2/sqrt(c) * artanh(sqrt(c)||(-p)⊕_c q||)`.

The area term is not an alternate distance: it is the integrated Gaussian curvature over the geodesic simplex (Gauss–Bonnet: `A_c=(π-θ_p-θ_q-θ_z)/c` for a nondegenerate triangle). It changes sign under reflection and depends jointly on the three vertices. The distance term is only a tie-breaking/locality regularizer; require a pre-registered `β` small enough that area contributes at least half the variance of `S` over the 256 candidates on validation data. If that condition fails, the candidate is not a hyperbolic-area test.

**Curvature, basepoint, and semantics.** Curvature is constant `-c`; basepoint is the current residual host `p` (not a learned per-code basepoint). `Log_p(q)` is the actual residual displacement; `Log_p(z_k)` is the candidate ray from the same host. Area measures the signed curvature enclosed between those two geodesics, not radius or a Euclidean residual coordinate. Use the existing residual quantizer update to decode the chosen code and update the next host; no token-count/layout changes.

**Gradient flow.** Backpropagate the frozen Stage-2 reconstruction/quantization losses through `q`, the selected-code straight-through path, and `z_{s,k}` as usual. Add a small differentiable soft-assignment area term `-τ logsumexp_k(-S_k/τ)` during training, anneal `τ` to the frozen hard-assignment regime, and stop gradients through the discrete argmin. Gradients therefore reach item encoder, selected codeword, and the candidate set through triangle area and geodesic distance; no gradients to `c` or `p` as a separately learned basepoint.

**Required invariants.** (i) Area is zero for collinear geodesic triples and reverses sign under swapping `q,z`; (ii) `A_c` agrees with the Gauss–Bonnet angle formula within numerical tolerance; (iii) score is invariant to a common hyperbolic isometry applied to all three points, with orientation transformed consistently; (iv) a Euclidean `c→0` control makes area vanish, so assignment must demonstrably depend on nonzero curvature; (v) exactly 256 codes/stage and exactly three emitted indices.

**Falsifier and metric prediction.** Kill this mechanism if signed area is numerically negligible versus the distance term, if assignment is unchanged by replacing `A_c` with zero (measure per-stage assignment agreement), or if host-conditional code usage collapses / Hard-L1 ER fails to exceed threshold. The hypothesis is that signed, host-relative wedge structure splits residuals that have similar radii but different geodesic directions, increasing Hard-L1 ER; if those distinctions align with user-item neighborhoods, Stage-3 Recall@10 and NDCG@10 may also rise. This is explicitly a risky prediction, not a guaranteed win: reject unless **all three** frozen thresholds are strictly beaten.

## MECHANISM 2 — Non-associative gyro-composition quantization

**Operator and domain.** Use Möbius addition on the same Poincaré ball:

`u ⊕_c v = ((1+2c<u,v>+c||v||²)u + (1-c||u||²)v) / (1+2c<u,v>+c²||u||²||v||²)`.

For each stage, let `p` be the current decoded host, `r=Log^c_p(x)` the target residual in `T_p`, and `v_k=PT^c_{o→p}(Log^c_o(z_{s,k}))` a code tangent vector transported from the origin. Set `u_k=Exp^c_p(v_k)`. Use a learned bounded tangent transport `T_s` (initialized to identity, constrained to be invertible) on residual vectors, then score

`G_k = Log^c_p( ((p ⊕_c Exp^c_o(T_s Log^c_o(z_{s,k}))) ⊕_c Exp^c_p(r)) )`,
`S_k = ||G_k||²_{g_p} + γ || gyr[p, Exp_o(T_s Log_o(z_k))] Exp_p(r) - Exp_p(r) ||²_{g_p}`,

and choose `argmin_k S_k`. Here `gyr[a,b]w = -(a⊕b) ⊕ (a⊕(b⊕w))` is the Möbius gyration (implemented with the matching gyrovector convention). The second term explicitly penalizes or rewards, according to a fixed signed `γ` chosen before training, the non-associativity-induced rotation of the residual when composed after the code. This is an order-sensitive composition operation at the actual SID decision, not a post-hoc geometric label. Fix `γ>0` and report each score component; do not tune it on test metrics.

**Curvature, basepoint, and semantics.** Curvature is `-c`; the canonical chart basepoint is origin `o`, while residuals live at the current host `p`. `Log_o(z_k)` is the code's canonical tangent semantic, parallel transport places that semantic in the host tangent space, and `Log_p(x)` is the item residual. Möbius addition composes points, while gyration records the rotation forced by curvature-dependent non-associativity. Code order and residual order are therefore semantically consequential.

**Gradient flow.** Stage-2 reconstruction gradients flow through `Exp`, `Log`, parallel transport, Möbius addition, gyration, and `T_s` into encoder residuals and codewords; use a softmin relaxation for training and hard argmin for emitted SID. The discrete index itself is not differentiated. Keep the existing Stage-2 recipe otherwise fixed and use the existing straight-through estimator for the selected embedding.

**Required invariants.** (i) `u⊕0=0⊕u=u`; (ii) as `c→0`, Möbius addition tends to vector addition and gyration tends to identity; (iii) gyration is metric-preserving at the appropriate tangent space; (iv) numerically verify that the two association orders differ for non-collinear nonzero inputs and coincide in the Euclidean limit; (v) transport preserves tangent norms; (vi) all points remain strictly inside the ball; (vii) code count/token count remain 256/three.

**Falsifier and metric prediction.** Kill it if measured gyration is near identity on actual residual/code pairs, if removing the gyration term leaves assignments effectively unchanged, if `c→0` reproduces assignments, or if the ball-boundary/clamp rate is material. The hypothesis is that the order-sensitive residual rotation separates behaviorally distinct item residuals that a radius/distance quantizer conflates; expect a possible increase in Hard-L1 ER from changed code partitions and a possible downstream Recall@10/NDCG@10 gain from semantically useful order-sensitive codes. If it only perturbs assignments without improving downstream ranking, it fails. Strictly require all three frozen thresholds.

## WHY NOT EUCLIDEAN

Mechanism 1 selects by signed **curved geodesic simplex area**, which is an integrated curvature quantity with a handedness/sign and a nonzero Gauss–Bonnet angle defect. In the flat limit the area term disappears; replacing it with a Euclidean norm or dot product cannot preserve its isometry-invariant three-point area and orientation semantics.

Mechanism 2 selects using **Möbius composition plus gyration**, whose non-associative rotation is zero in the flat limit and nonzero for generic curved, non-collinear inputs. A Euclidean vector addition quantizer has associative composition and identity gyration; matching it requires removing the operation that contributes to the selected-code score. Neither proposal is a curvature-only sweep or a reparameterized norm comparison.

## FALSIFIERS

Run each candidate with the exact frozen data/splits and unchanged Stage-2/Stage-3 recipes. Report the three target metrics, the baseline-to-candidate assignment agreement per stage, code usage/perplexity, host-conditional assignment entropy, curvature-operation contribution to score variance, numerical invariant errors, and clamp rate. First apply the mechanism-specific falsifiers above. Final success is conjunctive and strict:

- Hard-L1 ER `> 16.38953942232631`
- Stage-3 Recall@10 `> 0.06406796775709883`
- Stage-3 NDCG@10 `> 0.03530651701940677`

Failure on any single metric, any frozen constraint change, or failure to demonstrate a non-negligible genuinely hyperbolic contribution rejects the candidate. No threshold relaxation or recipe/data/split modification is admissible.
