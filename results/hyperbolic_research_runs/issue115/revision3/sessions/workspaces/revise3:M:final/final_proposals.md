# Final hyperbolic SID proposals — Role M

## Frozen protocol and evaluation contract

Both candidates retain the frozen protocol: 32-dimensional latent space, three codebooks of 256 entries, exactly three integer SID tokens, unchanged data, splits, Stage-2/Stage-3 recipes, and strict task gates: Hard-L1 ER `>16.38953942232631`, Recall@10 `>0.06406796775709883`, and NDCG@10 `>0.03530651701940677`. All three gates must pass. These are experimental proposals, not claims of established performance or priority.

Fix curvature `-c`, `c>0`, and the open Poincaré ball `B_c={y∈R^32:c||y||²<1}`. Project model points before scoring to radius at most `(1-10^-6)/√c`; record projection fraction and pre-projection maximum radius. Any non-finite score, post-projection out-of-ball point, or per-stage clamp rate above `0.1%` kills the run. Points and tangent vectors are distinguished explicitly below.

### Shared assignment audit and pre-registered decision gates

In the evaluation path, after each frozen checkpoint is loaded and its validation-row stage hosts and codebook points are materialized, but before writing that stage's partition-statistics record, construct two `(n_validation, 256)` score matrices on the exact same rows and points. Call `assignment_audit.audit(full_scores, ablated_scores)` once per stage; serialize its complete return value in that stage's partition-statistics artifact alongside the checkpoint/row identifiers and partition diagnostics. Stage 1, 2, and 3 each receive an independent call and gate; do not pool stages to rescue a failure. The full and ablated matrices differ only by the named mechanism term, use identical projection and lowest-index tie-breaking, and are computed without retraining. The function's pre-registered thresholds are `GATE.min_flip_rate=0.05` (at least 5% actual assignment flips in **each** stage) and `GATE.max_tie_share=0.10` (at most 10% of examples tied, defined by full-score winner/runner-up gap `<=1e-12`). The audit reports the term's scaled contribution to the winner/runner-up gap as a diagnostic; the decision gate is flip rate plus tie-share cap, not a contribution-only or variance test. Failure of either audit threshold in any stage means that candidate has not demonstrated decision materiality and is withdrawn. Also report full/ablated partition health and all applicable task metrics. The audit is a same-checkpoint operator counterfactual, not a substitute for paired training controls when making task-causality claims.

## Candidate 1 — unsigned geodesic-simplex area assignment

At stage `s`, let host `p`, target `x`, and candidate code `z_{s,k}` be ball points. Set `q=Exp_p(Log_p(x))=x`; let `u=Log_p(q)` and `v_k=Log_p(z_{s,k})`, both tangent vectors at `p`. Let `Area_c(p,q,z)` be the **unsigned Riemannian area** of the geodesic triangle with those vertices. For nondegenerate triangles its magnitude may be computed by angle defect `(π-θ_p-θ_q-θ_z)/c`. Define its value as zero when a vertex is repeated or the vertices are collinear; use the declared zero subgradient at this degenerate set.

There is no signed area in this candidate: no orientation datum is used, and no sign, swap-sign-reversal, or reflection-sensitivity claim is made. Unsigned area is invariant under all hyperbolic isometries, including reflections, and is unchanged by swapping `q` and `z`. This is therefore a symmetric area penalty, not an ordered signed-area mechanism.

For fixed finite `β>0`, score
`S_k = -Area_c(p,q,z_{s,k}) + β d_c(q,z_{s,k})²`,
choose `argmin_k S_k`, and resolve ties by smallest code index. Use the existing hard assignment and straight-through path. Any soft-assignment relaxation uses this score and the stated zero subgradient on degeneracies.

**Ablation and audit:** on the same trained checkpoint, same validation rows, same hosts and codebook points, compute `full_k=-Area_c+βd_c²` and `ablated_k=βd_c²`. Do not retrain or change any other score term, point, projection, or tie rule for the audit. Call `assignment_audit.audit(full_scores, ablated_scores)` at the shared evaluation-path location above, separately for each of the three stages. Each stage must have flip rate `≥0.05` and tie share `≤0.10`; otherwise withdraw. Report area and distance components, audit contribution diagnostic, assignment flips, and full/ablated partition statistics. A separate paired cold area-zero training arm may be used to assess task impact, but cannot replace this fixed-point audit.

**Controls and kill criteria:** with fixed bounded chart coordinates as `c→0`, area tends to zero and the score tends to Euclidean squared distance; choices must agree wherever the Euclidean minimum has positive margin. This is a correctness control, not finite-curvature evidence. The candidate is withdrawn if any stage fails the audit gates, any numerical kill criterion is triggered, or the full official evaluation fails any strict task gate. The unsigned area ablation tests the mechanism actually specified here.

## Candidate 2 — origin-chart ordered composition with tangent gyration

Let `o=0`; host `p`, target `x`, and candidate code `z_{s,k}` are ball points. Define tangent vectors at `o`: `h=Log_o(p)`, `r=Log_o(x)`, and `v_k=Log_o(z_{s,k})`. Möbius addition `⊕_c` acts only on ball points. The radial Levi-Civita transport `PT^{rad}_{o→p}` is diagnostic only and is not inserted into the score. For the point map `Gyr[a,b](y)=-(a⊕b)⊕(a⊕(b⊕y))`, use its tangent derivative `D_oGyr[a,b]` at `o`.

For each code, form the ball point `m_k=(p⊕_c z_{s,k})⊕_c x` and tangent vector at `o`, `δ_k=D_oGyr[p,z_{s,k}](r)-r`. With fixed `γ>0`, score
`S_k = ||Log_o(m_k)||²_{g_o} + γ||δ_k||²_{g_o}`,
where both norms use the Poincaré metric at `o`, `g_o=4I`; choose the minimum with lowest-index tie-breaking. Apply the numerical projection policy after point operations and count any such projection toward the clamp rate.

**Ablation and audit:** this is a **same-parameter, same-point counterfactual**. For each validation example and stage, hold fixed the trained full-model checkpoint, all parameters, hosts, targets, codebook points, chart, projection, and tie rule. Compute the full scores above and ablated scores `S^0_k=||Log_o(m_k)||²_{g_o}` by removing **only** `γ||δ_k||²`; do not recompute or alter any points. Call `assignment_audit.audit(full_scores, ablated_scores)` at the shared evaluation-path location above, once separately for each stage. Require flip rate `≥0.05` and tie share `≤0.10` in every stage, or withdraw. Report `δ` and term contributions as diagnostics, but a nonzero contribution without the registered flips does not pass. A retrained `γ=0` arm is an **additional** cold-training control for task impact, with matched seed/order/recipe; it is never a substitute for this same-parameter, same-point audit.

**Controls and kill criteria:** as `c→0` at fixed chart inputs, the score tends to `||h+v_k+r||²_{g_o}` with `g_o=4I`; verify agreement for every positive-margin case using that same metric convention. This is a correctness control, not evidence of finite-curvature mechanism utility. Withdraw if any stage fails the audit gate, the clamp/non-finite criteria trigger, or the full official evaluation fails any strict task gate.

## Prior-art boundary

The supplied review identifies HiHPQ (DOI `10.1609/aaai.v38i5.28261`) as abstract-level adjacent work in hyperbolic product quantization, not verified evidence of either exact assignment operator. The scan is bounded and does not establish priority. Do not claim novelty: the exact prior-art status remains unresolved, and task-gate success would establish performance under this protocol, not priority or non-obviousness.

## Status

**Candidate 1 STATUS: READY_FOR_JOIN_R** — revised to explicitly unsigned area, with all signed-area claims removed; the fixed-point ablation and per-stage audit gates are specified. Advancement remains conditional on passing the audit, numerical, and task gates; no novelty claim is made.

**Candidate 2 STATUS: READY_FOR_JOIN_R** — its ablation is explicitly same-parameter and same-point, removing only the `γ||δ||²` term; retrained `γ=0` is additional only. Evaluation-path audit call and per-stage thresholds are specified. Advancement remains conditional on passing the audit, numerical, and task gates; no novelty claim is made.
