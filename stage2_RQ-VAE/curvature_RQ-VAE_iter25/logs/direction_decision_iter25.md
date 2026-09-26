# Iter25 Direction Decision

## Evidence and context

The Iteration Bridge identifies a **hypothesis**, not a proven cause: Iter24's inverse-square-root target-frequency reweighting activated, but its run had lower conditional entropies, more three-token collisions, and lower Stage3 test recall than Iter18. The bridge forbids another behavior-loss change and selects the unresolved mapping from branching and raw-residual scales to the learnable cyclic layer prior.

The applied Iter11 prior was `[0.001, 0.932889, 1.0]`, because its `layer_norms.json` was missing and training used a fallback. Iter25's user-accepted objective replaces that mapping with:

```text
h_l = log(B_l) / max_j log(B_j)
r_l = m_l_raw / max_j m_j_raw
u_l_prior = 0.01 + 0.98 * (h_l + r_l) / 2
B = [19.32, 1.46, 1.015]
m_raw = [1.000, 0.10941, 0.09331]
u_l_prior = [0.990000, 0.126233, 0.058186]
```

The existing learnable layer scales and cyclic `c_l(t)` remain; only the prior values change. This replaces the Iter11 mapping, so it is not a pure add-branching ablation. Iter24 and Iter18 proxy/recall deltas are recorded in `iteration_bridge.md`.

## Candidate scorecard

`✓` = pass; `△` = partial; `✗` = fail. Criterion (e) is mandatory.

| Criterion | P1: layer-wise curvature prior | P2: cyclic `c_l(t)` | P3: curvature-adaptive margin | Pre-registered bounded prior |
|---|---|---|---|---|
| (a) Explicit path for Iter24 hierarchy/collision regression | △ — a per-layer prior changes scale allocation, but does not reverse the reweighting that the bridge associates with Iter24's regression | ✗ — cyclic curvature is retained by the bridge; Iter12 already changed the period and regressed | ✗ — an added margin loss is outside the prescribed mapping and has no demonstrated path to the observed conditional-entropy gap | △ — same limitation as P1; directly changes the unresolved layer-prior mapping, not the Iter24 behavior loss |
| (b) 2023+ empirical evidence | △ — hyperbolic layer curvature and hierarchical embedding are studied, but not this learned prior in RVQ | △ — only adjacent curvature-scheduling evidence in Agent A's report; no direct RVQ/recommender cyclic-`c` result | △ — only adjacent branching/recommender evidence in Agent A's report; no direct RVQ-margin result | △ — primary sources support the surrounding ideas, not this exact formula or SID pipeline |
| (c) Novelty vs mechanisms tried | △ — per-layer curvature is an existing family; this exact mapping is new | ✗ — cyclic schedule is already in Iter11 and the period variant was tried in Iter12 | ✓ — no margin-only RVQ experiment appears in the ledger, but it is not the accepted objective | △ — the exact branching-log plus raw-residual decomposition is untried; the learnable scales themselves are not new |
| (d) Stage1-only curvature ceiling risk | ✓ — no Stage1 or embedding changes | ✓ — no Stage1 changes | ✓ — no Stage1 changes | ✓ — only Stage2 scale initialization/anchor changes |
| (e) Gap-closing relevance (mandatory) | △ — generic per-layer curvature does not specify the bridge's unresolved mapping | ✗ — does not address the bridge's targeted prior mapping | ✗ — introduces a different loss and does not target the prescribed prior mapping | ✓ — implements the exact bridge objective: combine branching capacity and measured residual scale into the learnable cyclic-prior anchor |
| (f) Curvature relevance / keyword compliance | ✓ — layer-wise curvature | ✓ — cyclic curvature | ✓ — curvature-dependent margin | ✓ — prior controls each layer's cyclic curvature scale |

## Recommendation

**Unique recommendation: implement the pre-registered bounded branching-plus-raw-residual prior** `[0.990000, 0.126233, 0.058186]`, initializing and anchoring the existing learnable layer scales. This is P1 in the concrete form already accepted by the user and selected by the bridge. It is exactly implementable without another loss, parameter, or code path. Preserve Iter11's cyclic formula, commitment/Sinkhorn settings, optimizer, warm-start, and Stage3 protocol.

P2 is already present and its period-only variant regressed; P3 would add an unrequested margin mechanism. Neither replaces the unresolved prior mapping named in the bridge. Do not use Stage2 SID metrics as a progression gate. Only complete Stage3 `test_R@10 > 0.065` meets the adoption target.

## Literature evidence and limitations

Primary-source checks support only adjacent premises; **none directly evaluates the Iter25 formula in an RVQ recommender**:

- Chen, King, Yang, Zhou, and Ying, *Hyperbolic Representation Learning: Revisiting and Advancing*, ICML/PMLR 2023: https://proceedings.mlr.press/v202/yang23u/yang23u.pdf. Its hyperbolic layer equations use layer-specific curvature notation, and the paper studies hierarchy alignment; it does not establish this learned prior.
- van Spengler, Berkhout, and Mettes, *Poincare ResNet*, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/van_Spengler_Poincare_ResNet_ICCV_2023_paper.html. The abstract reports identity-based initialization that preserves norms in deep hyperbolic residual networks; it does not study residual-codebook medians.
- Kratsios, Hong, and Sáez de Ocáriz Borde, *Capacity Bounds for Hyperbolic Neural Network Representations of Latent Tree Structures*, arXiv 2023: https://arxiv.org/abs/2308.09250. It proves capacity results for embedding finite weighted trees in hyperbolic spaces; it does not prescribe `log(B_l)` in this model.
- Welz, Flek, and Karimi, *Multi-Hop Reasoning for Question Answering with Hyperbolic Representations*, arXiv 2025: https://arxiv.org/abs/2507.03612. Its curvature ablation reports that data-derived δ-hyperbolicity initialization outperforms random initialization in its multi-hop QA setup; this is not an RVQ or recommender result.

Agent A's search report remains the record of retrieved candidates. Two URL-only results there are unverifiable and are not used as evidence. The exact Iter25 composition is an extrapolation from adjacent literature and the project's measured bottleneck, not a published result.

## Risks and falsifiable claims

1. **Persistent downstream ceiling:** the failed-mechanism ledger records seven class-3 outcomes (Iter7–13); this prior may still fail to improve Stage3.
2. **Mechanism mismatch:** Iter24's behavior-loss reweighting and Iter25's scale prior are distinct. If the behavior-loss change is causal, changing the prior may not recover recall.
3. **Learnable-scale drift:** the current implementation clamps the scale where it is consumed and regularizes it toward its prior, but has no momentum-restricted per-step drift. Iter25 will not add clipping because that would be a second novelty; checkpoint scales must be reported.
4. **Descriptive proxies only:** H(L1|L0), H(L2|L0), code utilization, and collision rates may move independently of downstream recall and cannot stop the run.
5. **Prior-mapping confound:** because Iter11 actually used its fallback `[0.001, 0.932889, 1.0]`, Iter25 changes the applied mapping as well as adding the branching input. Treat comparisons as a prior-mapping experiment, not a one-factor causal proof.

The direct-effect checks are the exact initialized/anchored prior values, bounded finite cyclic curvatures with the expected layer ordering at a fixed cycle phase, nonzero gradient/update through the learnable scales, and the required same-seed 200-step ON/OFF loss difference. The proxy prediction is recovery versus Iter24 in H(L1|L0), H(L2|L0), and three-token collision rate; these remain descriptive, not gates.