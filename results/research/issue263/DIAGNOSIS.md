# Issue #263 — diagnosis before implementation

## Observed failure history

- #260–#262: the end-to-end GNN + gate + three-curvature RQ-VAE did not produce a
  stable recommended-recall improvement over the frozen TIGER baseline.
- #260 (R05): a token-based role classifier was only +2.6 pp above a degree-only
  feature baseline; that is observational and does not prove the tokens carry the
  intended directed topology.
- #262: visible-graph shared-in correlation (+0.958) was not the true directed
  held-out label; the corrected structure-test association changes sign. Curvature
  MAE and magnitude discrimination are not interchangeable.
- #221: with the induced adjacency supplied, directed graphlet/orbit recognition is
  strong (subtype 0.71 / unconditional orbit 0.61); without adjacency, item-only
  prediction is at chance. The information is in the local graph, not in a single
  scalar degree or the 768-d semantic vector.

## Candidate root causes

- **H1 — optimization entanglement.** The GNN, gate and quantizer are trained
  together; the quantization objective and assignment floor can dominate the GNN
  update, so the GNN is never optimized as a stable directed-structure encoder.
- **H2 — missing explicit multi-role/distribution supervision.** The parent GNN
  receives no direct role/orbit distribution or curvature-distribution target;
  it can satisfy the RQ-VAE objective with shortcuts (frequency/degree) and a
  single dominant structure.
- **H3 — online cost/instability.** Recomputing complete 1-hop/2-hop egos online
  every step is expensive and couples graph data to the optimizer; repeated
  online GNN work is a plausible source of both cost and unstable gradients.

## Competing explanations / null

- **N1 (null):** the parent GNN already encodes the needed topology; freezing it
  cannot help because the bottleneck is Stage3/RQ-VAE, not topology.
- **N2:** the pretrained GNN learns only global frequency/degree; role-distribution
  diagnostics should then not beat a degree-binned frequency baseline.
- **N3:** the frozen tokens can be useful, but the gate cannot exploit them; gate
  weights and downstream SID would remain unchanged.

## Evidence available before new training

- Existing directed graph and full 1-hop/2-hop direction ego assets; existing
  role tables (`own_orbit`, `neighbour_orbit`, `own_subtype`) and DGDV counts;
  existing curvature target table and offsets.
- Existing parent R01 source/checkpoint/SID (R2 102, test R@10 0.052734) and TIGER
  baseline (test R@10 0.0555) are frozen references, not retrained.
- Existing #262 full curvature table and quantisation/gradient endpoint receipts
  are reused as prior diagnostics, not as the new candidate's validation.

## Six questions required before formal Stage2

1. **Which problem?** The GNN is trained end-to-end with quantization/gating and
   receives no explicit multi-role/curvature-distribution supervision; its
   structural signal is unstable and its online cost is high.
2. **Evidence?** #260–#262 above: no stable recall gain, R05 role probe only +2.6 pp,
   #262 visible-label correction, #221 chance with no adjacency. These are
   observations and prior evidence, not a new causal proof.
3. **Why can the mechanism act?** Pretraining the same GNN on full train-graph
   role/orbit and curvature histograms directly shapes its four ego tokens; freezing
   removes quantizer/gate gradients from the GNN and Stage2 reads the same tokens.
4. **Substantive difference?** Parent trains GNN online with RQ-VAE; candidate trains
   it once with explicit distribution targets, then removes heads, sets
   `requires_grad=False`, and Stage2 has zero GNN forward/backward calls. This is not
   a new loss on the same online path.
5. **Metrics?** Held-out role KL/CE, minority-role/top-1 and curvature histogram
   KL; frozen cache identity; Stage2 SID collision <0.40; Stage3 test R@10 >0.065 and
   comparison to frozen TIGER; four-card epoch time.
6. **If it fails?** We learn whether structure is limited by representation,
   objective, or downstream gate/SID; a negative result is recorded with the
   above diagnostics, not hidden or converted into tuning.

## Status

- H1/H2/H3 are **UNTESTED** before pretraining.
- N1/N2/N3 are the explicit nulls that the diagnostics must distinguish.
- No formal Stage2 will start if pretraining/cache diagnostics fail or if the
  GNN is called online in Stage2.
