# Issue 247 frozen contract

2026-11-23. No training, PPR, new loss, baseline modification or historical asset overwrite.

## Acceptance
C1–C10 are the ten ordered items in `issue.json`, verbatim authoritative acceptance. RUN_LOG records its body hash. Every item requires actual readable assets and numerical evidence; reports alone are insufficient.

## Paper and official implementation crosswalk
Source: https://arxiv.org/html/2412.11085v1 Appendix A.1, Eq. A.1. Official source: https://github.com/Vic-GoodLuck/GraphMoRE commit 1948d3c5f19f7251cadd8d1118c06f0e294f96b3.

|Module|Paper|Official implementation|Project choice|
|---|---|---|---|
|Sectional statistic|A.1: mean_a[d(a,m)^2+d(b,c)^2/4-(d(a,b)^2+d(a,c)^2)/2]|Public tree has no sectional-statistic implementation; utils.cal_shortest_dis uses networkx.Graph shortest paths|Implement A.1 literally, independently compare scipy BFS distances with networkx shortest paths and doubled-distance variance identity; never substitute Ricci|
|Triples|m and two neighboring nodes b,c|No released triple sampler|Distinct unordered neighbor pair b<c, one deterministic uniform pair per eligible center; retain pair population and coverage|
|References|a ranges over V, uniform mean|No released reference sampler|One fixed seeded permutation of all item indices; prefixes 16,64,256. Exclude only unreachable references; include a=m,b,c as A.1 does; report effective count and component-conditional estimand for disconnected graphs|
|Distance|Unweighted shortest paths|utils.cal_shortest_dis builds nx.Graph|Visible directed support projected to undirected support; direction retained separately. Not directed sectional curvature|
|Normalization|None beyond reference mean|None released|Squared-hop units, no distance normalization, no extra division by d(a,m), no per-reference averaging across triples|
|Topology encoding|Eq.3–5 multi-scale ego + GNN pooling|models/backbone|Only deterministic sparse ego input; no GNN or learned embedding|

Public implementation absence is a provenance limitation, not an invented official comparator. Numerical comparison uses the paper's equation implemented with networkx, not a claim that official curvature code exists.

## Shapes, masks, boundaries
Graph: scipy CSR [N,N], nonnegative finite counts, zero diagonal. ID map: integer item IDs [N], unique and exactly aligned with semantic rows [N,768]. Triple table: int64 [T,3]=(m,b,c), b<c, center adjacent to both. Reference IDs int64 [R], unique. Per-reference contributions float64 [T,R]. Formula evaluated only when all distances finite; unreachable => missing mask, never zero imputation. Empty neighborhoods => no triple, count=0, per-item summaries NaN with valid=false. a=m,b,c remains valid. Coincident triple vertices invalid; duplicate triples/references rejected; NaN/negative distances rejected. Zero classification tolerance 1e-12; reference comparison atol=1e-12, rtol=1e-12. Population std ddof=0; quantiles 0,.25,.5,.75,1. Mean is over finite references per triple, then equal-weight triples per item. Sensitivity uses identical triple sample and nested references, not model tuning.

Known answers for full-reference mean: path 0-1-2 center1 pair(0,2):0; star K1,3 center0 pair(1,2):-0.5; triangle center0 pair(1,2):0.25; four-cycle center0 pair(1,3):1. Including a=m gives 0 on path, not missing. Isolated vertex has no triple. Reversing directed edges leaves this undirected statistic unchanged.
Training interface must not expose held-out targets or full-description features. Evaluation interface is separate and explicitly requested. Train banks reuse v2 strict blocking; visible graph may contain already-seen user-heldout edges, which are not strict targets. Ego loading uses only visible CSR; exact 1/2-hop deterministic on demand; sampled v2 assets remain separately named. Reuse 3-node counts and 3/4-node instance banks without re-enumeration; sampled orbit participation is not full-population counts.

TIGER: bind only Stage3 JSON whose SHA appears in run logs. Checkpoint and NPY remain different identities; reconstruction MSE for Stage3 SID is NA unless checkpoint provenance matches. Recall requires log hash and SID hash binding. Prefix3 collision=1-distinct/N; duplicate share=sum(size of groups with size>1)/N; entropy in bits.

## Independent reference and mutations
Production uses scipy sparse shortest_path + per-reference parallelogram arithmetic. Reference uses networkx BFS and identity (4*d_am^2+d_bc^2-2*d_ab^2-2*d_ac^2)/4, Fraction rational known answers, separate graph construction. No production import. Mutations: sign, quarter coefficient, wrong reduction, missing-reference zero fill, wrong source graph routing, wrong ID alignment; each must trigger a named invariant assertion, not crash. No training loss/gradients apply: MODULE_CHECKS records numerical modules and marks gradient checks NA (no trainable modules).
