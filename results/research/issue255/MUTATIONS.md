# Issue255 designated mutation checks

Acceptance: every deliberately incorrect value or route must raise the assigned assertion. An import error, a crash or a timeout is not a catch. The actual execution outcome is in the candidate's `tests/test_mutations.py` output and in `TEST_RESULTS.json`.

| # | Mutation | Category | Designated detector |
|---|---|---|---|
| 1 | The squared-distance coefficient is doubled | numeric | `geometry.expert_squared` hand answer |
| 2 | The balanced rule is replaced by argmin | structural | the balanced hard IDs must be `[1,0]` |
| 3 | The selection cost is detached before the soft assignment | routing | the selection cost must receive a nonzero gradient |
| 4 | The A.1 overall sign is flipped | numeric | the full-reference A.1 value must be 4/3 |
| 5 | The invalid-target mask is ignored | structural | an invalid target cannot contribute to the centre loss |
| 6 | The hard forward is replaced by soft codewords | routing | the hard prefix must equal the selected codewords |
| 7 | The geometry weights are normalised over the item axis | structural | the weights must sum to one over the experts |
| 8 | The subgraph attention mask is ignored | structural | a masked subgraph must receive exactly zero attention |
| 9 | The expert query is detached from the attention | routing | the query must receive a nonzero gradient |
| 10 | The layer weights are routed to the first level for every level | routing | every level must own its own expert mixture |
| 11 | The expert weights are normalised over the level axis | structural | each level's weights must sum to one over the experts |
| 12 | Every level reuses the final cumulative prefix | routing | each level must carry its own cumulative prefix |

Mutations 1–9 are the inherited #253 coverage, retargeted to the layer-wise gate where the single-level attention was replaced. Mutations 10–12 are new and cover the mechanism this issue adds: the per-level routing, the per-level expert normalisation and the cumulative prefix construction.

Mutation 10 and 11 are detected by the production attention and not by a re-implementation: the mutant replaces `LayerExpertAttention.forward` or the softmax axis inside `model/layer_gate.py`. Mutation 12 replaces `cumulative_prefixes` in the same module.

The red phase is recorded in the repository history: the first version of the mutation suite patched the producer module while the test held an imported callable alias, so the mutant never reached the consumer; that version is not counted as a catch. Every row above was executed with the current source and caught by its designated assertion.
