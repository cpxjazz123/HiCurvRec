# Issue #258 mutation tests (designated assertions, not crashes)

Three production modules are mutated in place while a designated assertion is active. Every mutation must raise `AssertionError` from that assertion; a crash, import error or timeout would not count as a catch.

| evidence file | mutation | category | first line of the failing designated assertion | caught |
|---|---|---|---|---|
| MUTATIONS.json | squared-distance coefficient doubled | numeric | `Tensor-likes are not close!` | YES |
| MUTATIONS.json | balanced rule replaced by argmin | structural | `Lists differ: [0, 0] != [1, 0]` | YES |
| MUTATIONS.json | cost detached before soft assignment | routing | `0.0 not greater than 0.0 : selection cost must receive nonzero gradient` | YES |
| MUTATIONS.json | A.1 overall sign flipped | numeric | `-1.3333333333333335 != 1.3333333333333333 within 7 places (2.666666666666667 difference) : full-reference A.1 must be 4/3` | YES |
| MUTATIONS.json | invalid mask ignored | structural | `252.25 != 4.5 within 7 places (247.75 difference) : invalid target cannot contribute to centre loss` | YES |
| MUTATIONS.json | hard forward replaced by soft codewords | routing | `False is not true : hard forward must equal real selected codewords` | YES |
| MUTATIONS.json | geometry weights normalized over items | structural | `Tensor-likes are not close!` | YES |
| MUTATIONS.json | subgraph attention mask ignored | structural | `2.873314575544806 != 0.0 : masked subgraph must receive zero attention` | YES |
| MUTATIONS.json | expert Query detached from attention | routing | `0.0 not greater than 0.0 : expert Query must receive gradient` | YES |
| LAYER_MUTATIONS.json | layer weights routed to the first level | routing | `True is not false : every level must own its own expert mixture` | YES |
| LAYER_MUTATIONS.json | expert weights normalised over the level axis | structural | `Tensor-likes are not close!` | YES |
| LAYER_MUTATIONS.json | every level reuses the final prefix | routing | `True is not false : each level must carry its own cumulative prefix` | YES |
| ROUTING_MUTATIONS.json | role mixture applied at level 3 (R04) instead of level 2 | routing | `Tensor-likes are not equal!` | YES |

Total: **13 mutations, 13 caught** (numeric 2, structural 5, routing 6).

The R05-specific routing mutation replaces the production `level_weights` with the R04 routing (role mixture on level 3). The pre-mutation checker passes on the real code; under the mutant it fails on the level-2 role assertion, so the new level placement is actually guarded.

Commands: `python3.9 -m pytest tests/test_mutations.py -q` (part of the full candidate suite).
