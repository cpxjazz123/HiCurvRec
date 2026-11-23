# Mutation detection evidence

`MUTATIONS.json` is the machine-readable output of candidate `tests/test_mutations.py`. All9 mutants were caught by designated assertions; no crash, timeout or import failure was counted as detection.

| Mutation | Class | Designated assertion |
|---|---|---|
| Squared-distance coefficient doubled |numeric |origin displacement squared must be0.25 for every expert |
| Balanced hard rule replaced by argmin |structure |balanced hard IDs must be `[1,0]` |
| Selection cost detached |routing |selection cost must receive nonzero gradient |
| A.1 sign reversed |numeric |full-reference A.1 must be4/3 |
| Invalid target mask ignored |structure |invalid target cannot contribute to centre loss4.5 |
| Hard forward replaced by soft codewords |routing |hard forward must equal actual selected codewords |
| Geometry softmax over items |structure |each item's expert weights must sum to1 |
| Attention token mask ignored |structure |masked subgraph must receive exactly zero attention |
| Expert Query detached |routing |expert Query must receive gradient |

The independent mathematical references live in candidate `tests/reference.py`; comparison assertions in `tests/test_reference.py`. Minimal fixed-answer cases are retained separately in `tests/test_known_answers.py` and `test_curvature_loss.py`; attention uses a scalar hand-worked Q/K/V calculation in `test_attention.py`, independent of production tensor operations. Finite differences apply to smooth cost/attention/allocation functions, not to discrete hard-index jumps.
