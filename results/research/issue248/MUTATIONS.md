# Mutation testing — issue 248

Harness: `tools/graph_infra/tests/test_mutation_v4.py`
Evidence: `results/research/issue248/MUTATIONS.json` (written by the run; the assertion text is captured).

Verdict rule: a mutation counts as caught only when the **unmutated** copy passes the designated
assertion and the **mutated** copy fails it with the expected message. Crashes, import errors and
timeouts are not catches.

| # | Mutation | Category | Designated assertion | Observed failure |
|---|---|---|---|---|
| 1 | the `g(b,c)²` quarter coefficient is doubled | numeric | closed form keeps the quarter coefficient | `1.4 != 0.4 within 12 places` |
| 2 | the sign of the parallelogram residual is flipped | numeric | closed form keeps the residual sign | `-0.4 != 0.4 within 12 places` |
| 3 | the reference count is replaced by its square | numeric | closed form divides by the reference count | `0.88 != 0.4 within 12 places` |
| 4 | the component size is replaced by one | structural | closed form uses the component size | `-2.0 != 0.4 within 12 places` |
| 5 | the distance histogram loses its first node | structural | histogram reproduces S2 | assertion failure |
| 6 | `g(b,c)` ignores whether `b` and `c` are adjacent | structural | g(b,c) respects adjacency | assertion failure |
| 7 | the reference set becomes the whole graph instead of the centre's component | routing | references stay inside the component | assertion failure |
| 8 | the structure class ignores the edge direction | routing | the structure class keeps direction | assertion failure |

Coverage: numeric 3, structural 3, routing 2 — **8/8 caught**, each by its named assertion, none
detected only by a crash. Every mutation corresponds to a defect class that has a standing
regression test in `tools/graph_infra/tests/test_full_curvature_v4.py`,
`tests/test_v4_assets.py` or `tests/test_v4_reload.py`.
