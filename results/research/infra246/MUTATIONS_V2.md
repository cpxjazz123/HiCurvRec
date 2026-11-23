# Issue #246 变异测试 (v2)

共 14 个变异；被抓到 14 个。
判定 = 未变异副本先通过(returncode 0) 且 变异副本 returncode 1 且输出含 `FAIL` 行；崩溃不计。

| 变异 | 类别 | 结果 | 抓到的断言 |
|---|---|---|---|
| `held-out-uses-the-v1-weighted-formula` | numeric | **CAUGHT** | FAIL test_held_out_support_uses_boolean_semantics: AssertionError: the held-out direction must survive |
| `ego-role-from-a-truncated-sample` | structural | **CAUGHT** | FAIL test_direction_reads_real_csr_membership_beyond_the_sample_budget: AssertionError: every included out-neighbour mus |
| `ego-hop-equals-traversal-depth` | structural | **CAUGHT** | FAIL test_traversal_depth_and_true_hop_can_differ: AssertionError: the sampled traversal depth must differ from the true |
| `fork-back-to-universal-independence` | structural | **CAUGHT** | FAIL test_fork_relaxed_accepts_two_adjacent_plus_one_isolated_successor: AssertionError: |
| `canonical-code-not-minimised` | numeric | **CAUGHT** | FAIL test_canonical_classes_are_the_sixteen_unlabeled_digraphs: AssertionError: 64 |
| `subgraph-summary-flips-the-pass-rule` | routing | **CAUGHT** | FAIL test_subgraph_summary_counts_centres_without_valid_subgraphs: AssertionError: |
| `subgraph-summary-worst-becomes-the-best` | numeric | **CAUGHT** | FAIL test_subgraph_summary_distortion_known_answer: AssertionError: |
| `prefix-at-k-drops-the-length-check` | structural | **CAUGHT** | FAIL test_prefix_at_k_requires_both_lengths: AssertionError: a code shorter than k cannot share a 2-prefix |
| `ppr-drops-the-dangling-correction` | numeric | **CAUGHT** | FAIL test_mass_conserves_with_a_dangling_row: AssertionError: 0.38587499999999997 |
| `join-accepts-extra-pairs` | structural | **CAUGHT** | FAIL test_join_rejects_every_bad_input: AssertionError: extra model pair must raise |
| `reference-flat-limit-reverts-to-plain-norm` | numeric | **CAUGHT** | FAIL test_euclidean_known_answer: flat limit of the family: 2 * (3-4-5): 5.0 != 10.0 |
| `manifold-flat-limit-reverts-to-plain-norm` | numeric | **CAUGHT** | FAIL test_geometry_boundaries_and_gradients: AssertionError: (0.0, 0.5) |
| `sphere-exponential-uses-the-half-angle` | numeric | **CAUGHT** | FAIL test_geometry_boundaries_and_gradients: AssertionError: (1e-12, 0.5002442101119368) |
| `curvature-transport-ignores-the-marginals` | numeric | **CAUGHT** | FAIL test_orc_known_answers: RuntimeError: transport LP failed: The problem is infeasible. (HiGHS Status 8: model_status |
