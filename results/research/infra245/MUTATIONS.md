# Issue #245 变异测试

共 17 个变异；被抓到 17 个。
判定 = 未变异副本先通过(returncode 0) 且 变异副本 returncode 1 且输出含 `FAIL` 行；崩溃不计。

| 变异 | 类别 | 结果 | 抓到的断言 |
|---|---|---|---|
| `dist-coefficient-twice` | numeric | **CAUGHT** | FAIL test_distance_is_not_double_scaled: (0.404073566198349, 0.20203678988190454) |
| `sphere-distance-halved` | numeric | **CAUGHT** | FAIL test_distance_matches_reference_and_known_answers: dist k=1.0 |
| `sphere-logmap-drops-radius` | numeric | **CAUGHT** | FAIL test_exp_log_match_reference: |
| `sphere-proju-hyperbolic-clamp` | structural | **CAUGHT** | FAIL test_exp_log_match_reference: |
| `mobius-add-sign` | numeric | **CAUGHT** | FAIL test_distance_matches_reference_and_known_answers: dist k=-3.0 |
| `expmap0-detached` | routing | **CAUGHT** | FAIL test_gradients_flow_through_the_manifold_ops: element 0 of tensors does not require grad and does not have a grad_f |
| `drop-out-edges` | structural | **CAUGHT** | FAIL test_gradients_flow_through_the_manifold_ops: parameters without gradient: ['layer1.w_out.weight', 'layer2.w_out.we |
| `ego-drops-induced-edges` | structural | **CAUGHT** | FAIL test_merge: |
| `pairs-bucket-by-source` | structural | **CAUGHT** | FAIL test_shared_successor_is_not_a_chain: {(4, 5), (1, 2)} |
| `metrics-spearman-as-pearson` | routing | **CAUGHT** | FAIL test_spearman_perfect_and_tied: monotone: 0.9569321102069627 != 1.0 |
| `bfs-depth-ignored` | numeric | **CAUGHT** | FAIL test_depth_cap_marks_beyond_as_unreachable: |
| `join-accepts-missing` | structural | **CAUGHT** | FAIL test_join_rejects_missing_duplicate_and_misaligned: join accepted a bad input: missing |
| `chain-drops-induced-check` | structural | **CAUGHT** | FAIL test_triangle_is_not_a_chain: |
| `fork-accepts-adjacent-successors` | structural | **CAUGHT** | FAIL test_fork_rejects_adjacent_successors: |
| `ppr-dangling-mass-dropped` | numeric | **CAUGHT** | FAIL test_ppr_converges_and_normalises_the_mass: |
| `two-hop-ignores-reverse-edge` | structural | **CAUGHT** | FAIL test_two_hop_excludes_the_reverse_direction: |
| `undirected-exclusion-ignored` | structural | **CAUGHT** | FAIL test_exclude_undirected_drops_the_mirror_image: |
