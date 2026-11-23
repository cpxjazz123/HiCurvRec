# Issue #243 变异测试

共 9 个变异；被抓到 9 个。

| 变异 | 类别 | 结果 | 抓到的测试 |
|---|---|---|---|
| `dist-coefficient-twice` | numeric | **CAUGHT** | FAIL test_distance_is_not_double_scaled: (0.404073566198349, 0.20203678988190454) |
| `drop-out-edges` | structural | **CAUGHT** | FAIL test_gradients_flow_through_the_manifold_ops: parameters without gradient: ['layer1.w_out.weight', 'layer2.w_out.we |
| `proju-no-clamp` | numeric | **CAUGHT** | FAIL test_exp_log_match_reference: |
| `expmap0-detached` | routing | **CAUGHT** | FAIL test_gradients_flow_through_the_manifold_ops: element 0 of tensors does not require grad and does not have a grad_f |
| `spearman-as-pearson` | routing | **CAUGHT** | FAIL test_spearman_with_ties: monotone -> spearman 1: 0.9569321102069627 != 1.0 |
| `mobius-add-sign` | numeric | **CAUGHT** | FAIL test_distance_matches_reference_and_known_answers: dist k=-3.0 |
| `sphere-distance-halved` | numeric | **CAUGHT** | FAIL test_distance_matches_reference_and_known_answers: dist k=1.0 |
| `sphere-logmap-drops-radius` | numeric | **CAUGHT** | FAIL test_exp_log_match_reference: |
| `sphere-proju-hyperbolic-clamp` | structural | **CAUGHT** | FAIL test_exp_log_match_reference: |
