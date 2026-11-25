# Replay of TIGER_module_budget_prefix_rq (issue256_r01_formal)

Status: **PASS**

| check | expected | actual | tolerance | status |
|---|---|---|---|---|
| sid_chain:distinct_l3_triples | 22972 | 22972 | 0 | PASS |
| sid_chain:distinct_l2_prefixes | 8559 | 8559 | 0 | PASS |
| sid_chain:l3_collision | 0.06568511815186884 | 0.06568511815186884 | 1e-10 | PASS |
| sid_chain:l3_refinement_gain_bits | 2.038433479685075 | 2.0384334796889796 | 1e-10 | PASS |
| independent_reduction:test_recall@5 | 0.034105746966346905 | 0.034105746966346905 | 1e-10 | PASS |
| independent_reduction:test_recall@10 | 0.05273420498267727 | 0.05273420498267727 | 1e-10 | PASS |
| independent_reduction:test_ndcg@5 | 0.022554059459244956 | 0.02255405945924502 | 1e-10 | PASS |
| independent_reduction:test_ndcg@10 | 0.02852344807984429 | 0.028523448079844565 | 1e-10 | PASS |
| artefact:test_final | 195392578b47bc29bc967e61e3cc9561b74b1cfe9f243136720d2753141dab21 | 195392578b47bc29bc967e61e3cc9561b74b1cfe9f243136720d2753141dab21 |  | PASS |
| artefact:sid | 201d2cc537dfdcc27060302489324d818356fd6327b88de89584943108a5758f | 201d2cc537dfdcc27060302489324d818356fd6327b88de89584943108a5758f |  | PASS |
| artefact:stage2_checkpoint | 680439b57bfdbec622a5a109b0d589ab21566f71236fc007eb19d59b6108fc0c | 680439b57bfdbec622a5a109b0d589ab21566f71236fc007eb19d59b6108fc0c |  | PASS |
| artefact:stage2_export_manifest | a90c264227101d5ef888f5f6306d95263fab18903313a5cd03b4e703a310b49d | a90c264227101d5ef888f5f6306d95263fab18903313a5cd03b4e703a310b49d |  | PASS |
| artefact:archive_binding | f26791dce14cbc6d2c3336abd354937861e2a20bb880c42efb77e909b968b4eb | f26791dce14cbc6d2c3336abd354937861e2a20bb880c42efb77e909b968b4eb |  | PASS |
