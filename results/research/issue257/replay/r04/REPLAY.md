# Replay of TIGER_role_conditioned_l3_rq (issue256_r04_formal)

Status: **PASS**

| check | expected | actual | tolerance | status |
|---|---|---|---|---|
| sid_chain:distinct_l3_triples | 22995 | 22995 | 0 | PASS |
| sid_chain:distinct_l2_prefixes | 8781 | 8781 | 0 | PASS |
| sid_chain:l3_collision | 0.06474966445682673 | 0.06474966445682673 | 1e-10 | PASS |
| sid_chain:l3_refinement_gain_bits | 1.9957073319719747 | 1.9957073319757797 | 1e-10 | PASS |
| independent_reduction:test_recall@5 | 0.03405351764480579 | 0.03405351764480579 | 1e-10 | PASS |
| independent_reduction:test_recall@10 | 0.05151552081338463 | 0.05151552081338463 | 1e-10 | PASS |
| independent_reduction:test_ndcg@5 | 0.02247801380083515 | 0.022478013800835216 | 1e-10 | PASS |
| independent_reduction:test_ndcg@10 | 0.028115151372565275 | 0.028115151372565594 | 1e-10 | PASS |
| artefact:test_final | 714d6f1a73ebf68d81e80683c99171782d6833f7abb5021d1e600be5bd9d79cf | 714d6f1a73ebf68d81e80683c99171782d6833f7abb5021d1e600be5bd9d79cf |  | PASS |
| artefact:sid | 26368f3e7690c6831b523eb82e0755aea58885336e0e23725ba5f7fb4ad80a18 | 26368f3e7690c6831b523eb82e0755aea58885336e0e23725ba5f7fb4ad80a18 |  | PASS |
| artefact:stage2_checkpoint | 5c6b1557f004a194010092f3d9cb7ff471940ee32a19dfd6c0feaf01b6c1a638 | 5c6b1557f004a194010092f3d9cb7ff471940ee32a19dfd6c0feaf01b6c1a638 |  | PASS |
| artefact:stage2_export_manifest | ebc490960412336d6026a438b13e875a539928cb7bc70fda059fa73c5ba6c3dd | ebc490960412336d6026a438b13e875a539928cb7bc70fda059fa73c5ba6c3dd |  | PASS |
| artefact:archive_binding | 4a1e8a22a710544895d19cee5a162549223765160546728fd9cbf1bfa6ccf69d | 4a1e8a22a710544895d19cee5a162549223765160546728fd9cbf1bfa6ccf69d |  | PASS |
