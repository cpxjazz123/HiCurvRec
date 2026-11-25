# Replay of TIGER_role_conditioned_l2_rq (issue256_r05_formal)

Status: **PASS**

| check | expected | actual | tolerance | status |
|---|---|---|---|---|
| sid_chain:distinct_l3_triples | 23228 | 23228 | 0 | PASS |
| sid_chain:distinct_l2_prefixes | 9609 | 9609 | 0 | PASS |
| sid_chain:l3_collision | 0.05527311180705252 | 0.05527311180705252 | 1e-10 | PASS |
| sid_chain:l3_refinement_gain_bits | 1.851570349766611 | 1.8515703497702383 | 1e-10 | PASS |
| independent_reduction:test_recall@5 | 0.033270077821689094 | 0.033270077821689094 | 1e-10 | PASS |
| independent_reduction:test_recall@10 | 0.05034906596563311 | 0.05034906596563311 | 1e-10 | PASS |
| independent_reduction:test_ndcg@5 | 0.021846481598707893 | 0.021846481598707935 | 1e-10 | PASS |
| independent_reduction:test_ndcg@10 | 0.02732772020207201 | 0.02732772020207228 | 1e-10 | PASS |
| artefact:test_final | ed2d61223dd58af4fd77166a6ecdd1ec2c4aa6129ef534c1d60be6866db97cb9 | ed2d61223dd58af4fd77166a6ecdd1ec2c4aa6129ef534c1d60be6866db97cb9 |  | PASS |
| artefact:sid | dc206dba7fbdadc28bbdedb582c7c755e412d70bdfec930b380a2bd5ea9d5ba6 | dc206dba7fbdadc28bbdedb582c7c755e412d70bdfec930b380a2bd5ea9d5ba6 |  | PASS |
| artefact:stage2_checkpoint | f93da1a7466d62f07c6482c64cf1a8efdd9dd1fa7ccdf7891be98d6d37bc7f26 | f93da1a7466d62f07c6482c64cf1a8efdd9dd1fa7ccdf7891be98d6d37bc7f26 |  | PASS |
| artefact:stage2_export_manifest | 071d75459ee43b4dbbec72dc50a91d0035c8b01b43e6d009d077b85b35d147cb | 071d75459ee43b4dbbec72dc50a91d0035c8b01b43e6d009d077b85b35d147cb |  | PASS |
| artefact:archive_binding | 28055c2832260b2c2a13620e3dcd74cc3a6cbfa636652a194deb3295ffd540ae | 28055c2832260b2c2a13620e3dcd74cc3a6cbfa636652a194deb3295ffd540ae |  | PASS |
