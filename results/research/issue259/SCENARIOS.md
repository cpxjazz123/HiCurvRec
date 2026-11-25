# #259 scenario matrix

| scenario | run | reuse | expected run | expected reuse | plan ms |
|---|---:|---:|---:|---:|---:|
| first_run | 1 | 0 | 1 | 0 | 8.691 |
| no_change_cross_round | 0 | 1 | 0 | 1 | 10.438 |
| single_unit_change | 1 | 1 | 1 | 1 | 21.571 |
| shared_dependency_change | 2 | 0 | 2 | 0 | 29.486 |
| missing_log | 1 | 0 | 1 | 0 | 12.419 |
| missing_asset | 1 | 0 | 1 | 0 | 5.746 |
| unknown_dynamic_import | 1 | 0 | 1 | 0 | 11.07 |
| deleted_dependency_first | 1 | 0 | 1 | 0 | 11.963 |
| deleted_dependency_after_receipt | 0 | 1 | 0 | 1 | 10.842 |
| mandatory_gate | 1 | 0 | 1 | 0 | 10.318 |
| runtime_scoped | 1 | 1 | 1 | 1 | 15.906 |

Total: 10 run, 4 reuse, all expected: True
