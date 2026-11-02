# Convergence of the level curvature estimates

nested edge ladder 128 edges, null replicates up to 8 (32 edges each), 73 arms.

## Level 1

| edges | mean curvature | sd across arms | Spearman vs recall | p | 95% CI |
|---|---|---|---|---|---|
| 8 | +0.4330 | 0.0526 | +0.185 | 0.116 | [-0.078, +0.444] |
| 16 | +0.4329 | 0.0421 | +0.179 | 0.124 | [-0.086, +0.426] |
| 32 | +0.4315 | 0.0375 | +0.173 | 0.154 | [-0.094, +0.417] |
| 64 | +0.4317 | 0.0364 | +0.169 | 0.160 | [-0.108, +0.411] |
| 128 | +0.4317 | 0.0359 | +0.127 | 0.298 | [-0.156, +0.370] |

| edges | Δmean vs max | Δrho vs max |
|---|---|---|
| 8 | +0.0013 | +0.059 |
| 16 | +0.0011 | +0.052 |
| 32 | -0.0002 | +0.046 |
| 64 | -0.0001 | +0.042 |
| 128 | +0.0000 | +0.000 |

| criterion | value | converged |
|---|---|---|
| |mean(k/2) - mean(k)| / SE | 0.02 | True |
| |rho(k/2) - rho(k)| | 0.042 | True |
| null std change at the last replicate step | 0.011 | True |

estimator SE at the largest sample: 0.0042; across-arm sd: 0.0359; signal-to-noise 8.59.

| null replicates | null mean | null sd |
|---|---|---|
| 1 | +0.4967 | 0.0261 |
| 2 | +0.4957 | 0.0248 |
| 3 | +0.4958 | 0.0255 |
| 4 | +0.4959 | 0.0255 |
| 6 | +0.4959 | 0.0260 |
| 8 | +0.4958 | 0.0258 |

## Level 2

| edges | mean curvature | sd across arms | Spearman vs recall | p | 95% CI |
|---|---|---|---|---|---|
| 8 | -0.1486 | 0.1388 | +0.056 | 0.640 | [-0.206, +0.317] |
| 16 | -0.1489 | 0.1363 | -0.008 | 0.938 | [-0.280, +0.251] |
| 32 | -0.1473 | 0.1371 | -0.065 | 0.576 | [-0.332, +0.199] |
| 64 | -0.1564 | 0.1251 | -0.012 | 0.929 | [-0.273, +0.263] |
| 128 | -0.1566 | 0.1258 | -0.030 | 0.811 | [-0.289, +0.241] |

| edges | Δmean vs max | Δrho vs max |
|---|---|---|
| 8 | +0.0079 | +0.086 |
| 16 | +0.0077 | +0.023 |
| 32 | +0.0093 | -0.035 |
| 64 | +0.0002 | +0.019 |
| 128 | +0.0000 | +0.000 |

| criterion | value | converged |
|---|---|---|
| |mean(k/2) - mean(k)| / SE | 0.01 | True |
| |rho(k/2) - rho(k)| | 0.019 | True |
| null std change at the last replicate step | 0.001 | True |

estimator SE at the largest sample: 0.0079; across-arm sd: 0.1258; signal-to-noise 15.89.

| null replicates | null mean | null sd |
|---|---|---|
| 1 | -0.1463 | 0.1439 |
| 2 | -0.1476 | 0.1426 |
| 3 | -0.1483 | 0.1425 |
| 4 | -0.1480 | 0.1430 |
| 6 | -0.1482 | 0.1422 |
| 8 | -0.1485 | 0.1424 |

## Level 3

| edges | mean curvature | sd across arms | Spearman vs recall | p | 95% CI |
|---|---|---|---|---|---|
| 8 | -0.2363 | 0.0881 | -0.063 | 0.597 | [-0.290, +0.194] |
| 16 | -0.2392 | 0.0865 | -0.077 | 0.519 | [-0.329, +0.170] |
| 32 | -0.2442 | 0.0860 | -0.083 | 0.492 | [-0.336, +0.178] |
| 64 | -0.2434 | 0.0827 | -0.106 | 0.373 | [-0.359, +0.148] |
| 128 | -0.2431 | 0.0823 | -0.115 | 0.335 | [-0.357, +0.146] |

| edges | Δmean vs max | Δrho vs max |
|---|---|---|
| 8 | +0.0068 | +0.051 |
| 16 | +0.0039 | +0.037 |
| 32 | -0.0010 | +0.031 |
| 64 | -0.0002 | +0.008 |
| 128 | +0.0000 | +0.000 |

| criterion | value | converged |
|---|---|---|
| |mean(k/2) - mean(k)| / SE | 0.02 | True |
| |rho(k/2) - rho(k)| | 0.008 | True |
| null std change at the last replicate step | 0.005 | True |

estimator SE at the largest sample: 0.0077; across-arm sd: 0.0823; signal-to-noise 10.67.

| null replicates | null mean | null sd |
|---|---|---|
| 1 | -0.2418 | 0.0892 |
| 2 | -0.2444 | 0.0876 |
| 3 | -0.2448 | 0.0873 |
| 4 | -0.2442 | 0.0866 |
| 6 | -0.2441 | 0.0857 |
| 8 | -0.2435 | 0.0854 |

