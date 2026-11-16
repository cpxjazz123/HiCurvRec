# Sentence-T5 kNN K sweep: graphlets and semantic consistency

Original sentence-T5 embedding, L2-normalised, global cosine top-K, directed edges
symmetrised as the undirected union. K = 5, 10, 15, 20.

PGD global 4-node counts are used; the PGD **local** 3-star column is not used.
The PGD global-count implementation is checked against brute force on a random graph.

Categories in the parquet are null, so semantic consistency uses cluster proxies:
**coarse = KMeans(16)**, **fine = KMeans(256)** on the same embedding.
Expected random is the empirical within-cluster pair probability; lift = rate / expected.

## Graphlet distribution

Connected 4-node induced graphlets; shares sum to 1 across the six connected types.

| graph | edges | mean deg | triangles | 3-node triangle share | K4 | diamond | C4 | paw | K1,3 | P4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| behavior_reference | 315391 | 25.66 | 546438 | 0.0133 | 0.0001 | 0.0020 | 0.0016 | 0.0467 | 0.6247 | 0.3248 |
| knn_k5 | 98547 | 8.02 | 83210 | 0.1159 | 0.0049 | 0.0256 | 0.0064 | 0.1687 | 0.2725 | 0.5220 |
| knn_k10 | 198415 | 16.14 | 382051 | 0.1286 | 0.0073 | 0.0317 | 0.0068 | 0.1877 | 0.2536 | 0.5129 |
| knn_k15 | 298627 | 24.29 | 904515 | 0.1335 | 0.0083 | 0.0344 | 0.0071 | 0.1947 | 0.2485 | 0.5071 |
| knn_k20 | 399072 | 32.46 | 1654443 | 0.1366 | 0.0089 | 0.0360 | 0.0072 | 0.1987 | 0.2442 | 0.5050 |

## Semantic consistency

| graph | mutual directed | coarse rate | coarse lift | fine rate | fine lift | edge cosine mean |
|---|---:|---:|---:|---:|---:|---:|
| behavior_reference | nan | 0.2437 | 3.230 | 0.0535 | 10.071 | 0.7257 |
| knn_k5 | 0.3968 | 0.7964 | 10.556 | 0.7178 | 135.013 | 0.8993 |
| knn_k10 | 0.3860 | 0.7806 | 10.346 | 0.6799 | 127.886 | 0.8886 |
| knn_k15 | 0.3806 | 0.7699 | 10.204 | 0.6511 | 122.470 | 0.8821 |
| knn_k20 | 0.3769 | 0.7622 | 10.102 | 0.6259 | 117.717 | 0.8773 |

The behavior reference is not a KNN graph and has no reciprocal directed fraction.

## Reading

- Graphlet shares are conditional on connected 4-node subgraphs, so raw edge
  density changes do not mechanically force a share change.
- Coarse/fine rates are not directly comparable in absolute value; the lift over
  the same clustering's random pair baseline is the comparable quantity.
- This is an embedding-space diagnostic only: no SID, no RQ-VAE, no Stage 3.
