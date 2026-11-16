# Pre-quantisation z-kNN vs post-quantisation q-kNN vs behaviour graph

No semantic clustering. Both z and q graphs use global k=10 candidates and are trimmed to the smaller graph's edge count by shortest edge distance, so the z/q comparison has the same number of edges.

## Graph recovery of behaviour edges

| arm | graph | edges | behavior recall | precision | Jaccard | degree Spearman | components | triangles |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | z-kNN | 186044 | 0.0164 | 0.0278 | 0.0104 | 0.0486 | 1098 | 323124 |
| pure_euclidean_TIGER | q-kNN | 186044 | 0.0113 | 0.0192 | 0.0072 | 0.0172 | 10 | 390495 |
| mixed_F0 | z-kNN | 184398 | 0.0149 | 0.0254 | 0.0095 | 0.0366 | 486 | 323737 |
| mixed_F0 | q-kNN | 184398 | 0.0107 | 0.0182 | 0.0068 | 0.0061 | 12 | 383166 |

## ORCA roles: behavior vs z / behavior vs q

| arm | pair | top ARI | top NMI | same top | role JSD mean | role JSD median | multi before | multi after | role-pair JSD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | behavior vs z | -0.0002 | 0.0002 | 0.4149 | 0.3237 | 0.2733 | 0.3099 | 0.0101 | 0.7965 |
| pure_euclidean_TIGER | behavior vs q | -0.0003 | 0.0002 | 0.4287 | 0.3326 | 0.2839 | 0.3099 | 0.0903 | 0.7005 |
| mixed_F0 | behavior vs z | -0.0014 | 0.0006 | 0.4257 | 0.3457 | 0.3017 | 0.3099 | 0.0042 | 0.7965 |
| mixed_F0 | behavior vs q | -0.0003 | 0.0002 | 0.4316 | 0.3571 | 0.3173 | 0.3099 | 0.0166 | 0.7834 |

## Quantisation step: z-kNN vs q-kNN

| arm | z edges | q edges | edge intersection | edge Jaccard | top ARI | top NMI | same top | role JSD mean | multi agreement | role-pair JSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_euclidean_TIGER | 186044 | 186044 | 56713 | 0.1798 | 0.2486 | 0.1167 | 0.9226 | 0.0358 | 0.9037 | 0.0299 |
| mixed_F0 | 184398 | 184398 | 62721 | 0.2049 | 0.3123 | 0.1609 | 0.9710 | 0.0186 | 0.9807 | 0.0046 |
