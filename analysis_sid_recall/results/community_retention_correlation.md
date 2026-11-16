# Community-internal vs cross-community L1 retention and recall

Source: `analysis_sid_recall/results/community_retention.json` (historical, 68 arms, 15 Louvain communities, L1 SID code retention, AF3 quartiles). The one arm without `test_recall@10` is excluded; n=67.

L1 retention is the share of real behaviour edges whose endpoints receive the same first SID code. `lift` subtracts the same-size shuffled-L1 null. Correlations are across arms; partial correlations control the arm's L3 SID collision rate.

## Overall

| quantity | mean | Spearman | p | Pearson | p | partial Spearman | partial Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| inside_retention | 0.2256 | +0.3518 | 0.0038 | +0.0189 | 0.8816 | +0.5084 | +0.3796 |
| inside_null | 0.0844 | -0.0663 | 0.5829 | -0.2126 | 0.0824 | +0.0059 | +0.0077 |
| inside_lift | 0.1412 | +0.5932 | 0.0002 | +0.5207 | 0.0002 | +0.6407 | +0.6273 |
| between_retention | 0.1160 | +0.0407 | 0.7235 | -0.1997 | 0.0970 | +0.1957 | +0.0429 |
| between_null | 0.0845 | -0.0694 | 0.5643 | -0.2127 | 0.0822 | +0.0005 | +0.0075 |
| between_lift | 0.0315 | +0.2675 | 0.0312 | +0.0791 | 0.5223 | +0.3757 | +0.2261 |
| gap_retention | 0.1096 | +0.6052 | 0.0002 | +0.5565 | 0.0002 | +0.6313 | +0.6289 |
| gap_lift | 0.1097 | +0.6024 | 0.0002 | +0.5551 | 0.0002 | +0.6280 | +0.6273 |

## AF3 quartiles: inside and between lift

| quantity | Spearman | p | Pearson | p | partial Spearman | partial Pearson |
|---|---:|---:|---:|---:|---:|---:|
| inside_Q1_retention | +0.1449 | 0.2396 | -0.1432 | 0.2266 | +0.3136 | +0.1708 |
| inside_Q1_lift | +0.3495 | 0.0040 | +0.0794 | 0.5159 | +0.4700 | +0.2731 |
| inside_Q2_retention | +0.3026 | 0.0132 | +0.0015 | 0.9880 | +0.4759 | +0.3589 |
| inside_Q2_lift | +0.5544 | 0.0002 | +0.4626 | 0.0004 | +0.6220 | +0.5925 |
| inside_Q3_retention | +0.4139 | 0.0008 | +0.0712 | 0.5617 | +0.5467 | +0.4278 |
| inside_Q3_lift | +0.5865 | 0.0002 | +0.5944 | 0.0002 | +0.6273 | +0.6719 |
| inside_Q4_retention | +0.4842 | 0.0002 | +0.1226 | 0.3105 | +0.5887 | +0.4667 |
| inside_Q4_lift | +0.6507 | 0.0002 | +0.7032 | 0.0002 | +0.6683 | +0.7277 |
| between_Q1_retention | -0.0372 | 0.7508 | -0.2680 | 0.0346 | +0.0610 | -0.0624 |
| between_Q1_lift | +0.1547 | 0.2140 | -0.2905 | 0.0272 | +0.2366 | -0.1964 |
| between_Q2_retention | +0.0242 | 0.8384 | -0.1916 | 0.1110 | +0.1691 | +0.0496 |
| between_Q2_lift | +0.2442 | 0.0496 | +0.1786 | 0.1448 | +0.3439 | +0.3168 |
| between_Q3_retention | +0.0884 | 0.4657 | -0.1632 | 0.1622 | +0.2684 | +0.0889 |
| between_Q3_lift | +0.4516 | 0.0004 | +0.5085 | 0.0002 | +0.5378 | +0.5923 |
| between_Q4_retention | +0.1775 | 0.1502 | -0.1322 | 0.2543 | +0.3819 | +0.1380 |
| between_Q4_lift | +0.6040 | 0.0002 | +0.7073 | 0.0002 | +0.6696 | +0.7417 |
| gap_Q1_lift | +0.4939 | 0.0002 | +0.2791 | 0.0220 | +0.5483 | +0.4355 |
| gap_Q2_lift | +0.5686 | 0.0002 | +0.4708 | 0.0004 | +0.6053 | +0.5770 |
| gap_Q3_lift | +0.5961 | 0.0002 | +0.5744 | 0.0002 | +0.6247 | +0.6447 |
| gap_Q4_lift | +0.6346 | 0.0002 | +0.6722 | 0.0002 | +0.6444 | +0.6929 |
