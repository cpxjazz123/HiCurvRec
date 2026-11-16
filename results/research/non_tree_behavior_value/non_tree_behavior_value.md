# Non-tree text similarity and held-out behaviour

Positives: (last history item -> target) in TRAIN/VALID/TEST.
Negatives: same training-popularity bin, excluding the user's history and target.
Non-tree residual = cosine of item vectors after subtracting each item's fine-category mean.
This removes the category leaf direction; singleton fine categories have zero residual.

Pairs: train **1298256**, valid **330252**, test **330390**.

## Models

| split | model | n | AUC | AP | log loss |
|---|---|---:|---:|---:|---:|
| valid | hierarchy_controls | 330252 | 0.6686 | 0.3762 | 0.4302 |
| valid | + raw cosine | 330252 | 0.6924 | 0.3959 | 0.4240 |
| valid | + non-tree residual | 330252 | 0.6726 | 0.3841 | 0.4285 |
| valid | + both | 330252 | 0.6917 | 0.3967 | 0.4240 |
| test | hierarchy_controls | 330390 | 0.6561 | 0.3625 | 0.4329 |
| test | + raw cosine | 330390 | 0.6830 | 0.3826 | 0.4268 |
| test | + non-tree residual | 330390 | 0.6609 | 0.3703 | 0.4313 |
| test | + both | 330390 | 0.6823 | 0.3833 | 0.4268 |

## Paired user bootstrap on test

| candidate | mean ΔAUC | 95% CI | fraction positive |
|---|---:|---:|---:|
| + raw cosine | +0.0269 | [+0.0259, +0.0279] | 1.000 |
| + non-tree residual | +0.0048 | [+0.0042, +0.0054] | 1.000 |
| + both | +0.0262 | [+0.0252, +0.0272] | 1.000 |

## Within tree relation (test AUC)

| relation | base | + raw cosine | + non-tree residual | + both |
|---|---:|---:|---:|---:|
| same_fine | 0.6047 | 0.6443 | 0.6590 | 0.6565 |
| same_coarse_diff_fine | 0.6373 | 0.6712 | 0.6469 | 0.6709 |
| cross_coarse | 0.5995 | 0.6365 | 0.6051 | 0.6353 |
