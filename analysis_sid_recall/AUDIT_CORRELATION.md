# The curvature-recall statistics by pair-verification group

Pairs are classified by `pair_audit.py`: the SID table and the recall file were introduced by the same commit, and the stage-2 and stage-3 directories carry the same name. The join key is `sid_sha256` (182 distinct SID contents; 168 distinct variant names).

Self-check: the ALL group reproduces `curvature_summary.json` to 0.0000 in Spearman.

| group | pairs | families | recall median [min, max] |
|---|---|---|---|
| ALL | 182 | 115 | 0.0580 [0.0306, 0.0643] |
| NAME_OK | 110 | 107 | 0.0579 [0.0306, 0.0637] |
| SAME_COMMIT | 86 | 78 | 0.0580 [0.0423, 0.0634] |
| VERIFIED | 73 | 73 | 0.0580 [0.0423, 0.0634] |
| CODE_PATH_OK | 178 | 114 | 0.0580 [0.0306, 0.0643] |
| UNVERIFIED | 109 | 42 | 0.0576 [0.0306, 0.0643] |
| VERIFIED_BEST_PER_FAMILY | 73 | 73 | 0.0580 [0.0423, 0.0634] |

## z_curv_L1

| group | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| ALL | 167 | -0.091 | 0.241 | [-0.232, +0.066] |
| NAME_OK | 100 | -0.120 | 0.238 | [-0.325, +0.087] |
| SAME_COMMIT | 82 | +0.053 | 0.629 | [-0.192, +0.292] |
| VERIFIED | 70 | +0.150 | 0.212 | [-0.117, +0.389] |
| CODE_PATH_OK | 167 | -0.091 | 0.241 | [-0.232, +0.066] |
| UNVERIFIED | 97 | -0.212 | 0.042 | [-0.395, -0.014] |
| VERIFIED_BEST_PER_FAMILY | 70 | +0.150 | 0.212 | [-0.117, +0.389] |

## z_curv_L2

| group | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| ALL | 179 | +0.150 | 0.035 | [+0.001, +0.306] |
| NAME_OK | 108 | +0.204 | 0.039 | [-0.001, +0.397] |
| SAME_COMMIT | 86 | +0.074 | 0.501 | [-0.144, +0.298] |
| VERIFIED | 73 | +0.129 | 0.278 | [-0.128, +0.365] |
| CODE_PATH_OK | 176 | +0.161 | 0.036 | [+0.009, +0.311] |
| UNVERIFIED | 106 | +0.167 | 0.086 | [-0.035, +0.362] |
| VERIFIED_BEST_PER_FAMILY | 73 | +0.129 | 0.278 | [-0.128, +0.365] |

## z_curv_L3

| group | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| ALL | 181 | +0.145 | 0.049 | [+0.006, +0.285] |
| NAME_OK | 110 | +0.103 | 0.288 | [-0.105, +0.294] |
| SAME_COMMIT | 86 | +0.029 | 0.800 | [-0.197, +0.243] |
| VERIFIED | 73 | -0.044 | 0.706 | [-0.274, +0.184] |
| CODE_PATH_OK | 178 | +0.140 | 0.077 | [-0.007, +0.284] |
| UNVERIFIED | 108 | +0.234 | 0.017 | [+0.038, +0.413] |
| VERIFIED_BEST_PER_FAMILY | 73 | -0.044 | 0.706 | [-0.274, +0.184] |

## z_curv_L1_minus_L3

| group | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| ALL | 167 | -0.095 | 0.225 | [-0.237, +0.060] |
| NAME_OK | 100 | -0.121 | 0.225 | [-0.332, +0.085] |
| SAME_COMMIT | 82 | +0.051 | 0.644 | [-0.197, +0.289] |
| VERIFIED | 70 | +0.141 | 0.239 | [-0.128, +0.379] |
| CODE_PATH_OK | 167 | -0.095 | 0.225 | [-0.237, +0.060] |
| UNVERIFIED | 97 | -0.225 | 0.034 | [-0.400, -0.028] |
| VERIFIED_BEST_PER_FAMILY | 70 | +0.141 | 0.239 | [-0.128, +0.379] |

## sign_change_L1_L3

| group | n | Spearman | p | 95% CI |
|---|---|---|---|---|
| ALL | 167 | -0.087 | 0.279 | [-0.259, +0.116] |
| NAME_OK | 100 | -0.074 | 0.461 | [-0.324, +0.208] |
| SAME_COMMIT | 82 | +0.175 | 0.111 | [-0.162, +0.400] |
| VERIFIED | 70 | +0.106 | 0.420 | [-0.255, +0.381] |
| CODE_PATH_OK | 167 | -0.087 | 0.279 | [-0.259, +0.116] |
| UNVERIFIED | 97 | -0.159 | 0.127 | [-0.349, +0.062] |
| VERIFIED_BEST_PER_FAMILY | 70 | +0.106 | 0.420 | [-0.255, +0.381] |
