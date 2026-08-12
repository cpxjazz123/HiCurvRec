# Issue #144 Verdict: 无效 (INVALID)

## A/B 六指标 (Stage4, beam=20, 全量 24772)

| metric | A (共享曲率) | B (codeword-specific) | Δ |
|---|---|---|---|
| R@5 | 0.0811 | 0.0786 | -0.0025 |
| R@10 | 0.0999 | 0.0959 | -0.0039 |
| R@20 | 0.1229 | 0.1210 | -0.0019 |
| NDCG@5 | 0.0680 | 0.0666 | -0.0014 |
| NDCG@10 | 0.0740 | 0.0721 | -0.0019 |
| NDCG@20 | 0.0799 | 0.0785 | -0.0014 |

## Paired 分析 (同一批 test 样本, bootstrap 10000 次)

- ΔR@10 (paired) = -0.0040, 95% CI = [-0.0060, -0.0019]
- McNemar: only_B=296, only_A=394, p=0.0002
- first_error=1: A=0.8855 → B=0.8829
- 固定 L0 桶 [200,1000) (n=6717): A=0.1708 → B=0.1702

## Gate 判定

- B R@10 > A: False
- paired CI 不跨 0: False
- first-error/固定桶至少一项改善: True
- 另一项不显著恶化: True
- B 无尺度捷径: True
- TARGET (R@10 > 0.1020): False
