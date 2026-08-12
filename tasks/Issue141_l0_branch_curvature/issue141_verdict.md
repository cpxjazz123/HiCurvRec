# Issue #141 Verdict: 机制有效 (MECHANISM_EFFECTIVE)

## A/B 六指标 (Stage4, beam=20, 全量 24772)

| metric | A (共享曲率) | B (codeword-specific) | Δ |
|---|---|---|---|
| R@5 | 0.0771 | 0.0796 | +0.0024 |
| R@10 | 0.0968 | 0.0990 | +0.0022 |
| R@20 | 0.1214 | 0.1242 | +0.0027 |
| NDCG@5 | 0.0659 | 0.0669 | +0.0010 |
| NDCG@10 | 0.0722 | 0.0732 | +0.0009 |
| NDCG@20 | 0.0785 | 0.0795 | +0.0011 |

## Paired 分析 (同一批 test 样本, bootstrap 10000 次)

- ΔR@10 (paired) = +0.0022, 95% CI = [0.0000, 0.0044]
- McNemar: only_B=405, only_A=351, p=0.0539
- first_error=1: A=0.8798 → B=0.8793
- 固定 L0 桶 [200,1000) (n=6717): A=0.1699 → B=0.1694

## Gate 判定

- B R@10 > A: True
- paired CI 不跨 0: True
- first-error/固定桶至少一项改善: True
- 另一项不显著恶化: True
- B 无尺度捷径: True
- TARGET (R@10 > 0.1020): False
