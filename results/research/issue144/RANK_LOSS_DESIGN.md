# Supervising the quantized SID directly (proposal, not an admitted candidate)

## Why
The retention sweep (`analysis_sid_recall/sid_retention.py`, 56 arms) says L1
keeps 18.4% of behaviour edges against a same-size null (lift +9.5pp), L2 4.0%
(+1.9pp), L3 2.0% (+0.55pp).  Retention rises with the edge's own Forman-Ricci
value, and the L1 lift correlates with recall at rho=0.40.  So the quantizer is
not blind to behaviour, but it is far from preserving it, and the fine levels
keep almost nothing.  That motivates a loss that asks for *relative order* after
quantization rather than forcing neighbours into one code.

## Loss
With `s_l(i,j)` the similarity of two items after quantizing to level l:

    L_rank^(l) = E_{(i,j) in E, k not in N(i)} log(1 + exp[m - s_l(i,j) + s_l(i,k)])

    L = L_RQVAE + lambda_final * L_rank^(3) + lambda_coarse * L_rank^(1)

- `s_3` is read from the cumulative quantized vector (sum of the three levels'
  chosen codewords), or from the soft assignment when the straight-through path
  is used; `s_1` from the first level only.
- Negatives: degree/heat-matched non-neighbours, two-hop non-adjacent items as
  weaker negatives, far non-neighbours as strong negatives.  Matching matters:
  an unmatched sampler lets the model win by pulling popular items together.
- Edge batches are stratified by the edge's AF3 quartile and each stratum's loss
  is averaged before the total, so low-curvature edges keep their share; no
  extra weight is given to high-AF3 edges, whose prefix retention is already
  higher (L1 lift by quartile: 3.6 / 6.0 / 8.0 / 10.9 pp on the baseline,
  6.1 / 10.3 / 13.4 / 16.0 on `mixed_rq_F0`).
- Start from small lambdas; keep reconstruction, commitment and codebook terms
  untouched.

## What must be checked before any training
1. A frozen `math_spec.json` with the differentiable surface (the similarity, the
   logistic ranking term, the negative sampler as a *discrete* piece listed in
   `excluded`), and `reference_contract.json` statements equal to it.
2. TDD: a red test for each new piece (ranking term value and gradient,
   stratum balance, negative sampler never draws a true neighbour, switch wiring
   behaviour-off / geometry-off / both-on).
3. The four suites and the real-entry smoke the framework requires.

## Readings that decide whether it worked
- neighbour-vs-non-neighbour ranking AUC **after** quantization, per level;
- L1 behaviour-edge retention against the same-size null (the null must be
  recomputed per AF3 quartile, which `sid_retention.py` now reports);
- codebook usage per level and L3 collision rate, to catch quantization damage;
- Stage 3 `test_recall@10`, as always.
