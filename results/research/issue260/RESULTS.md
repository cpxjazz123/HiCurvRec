# R06b results

## Stage2 (formal, complete)

- arm: `TIGER_topology_residual_l2_rq`
- dispatch: `stage2_127a3cca78dc49bf899ded006b2a11ca`
- 3000 epochs, 4 cards, global batch 4096, seed 42; 2010.8 s.
- final collision: `0.10054093626713301` (< 0.40).
- layer record: `results/stage2_RQ-VAE/TIGER_topology_residual_l2_rq/stage2_metrics/c5f0e739b11b3ee79237e22c2354f18b2dc6f18b8b78b03a3978d77bfec0691b.json`
  - distinct L2 prefixes 7714; distinct triples 22115; L3 refinement gain 2.2908 bits.
  - L1 hard ER 8.2843; conditional log enrichment 1.9128 bits.

## Graphlet-role evidence (causal, same checkpoint)

`token_ablation_r06b.json`: switch off the topology-token residual on the trained checkpoint:
- L1 changed `0 / 24587`; L2 changed `12831` (52.19%); L3 changed `12571` (51.13%).
- per-role L2 changed fraction: P4:0 `0.413`, P4:1 `0.459`, K1_3:0 `0.482`,
  P3:1 `0.523`, P3:0 `0.529`, K3:0 `0.531`, C4:0 `0.551`, K1_3:1 `0.575`.
- The offset norm is `0.03928`, the mean level-2 codeword norm; L1 is untouched by construction.

## Stage3

- protocol: 150 epochs, beam 20, seed 42, frozen trainer; SID collision gate passed
  (`quant_error 0.0086`, `effective_rank 5.05`).
- final `test_R@10`: **0.049600445690210486** (R05 comparator 0.05034906596563311; delta -0.0007486).

## Verdict

- **Mixed curvature is better than either pure half** (matched arms, same rows):
  F0 (E16×H16, α=1/2) recall@10 **0.057226** vs pure Euclidean E **0.054197**,
  Δ **+0.00303**, 95% paired bootstrap CI **[+0.00152, +0.00456]**, McNemar
  p **1.06e-4**. This is evidence from the matched geometry arms, not from R06b.
- **Different graphlets requiring different geometric biases is not supported**:
  ROLE vs GLOBAL Δ **+0.00016**, CI [-0.00136, +0.00169], p 0.855; ROLE vs
  no-routing F0 Δ **-0.00185**, CI [-0.00334, -0.00033], p 0.016; role-conditioned
  geometry preference heterogeneity p=0.31; edge-class and five-node graphlet
  held-out gains are zero/negative.  R06b adds an eighth independent negative:
  the per-role L2 change is real (0.413-0.575) but test_R@10 is **0.049600**,
  below R05 **0.050349** and far below **0.065**.
- Therefore the defensible conclusion is: **mixed curvature is the useful
  geometry; graphlet-specific geometric bias does not yield a downstream gain
  on this data.**  The second sentence of the original target is not supported.
