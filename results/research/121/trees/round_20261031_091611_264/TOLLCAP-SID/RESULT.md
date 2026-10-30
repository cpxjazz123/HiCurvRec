# TOLLCAP-SID -- the round-four result

`test_recall@10 = 0.051063` (recall@5 0.034036, ndcg@10 0.028065, 150 epochs,
frozen protocol, checkpoint `.../TIGER_TOLLCAP-SID/ckpt/Amazon_2023_Instruments/
Oct-31-2026_09-40-07/HG_Rec_best.pth`).

## The mechanism was implemented as specified

A capacity-frozen, behaviour-funded relabelling of the FROZEN baseline SID: the
codebooks, the cell set and every cell's size stay bit-identical (asserted:
L1/L2/L3 = 247/7414/20223 against 247/7414/20223), and 13,845 items (56.3%) move
inside their level-1 group through one optimal-transport solve whose cost is
`toll - lambda (A(i,s) - A(i,t(i)))` with `lambda = eps_rec/sigma_A = 0.0742`.
The geometry is taken in the quantizer's input space (the frozen encoder's
32-dimensional latents), the behaviour term from the training transitions only,
and the disambiguation token is recomputed inside each new cell (without that, the
full SID is not unique and the Stage 2 admission correctly refuses the file).

## The result is negative, and it is informative

| round | mechanism | prefixes | test_recall@10 |
| --- | --- | --- | --- |
| one | curved chain on the frozen text geometry | 251/1769/2201 | 0.05439 |
| two | geodesic step to the successor barycentre | 230/1251/1298 | 0.05165 |
| three | fourth level from the PPMI successor profile | 208/2766/6235/6302 | 0.04823 |
| four | capacity-frozen behaviour-funded relabelling | 247/7414/20223 | 0.05106 |
| gate | -- | -- | > 0.065 |
| frozen baseline | plain three-level text-quantised RQ-VAE | 247/7414/20223 | about 0.0679 |

Round four is the strongest test yet of the hypothesis, because it is the only one
that leaves the partition, the vocabulary and every prefix count exactly as the
baseline has them, and it moves only items inside the existing cells. It still
loses 1.7 points of recall against that baseline. The diagnostics say why the move
went that way: Hard-L1 ER drops 16.39 -> 6.47 and the level-2 prefix count is
unchanged, but the validation L1 NEGATIVE hit rate rises 0.0048 -> 0.0134 (2.8x)
while the positive rate rises only 0.0794 -> 0.0864 (1.09x). In plain terms the
transport buys behaviour affinity by making items that are far apart in the text
geometry share a prefix, and Stage 3 pays for that with false prefix hits.

The four-round conclusion: the frozen plain text quantiser (about 0.0679) beats
every behaviour-driven or curved modification tried so far. A mechanism that only
rebuilds or relabels the SID loses the linear separation the plain quantiser
provides; reaching 0.065 by SID construction alone is not supported by these four
runs.
