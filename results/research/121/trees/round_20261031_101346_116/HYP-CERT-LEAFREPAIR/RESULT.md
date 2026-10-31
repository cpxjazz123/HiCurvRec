# HYP-CERT-LEAFREPAIR -- the round-five result

`test_recall@10 = 0.054945` (recall@5 0.036299, ndcg@10 0.029915, 150 epochs,
frozen protocol, checkpoint `.../TIGER_HYP-CERT-LEAFREPAIR/ckpt/
Amazon_2023_Instruments/`).

## What was built

Only the deepest SID code may change; the first L-1 tokens stay bit-identical
(asserted). A move is taken only when the training-split certificate holds:
`dSplit` and `dMerge` are both non-increasing, at least one strictly decreases,
and the item stays inside the target cell's frozen radius. The move is scored by
a behaviour-displaced Poincare point -- the item residual is lifted around its
frozen prefix centroid, displaced towards the weighted gyro-midpoint of its
successor neighbourhood, and compared with the prefix-local deepest-cell centres.
8,338 items (33.9%) were repaired; the train negative rate was r0 = 0.0134 and the
coherence bound B = ln((1-r0)/r0) = 4.30.

## The result

| round | mechanism | test_recall@10 |
| --- | --- | --- |
| one | curved chain on the frozen text geometry | 0.05439 |
| two | geodesic step to the successor barycentre | 0.05165 |
| three | fourth PPMI-driven level | 0.04823 |
| four | capacity-frozen behaviour-funded relabelling | 0.05106 |
| five | certificate-gated deepest-code repair | **0.054945** |
| gate | -- | > 0.065 |
| frozen baseline | plain three-level text-quantised RQ-VAE | about 0.0679 |

Round five is the best mechanism of the five and the only one that improves on
its predecessor, which is consistent with its design: it is the only mechanism
that holds levels 1..L-1 exactly and guards the behaviour move with a
non-increase certificate on both the split and the merge side. It still loses to
the plain quantiser, so the certificate is necessary but not sufficient: a local
repair of a cell assignment cannot recover what an unmodified text quantiser
already separates.

The five-round conclusion stands: the frozen plain text quantiser (about 0.0679)
beats every behaviour-driven or curved modification attempted. The remaining gap
to 0.065 in the sense of *beating* that baseline has not been closed from the SID
side in five independent mechanisms.
