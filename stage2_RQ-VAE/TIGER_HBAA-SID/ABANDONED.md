# Abandoned: the HBAA training objective

This tree is a **fresh copy of the frozen baseline** (`stage2_RQ-VAE/TIGER_RQ-VAE`)
with one addition: `model/hbaa.py`, a differentiable reading of the Issue #126
HBAA metric used as a Stage-2 training signal (the level-1 assignment's soft
weights, the prefix prototypes as a globally reduced barycentre approximation,
the history's query point, and the Gromov product at the origin).

**It was never verified, never trained and never launched.** The work was stopped
at the operator's request while the trainer smoke test was still running, so:

* there is no verification record for this tree, and the V2 gate would refuse it;
* there is no Stage 2 run, no exported SID and no Stage 3 result for it;
* the objective's gradient path, its DDP consistency and its weight rule were
  written but not tested.

Nothing in the repository may treat this tree as a candidate. It is kept only so
that the attempt is on the record; deleting it would throw away the code that
made the attempt possible.

The measurements that motivated the attempt are unaffected and stand: the HBAA
metric and the Issue #126 Q_H are the two constructions that track the recall of
the confirmed SID+recall pairs (`results/research/issue126/metric_round/REPORT.md`),
while every construction of the partition's own geometry does not.
