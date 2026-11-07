# Issue #139: structural fidelity, the core comparison

Measured on the reviewed source, on **one** footing for both arms: the same
training graph, the same 940,498-pair support, the shared readout frozen by F0 at
its 1500th epoch (tau = 0.507976, intercept = -0.041207), the
same Huber scale (the graph's own kappa IQR). Produced by
`results/research/issue139/measure_structure.py`; the raw record is
`structural_fidelity.json`.

| quantity | F0 (weight fixed at 1/2) | G (weight chosen by AF3) | G - F0 |
|---|---|---|---|
| **D_AF3 of the final hard SID** | **0.296474** | **0.295375** | **-0.001100** |
| D_AF3 of the continuous encoder output | 0.298475 | 0.297905 | -0.000570 |
| quantization cost (discrete - continuous) | -0.002001 | -0.002531 | -0.000529 |
| adjacency Brier (over the readout probability) | 0.2519 | 0.2513 | -0.0006 |
| neighbour retention in the SID readout | 0.0108 | 0.0114 | +0.0005 |

## The hypothesis

`D_AF3(G, final) < D_AF3(F0, final)` — **supported**: G is lower by
0.001100, a relative improvement of
0.37%.

The three independent quantities agree in direction: the distortion, the
adjacency Brier and the neighbour retention all favour G. The margin is small,
and it should be read as "the weight selection moves the quantization in the
right direction in this run", not as a strong effect.

## What the decomposition says

For both arms the discrete representation has a *lower* distortion than the
continuous one (`-0.002001` and `-0.002531`): here the
codebook acts as a smoother, so the loss that remains is mostly the encoder's
own representation error, not something quantization adds. G's quantization
smooths slightly more than F0's, which is part of where its advantage comes
from.

## A caveat that changed the answer

The first run of this measurement reported the opposite conclusion
(D(G) - D(F0) = +0.000734, "not supported"). The cause was an alignment error,
not a number: the graph's node ids are *training-item indices* while the readout
is rowed by *item id*, so the two sides were comparing unrelated rows — and the
result looked plausible. Gathering the readout through the training ids flips the
sign. Both numbers are recorded here because the wrong one is the more
instructive.

The neighbour retention is low in absolute terms (~1%) for both arms: the
neighbours of an item in the SID readout barely overlap its co-occurrence
neighbours. Only the comparison between arms is meaningful; the absolute value
says the discrete readout is a poor neighbour map, which the third-level
collision already said.

## What remains to be judged elsewhere

ER and the collision rates are side-effect checks, not the hypothesis. Whether
this structural improvement has recommendation value is a Stage 3 question, and
no arm of this issue has been run there.
