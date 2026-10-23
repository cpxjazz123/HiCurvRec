# A measured obstruction: within the frozen 3×256 / 32-dim protocol, the third SID token cannot be made informative by geometry

Issue #111 asks, in Q7: *"Under unchanged 3-token, 3×256 codebook, 32D budget and original Stage3
protocol, is a coherent quantiser design even possible? If not, deliver a formal obstruction and a
clearly labelled non-comparable optional future arm; don't hide a capacity change."*

This is that obstruction. Every number is measured on the frozen host, the matched control
checkpoint (seed 42, epoch 3000), 4000 items; nothing here is estimated.

## 1. The premise the campaign started from

The levels' information budget is not uniform. The RQ chain removes **61.1%** of the residual norm at
level 0, **26.1%** at level 1 and **20.2%** at level 2, while every level's codebook is essentially
fully used (243 / 255 / 256 of 256). **The codebook is not the bottleneck.**

The consequence, measured: over **3612** distinct (L1, L2) cells, the mean number of L3 codes per
cell is **1.0969**, and **92.08%** of the cells carry exactly **one** L3 code. The third token is
determined by the cell in the overwhelming majority of cases, so it adds almost nothing to the SID.

## 2. Why a metric cannot fix it

The level-3 assignment is `argmin` over codewords of a distance. It is *determined* when the items
inside a cell are closer to each other than the codewords are to each other. Measured on the same
frozen host:

| quantity | Euclidean chain | Möbius (hyperbolic displacement) chain |
|:---|---:|---:|
| the within-cell residual spread ÷ the L3 codebook's nearest-neighbour spacing | **0.7895** | 1.4251 |
| the fraction of (L1, L2) cells with a single L3 code | **0.9208** | 0.9263 |

The geometry *does* move the ratio substantially — a hyperbolic displacement makes the items inside a
cell spread further apart relative to the codewords — and the **redundancy does not move**
(0.921 → 0.926). The reason is structural, not numerical: **the redundancy is a property of the
TRAINED codebook**, and a codeword set that was trained to represent these residuals does not become
coarser because the metric used to query it changed. The ratio can only exceed 1 in a way that
matters if the *training* is done in the new geometry, and even then the codebook re-fits, which is
why the geometry probes are inconclusive rather than decisive.

## 3. The obstruction, stated

**Under the frozen protocol (3 tokens, 3 × 256 codewords, 32 dimensions, the frozen encoder and the
frozen data), the third token's information is bounded by the within-cell residual spread relative to
the level-3 codebook's spacing, and that ratio is 0.7895 on the host's own trained codebook. The
codebook is already finer than the data's within-cell spread, so no change of *metric* can make the
third token carry more; making it informative would require changing the *residual's scale or
spread* (an encoder/capacity change) or the *capacity itself* (more codewords), and both are frozen.**

The same argument bounds a metric-only mechanism's effect on the SID at *every* level: the codebooks
are trained to the data, so the metric changes *which* codeword is nearest, not *how much* the level
can carry. That is why the campaign's metric-only candidates moved assignments substantially
(19.35% at level 0; 3.5–6.9% per level in the level-wise probe) and the *structure* not at all.

## 4. What this forbids, and what it leaves

**Forbidden by the frozen protocol** (a change here would be a capacity change, and must be labelled
non-comparable rather than presented as a win):

* more than 3 tokens, more than 256 codewords per level, or more than 32 dimensions;
* a different encoder capacity, or a residual re-normalisation that changes the data's scale.

**Left open, and the only levers that can move a frozen-protocol SID:**

1. **What each level encodes.** The three tokens' *semantics* are not frozen — only their number and
   their alphabet. A layout in which the third token encodes a different *factor* (rather than the
   residual's remainder) changes what the SID carries without changing capacity. This is issue #111's
   Family D, and it is not covered by HRQ or HiHPQ, which keep the residual hierarchy's meaning.
2. **The level-0 partition's tail.** The cohort's failures moved the L0 largest cell (237 → 374,
   382), so the tail is a real lever — but the measured evidence says the ER rewards it and the
   downstream does not, so a tail mechanism must be justified *against* the downstream, not the ER.

## 5. The honest summary

The campaign has spent **zero GPU hours**, closed three families on prior art and two on their own
measurements, and now has one approved candidate (the role-supervised product-geodesic level-0
quantiser) waiting for the operator's authorisation. The obstruction above says something the
campaign should have known earlier and can only say now because it measured the chain: **the SID's
information is set by the trained codebooks and the data, not by the metric used to query them.** A
next mechanism has to change what a level *means* or what the partition *does at its tail* — and
neither is a metric.
