# round_20261027_103131_805 — cogeo_region_argmax (iteration 6 of the hyperbolic x behaviour line)

## The mechanism

The selected proposal of the round: the first-level code of an item is chosen by a score
that combines the Poincare geodesic distance from the item's latent to the codewords, the
**symmetric** co-consumption agreement between the item and the region (the mean
co-consumption row of the items the code currently holds), and a bounded **directed**
tie-break (the incoming transition mass of the item spread over the radar of its
predecessors). The score is differentiable and drives the straight-through path of the
quantiser, so the behaviour and the curvature train the encoder and the codebook and not
only select a code.

## The review (five verifiers per revision) and what the operator did

| revision | what the reviewers found | what was done |
|---|---|---|
| 0 | the coupled score only fitted the Euclidean argmin; `raw_module` used before assignment | the coupling was made real and the unwrapping fixed |
| 2-3 | the coupled score was auxiliary, then disconnected | the score was wired into the first-level quantiser |
| 5-8 | a capped predecessor list, a co-consumption signal that lost its multiplicities, no differentiable path, a 2.24 GiB dense allocation per epoch, the hook installed before its profiles existed | all fixed by the operator, each with an executable check |
| 9 | the ablation switches existed only inside the smoke | `COGEO_MODE` became a source constant, so an ablation is a real training arm |

The operator applied every finding that touched the mechanism and declined the rest (a
line-continuation indentation, a NEEDS_FIX without a stated finding, the default Stage 3
code path, which the MCP's `stage3_start` takes as an argument).

Two findings were regressions the operator introduced while making the arm fast, and the
verifiers caught both: the vectorised construction had dropped the multiplicity of every
co-consumption pair and had reduced the transition table to the last edge of every sequence.
The tree now carries an equivalence check that compares the vectorised counts with the
original dictionary construction value by value.

## Runtime

| phase | before | after |
|---|---|---|
| start-up (signal + KMeans + first refresh) | ~57 min | **~20 s** |
| training | 2.36 s/epoch | **0.251 s/epoch** (3000 epochs in 785 s) |

The co-consumption and transition counts are built with numpy instead of 4.7 million
dictionary increments (386 s -> 3.7 s), the region profiles and the radar are one bincount
per epoch, and the agreement of every item with every region is one sparse product per
epoch instead of a per-step gather.

## The result: the first level collapses

| quantity | this arm | baseline | best arm in the 67-table sample |
|---|---|---|---|
| L1 codes | **3** | 247 | 204-248 |
| largest L1 bucket | 0.642 | ~0.02 | — |
| L2 prefixes | 394 | — | 15,000-24,000 |
| L3 triples | 9,216 | ~20,000 | 15,000-24,000 |
| extension alphabet | 73 | 42 | 7-10 |
| `cooccur_overlap_L3` | **0.0601** | **0.0756** | 0.168 |
| `adjacency_L3` | 0.0237 | 0.0506 | 0.23-0.27 |

The gate (the target metric has to clear the plain baseline's 0.0756 before any Stage 3
budget is spent) was **not met**, so no Stage 3 run was made.

## What the line now shows

Four designs in this line, all of which let the behaviour signal decide the code, all
collapsed the partition, and the more strongly the behaviour signal drove the assignment
the worse the partition got:

| variant | L1 codes | largest L1 bucket | `cooccur_overlap_L3` | Stage 3 |
|---|---|---|---|---|
| replace (iteration 1) | 116 | 0.985 | 0.0584 | 0.058497 |
| balanced (iteration 3) | 104 | 0.693 | 0.0571 | not run (gate) |
| cogeo, differentiable (iteration 6) | **3** | 0.642 | 0.0601 | not run (gate) |

The reconstruction loss is the only counterweight, and it is too weak: a score that
decides the code outright wins over the reconstruction, and the items whose co-consumption
profile agrees pick the same region. The next design has to keep the Euclidean term in
charge of the ordering and use the behaviour term as a bounded bias, with a balance
constraint on the assignment; and it has to be screened on the L1 code count after a few
hundred epochs, which costs minutes instead of a full run.

## The item-led design, as prescribed, still fails — and the collapse is not the behaviour term

| design | L1 codes | largest L1 | `cooccur_overlap_L3` | Stage 3 `test_recall@10` |
|---|---|---|---|---|
| baseline `TIGER_RQ-VAE` | 247 | ~0.02 | 0.0756 | **0.055537** |
| context replaced the assignment | 116 | 0.985 | 0.0584 | 0.058497 |
| Sinkhorn-balanced assignment | 104 | 0.693 | 0.0571 | not run |
| coupled assignment, differentiable | 3 | 0.642 | 0.0601 | not run |
| item-led + bounded context, usage pull 0.5 | 1 | 1.000 | 0.0275 | not run |
| **item-led + bounded context, usage pull 0.02** | **2** | **1.000** | **0.0339** | **0.039834** |

The last row is the design the operator asked for: the squared Euclidean distance of the
item's own latent always orders the codes, the behaviour context is the clipped, gated
contrast of the candidate code against the positive and the hard-negative context, and a
soft usage pull keeps the first-level usage near the no-mechanism usage. It **also**
collapsed, and it is the worst Stage 3 of the line — 0.0398, 0.0157 below the baseline.

Three facts pin the cause down:

1. The behaviour term is bounded to +/- 1 cost unit and the usage pull is 0.02, so neither
   can move an item that the Euclidean term has already placed. The collapse therefore does
   **not** come from the behaviour signal any more.
2. The two runs of the item-led design differ only in that constant (0.5 vs 0.02) and their
   final collisions were 0.99996 and 0.9367: the collapse tracks the usage pull, which means
   it is being driven by the score path rather than by the behaviour term.
3. The remaining difference from the baseline is the **straight-through path**: the coupled
   score replaces the hard code in the soft estimate (`x_q = x_q + soft - soft.detach()`) and
   the soft reconstruction enters the loss.

The decisive next experiment is cheap and needs no new mechanism: train the arm with
`COGEO_MODE = "behaviour_off"`, which by construction reduces the score to the plain squared
Euclidean distance and drops the usage pull. If that ablation also collapses, the culprit is
the straight-through path and the behaviour terms are exonerated; if it trains like the
baseline, the culprit is in the coupled score. Twelve minutes of Stage 2 answers it.

## The framework, as asked for in this session

* five ideation agents, three scorers, **five verifiers** per implementation;
* the verifiers **report**, the main agent **decides**: `research_round_accept`,
  `research_round_plan_repair`, `research_round_apply_repair`;
* the verification and repair-planner agents are **read-only** (no shell, no edit, no write;
  their answer is read back from the opencode session stream and the framework writes it);
* the agents of one phase run **concurrently**: a review round fell from 176-210 s to 48 s;
* the repair cap counts the **review rounds**, not the source revisions, so a performance fix
  no longer costs a candidate a repair round.
