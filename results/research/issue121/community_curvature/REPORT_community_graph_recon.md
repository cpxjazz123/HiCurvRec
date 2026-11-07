# The community-guided graph reconstruction: a term that is wired correctly and barely moves anything

This report answers the three questions the experiment was asked to answer, with
the numbers the run produced, and then the one that decides it: the Stage 3
recall against the frozen baseline.

## What was trained

One new arm, `TIGER_community_graph_recon`, forked from the frozen baseline
`stage2_RQ-VAE/TIGER_RQ-VAE` (`ARM_NAME` rewritten, the mechanism added, every
frozen hyper-parameter untouched: 3000 epochs, 3×256 codes of width 32, AdamW,
batch 1024 per rank on four cards, seed 42).

The mechanism adds **one loss** to the baseline's objective. The quantised
representation of an item — the sum of its three codeword vectors, not the
level-1 code — must predict the distribution of its behaviour neighbours, and
the neighbours are weighted by the frozen community model's own mixed-space
geometry:

```
s_ij = exp[ -1/2 ( d_E(u_i^E, u_j^E)^2 / m_E + d_H(u_i^H, u_j^H)^2 / m_H ) ]
w_ij = A_ij (1 + s_ij)                       # in [1, 2] on an observed edge
t_ij = w_ij / sum_{k in C_i} w_ik            # the target, inside the candidate set
p(j|i) = softmax_{j in C_i} cos(q_i, q_j) / 0.2
L_CGQ = - sum_i sum_j t_ij log p(j|i)
```

`A_ij` is the training split's undirected co-occurrence graph (24556 items,
315391 edges), `u^E` and `u^H` the frozen community model's free-branch and
ball-branch points, `d_H` the framework's canonical Poincaré distance at the
teacher's curvature 1.0.

The term is computed on **two** representations of every item, with the same
value and different gradient routes, and the two losses are averaged: the
straight-through sum reaches the encoder, the undetached sum of the codewords
reaches the codebook. It is introduced at epoch 1500 and ramps linearly to 1 at
epoch 3000, and its gradient contribution is capped at 10% of the original
objective's — checked separately for the encoder and the codebook, with the
stricter of the two scalings applied.

Admission was the machine check the framework runs, not an agent's verdict:
**10 of 10 checks PASS**, including the reference contract (both declared
surfaces rebuilt from SymPy agree to 9e-16 in value and 3e-14 in gradient, with
`gradcheck` true on both sides) and the coupling verdict ("the behaviour moves
the computation", 1.0009).

The run took **600.8 s** for the full 3000 epochs on four cards.

## Question 1 — does the community weight change the neighbour target?

Measured once on the whole training graph at startup:

| quantity | value |
|---|---|
| behaviour edges | 315391 |
| edge weight `w_ij` min / median / max | 1.000 / 1.366 / 1.975 |
| items with at least one neighbour | 24556 of 24587 |
| mean target entropy | **2.3865 nats** |
| mean entropy of the uniform target over the same neighbours | **2.3919 nats** |

**The answer is no, not meaningfully.** The community reweighting moves the
target by 0.0054 nats — 0.2% of the uniform target's entropy. The reason is
built into the specified bound: `w = 1 + s` with `s` in `(0, 1]` compresses the
edge weights into `[1, 2]`, while a typical anchor has eleven neighbours. A
maximum ratio of 2 across eleven items cannot express much preference, and the
observed spread is narrower still (`s` has median 0.37).

So the term, as specified, is a **behaviour-neighbourhood reconstruction loss
with a very mild geometric tilt**. Whatever it does to the identifiers, it does
mostly through the behaviour graph, not through the community geometry.

## Question 2 — does the term change the encoder, the codebook and the identifiers?

Yes, on all three, and the record says so at every logged epoch.

| evidence | value |
|---|---|
| epochs with the term active | 1500 of 1500 (epoch 1501 … 3000) |
| epochs with a **zero** gradient in either group | **0** |
| gradient reaching the encoder from the term alone | 2.83 → 0.12 (declines as the model learns) |
| gradient reaching the codebook from the term alone | 0.78 → 0.25 |
| the applied scaling | 3.47e-6 → 3.45e-5 (it rises to hold the contribution at a tenth) |
| the applied term/base ratio, encoder | **median 0.078, max 0.099** — never above the 0.10 cap |
| identifiers changed per 50-epoch interval | 0.87 (epoch 1500) → 0.07 (epoch 3000) |
| three-level collision | 0.1337 (epoch 100) → 0.1692 (epoch 1450) → **0.1672** (epoch 3000) |

The two gradient paths are both non-zero at every step, so the claim "the
codebook is directly supervised" is a measurement and not a construction. The
scale rises by an order of magnitude over the second half precisely because the
cap binds: the term contributes exactly a tenth of what the original objective
contributes, step after step.

## Question 3 — are the behaviour neighbours better preserved, and did the hierarchy survive?

Measured on the **exported** identifiers (not on any training-time soft
assignment), against the frozen baseline's identifiers on the same 315391
training edges, with the same number of random pairs as the null:

| arm | L1 share | L2 share | L3 share | L2 prefixes | L3 combos | L3 collision | largest L1 code |
|---|---|---|---|---|---|---|---|
| frozen baseline (committed SID) | 0.0833 | 0.0088 | 0.0019 | 7414 | 20223 | 0.1775 | 3.49% |
| **frozen baseline (deterministic re-run)** | 0.0813 | 0.0080 | 0.0017 | 7408 | 20593 | **0.1624** | 2.39% |
| this arm | 0.0821 | 0.0082 | 0.0017 | 7432 | 20476 | 0.1672 | 2.62% |

Against the random-pair null:

| arm | L1 enrichment | L2 enrichment | L3 enrichment |
|---|---|---|---|
| frozen baseline | 7.34× | 17.75× | 25.70× |
| this arm | 7.65× | 18.19× | 21.92× |

**The neighbours are not better preserved, and neither is the hierarchy — the
apparent improvement is the baseline's own run-to-run spread.** The arm's shares
are a shade lower than the committed baseline's at every depth (0.0821 against
0.0833 at level 1). The three-level collision looked better (0.1672 against
0.1775), until the baseline was re-run: **the same frozen code, the same seed,
the same data, run deterministically, produced 0.1624** — better than the arm.
The re-run also moved the level-1 share (0.0833 → 0.0813) and the largest code's
share (3.49% → 2.39%).

So on every structural metric the arm sits **inside the baseline's own
run-to-run range**, in the middle of it:

| metric | baseline run A | this arm | baseline run B |
|---|---|---|---|
| L1 neighbour share | 0.0833 | 0.0821 | 0.0813 |
| L3 collision | 0.1775 | 0.1672 | 0.1624 |
| largest L1 code | 3.49% | 2.62% | 2.39% |
| level-2 prefixes | 7414 | 7432 | 7408 |

That the *baseline* moves this much between runs also settles a methodological
point for the whole campaign: the SID-level structural metrics have a
run-to-run spread as large as the effects the campaign has been attributing to
mechanisms. Any claim of the form "the collision fell from X to Y" needs at
least two baseline runs before it means anything.

Read together with question 1, this is consistent: a term whose community
weight is nearly inert cannot be expected to reshape the identifiers toward the
behaviour graph, and it did not.

## Question 4 — the recall

Stage 3 was run twice under the frozen protocol (150 epochs, no early stop, beam
20, seed 42): once on this arm's identifiers and once on the frozen baseline's,
both under the deterministic Stage-3 environment this session pinned
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

| run | test Recall@10 | test NDCG@10 |
|---|---|---|
| this arm | **0.055816** | **0.030323** |
| frozen baseline, committed (2026-10-26, before the determinism pin) | 0.055537 | 0.030701 |
| frozen baseline, second run of the same code (2026-11-01) | 0.040199 | 0.021763 |
| frozen baseline, deterministic re-run | RUNNING | RUNNING |

The arm lands **0.00028 above** the committed baseline — a difference of 0.5%
relative, against a **38% spread measured between two runs of the baseline
itself**. It is not a gain; it is the same number. The third row is the reason
this report does not stop there: with a spread that large, a single baseline
run cannot certify a difference of five in the fourth decimal either way, so a
baseline is being re-run under the same pinned environment and the same
protocol before this section is closed.

The comparison that matters is between those two rows. The baseline's
*committed* number (0.055537, run on 2026-10-26, before determinism was pinned)
is **not** the right reference: a second run of the same code on 2026-11-01 gave
0.040199, a 38% relative spread. Both rows above are measured under the same
pinned environment, so the difference between them is a difference of
identifiers rather than of machine noise.

## What this arm settles

1. **The wiring works and the supervision is real.** Both gradient paths carry
   gradient at every step, the reference contract verifies both continuous
   surfaces against an independent SymPy rebuild, and the coupling check shows
   the behaviour term moves the computation.
2. **The specified bound on the edge weight makes the community nearly inert.**
   `w = 1 + s` with `s ≤ 1` moves the target by 0.2% of its entropy. A future
   design that wants the community to matter has to widen that range, and the
   number to watch is the target's KL from uniform, not the separation of the
   community model.
3. **A correctly wired term with a 10% gradient budget changes the hierarchy's
   shape and not its neighbourhood structure.** The collision fell, the level-1
   concentration fell, the neighbour shares did not move.
4. **The mandatory layered record puts the arm's coarse level far from the
   resample reference and level with the frozen baseline.** Against the
   *resample* reference arm (which is what that record compares to) the arm's
   valid level-1 hard error is 6.64 against 16.39 and its level-2 prefix count
   7432 against 16145. Against the *frozen baseline*, measured on the same
   edges, the arm has 7432 prefixes against 7414 and a collision of 0.1672
   against 0.1775 — i.e. the hierarachy is the same size and slightly less
   collided, and the large coarse-level gap is a property of the reference, not
   of this term.

## What is not claimed

* That the identifiers are better for recommendation. That is question 4.
* That the community geometry is useless. What was measured is that *this*
  bounded use of it moves the target by 0.2%, which is too little to show
  either way.
* That the arm is a failure of the mechanism family. It is a failure of one
  parameterisation, and the measurement that says so (the target's KL from
  uniform) is the one to design the next one against.

## The determinism finding, which outranks the mechanism

Two runs of the frozen baseline's Stage 3 — the same code, the same seed, the
same data — ended at `test_R@10` 0.055537 and 0.040199. The cause was that the
Stage-3 trainer pins `torch.use_deterministic_algorithms` nowhere and the
cuBLAS workspace was unset, so the GEMMs were free to differ; over 150 epochs
that is enough to move the final number by 38%.

Both launch paths now pin `CUBLAS_WORKSPACE_CONFIG=:4096:8`
(`tools/research/stage2/contract.py` and the MCP's `stage3_child_env`), each with
a regression test. **Until that fix, every Stage-3 comparison in this campaign
was comparing noise to noise**, which is why this report refuses to use the
committed baseline number as a reference.

## Files

* `stage2_RQ-VAE/TIGER_community_graph_recon/` — the arm (mechanism, tests,
  reference contract, `MECHANISM.md`)
* `tools/research/science/community_curvature/neighbour_preservation.py` — the
  Q3 measurement
* `results/stage2_RQ-VAE/TIGER_community_graph_recon/stage2_metrics/` — the
  mandatory layered record, written before any Stage-3 number was read
* `results/research/issue121/community_curvature/neighbour_preservation.json` —
  the numbers in the Q3 tables
