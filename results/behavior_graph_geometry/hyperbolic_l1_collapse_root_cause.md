# Hyperbolic L1 Codebook Collapse: Root Cause Analysis and Repair

Companion report to the behaviour-graph geometry study
(`results/behavior_graph_geometry/research_report.md`). Same repository, same seed 42,
single seed throughout. This report covers the RQ-VAE side only: what was wrong with
the hyperbolic L1 quantiser, what was fixed, what the fix bought, and what was falsified
on the way.

Provenance: git `45fbfb0` for the measurement commits, `a1c8fac` HEAD. Experiments run
in `stage2_RQ-VAE/TIGER_FORK_RQ-VAE/`, Stage 3 through `stage3_T5Train/scripts/run_stage3_fork_arms.py`.
Artefacts: `results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE/hyper_l1_*`, `results/stage3_T5Train/tiger_fork_*`.

---

## 1. The symptom

Every arm of the hyperbolic L1 line trained to a single L1 codeword:

| arm | L1 codewords used | full-SID collision | validation-set L1 enrichment |
|---|---|---|---|
| `hyper_l1_s42` (geodesic L1 selection) | **1 / 256** | 0.6208 | **1.00** |
| `hyper_l1_br_euclid_s42` | **1 / 256** | 0.6302 | **1.00** |
| `hyper_l1_ang_s42` | **1 / 256** | 0.7973 | **1.00** |
| `nce_euclid_group` (healthy Euclidean control) | 236 | 0.0508 | 6.87 |
| `Exp1a_resample` | 241 | 0.0363 | 16.39 |

L1 enrichment of 1.00 is not a weak result, it is a constant: every item pair shares
L1, so positive sharing = negative sharing = 1.0, the gap is exactly 0 and the
enrichment ratio is exactly 1.

## 2. Confirmed root cause: the geodesic L1 nearest-neighbour rule

**The Poincare distance grows with both radii.** For an item pinned near the ball
boundary, the nearest codeword is therefore the one with the smallest norm whatever its
angle, and the rule stops discriminating by direction at all. Three measurements, in
order:

**(a) The rule alone destroys a healthy codebook.** Taking the *same* frozen codebook
and the *same* embeddings and assigning them three ways:

| arm | rule | codewords kept | top-10 share | chosen-codeword norm mean/std | Spearman(item radius, chosen radius) |
|---|---|---|---|---|---|
| healthy control | euclidean | 236 | 0.2215 | 0.1980 / 0.1474 | 0.8670 |
| healthy control | **geodesic** | **51** | **0.7967** | 0.0896 / 0.0272 | 0.1185 |
| healthy control | angular only | 247 | 0.1191 | 0.3497 / 0.2652 | 0.2764 |
| collapsed arm | euclidean | 98 | 0.9787 | 0.0355 / 0.0497 | 0.3293 |
| collapsed arm | **geodesic** | **1** | **1.0000** | 0.0193 / 0.0000 | -0.0329 |
| collapsed arm | angular only | 201 | 0.6319 | 0.1481 / 0.1033 | 0.0732 |

The geodesic rule on a *healthy* codebook keeps 51 of 256 and puts 79.67% of items on ten
of them, and it systematically selects the smallest-norm codewords. The angular channel
on the same codebook keeps 247.

**(b) The collapse is present in the parameters, not in the distance computation.**
Assigning the collapsed `hyper_l1_s42` model with the plain Euclidean rule already puts
**90.06%** of items on one codeword, and its L1 codebook has contracted (prototype norm
mean 0.5408 → 0.2267, max 1.8393 → 0.5637). The geodesic metric turns a near-total
concentration into an absolute one; it did not create it.

**(c) The time order rules out the encoder.** On an instrumented re-run of the parent
configuration, with the behaviour weight still exactly 0 (the ramp opens at epoch 1500):

| epoch | L1 codes | L1 top share | encoder norm | prototype norm | recon | ball radius |
|---|---|---|---|---|---|---|
| 100 | **17** | 0.3158 | 0.0760 | 0.1799 | 1.21e-04 | 0.7193 |
| 200 | **6** | 0.3568 | 0.0829 | **0.1822** | 1.10e-04 | 0.7566 |
| 400 | 5 | 0.3916 | 0.0884 | 0.1822 | 1.02e-04 | 0.7833 |
| 1000 | 4 | 0.4268 | 0.0959 | 0.1821 | 9.62e-05 | 0.8154 |
| 1300 | 4 | 0.4201 | 0.0984 | 0.1821 | 9.46e-05 | 0.8251 |

L1 occupancy collapses first and fastest (17 → 6 codewords between epoch 100 and 200).
The encoder norm only drifts (+29% over 1300 epochs, +9% across the collapse window).
The reconstruction is still *improving*. The prototype norm freezes at 0.1822 from epoch
200 on — the codebook stops following the encoder exactly when the assignment
concentrates.

**Confirmation by intervention.** Replacing *only* the L1 selection distance with the
ball's radius-free angular channel — codebook objective, L2/L3, capacity, budget, seed,
normalizer and initialisation order all untouched — moved:

| quantity | before (`hyper_l1_s42`) | after (`hyper_l1_angsel_s42`) |
|---|---|---|
| L1 codewords in use | 1 / 256 | **256 / 256** |
| L1 top-codeword share | 1.0000 | 0.0193 |
| full-SID collision | 0.6208 | **0.0450** (Euclidean control 0.0508) |
| validation L1 enrichment | 1.00 | **11.79** |
| validation L1+L2 enrichment | 6.15 | **15.60** |
| validation L1 positive / negative sharing | 1.00 / 1.00 | 0.0752 / 0.0064 |

The collapse is caused by the selection rule. This is established by intervention on the
mechanism, not by correlation with it.

## 3. The fix, and what it did not buy

Stage 3 under the frozen protocol, n_eval = 57439, seed 42:

| arm | R@10 | NDCG@10 | L1 codes | collision | validation L1 ER |
|---|---|---|---|---|---|
| Exp1a_resample | 0.0636327 | 0.0347812 | 241 | 0.0363 | 16.39 |
| nce_euclid_group | 0.0590191 | 0.0324047 | 236 | 0.0508 | 6.87 |
| hyper_l1_s42 (collapsed) | 0.0614043 | 0.0334570 | 1 | 0.6208 | 1.00 |
| hyper_l1_br_euclid_s42 | **0.0625011** | 0.0338817 | 1 | 0.6302 | 1.00 |
| **hyper_l1_angsel_s42** (collapse fixed) | 0.0613695 | 0.0332098 | **256** | **0.0450** | **11.79** |

**Recall@10 went from 0.0614043 to 0.0613695 — a change of -0.0000348, -0.06%, noise on
a single seed.** Codebook utilisation went from the worst arm in the tree to better than
the healthy Euclidean control, and nothing downstream moved.

### Two inference chains are now experimentally falsified

- *"Fixing the L1 codebook utilisation improves Recall@10"* — falsified here.
- *"Raising L1 and L1+L2 enrichment raises Recall@10"* — falsified here: L1+L2
  enrichment rose 2.5× (6.15 → 15.60) and L1 enrichment rose from 1.00 to 11.79, with
  Recall@10 flat.

The starting evidence for a hyperbolic mechanism was a cross-sectional correlation
over 38 trained models (L1+L2 enrichment vs Recall@10, Spearman 0.891). Across models a
higher enrichment accompanies a better model; moving the enrichment alone inside one
model does not move Recall@10. That correlation is not an intervention lever, and it
should not be cited as one.

## 4. Causes examined and excluded

| candidate | how it was tested | verdict |
|---|---|---|
| Poincare distance formula | d(0,r) = 2·atanh(r) to 1e-9 at r = 0.1/0.4/0.8; matches the closed form acosh(1 + 2c‖x−y‖²/((1−c‖x‖²)(1−c‖y‖²))); pair form equals scalar form on all 28 pairs; d(expmap0(u),0) = 2‖u‖ | **correct** |
| float32 instability | float32 and float64 pick the same L1 codeword on 512/512 items; near-boundary distances finite and monotone | **not the cause**; two benign properties remain (expmap0 saturates to exactly the boundary and is rescaled; the acosh floor is 4.472e-4 with exactly zero gradient below it) |
| `argmin` non-differentiability | the one-hot selection tensor has `grad_fn=None` | **real but not the cause**: confirmed as a mechanism fact, and it is what finding 2(b) measures the consequence of |
| KMeans initialisation order | resolve the normalizer from the initial encoder before `init_codebook`, so centres and the first assignment share the training geometry. MVG PASS (L1 occupancy at init 256 → 171 codewords, 32.18% of initial assignments change) | **falsified as the cause**: the repair changes the initialisation substantially and the L1 level still collapses, now by epoch 200 |
| geometry-consistent codebook objective (ball MSE) | train with `mse(enc(e_k), sg[enc(z)]) + β·mse(enc(z), sg[enc(e_k)])` | **falsified in this formulation**: diverges from epoch 45, loss 6.45e-05 → 8.76e-09 → 5.42e+09, collision 0.9996. The normalizer inside `encode_ball` multiplies the codebook's effective step by ~142, coinciding with the learning-rate warmup peak (1.000e-03 at epoch 50) |
| dividing that loss by normalizer² | same formulation rescaled | **does not fix it**: the encoder-gradient share stays at 31.2–38.0 against the pre-map objective's 11.8–14.7, because the normalizer² is carried by the exponential map's Jacobian, so dividing it out cancels exactly the factor that made the term heavy. Both formulations are the same experiment as far as stability is concerned |
| Riemannian log map and squared-geodesic objectives | metric + direction checks | **falsified as written**: their encoder-gradient share is 1.97e+05 and 1.25e+06 against the pre-map objective's 14.7 |
| boundary saturation as the cause | the *healthy* Euclidean control's codewords sit at ball radius 0.9814, essentially on the boundary, and are not collapsed | **not sufficient**; but under the frozen initialisation normalizer both arms saturate badly (46.02% of the control's items and 76.95% of its codewords past radius 0.99), so it is a contributing regime, not the mechanism |

## 5. Confirmed secondary defect: the hard-negative hinge was inverted

The registered hard-negative term was

```
L_hard = mean relu(margin + d_shared - d_alt)
```

whose gradient with respect to `d_shared` is **+1**, so gradient descent *decreases*
the distance to the shared codeword: the term actively deepened the wrong sharing it was
meant to break. The form that pushes out is

```
L_hard = mean relu(margin + d_alt - d_shared)      # gradient w.r.t. d_shared is -1
```

and the positive-preservation term is already correct and must stay
`relu(margin + d_shared - d_alt)`. This was found by measuring the gradient direction
against the intended motion, not by inspecting the sign. The `hyper_l1_br_*` arms
therefore tested a term that behaved as a "pull everything together" constraint; their
behaviour must be re-read with that in mind, and the corrected form has not been trained
yet.

## 6. Still open, and what would settle it

1. **Whether a corrected boundary term helps at all.** The inversion in §5 means the
   boundary-repulsion family has not actually been tested. This is the next single
   variable.
2. **Why the hyperbolic arm degrades with dimension.** Under the frozen initialisation
   normalizer the representation saturates near the ball boundary, where the geodesic
   gradient vanishes. Whether rescaling the normalizer as the encoder grows, or
   parameterising in the tangent chart, removes that, is untested.
3. **Whether the L1+L2 prefix carries the recoverable signal.** The fix moved L1+L2
   enrichment 2.5× with no Recall effect, which weakens the prefix hypothesis but does not
   rule it out; a direct L2/L3-level intervention would test it.
4. **Seed sensitivity.** Every number here is one seed. A -0.06% difference is
   indistinguishable from noise *and* from a small real effect; separating them needs
   more than a second run of the same configuration.

## 7. Bottom line

The hyperbolic L1 quantiser's collapse was a **geometry bug in the assignment rule**,
confirmed by a single-variable intervention that took L1 from 1 of 256 codewords to 256
of 256. It is fixed. Restoring the L1 level did **not** improve Recall@10, which
falsifies the two inference chains the programme was built on and removes the
behaviour-prefix enrichment as a lever. Combined with the behaviour-graph study
(`research_report.md`), which finds no tree-like metric structure beyond the degree
sequence and no embedding advantage for Poincare at d = 8/16/32, the case for a
hyperbolic mechanism in the L1 prefix is not supported by the evidence collected here.