# Preregistration — candidate `geometry-asymmetric-next-item-hyperbolic-semantic-id`

Frozen proposal: `results/research/issue117/runs/run_f8049f2ce2774d7c/agents/geometry/candidate.json`
sha256 `c8701416d4b694141d60366f61d3b70b4fbbc958f8188447d822f8d36e6a283d`.
Independent review: `.../agents/review/review.json` sha256
`8685bd9ae83ddbfa0470f88632ffb1e9e1cb25f4c165feea5b1c2fec7c229148`, verdict
**CPU_ELIGIBLE** with two MATERIAL items. This document resolves them and is
the contract every later step is measured against.

## 1. What the measured behaviour does and does not say

The DATA evidence (`behavior.json`, sha256
`793d52dce39a5a8aa3963ff566ae50612114b9d20d31227dfed3d1e9fc653b9b`) is
train-only and descriptive: 57 439 users, 396 958 events, 339 519 transitions,
323 633 distinct directed pairs, of which **314 605 (92.66 %) never occur in
reverse**, mean next-item entropy 4.844 bits.

This establishes directional asymmetry of the training transitions. It does
**not** establish a tree, a hierarchy, a causal next-item benefit, or that
negative curvature helps. The candidate's own text says so, and the review
marked it MATERIAL. Consequently:

* the 92.66 % figure is **never** an effect size and never a trigger;
* the effect must be measured per example on held-out data, stratified;
* if the per-stratum effect does not appear, the candidate is falsified even
  when the aggregate is favourable.

## 2. The exact mechanism implemented in Stage 2

Item semantic-ID geometry only; the history encoder and the autoregressive
code-generation pathway are untouched and live in Stage 3, whose protocol is
frozen by CLAUDE.md §22 and is **not** modified by this candidate.

Curvature -1 ball `B^32 = {x in R^32 : ||x|| < 1}`:

```
exp_0(v) = tanh(||v||) * v/||v||            (radius clamped below 1-epsilon)
log_0(x) = artanh(||x||) * x/||x||
a (+) b  = ((1+2<a,b>+||b||^2)a + (1-||a||^2)b) / (1+2<a,b>+||a||^2||b||^2)
D(x, y)  = 2 * artanh(|| (-x) (+) y ||)
```

Implementation: `stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id/`
(`model/hyperbolic.py`, `model/layers.py::HyperbolicVQLayer/HyperbolicRQLayer`,
`model/model.py` geometry switch). The codebook parameters stay unconstrained
tangent vectors, exactly as in the Euclidean host; the geometric position of a
code is `exp_0` of its parameter, the assignment uses `D`, and the residual
handed to the next level is taken in the tangent space at the origin
(`log_0(code)`).

Numerics that the review asked for:

* `exp_0` clamps the radius below `1 - epsilon`, so `artanh` never sees the
  boundary;
* `log_0(exp_0(v)) == v` to 1e-13 in float64;
* `D(x, x) == 0` exactly;
* the distance folds the Möbius offset into the ball with a smooth `tanh`
  bound instead of a hard clamp, because a clamp zeroes the gradient of every
  pair that reaches the boundary; gradients stay finite and non-zero.

In the small-radius limit the operator reduces to the Euclidean one, which is
what makes the baseline host the capacity-matched control.

## 3. Frozen protocol and the matched controls

| Item | Value |
|---|---|
| latent dim | 32 |
| SID | 3 levels x 256 codes, strictly 3 integer tokens, global and stable per item |
| Stage 2 epochs | 3000, no early stop, no hyperparameter sweep |
| Stage 3 | unchanged frozen protocol: 150 epochs, seed 42, beam 20, `NO_EVAL=True`, `SKIP_TEST=False` |
| splits | frozen train / valid / test of `results/stage0_build_parquet/` |
| inputs | train interactions and train-prefix behaviour evidence only; no validation/test event may enter a representation, a trigger or a model choice |

Controls, all with the same data, the same capacity, the same optimiser, the
same budget and the same evaluation:

1. **Euclidean control** — the existing baseline tree
   `stage2_RQ-VAE/TIGER_RQ-VAE/` (identical code except the geometry switch),
   which is the small-radius limit of the candidate.
2. **Directed-graph control** — a third tree with the same capacity whose item
   representation is informed only by train-prefix transition edges, to
   separate "a graph of interactions" from "negative curvature".

The graph control is **not** implemented yet. Until it exists and is verified
in the same way, the comparison is candidate-versus-Euclidean only, and the
result must be reported as such.

## 4. Predeclared measurement

On validation histories built only from allowed past events:

* primary metric: next-item ranking of the held-out target, reported with the
  Stage-3 frozen evaluation, i.e. `test_R@10` from the frozen Stage-3 protocol;
* secondary: NDCG@10, and the same metrics stratified by
  (a) whether the source-destination pair has an observed reverse transition in
  the training prefixes, (b) item frequency, (c) history length;
* every stratification threshold is computed from training prefixes only.

The claim counts only if the candidate improves the primary metric over **both**
controls, and the gain survives on the asymmetric-transition stratum. A gain
that comes from extra capacity, from leakage, or from tuning after seeing the
result does not count.

## 5. Falsifiers

* the candidate does not improve the predeclared ranking metric over the
  Euclidean control, or the gain disappears on the prespecified asymmetric
  stratum -> hypothesis falsified;
* the graph control matches the candidate -> the effect is a graph effect, not
  a curvature effect;
* the per-example trigger frequency is too low to support the mechanism ->
  the candidate is out of scope regardless of any aggregate number.

## 6. Operator corrections applied to the drafted suites

The suites are authored by independent Agents and are promoted into this
directory by the repository operator, which may correct a defect but never
weaken a suite. Two corrections were made and are recorded here so the
independent reviewer can judge them:

1. **test_math.py, the collinear distance reference.** The MATH Agent derived
   the reference correctly in its own comment -- `||(-x) (+) y|| = 5/23`, so
   `D = 2*atanh(5/23) = ln(14/9)` -- but hard-coded the rounded decimal
   `0.442169`, which is 3.4e-4 off the value its own derivation gives
   (`ln(14/9) = 0.44183275227903923`, which the implementation reproduces
   exactly). The reference is now the exact expression
   `math.log(14.0/9.0)` and the tolerance is 1e-12. No assertion was removed
   and no expectation was loosened.
2. **The RUNTIME Agent's first draft was pytest-style** (`tmp_path` /
   `monkeypatch` fixtures, plain test functions). The executor is
   `python -m unittest discover`, which collects only `unittest.TestCase`
   classes, so such a suite runs ZERO tests and is refused. The role guidance
   now states the contract, and the suites under review are the re-authored
   ones.
3. **test_entrypoint.py, the trainer import path.** The draft put only the
   tree's PARENT directory on `sys.path` and then imported the trainer as a
   package member, but `train_rqvae.py` itself does `from model import RQVAE`,
   a top-level import, so the suite failed at import with
   `ModuleNotFoundError: No module named 'model'` and never reached its
   assertions. The tree directory is now on the path as well. No assertion was
   removed, added or loosened.

## 7. What is authorized now, and what is not

The review verdict is CPU_ELIGIBLE: it authorizes the implementation and the
mechanism-specific CPU checks (the four reviewed suites), and nothing else. It
is not GPU approval. Stage 2 may start only after

1. `research_verify_start` -> three independent agents author the four suites;
2. the suites are committed under `tools/research/verification/cases/<candidate_id>/`;
3. `research_verify_review` freezes source, proposal and test hashes;
4. `verification_check(source_dir, verification_id)` really re-runs all four
   suites with the project's Stage 2 interpreter and passes;
5. the resource authorization for the 4-GPU run is present. That standing
   authorization is granted by the repository owner in CLAUDE.md §10 and
   recorded on issue #117; it waives the per-run human approval and nothing
   else, so the four suites, the source/test hashes and the real trainer smoke
   still have to pass on their own.

Until then the honest status is WAITING, not RUNNING.
