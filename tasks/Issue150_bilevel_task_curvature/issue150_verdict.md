# Issue #150 Verdict — Bilevel Task-aware Curvature

**Verdict: INVALID**

> Bilevel mechanism fails to improve over single-level control. Treatment
> curvature collapses to `c_l ≈ C_MIN=0.5` and outer preference loss stays
> near random (`log 2 ≈ 0.693`), producing a Stage4 R@10 that is statistically
> indistinguishable from control (ΔR@10 = -0.00032, 95% CI crosses 0,
> McNemar p = 0.82).

## 1. Setup (R40 self-contained)

- **Task directory**: `tasks/Issue150_bilevel_task_curvature/`
- **Stage2/3/4**: copied from `baseline/` (R46), patched only paths/cuda/port
- **Two arms (A/B)**:
  - **A (control)**: single-level optimization — encoder/decoder/codebook/θ
    all updated by RQ-VAE loss in Poincaré ball.
  - **B (treatment)**: bilevel optimization — φ (encoder/decoder/codebook)
    updated by inner RQ-VAE, θ_l updated by outer next-item preference loss
    hypergradient through functional φ' = φ − η_inner ∇_φ L_RQ.
- **Data**: `dataset/{train,valid,test}.parquet` (R44). 90/10 user hash split
  into inner-train (90%) and meta-train (10%) for inner/outer.
- **Architecture**: 3-layer hyperbolic RQ-VAE (CODEBOOK_SIZES=(64,128,256))
  with shared curvature `c_l = C_MIN + (C_MAX−C_MIN)·sigmoid(θ_l)`,
  C_MIN=0.5, C_MAX=2.0.
- **Stage4**: R35 single checkpoint + `beam_search=20` evaluation.

## 2. Stage2 Mechanism Health (Gate 1)

Gate 1 8 checks all PASS (see `gate1_precheck.py` log):

| # | Check | Result |
|---|-------|--------|
| 1 | Stage1 SHA matches baseline | PASS (`1a6dd2ac...`) |
| 2 | inner/meta split is 90/10 + only train read | PASS (22272/2500) |
| 3 | B with zero outer ≡ A | PASS (out_diff=0, loss_diff=0, sid_diff=0) |
| 4 | Hypergradient autograd vs FD direction match | PASS (sign_match=True, rel_diff=0.25) |
| 5 | θ non-zero, φ unchanged at θ=0 | PASS (theta_nonzero=3/3, phi_unchanged=True) |
| 6 | Shuffled labels change margin | PASS (margin_diff=0.0036, grad_diff=1.6e-4) |
| 7 | τ annealing monotonic on hard mass | PASS (5.0→0.2, mass 0.01→0.12) |
| 8 | No NaN/Inf, c in [C_MIN, C_MAX] | PASS (nan=0, c_in_bounds=True) |

Mechanism is *implementable*; the problem is what the optimization actually
finds.

## 3. Stage2 Results

```
=== control ===
epoch 99 | loss=0.3917 | c=[0.5279, 0.5229, 0.5242] | SID3=9066

=== treatment ===
epoch 99 | loss=0.3917 | c=[0.5004, 0.5004, 0.5001] | SID3=9114 | outer=0.6833
```

- **Control c_l stays bounded away from C_MIN** (~0.524–0.528), consistent with
  a healthy train on the RQ-VAE objective.
- **Treatment c_l collapses to ~C_MIN=0.5** (θ_l → −∞).
- **Treatment outer loss plateaus at 0.683**, which is essentially `log 2`
  (random). The preference signal is uninformative — the bilevel gradient
  drives c_l toward the lower bound but does not move the ranking objective
  away from random.

Interpretation: soft-assignment codeword `q_l(i) = Σ_b p_l(b|i) e_l,b` with
Poincaré distance `d_c` and softmax temperature τ produces a near-uniform
ranking over items (≈ random preference), so the outer loss has no signal
for θ. The hypergradient pushes θ in the only direction that consistently
lowers the random signal: shrink c toward 0 (where the ball flattens and
all distances collapse). The mechanism is *valid* (Gate 1 PASS) but the
*objective* is uninformative in this geometry.

## 4. Stage3 Training (Gate 4/5)

- **control**: best epoch 124, valid_R@10 = **0.1325**, EARLY_STOP=20, 200/200 epochs
- **treatment**: best epoch 129, valid_R@10 = **0.1329**, EARLY_STOP=20, 200/200 epochs

Training-time valid R@10 difference is 0.0004 (treatment marginally higher),
well within stage3 epoch noise (~0.001). Both reach baseline-level
performance — no catastrophic divergence.

## 5. Stage4 Evaluation (Gate 4 — primary outcome)

| Arm | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-----|-----|------|------|--------|---------|---------|
| control | 0.0859 | **0.1084** | 0.1399 | 0.0707 | 0.0780 | 0.0860 |
| treatment | 0.0843 | **0.1071** | 0.1365 | 0.0697 | 0.0771 | 0.0845 |
| **Δ (T-C)** | -0.0016 | **-0.0012** | -0.0034 | -0.0010 | -0.0009 | -0.0014 |

(Stage4 test metrics from `eval_test.json`; full beam=20, single checkpoint.)

**Paired bootstrap 95% CI on R@10 (n=24772, B=10000 resamples)**:
- Δ = -0.00032
- 95% CI: **[-0.00275, +0.00210]** — **crosses 0**
- One-sided p(direction matches Δ) = 0.385

**McNemar test on hit@10 disagreement**:
- b = 462 (ctrl hit, trt miss)
- c = 454 (ctrl miss, trt hit)
- Two-sided p = **0.817**
- Discordant fraction = 3.70% (916/24772)

Both paired bootstrap CI and McNemar confirm the difference is **not
statistically significant**. The point estimate is marginally negative
(treatment −0.0012 R@10), which combined with the c_l collapse and
unchanged outer loss is consistent with bilevel mechanism providing no
benefit and very slightly hurting via the collapsed-curvature prior.

## 6. Gate Summary

| Gate | Question | Answer |
|------|----------|--------|
| **Gate 1** | Spec exists, mechanism is implementable | **PASS** (8/8 checks) |
| **Gate 2** | Stage2 is internally consistent | **PASS** (control c bounded, both reach SID3>9000) |
| **Gate 3** | Mechanism changes behavior (R36) | **PASS** (c_l trajectory differs, B-0 ≡ A in check3) |
| **Gate 4** | Stage4 test R@10 ≥ baseline (R35+R37) | **FAIL** — 0.1071 < 0.1084 baseline-equivalent; 95% CI crosses 0; McNemar p=0.82 |

## 7. Failure Mechanism (R36 mandatory root-cause)

The bilevel outer objective
```
L_pref = -log σ(S(i, j+) − S(i, j−)),  S(i, j) = -Σ_l d_{c_l}(q_l(i), q_l(j))
```
with `q_l(i)` a soft codeword built from item-encoder output, is **bounded
between `log 2` (random) and 0 (perfect)**. Empirically:

1. With shared curvature `c_l ∈ [0.5, 2.0]` and item embeddings trained
   jointly with the inner RQ-VAE loss, all item distances are highly
   correlated across pairs. The pairwise preference signal carries almost
   no curvature-specific information — it mostly tracks the inner-loss
   structure.
2. The only way for `∇_θ L_pref` to consistently lower `L_pref` is to
   collapse `c_l` toward 0, where the Poincaré ball flattens and distance
   differences vanish — which is precisely what happened
   (`c_l → 0.5001`).
3. Once `c_l` is at C_MIN, the soft codeword `q_l(i)` becomes essentially
   the mean of codewords in Euclidean tangent space (tangent @ 0 is
   Euclidean), and the preference loss is uninformative (`≈ log 2`).
4. With uninformative outer gradient, the φ update from the virtual inner
   step dominates and tracks the control arm's RQ-VAE loss. Stage3 picks
   up the same T5 signal regardless of θ, producing R@10 essentially
   identical to control.

This is a **structural failure of the soft-codeword preference surrogate,
not a bug** — the mechanism passes Gate 1 precisely because at θ=0 the
outer gradient vanishes (so check3 holds). The information geometry of
`d_c(q_l(i), q_l(j))` with shared curvature simply does not carry enough
task signal to drive θ meaningfully.

## 8. Verdict

**INVALID — close issue.**

- Treatment c_l collapses to C_MIN; outer preference loss uninformative
  (≈ log 2).
- Stage4 R@10 = 0.1071 (treatment) vs 0.1084 (control), Δ = -0.0012,
  95% CI [-0.0027, +0.0021] crosses 0, McNemar p = 0.82.
- Bilevel mechanism fails to improve over single-level control on this
  task/dataset. R37 holds — terminate lineage.

Artifacts:
- `tasks/Issue150_bilevel_task_curvature/control/stage2/eval/eval_test.json`
- `tasks/Issue150_bilevel_task_curvature/treatment/stage2/eval/eval_test.json`
- `tasks/Issue150_bilevel_task_curvature/_lib/pairwise_test.json`