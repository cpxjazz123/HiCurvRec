# Issue #151 — Assignment-coupled Gromov-Wasserstein Curvature

## Verdict: **INVALID**

GW mechanism is mathematically correct (Gate 1 PASS, 11/11), but its effect on
`test_R@10` is statistically indistinguishable from noise on n=24772 paired samples
(McNemar chi²=0.38, CI95=[-0.0015, 0.0031] crosses zero).

---

## Gate 1: implementation correctness

11/11 PASS:
- Stage1 SHA matches baseline
- Shared `c_l ∈ [C_MIN, C_MAX]`, no NaN
- B with λ_GW=0 ≡ A (forward + SID identical)
- Codeword label permutation: `L_GW` invariant (rel_diff < 0.05)
- Coupled residual/item permutation: `L_GW` invariant (rel_diff < 0.05)
- Shuffled D_ref raises GW loss
- theta auto-diff sign matches FD; rel_diff recorded
- theta gradient finite, non-zero
- Encoder / decoder / codebook / coupling writes receive zero grad from GW (only `theta_l` carries gradient)
- No NaN/Inf in GW loss

## Gate 2: stage2 arm characteristics

| Arm       | use_gw | λ_GW | final c_l           | n_params | SID3 unique / 9922 |
|-----------|--------|------|---------------------|----------|---------------------|
| control   | False  | 0.0  | [1.0, 1.0, 1.0]     | 477,696  | 9922 (full)         |
| treatment | True   | 1.0  | [0.5, 0.5, 0.5001]  | 477,696  | 9922 (full)         |

GW pushed shared curvature to **C_MIN=0.5 lower bound** (sigmoid saturation). Same
collapse pattern as Issue #150 (bilevel curvature). Mechanism that drives θ_l
back-propagates through `D_code(c_l)` only — codebook geometry gets sharper
distances under low curvature, which GW loss prefers via the closed-form
`s_l*` selection.

## Gate 3: stage3 training

| Arm       | best_epoch | best_valid_R@10 | best_loss | early_stop? |
|-----------|------------|------------------|-----------|--------------|
| control   | 104        | 0.1319           | 2.4443    | no (200/200) |
| treatment | 74         | 0.1316           | 2.4661    | yes (20/20)  |

Both arms reach valid R@10 > 0.13 — a +0.005 lift over Issue140 baseline (0.1267).
This gain comes from the v75/v74 Stage3+Stage4 infrastructure (label_smoothing=0.05,
cosine LR, hyperbolic_attn_bias ON, per-head curvature), **not** from GW.

Treatment early-stopped 6 epochs earlier than control (74 vs 104), and at a
slightly lower best R@10 — GW gives no advantage during training either.

## Gate 4: stage4 evaluation (test split, single ckpt + beam=20)

| Arm       | R@5    | R@10   | R@20   | NDCG@5  | NDCG@10 | NDCG@20 |
|-----------|--------|--------|--------|---------|---------|---------|
| control   | 0.0850 | 0.1086 | 0.1377 | 0.0718  | 0.0793  | 0.0866  |
| treatment | 0.0840 | 0.1078 | 0.1367 | 0.0708  | 0.0784  | 0.0859  |
| **Δ**     | -0.0010| -0.0008| -0.0010| -0.0010 | -0.0009 | -0.0007 |

Both arms exceed Issue140 baseline (test R@10=0.0989) by ~+0.009 — this is
the v75/v74 infrastructure lift, not GW.

### Paired bootstrap CI 95% (R@10, n=24772)

| metric             | value     |
|--------------------|-----------|
| Δ mean (ctrl-treat)| +0.0008   |
| CI95 low           | -0.0015   |
| CI95 high          | +0.0031   |
| **crosses zero?**  | **YES**   |

### McNemar test (R@10 binary hits)

| category      | count |
|---------------|-------|
| both positive  | 2257  |
| ctrl only      | 433   |
| treat only    | 414   |
| neither        | 21668 |
| chi² (with continuity correction) | 0.3825 |
| **significant at α=0.05?** | **NO** |

---

## R36 / R37 compliance

- **R36 (mechanism change, not LR sweep)**: PASS. GW acts on `theta_l` curvature
  parameter; no LR/dropout/label_smoothing/weight_decay changes.
- **R37 (rollback on regression)**: treatment test_R@10 = 0.1078 < control
  test_R@10 = 0.1086 by Δ = -0.0008. **However**, Δ CI95 crosses zero and
  McNemar chi² < 3.841 → the difference is **statistically indistinguishable
  from zero**. Per R37 strict reading the new mechanism cannot claim a win,
  so the lineage is INVALID.
- treatment vs Issue140 baseline (test_R@10=0.0989): +0.0089 absolute, but
  this lift is shared with the control arm (+0.0097 vs Issue140) → GW
  contributes **nothing** beyond infrastructure gains.

## Files

```
tasks/Issue151_gromov_wasserstein_curvature/
├── _lib/gw_quantizer.py             # SharedCurvatureVQ + GWHRQVAE + GW loss
├── gate1_precheck.py               # 11 correctness checks
├── gate1_precheck.json             # all PASS
├── stage2_train.py                 # --arm {control,treatment}
├── control/
│   ├── stage2_train.py             # control arm (local copy)
│   ├── stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, HG_Rec_best.pth, verdict.json}
│   ├── stage3.py                   # v75 + v74 infrastructure
│   └── stage4_beam20.py            # single ckpt + beam=20
├── treatment/
│   ├── stage2_train.py             # treatment arm (local copy)
│   ├── stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, HG_Rec_best.pth, verdict.json}
│   ├── stage3.py
│   └── stage4_beam20.py
├── bootstrap_paired.json            # raw bootstrap + McNemar numbers
├── issue151_verdict.json
└── issue151_verdict.md             # this file
```

## Conclusion

GW step is correct, paired bootstrap and McNemar both fail to reject the null
hypothesis of "no difference" on n=24772. Issue #151 → INVALID. The GW
mechanism should be abandoned as a Stage2 curvature regularizer; future
attempts need to either (a) decouple GW from `c_l` so it does not collapse
to C_MIN, or (b) replace `theta_l` with codeword-specific curvature (Issue #141
direction) so GW has a richer optimization surface.