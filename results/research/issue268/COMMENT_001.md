## Geometry gate: per-triple κ is not identifiable; no Stage2/Stage3

Frozen inputs: `Z_ref` = #265 final encoder (sha256 `a2643a9b…`, 24587×32),
graph-unit scale `S=0.1058943657241384` (median visible-edge Euclidean length),
`w=Z_ref/(2S)`, legal κ interval `[-0.125, 1.0]` from the coordinate box
(`max||w||=2.8118`).

Full fit over all **28,391,948** legal rows (`results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/kappa_cache/`):

| statistic | value |
|---|---:|
| mean J at κ=0 | 0.268071 |
| mean J at fitted κ_t | 0.242248 |
| improvement median | 9.568 % |
| κ_t at the floor `-0.125` | **99.78 %** |
| κ_t mean / std | -0.124651 / 0.010070 |
| flat fraction | 0.00046 % |

Independent transfer (`analysis/TRANSFER_VALIDATION.json`): seed-269 subset
20,000 rows J0 `0.280895`, per-triple `0.253693`, one global κ `-0.125`
`0.254044` → per-triple beats global on **0.18 %** of rows; full-reference audit
(24,471 refs, 512 rows) → **0.0 %**. Cache recomputation from raw pairs matches
to `5.5e-8` / `3.9e-8` (`analysis/CACHE_VERIFICATION.json`).

Global-scale control (`analysis/SCALE_CONTROL.json`): one global Euclidean scale
`S*=0.068284` reaches **J=0.174575**, better than the whole per-triple family
(`0.253693`); per-triple free scale would be `0.163897`. At `S*` the per-triple
gain is only `2.27 %` median with `38.3 %` boundary solutions
(`analysis/probe_J_scale_0p068284.json`).

Conclusion: the curvature gain is a **global ruler** effect; triplet-specific
curvature is boundary-dominated and not distinguishable from a single global
curvature. Per the issue's own gate ("若无法证实有效几何适配，继续低成本诊断，
不默认正式训练"), formal Stage2 and Stage3 are **not launched**; C13/C15 are
NOT APPLICABLE, not passed. The diagnostic and cache are delivered; see
`results/research/issue268/{DIAGNOSIS.md,VERDICT.md,CONTRACT.md}`.

Host note: the session started with four A100s and now exposes one A100-80GB;
the frozen launcher refuses a single-card formal Stage2. It did not affect this
decision (the gate stopped the mechanism before dispatch).
