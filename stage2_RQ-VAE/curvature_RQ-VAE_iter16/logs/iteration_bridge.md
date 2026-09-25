# iteration_bridge (Agent G — post iter15 → iter16)

## Dominant bottleneck (iter15)

Frozen residual `u_l` + cyclic `c(t)` still **regressed** downstream (test_R@10=0.0582); iter11 learnable `u_l` remains peak (0.0598). Cyclic time-mixing obscures whether fixed hierarchy geometry helps.

## Forbidden next directions

- iter15 freeze-only `u_l` retries; iter14 `b_l` without new hypothesis.
- Re-using unstable `(log B/m)^2` mapping from iter12 without the stabilized denominator.

## Next iteration objective (iter16)

**Closed-form fixed `[c0,c1,c2]`** from behavior branching `B_l` + residual `m_l` via
`s_l = log(1+B_l)/log(1+m_l/m_min)`, `z_l` standardized, `c_l = c_base*exp(α z_l)`;
**no cyclic, no learnable layer scale, no behavior multiplier** — isolate “formula decides curvature”.
