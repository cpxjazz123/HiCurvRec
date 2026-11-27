## Close: negative geometry-gate result, no formal training

**Verdict.** Per-triple continuous curvature is not identifiable under the frozen
reference representation. 99.78 % of the 28.4 M fitted values sit on the
coordinate-box floor (`κ=-0.125`), one global κ reproduces the per-triple result
(0.18 % subset / 0.0 % full-reference difference), and a single global Euclidean
scale (`S*=0.068284`, J `0.174575`) beats the entire per-triple family
(`0.253693`). The issue's own gate therefore stopped the mechanism before
training: no Stage2, no Stage3, no SID, no recall claim.

**Delivered.** `results/research/issue268/`:
`DIAGNOSIS.md`, `HYPOTHESES.json` (17 inherited/updated entries), `CONTRACT.md`,
`VERDICT.md`, `INFRA.md`, `INFRA_SNAPSHOT.json`, `fit_kappa.py`,
`validate_kappa.py`, `scale_control.py`, `verify_cache.py`, `zref_probe.py`;
evidence in `analysis/`; the frozen `Z_ref` and the full κ cache under
`results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/kappa_cache/` (manifest with
all input/output hashes, `111.04 s`, 341 MB).

**Acceptance.** C1–C8, C17–C19 PASS; C14 PARTIAL (distance side PASS, SID side
NOT APPLICABLE); C9–C13, C15, C16/R2 NOT APPLICABLE under the gate. No unrun item
is reported as a pass.

**Git.** `babc90ee` (evidence + diagnostics), `9232e519` (remove duplicated
public source copy), `20aff2bf` (restore an unrelated file); local `main` =
`origin/main`.

**Knowledge increment.** The gate never could move hard q; its channel was only
the curvature objective's scale. The frozen encoder's Euclidean geometry is
globally mis-scaled against graph hops, and that one scalar dominates everything
a per-triple metric table can do. Future geometry work must change the global
calibration or the reference representation, not add a per-triple parameter.
