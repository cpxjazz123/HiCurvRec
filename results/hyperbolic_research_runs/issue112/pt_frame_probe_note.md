# The path-B candidate: a design defect found before any implementation, and a probe whose own
# isometry check has not passed yet

## 1. The proposal as written is not well posed

`hyp_transported_residual` transports the RESIDUAL into the level-0 codeword's tangent frame and
then quantises it with the **shared** level-1 codebook. The transported frame depends on the item
(the codeword it was assigned), so the shared codebook has no fixed coordinates in that frame: the
proposal needs either a codebook per level-0 codeword (256x the capacity) or the **dual** — the
codebook stays canonical and is transported *into* the item's frame.

The dual is well posed and is what `probe_pt_frame.py` measures.

## 2. What the probe measured (4000 frozen items, the matched control checkpoint, epoch 3000)

| quantity | value |
|:---|---:|
| level-1 assignments changed by the transport | **0.0825** (the transport is causally active, not a no-op) |
| mean level-1 residual norm, ambient frame | 0.27369 |
| mean level-1 residual norm, transported frame | 0.28304 |
| the same, for the items above the median radius | 0.28816 → 0.30154 |
| the probe's OWN isometry check | **0.0679 — FAILED** |

## 3. What may and may not be concluded

**May be concluded**: the proposal's definition needs the dual (a design finding, independent of any
measurement), and the transport is causally active on the level-1 assignment (8.25% of the
assignments change).

**May NOT be concluded**: that the transported frame is worse for the quantisation. The probe's own
isometry check failed (0.0679 against a tolerance of ~1e-8), so the transport implementation in
the probe is not yet the verified one; the earlier sandbox measured the same operator's isometry
deviation at 8.7e-09 with a different convention, and the probe uses a third one. Until the probe
runs the VERIFIER's transport (`tools/hyperbolic_verifier/`), the error comparison above is not
evidence.

**Next**: re-run this probe with the verifier's parallel transport (the one whose isometry and
roundtrip are regression-tested), and only then decide whether the path-B family is worth a
quantiser rewrite. The horosphere family, by contrast, is closed: it was refuted twice with
verified measurements (a near-radial operator, and a level-matching constraint that fights the
host's assignment).
