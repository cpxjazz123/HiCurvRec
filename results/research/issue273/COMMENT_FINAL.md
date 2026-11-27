### Round 1 result (owner-directed close, 2026-11-28)

The candidate `TIGER_samecentre_delta_rq` (same-centre difference protection `L_delta`, one primary
change on top of the #269 fixed local supervision) was dispatched as
`stage2_3d8bc9a4ef5340e690b25efa20ad2c8d` and **completed the full 3000-epoch budget** (seed 42, the
single visible GPU, 1303.7 s, 0.45 s per epoch). The machine admission before the dispatch passed
(`entry`: one epoch runs and exports the identifier, levels 256/256/256, collision 0.048278;
`behavior_graph`: NOT_APPLICABLE). The candidate suite passes: 138 tests.

**Measured criteria**

| criterion | value | threshold | verdict |
|---|---|---|---|
| C12 L2 distinct prefixes | 5,966 | >= 10,918 | **FAIL** |
| C15 reconstruction | 9.792938e-05 | <= 8.589e-05 | **FAIL** |
| C1, C2, C3, C4, C5, C6, C7, C9, C10, C11, C13 | NOT_RUN | -- | the owner stopped the evaluation |
| collision (recorded only) | 0.141904 (three codes); 0.0 for the full four-token SID | -- | recorded |
| L3 distinct triples | 21,098 | -- | recorded |

The frozen evaluation (`results/research/issue273/evaluate_candidate.py`: the common-ruler J from the
candidate's own `model/global_constraint.py`, the local and within-centre statistics from
`within_centre_metrics.py`, the SID structure from the frozen #271 collector) was stopped by the
owner before it produced the table. The two criteria above were recomputed directly from the
exported SID and the final training record — nothing is estimated or copied from another arm
(`results/research/issue273/round1_sid_structure.json`).

**Calibration (frozen on the real first batch, 4096 items)**: `lambda_point` 5.883528956e-06
(inherited from #269), `lambda_delta` 5.955654956e-07, the realised global gradient share
**0.013784** (cap 0.02), the pointwise/delta gradient cosine **0.104792** — the two terms do not
fight each other.

**Conclusion: acceptance not met; this is not a PASS.** The round establishes that the mechanism is
implementable and budgeted as designed, and that it **costs** prefix capacity (L2 5,966 against the
#269's 6,848 and the threshold 10,918) and reconstruction accuracy (9.7929e-05 against the 8.589e-05
threshold the #269 met). The #273-specific claim — that the signed within-centre differences (C6)
are the statistic the #272 round-3 left missing — **remains untested**, because C6/C7 are exactly
what the terminated evaluation would have produced; the frozen artefacts needed for that single
measurement are still in place.

Artefacts: `results/stage2_RQ-VAE/_runs/stage2_3d8bc9a4ef5340e690b25efa20ad2c8d/TIGER_samecentre_delta_rq/result/`
(`item_sids.json` sha256 `f34bde31…`, `rqvae_best.pth` sha256 `a7f17036…`, the metrics stream); the
verdict is `results/research/issue273/VERDICT.md`; the tree is archived to
`stage2_RQ-VAE/_archive/TIGER_samecentre_delta_rq`. Stage3 was forbidden by this issue in every
outcome and was not run. The CLAUDE.md §63 layered metric record is marked **missing** (the run
wrote no `archive_binding.json` next to the SID); it is not filled with zeros.

One engineering fault was found and fixed on the way: the first dispatch (`stage2_98834b02…`) was
refused by the smoke because the delta calibration ran under `torch.no_grad()`, so the loss had no
graph; the fix is covered by a regression test.
