# Tables computed on the unaudited 182-pair table

Every file here was produced from `sid_recall.csv` **before** the pair audit of
2026-11-02, when that file still held 182 (SID, recall) rows. Only 73 of those
rows pass the two admission criteria of repo rule 64b (same introduction commit
and the same stage-2/stage-3 directory name; see
`analysis_sid_recall/PAIR_AUDIT_FINDINGS.md`).

They are archived, not deleted, because they are the record of the earlier
analyses (`CORRELATION.md` is their write-up). They must not be quoted as
current results: the current `sid_recall.csv` has the 73 verified pairs, and the
scripts in the parent directory regenerate every table from it.

The curvature measurements in `../curvature/` are different: they are
per-SID measurements (not pairings), and `AUDIT_CORRELATION.md` reports the
statistics on both the verified subset and the full table, as the rule requires.
