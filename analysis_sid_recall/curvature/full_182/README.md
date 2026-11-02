# The full-table curvature artefacts, frozen before the prune

These are the per-SID measurements and the report of the **unaudited 182-pair**
table. They are a snapshot taken on 2026-11-02, before `sid_recall.csv` was
pruned to its 73 verified pairs, and they are kept because repo rule 64b requires
the full-table numbers to be reported alongside the verified-subset ones.

The five files that stay one level up (`curvature/`) are the working copies:
`AUDIT_CORRELATION.md` reports both subsets, and a rerun of
`curvature_analysis.py` now reads the pruned table and writes the
verified-subset numbers there. Do not quote a number from this directory as a
current result.
