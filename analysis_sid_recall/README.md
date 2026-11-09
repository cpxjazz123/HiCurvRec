# analysis_sid_recall — one script, one report

`analyze.py` is the only analysis entrypoint in this directory. It reads the
mirror of the Cloudflare R2 archive (`r2/<NN>/`, one numbered directory per
archived mechanism) and writes `report.json` + `report.md` for every pair.

    python3 analysis_sid_recall/analyze.py            # analyse r2/<NN>/
    python3 analysis_sid_recall/analyze.py --download # refresh the mirror from r2:genrec
    python3 analysis_sid_recall/analyze.py --check    # verify the mirror against the ledger

Provenance of each number (arm identity, R@10, per-file sha256) is the
repository-side ledger `results/research/issue139/r2_ledger.json`; the mirror is
verified file-by-file against it. `r2/INDEX.md` lists what every number holds.

Per pair the report records, from the SID table alone: level-1 codes used,
effective number (`exp` of entropy, `l1_er`), largest level-1 share, distinct
level-2 prefixes, distinct level-3 triples, L3 collision (`1 - triples/items`),
L3 gain bits (`log2(items/triples)`), and prefix-overlap gaps (observed TRAIN
transitions minus random item pairs) at levels 1–3, next to the Stage 3 recall
from `recall.json`. Pairs archived before Stage 3 (`stage2_record.json`) are
listed without a recall and excluded from the correlations.

The cross-pair table is Spearman rho of each metric against `test_recall@10`
with a percentile bootstrap interval over pairs; n is printed with every row.
Twenty-odd arms from a handful of families are not a population — the intervals
say so, and no metric here is an admission gate (Stage 3 `test_recall@10` is the
only verdict, per CLAUDE.md §19/§25).

`graph_shape.py` -> `graph_shape.json` / `GRAPH_SHAPE.md`: the shape of the
TRAIN behaviour graph itself (tree-like branches next to dense communities).
