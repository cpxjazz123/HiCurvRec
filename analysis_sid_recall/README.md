# analysis_sid_recall — one script, every structure-retention method

`analysis.py` is the **only** analysis script in this directory, and `results/` is where every artifact
it writes goes (`results/<command>.json`, or `--out <path>`). Every method that used to live in its own
file now sits behind one entry point, sharing one definition of structure.

    analysis_sid_recall/
    ├── analysis.py          the script
    ├── README.md            this file
    ├── results/             every JSON/Markdown artifact, historical and regenerated
    ├── r2/                  the archived SID+recall mirror (input: one numbered directory per arm)
    ├── structure_roles*.npz the frozen ORCA role caches (input)
    └── __pycache__/

    python3 analysis_sid_recall/analysis.py --help
    python3 analysis_sid_recall/analysis.py roles
    python3 analysis_sid_recall/analysis.py distance --arms 67
        python3 analysis_sid_recall/analysis.py patterns --representations
    python3 analysis_sid_recall/analysis.py independence      # reads results/patterns.json, else results/role_pattern_gain.json
    python3 analysis_sid_recall/analysis.py k --k 3,4,5,6,8,10
    python3 analysis_sid_recall/analysis.py split --k 6,8,10 --role 2
    python3 analysis_sid_recall/analysis.py profile --role 2
    python3 analysis_sid_recall/analysis.py sid

## What "structure" means here, fixed once

* **Roles**: ORCA's 73 node orbits of graphlets up to five nodes on the TRAIN behaviour graph, grouped
  into six NMF roles (`structure_roles5.npz`, the 213 audit's fit). No role is named or mapped to a
  graphlet.
* **Patterns**: nine five-node shapes with their own distance conditions — K5, double triangle sharing a
  centre, K4 with a tail, K2,3 fan-out, P5 by centre and by hop count, C5, K1,4, and the T5 branch tree.
  Each instance is built strictly (all missing edges verified).
* **Nulls**: degree-bucket × popularity-bucket matched random groups (5 × 5), distinct members, five
  draws per instance, shared by every representation.
* **Aggregation**: a pattern's score reaches every role that owns one of its five nodes, but the average
  is taken **per item first** and then over the role's items, so a hub that sits in many graphlets counts
  once; a pattern enters a role's set only when at least 30 of the role's items touch it; weights inside a
  role are equal and were fixed before any recall was read.
* **Ties**: `psi(a,b)` scores a strict win 1, a tie 0.5, a loss 0; the discrete SID distance is never
  punished for a tie.

## Commands

| command | what it computes |
|---|---|
| `roles` | level-1/2/3 prefix retention per role on behaviour edges, and its correlation with recall |
| `distance` | where a role's behaviour neighbours sit in the SID ordering: win rate against one random item, and a score over 200 candidates (0.5 = chance) |
| `patterns` | per-pattern SID structure gain per arm, per-role aggregated rate and gain, and their correlations; `--representations` adds the z / q / SID comparison on TIGER, the only arm with a pre-quantization vector |
| `independence` | out-of-sample ridge on the six role rates: repeated 5-fold CV, marginal contribution of each role, permutation p-values, collinearity |
| `k` | the same rates under other NMF granularities, with marginals and how the cached Role 3 is cut |
| `split` | the cached Role 3's membership under each K |
| `profile` | graph invariants, the ORCA orbits that separate one role, and the patterns its items join |
| `sid` | arm-level SID quantities (codes, prefixes, triples, collision, gain bits) against recall |

Shared flags: `--arms N` (limit the archived pairs), `--instances N`, `--null-draws N`,
`--permutations N`, `--repeats N`, `--edges N`, `--k 3,4,6`, `--role R`, `--seed 42`, `--in`, `--out`.
The limits exist so a smoke run is cheap; every table prints to stdout and `--out` writes the JSON.

## Data and provenance

The pairs are the archived mirror `r2/<NN>/` (a Stage2 `item_sids.json` next to the Stage3 `recall.json`
produced from it). Arm identity and per-file sha256 live in
`results/research/issue139/r2_ledger.json`; every arm is verified against it before use, and `r2/INDEX.md`
lists what each number holds. Structure itself is computed on
`results/research/issue215/graphs/behaviour.in` with the role cache and ORCA table above, so it is
identical for every arm.

## JSON artifacts

Each file here is the record produced by the command in the bracket; the historical records are kept
as they were and are not regenerated.

All rows are under `results/`. | file (in `results/`) | command |
|---|---|
| `report.json`, `report.md`, `rank_arm_comparison.json`, `ranking_auc.json`, `sid_retention.json` | `sid` (the bootstrap-interval detail of the old `analyze.py` is in git history) |
| `structure_retention.json`, `structure_retention_5node.json`, `STRUCTURE_RETENTION*.md` | `roles` |
| `role_distance_vs_recall.json`, `role_retention_table.json` | `distance` |
| `role_pattern_gain.json`, `role_pattern_gain_representations.json`, `orca5_pattern_gain_vs_recall.json`, `structure_vs_recall.json`, `orca5_role_preservation.json` | `patterns` (new runs write `patterns.json`) |
| `role_independence.json` | `independence` |
| `nmf_k_sensitivity.json`, `role3_split_by_k.json` | `k` / `split` |
| `role3_profile.json` | `profile` |
| `graph_shape.json`, `closure_regions.json`, `community_retention.json`, `structure_predictability.json`, `retention_ceiling.json` | graph-shape studies from earlier rounds; **not** regenerated by `analysis.py`, their code is in git history |

No metric here is an admission gate: Stage 3 `test_recall@10` stays the only verdict (CLAUDE.md §19/§25).
