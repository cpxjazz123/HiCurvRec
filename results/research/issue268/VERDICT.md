# Issue #268 verdict

## Headline

The frozen per-triple continuous curvature is **not identifiable** under the
frozen reference representation. A per-triple κ reduces the graph-distance
distortion by 9.6 % relative to κ=0, but 99.78 % of the 28,391,948 fitted values
sit on the coordinate-box boundary and one global κ reproduces the per-triple
result (0.18 % of rows differ; the full-reference audit differs on 0.0 %). A
single **global Euclidean scale** — the weakest possible ruler change — reaches
J `0.174575`, better than the entire per-triple family (`0.253693`). The
mechanism was therefore stopped at the issue's own pre-training gate; no Stage2
or Stage3 was launched.

## Four required questions

1. **Does continuous κ reduce real graph-distance distortion under the fixed
   reference?** Only as a boundary/global effect. At the frozen scale the fit
   lowers J from `0.268071` to `0.242248` (median 9.57 %), but 99.78 % of rows
   choose the floor `κ=-0.125`, κ std is `0.0101`, and one global κ is
   indistinguishable. The global-scale control reaches `0.174575`, i.e. 45 %
   below the per-triple family. Triplet-specific curvature is not supported.
2. **Does fixed κ improve independent geometry fidelity and hard SID?** The
   independent fidelity gain is only the global-scale component; the hard-SID
   question has no experiment because the gate stopped before training.
3. **Does recommendation improve?** Not tested. No Stage2 candidate, no SID, no
   Stage3 result; nothing is claimed.
4. **Offline and online cost.** Full fit: `111.04 s` on one A100 for 28.4 M rows
   (201-point grid, exact node decomposition), cache `341 MB` (κ, J0, Jfit) or
   `114 MB` for κ alone. A Stage2 cache lookup would be one host gather per
   sampled row plus a `3×32×32` distance block per update; no training run
   exists to quote an inclusive wall time, so the 0.25 s/epoch target is not
   evaluated.

## Acceptance table

| ID | Issue requirement | Evidence | Verdict |
|---|---|---|---|
| C1 | Inherit #260-267 hypotheses; test gate-only vs curvature-information vs reference explanations | `HYPOTHESES.json` (17 entries, inherited from #267); `DIAGNOSIS.md` §2-§3 | PASS |
| C2 | Approve training graph, triples, references, shortest-path, mask; keep target definition | graph sha256 `097a719a…`; targets `8c0ff9f8…`; 28,391,948 rows; `CONTRACT.md` §1 | PASS |
| C3 | Keep κ_graph, κ_t, κ_hat separate; never use κ_t as the target | `CONTRACT.md` §1; quota check in `fit_kappa.py` (target table read-only) | PASS |
| C4 | Freeze Z_ref, global scale, coordinate map, κ domain, zero limit, references, budget | `artifacts/z_ref_manifest.json`; `CONTRACT.md` §2-§4 | PASS |
| C5 | J_t with Bachmann κ-stereographic, bounded 1-D search, κ=0 control, boundary/flat report | `CONTRACT.md` §4; `kappa_cache/manifest.json`; exact decomposition error `3.1e-8` | PASS |
| C6 | Full shard-processing fit with κ_t, row IDs, references, both distortions, dtype, coverage, hashes, runtime; subset disclosed and full-audited | `kappa_cache/` (3 arrays + manifest); `analysis/TRANSFER_VALIDATION.json` full-reference audit | PASS |
| C7 | Independent population: improvement, κ distribution/strata, stability, sensitivity; do not default to training if no fit | `analysis/TRANSFER_VALIDATION.json`, `analysis/CACHE_VERIFICATION.json`, `analysis/SCALE_CONTROL.json`, `DIAGNOSIS.md` §3 | PASS |
| C8 | Immutable cache, row identity, hash, legal values, same-source recomputation; no refit | `analysis/CACHE_VERIFICATION.json` (j0 err `5.5e-8`, jfit err `3.9e-8`, off-grid `1.4e-8`); cache hashes frozen | PASS |
| C9 | Stage2 mechanism: Euclidean selection, gate removed, per-triple κ distance, λ budget | Not implemented/trained: the mechanism is ineligible after C7 | **NOT APPLICABLE** (issue gate) |
| C10 | Gradient route: encoder/codebooks via STE, κ/reference/assets frozen | Same: no mechanism was trained; the frozen tests were not run | **NOT APPLICABLE** (issue gate) |
| C11 | TDD, reference, hand geometry, boundaries, mutation, index, gradient tests | Same: implementation was stopped at the gate | **NOT APPLICABLE** (issue gate) |
| C12 | Pre-train real-batch probe on the frozen parent checkpoint | Same: no candidate to probe | **NOT APPLICABLE** (issue gate) |
| C13 | One Stage2 3000-epoch four-card run, collision, export, performance | Not run: the mechanism is ineligible and the host currently exposes one A100-80GB; the frozen launcher refuses a single-card formal Stage2 | **NOT APPLICABLE** |
| C14 | Fixed Euclidean / equal-weight three-expert evaluation plus SID/reconstruction/prefix fidelity; own-κ diagnostics | Distance and transfer side: `analysis/TRANSFER_VALIDATION.json`, `SCALE_CONTROL.json`, `probe_J*.json`. SID side: no candidate | **PARTIAL** (distance side PASS; SID side NOT APPLICABLE) |
| C15 | One frozen Stage3 150-epoch run, §63 record, independent Recall/NDCG recompute | Not run: no Stage2 SID exists | **NOT APPLICABLE** |
| C16 | Deliver diagnostics, cache/manifest, code/tests, logs, reports, R2, VERDICT, push | Documents, cache, scripts, receipts, tests of the offline pipeline; no R2 because there is no checkpoint/SID/recall four-piece set | **PARTIAL** (all applicable deliverables PASS; R2/Stage2 source not applicable) |
| C17 | Answer the four questions explicitly | This file, §"Four required questions" | PASS |
| C18 | Record negative results, do not fake unrun items, do not open a next round or edit infra | `DIAGNOSIS.md` §5-§7; `RUN_LOG.md`; no infra or asset change | PASS |
| C19 | Disclose existing test-access risk | `DIAGNOSIS.md` §6 | PASS |

## Closure checks (skill §6.4)

1. **All production code implemented?** No. The candidate was stopped at the
   issue's pre-training geometry gate, before the mechanism was completed. This
   is recorded as NOT APPLICABLE, not as a pass.
2. **All necessary tests executed?** The applicable tests are the offline fit
   and validation scripts, the cache verification and the independent transfer
   audit; all executed with persisted logs and exit code 0. The mechanism tests
   were not run because no mechanism exists.
3. **All required training completed?** No: Stage2/Stage3 not run, for the reason
   in C13/C15. No run, checkpoint or metric is claimed.
4. **Infra boundary respected?** Yes: `FROZEN`; no `tools/`, `assets/`, launcher,
   admission, evaluator or archive change; only candidate-local research files
   and a partial candidate tree were written (the latter is archived, not run).
5. **Any executable task left?** Under the issue's own gate, no: the geometry
   adaptation is disproven; implementing and training the mechanism would be a
   known-empty exercise and is forbidden by the issue's "不默认正式训练".

## Knowledge increment

- The gate never could move hard q (#265's selection is Euclidean); its only
  channel was the curvature objective's scale.
- A frozen semantic encoder's Euclidean geometry is globally mis-scaled against
  graph hops; one scalar is worth far more than a per-triple metric table.
- Per-triple curvature fitted on a graph-distance distortion saturates the
  coordinate box (99.78 % at the floor) and is not distinguishable from one
  global curvature; future geometry work must fix the global calibration or the
  representation, not add a per-triple parameter table.
- The reusable artifacts are `Z_ref`, the full 28.4 M-row κ cache with both
  distortions, the fit/validation scripts and the negative evidence.
