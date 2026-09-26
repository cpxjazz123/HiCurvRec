# S05 One-Factor Amendment Packet — iter29, Round 2

STAGE_ID=S05_ONE_FACTOR
ROUND=2

## Trigger and authority

Round 1's canonical `logs/one_factor_diff_iter29.md` lists deletion of `scripts/sid_metrics_any.py` only if unused. New direct source inspection shows it is used: `modules/sid_quality.py::evaluate_sid_quality_full()` writes a temporary SID file, launches the project-local script with positional path/label arguments, parses `HitRate@K=50`, and is called by `curvature_RQ-VAE.py` at SID reporting steps (`modules/sid_quality.py` lines 175–200; `curvature_RQ-VAE.py` lines 768–780). The project rule in the task context §1 prohibits CLI arguments for project scripts (only `/tmp` reference scripts may take SID-path/label positional arguments). The project rule §2 records HitRate@50 as removed because it does not read SID and is non-discriminating; Stage2 metrics are descriptive only, never gates. `SidMetrics` and `format_metrics` still carry/print HitRate, and `should_early_stop` still requires it finite despite the project rule that `should_early_stop` always returns `(False, "")`.

These facts invalidate only the round-1 conditional deletion assumption. The canonical experiment remains viable and unchanged. S05 must amend the source-path scope to remove the forbidden/no-value CLI measurement path safely, with a caller migration; do not silently delete an active utility or leave a project-local positional-argument script.

## Canonical science and locked scope remain unchanged

Use the exact mapping, values, parent/control, inputs, and provenance limits from canonical S00–S04 and S05 round 1. Parent and sole control are iter26; parent commit is `612a5a41dfe524205b6afa46370ad0dce377882`. The only scientific delta remains the S02/S04 mapping from the same explicit inputs to `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`. The inherited `effective_eps = sk_eps * (c / c_cyclic_max)` rule remains unchanged; its realized value can differ only through new c. No optimizer/loss/Sinkhorn rule/parameter, architecture, input, seed, step, warm-start, Stage1, Stage3 trainer, or evaluation-protocol change.

The raw residual key remains plural `raw_residual_medians`; implementation must fail if absent and never use `residual_norm`, normalized scales, or any fallback. S03 is historical method/value provenance only, not checkpoint replay or historic byte identity. Do not rerun the old behavior writer.

## Authorized Stage2 reporting migration

Add this minimal project-policy cleanup to the round-1 expected S06 implementation paths:

- `curvature_RQ-VAE.py`: replace the `evaluate_sid_quality_full(sids, EMB_NPY)` call/import with the existing in-process `evaluate_sid_quality(sids)`; continue printing the existing SID-derived descriptive metrics. SID generation, reporting cadence, model/training behavior, and final SID export stay unchanged.
- `modules/sid_quality.py`: remove the subprocess/CLI HitRate@50 path (`evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, related imports); remove the non-SID `hitrate_k50` metric field/placeholder/formatting/report field; retain the valid SID-derived descriptive values already computed by `evaluate_sid_quality()` (full/per-layer Gini, unique count, L0/L1 unique pairs, H(L1|L0)); make `should_early_stop()` return exactly `(False, "")` as root policy requires. Update stale comments/docstrings to state metrics are descriptive and no gate exists. Keep `write_quality_report` and unrelated existing API only if still used/appropriate; do not expand unrelated cleanup.
- Delete `scripts/sid_metrics_any.py` after migrating its only iter29 caller. It is an obsolete project-local CLI implementation for the non-SID-dependent HitRate metric; no `/tmp` script is moved or created.

Evidence from repository search finds the only iter29 `evaluate_sid_quality_full` caller is this iter29 training entry; other iteration trees are independent copies and are outside this iteration's scope. Existing `evaluate_sid_quality(sids)` computes the SID-derived metrics and the caller already catches metric-collection exceptions so these outputs do not control training. No gate is added. This reporting cleanup must not alter SID values or training; it removes an invalid redundant temporary-file subprocess and HitRate log field.

## Round-2 candidate task

Agent A and Agent B independently assess this new callsite evidence against root rules and the canonical one-factor experiment. Each writes only `logs/deliberation/S05_ONE_FACTOR/round_2/agent_a.md` or `agent_b.md`, begins with the exact role/independence/source/stage header, and includes evidence, one-factor boundary, risks, self-rejection, and autonomous next action. Judge C adjudicates and updates canonical `logs/one_factor_diff_iter29.md` plus `logs/deliberation/S05_ONE_FACTOR/round_2/judge.md`. The updated one-factor diff must enumerate the reporting caller migration/removal as policy/hygiene only while retaining exactly one scientific delta. No implementation source edits, tests/build/formatter, GPU, or training in S05.