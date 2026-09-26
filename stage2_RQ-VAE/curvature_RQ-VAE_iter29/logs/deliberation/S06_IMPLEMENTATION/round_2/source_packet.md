# S06 Implementation Amendment Packet — iter29, Round 2

STAGE_ID=S06_IMPLEMENTATION
ROUND=2

## Why round 2 is required

S05 round 2 updated canonical `logs/one_factor_diff_iter29.md` after direct inspection found that `curvature_RQ-VAE.py` calls `modules/sid_quality.py::evaluate_sid_quality_full()`, which launches project-local `scripts/sid_metrics_any.py` with positional arguments. Root project rules prohibit CLI arguments to project scripts; root §2 removes HR@50 as non-SID-dependent/nondiscriminating, requires all Stage2 metrics descriptive-only/no hard gates, and requires `should_early_stop()` to always return `(False, "")`. The canonical S05 amendment authorizes a minimal policy/hygiene migration and is binding. Round-1 S06 candidates/plans predate this amendment and are superseded for the current active scope; do not apply their instruction to leave `modules/` untouched.

## Canonical research constraints

Use canonical S00–S05 and this updated one-factor diff as sole authority. The sole scientific change remains replacing iter26's mapping with the fixed S02/S04 mapping, on the same `[L0,L1,L2]` input vectors:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

`x=B/(B+2.0)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, `c=0.05+1.45*u`. FCCR-1 remains precomputed/fixed/non-trainable/time-invariant. S03 is historical method/value provenance only, not checkpoint replay or historic byte identity. The consumer must use only explicit plural `raw_residual_medians`, fail closed if absent, and never use `residual_norm`, normalized scales, or fallback. Do not rerun the old behavior writer. Keep parent iter26 protocol/control, optimizer/loss, architecture, inputs, warm-start, seed/steps, inherited `effective_eps=sk_eps*(c/c_cyclic_max)` rule/parameters, Stage1, Stage3 trainer/settings/evaluation unchanged. Stage2 metrics are descriptive only and never gates.

## Complete round-2 implementation scope

Read round-1 S06 packet for the full source baseline, provenance paths and MVG contract. Apply only the S05-authorized additions below to that implementation plan:

1. `curvature_RQ-VAE.py`: migrate the import/call from `evaluate_sid_quality_full(sids, EMB_NPY)` to existing in-process `evaluate_sid_quality(sids)`. Preserve checkpoint cadence, SID generation/raw SID write, formatting of SID-derived descriptions, warning-only metric exception handling, training loop and final 4-token Stage3 export.
2. `modules/sid_quality.py`: remove the subprocess/temporary-file/parser HitRate path (`evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, and imports used only by that path). Remove `hitrate_k50` from `SidMetrics`, its NaN placeholder, validation, formatting and report serialization. Keep/report existing valid SID-derived metrics: full and per-layer Gini, unique SID count, item count, L0/L1 unique pairs and H(L1|L0). Correct stale HitRate/gate comments/docs to state descriptive-only/no gate. Make `should_early_stop()` return exactly `(False, "")` without metric validation/gate branches. Keep `write_quality_report` or other helper APIs only if still appropriate; do not broaden cleanup.
3. Delete iter29 `scripts/sid_metrics_any.py` only after the active caller/import migration, so no project-local argument-taking script remains in the active iter29 path. Do not move/copy a replacement to `/tmp` or change another iteration's files.

The rest of round-1 S06 scope remains in force: add the no-argument shared formula/JSON writer; preserve branching and explicit raw medians while splitting behavior and raw-residual provenance; recompute/validate loaded formula versus both data and S04 contract; retarget iter29 paths/label; add only the iter29 Stage3 wrapper; extend MVG with `loss.backward()`, fixed-buffer/optimizer exclusion/invariance, and the matched iter26 counterfactual; remove only authorized stale wrappers/calibrator/grad shell script if unused. Do not introduce a permanent test unless a genuinely uncertain edge merits one.

The reporting migration is `POLICY_HYGIENE_DIFFERENCES`, not another scientific factor. `evaluate_sid_quality(sids)` already computes the remaining SID-derived metrics in-process; the removed HitRate path does not alter SIDs, model, training values, stopping steps or outcomes. Do not add a metric gate or use descriptive metrics to select the candidate.

## Verification plan boundaries

Candidates propose checks only; no code edits/checks/training. After Judge approves a plan and orchestrator applies it once, use bounded verification: run the no-argument CPU formula writer and inspect exact JSON outputs/provenance; test missing plural raw key fails even with legacy alias in a throwaway payload; run official preflight and deliberation gate in authorized S07; run one-checkpoint/one-batch GPU MVG with explicit `loss.backward()` and counterfactual in S08; smoke the SID-only metric path on deterministic in-memory data or actual Stage2 log later, confirming SID-derived fields print and no HitRate field/call remains. No project-wide suite/build/formatter. Exact locked hashes and root-required `RQVAE_OUT_DIR` grep are required before later Stage2 launch. No Stage2/Stage3 launch in S06.

## Candidate task

Fresh independent A/B workers produce complete implementation plans and bounded verification plans using this packet, round-1 S06 packet, updated canonical S05 diff/Judge, and current source evidence. Each writes only `logs/deliberation/S06_IMPLEMENTATION/round_2/agent_a.md` or `agent_b.md`; do not read the other candidate. Judge C adjudicates and writes updated `logs/implementation_plan_iter29.md` plus `logs/deliberation/S06_IMPLEMENTATION/round_2/judge.md`. The orchestrator applies the resulting canonical patch once.