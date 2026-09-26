ROLE=CANONICAL_SOURCE_PACKET
STAGE_ID=S13_GIT_CLOSURE
ROUND=2

# Trigger
S13 round 1 Judge accepted the iter29 commit scope as ready conditional on the closure gate and a fresh precommit check. The required closure-mode `deliberation_gate.py` then failed before Git staging/commit/push at S11:

`DELIBERATION_GATE_FAIL: .../S11_STAGE3_EVALUATION/round_2/judge.md does not declare canonical artifact stage3_outcome_iter29.md`

A separate first gate attempt failed earlier because S11 round-2 candidate `SOURCE_PACKET` headers used the `/home/...` symlink alias while the checker resolved the repository cwd to `/fs04/...`. Both candidate headers were normalized to the project-relative packet path; candidate analysis and S11 verdict content were not changed. The subsequent failure above is the current blocker.

# Primary source facts
- Skill `curvature-rqvae-iter` §2.7 S11 row lists `stage3_evaluation_plan_iter<N>.md` as the S11 canonical output; §2.7 S12 row lists `failure_attribution_iter<N>.md` and `gate_decision_iter<N>.md`. §17 requires `stage3_outcome_iter<N>.md` after Stage3 and the S10–S13 deliberation records.
- Current `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py` `POST_STAGE2` list at lines 35–45 instead requires both `stage3_evaluation_plan_iter<N>.md` and `stage3_outcome_iter<N>.md` under S11, while S12 list requires only failure attribution and gate decision. `check_stage` requires every canonical filename to be declared in that stage's `judge.md`.
- S11 round 2 Judge `logs/deliberation/S11_STAGE3_EVALUATION/round_2/judge.md` line 17 declares only the corrected Stage3 plan plus itself; it predates the authorized Stage3 run and did not adjudicate a completed test record. Its ACCEPT_A authorization and one-run boundary remain valid.
- The already-existing `logs/stage3_outcome_iter29.md` is an exact post-run outcome record with path, all Stage3 metrics, direct iter26 comparison, target result, and validity caveats. It is explicitly canonical in `logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/judge.md`, which adjudicated the output from primary test/metrics/checkpoint records.
- S12 round-1 A/B/Judge deliberation is complete; its Judge lists the outcome, failure attribution and gate decision as its canonical artifacts. No Stage3 retry/restart is allowed or needed. The current `logs/git_closure_iter29.md` records the approved scope and does not yet record the gate failure.

# Decision required
Independently decide the smallest truthful, contract-compliant resolution that lets the required closure checker validate this completed run without falsifying authorship, changing Stage3 outcomes, bypassing the checker, modifying the shared skill script, or exceeding two adjudication rounds per stage. S13 is now in round 2 (maximum). No Git staging/commit/push has occurred. Preserve the S12 scientific classification and S11 one-run authorization. If a narrow metadata/index correction to S11's canonical artifact listing is recommended, state exactly how to record that `stage3_outcome` was materialized after the run and separately adjudicated by S12; it must not claim S11 prelaunch candidates or Judge analyzed the future output. If a different remedy is better, explain its evidence and exact scope. No user input.

# Git closure facts unchanged
- The round-1 S13 audit verified complete required iter29 source + Stage2 + Stage3 artifacts, current Git branch `main`, upstream `origin/main`, only remote `origin https://github.com/cpxjazz123/HiCurvRec.git`, and precommit local/remote hash `612a5a41dfe524205b6afa46370ad0d0ce377882`.
- Required output exceptions exist in `.gitignore`; required files are unignored; iter29 source tree contains no Stage2 output `.pth/.npy`; unrelated root scripts and prior iteration logs must remain unstaged.
- Stage2 SID path correction: actual `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy` (not `out/sids_for_hgrec.npy`); Stage3 result file hashes confirmed in `logs/git_closure_iter29.md`. The Stage3 metric hash is `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`; both first-round S13 candidates had copied a different hash, and round-1 Judge documented that candidate-only transcription correction.

# Candidate contract
Agent A and B independently analyze the same packet, current gate failure, skill and exact S11/S12 records. They must state the minimal fix, whether closure remains safe, affected paths, risks, assumptions and rejection conditions. Write only `logs/deliberation/S13_GIT_CLOSURE/round_2/agent_a.md` or `agent_b.md`. Do not edit or stage files, do not run the closure checker as a workaround, and do not commit/push. No tests/formatters/linters.