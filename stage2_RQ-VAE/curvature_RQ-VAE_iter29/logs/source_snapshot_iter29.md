# Source Snapshot — iter29

## Controlling contract and hierarchy

- Root `CLAUDE.md` governs project paths, no-CLI-override rule, Stage2 quality-gate policy, training launchers, mandatory pre-training gradient-path check, GitHub-only `main` closure, and iter-specific artifact paths (`CLAUDE.md` §§0–13).
- `.claude/skills/curvature-rqvae-iter/SKILL.md` is the active research workflow. Its source hierarchy is root rules → skill → machine-readable contract → protocol manifest → local audits → historical files (§1).
- The active research contract remains **FCCR-1**: precompute fixed per-layer curvature from behavior branching and raw residual magnitude; curvature is non-trainable and time-invariant. Learnable/cyclic curvature, curvature regularization, curvature-conditioned optimizer changes, and new curvature-conditioned auxiliary losses are not authorized (§§3.1–3.3).
- The pre-deliberation iter29 draft that proposed curvature-conditioned behavior-loss scheduling violated FCCR-1. It was not canonicalized and is discarded; no code, contract, or downstream instruction may propagate from that proposal.
- Every stage requires independent A/B work on one source packet followed by Judge C; only the judge-approved canonical artifact propagates. Stage2 and Stage3 full runs occur once after their audits (§§2, 17–19).

## Goal and execution constraints

- The user's success condition is **`test_recall@10 >= 0.065`**. The lower-priority project/skill wording uses strict `> 0.065`; the user's explicit threshold controls this task. No outcome is guaranteed.
- Stage2 descriptive SID metrics are not performance gates; a valid, non-aborted run proceeds to Stage3 (`CLAUDE.md` §2; Skill §10).
- Stage2 outputs belong under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/`; source iteration directories may contain code/configs/scripts/logs/input copies, not `.pth`, `.npy`, or item SID outputs (`CLAUDE.md` §§0, 10). Stage3 outputs belong under the matching `results/stage3_T5Train/curvature_RQ-VAE_iter<N>/` (`CLAUDE.md` §11).
- No project script CLI arguments, worktrees, or non-GitHub remotes. Actual Stage2 execution requires the specified one-checkpoint/one-batch gradient-path check first; normal closure requires code and Stage2/Stage3 artifacts committed, pushed to `origin/main`, and matching local/remote hashes (`CLAUDE.md` §§1, 6, 8–13).

## Primary evidence

- **Iter26 is the most recent directly registered FCCR-1 experiment.** Its machine contract specifies closed-form, fixed, non-trainable curvature with inputs `behavior_branching` and `raw_residual_median`, and values `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]` (`stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/mechanism_contract_iter26.json`). Its preregistered inputs were branching `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`, raw residual `[1.0, 0.10941, 0.09331]`, and mapping `s_l=log1p(B_l)/log1p(m_l/min(m))`, `c_l=clip(0.5*exp(0.2*z_l),0.05,1.5)` (`hypothesis_iter26.md`).
- Iter26's exact Stage3 result is `R@10=0.057017009349048554`, `n_eval=57439` (`results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`). Its attribution reports contract/invariance and activation checks passed (`failure_attribution_iter26.md:18–22`). This is evidence about this registered mapping, not proof that every allowed FCCR-1 mapping fails.
- Iter26's protocol manifest declares iter18 as canonical baseline and directly rankable; exact iter18 `test_final.json` reports `R@10=0.05988962203380978`, `n_eval=57439` (`results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`). S01 must still verify protocol compatibility from primary records before any new direct ranking; do not rely only on equal `n_eval`.
- Iter25 reports `R@10=0.05882762582914048`, `n_eval=57439`, with a learnable layer-scale prior (`failure_attribution_iter25.md`; exact result at `results/stage3_T5Train/curvature_RQ-VAE_iter25/logs/Amazon_2023_Instruments/Sep-26-2026_16-34-15/test_final.json`). It is historical evidence, not an FCCR-1 fixed-curvature result.
- Iter28's contract is CAO-1 with cyclic learnable curvature plus SREMA codebook updates, not FCCR-1 (`mechanism_contract_iter28.json`; `protocol_manifest_iter28.md`). Its exact `R@10=0.05649471613363742`, `n_eval=57439` (`results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/test_final.json`) is a distinct-mechanism result, not evidence against FCCR-1. Historical recommendations in iter25/26/28 do not override the current contract.

## Review trigger and unresolved items

- The mandatory S14 Global Review was warranted by a discovered protocol/baseline inconsistency: iter25 has no protocol manifest at the registered path. The formal three-clean-protocol-valid-iteration trigger is not met: iter25 lacks that manifest and used learnable/cyclic curvature; iter26 is the only direct FCCR-1 fixed-curvature run; iter27 was aborted; iter28 used CAO-1/SREMA with cyclic, learnable curvature. The S14 Judge decision is recorded in iter28 `logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md`; finish with the canonical `logs/global_review_after_iter28.md` before choosing an iter29 mechanism.
- Exact iter18 protocol artifacts are incomplete in the iteration tree; iter26 declares the result comparable. S01 must independently verify datasets, Stage1 identity, Stage2/Stage3 settings, and source compatibility; mark comparisons `HISTORICAL_NONCOMPARABLE` if uncertainty remains.
- Global Git/remote/process claims in the shared packet were not treated as current facts. Recheck repository/remote state before any commit, push, or launch decision.

## Canonical next action

Complete the triggered S14 Global Review with independent A/B candidates and Judge C; then complete S01 protocol lock. Keep FCCR-1 unchanged unless a separately adjudicated between-iteration contract transition is required. Do not launch Stage2 before canonical S00–S09 artifacts, contract preflight, deliberation gate, and mandatory gradient-path check all pass.
