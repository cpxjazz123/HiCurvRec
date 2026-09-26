# S14 Global Review Source Packet — after iter28

STAGE_ID=S14_GLOBAL_REVIEW
ROUND=1
OBJECTIVE=Review protocol-valid experimental evidence after iter28, separate current-contract evidence from historical mechanisms, assess the recurring Stage3 ceiling and unresolved hypotheses, then recommend a concrete next research action autonomously.

## Controlling sources
- `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md` — current project constraints and Stage3 target.
- `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/SKILL.md` — source hierarchy, FCCR-1 contract, protocol rules, and Global Review trigger/output requirements (§§1, 3–5, 15, 17–19).
- `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/references/mechanism_pool.md` — current FCCR-1 active/deferred family scope, subordinate to the skill.

## Primary iteration records
- Iter25: `stage2_RQ-VAE/curvature_RQ-VAE_iter25/logs/failure_attribution_iter25.md`, `gate_decision_iter25.md`, `hypothesis_iter25.md`; exact Stage3 result `results/stage3_T5Train/curvature_RQ-VAE_iter25/logs/Amazon_2023_Instruments/Sep-26-2026_16-34-15/test_final.json`.
- Iter26: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/mechanism_contract_iter26.json`, `protocol_manifest_iter26.md`, `hypothesis_iter26.md`, `failure_attribution_iter26.md`; exact result `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`.
- Iter28: `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/mechanism_contract_iter28.json`, `protocol_manifest_iter28.md`, `failure_attribution_iter28.md`, `stage3_outcome_iter28.md`; exact result `results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/test_final.json`.
- Canonical baseline iter18 exact result: `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`.
- Stage3 code at iter26 declared commit `0052f4bff117e0aa16c9a1080be1014fbd417cc6`; iter28 manifest declares `bdcbbf97bd92ef6b5f0ab016cdfbae2b62c5d73f`. Verify source equality directly; `git diff --stat` showed no change to `stage3_T5Train/train_HG-Rec.py` across those revisions, but a complete protocol review must inspect the rest of the shared protocol claims.

## Review constraints
- The Skill requires Global Review after 3 clean protocol-valid iterations or recurring neutral/negative mechanism-family results; iter27 was aborted and must not count.
- Use only protocol-compatible metrics for direct ranking. If protocol compatibility is uncertain, label the comparison `HISTORICAL_NONCOMPARABLE`; same `n_eval` alone is insufficient.
- Separate mechanism status from promotion status. Do not use Stage2 proxies as promotion gates.
- FCCR-1 remains active unless changed by a higher-priority instruction or canonically adjudicated between-iteration transition. Historical iter25/26/28 recommendations do not override the skill.
- User's objective is `test_recall@10 >= 0.065`. The repo/skill text uses strict `>0.065`; the current user instruction controls this task's success threshold. Do not guarantee success.

## Candidate task
Independently reconstruct the clean protocol-valid evidence, identify which hypotheses were tested vs confounded/invalid, state whether the global trigger fires, assess evidence for/against the current FCCR-1 fixed mapping without overgeneralizing, and recommend one concrete next action within current contract. Include assumptions, evidence, risks, and self-rejection conditions. Do not read another candidate or edit canonical/global artifacts.
