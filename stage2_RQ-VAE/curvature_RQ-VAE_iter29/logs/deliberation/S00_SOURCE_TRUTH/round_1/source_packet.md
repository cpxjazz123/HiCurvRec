# S00 Source Packet — iter29

STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
OBJECTIVE=Independently extract controlling constraints, primary repository facts, and unresolved source conflicts for a new iteration. Do not select or implement a mechanism.

## Controlling sources to inspect directly
- `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md` — project rules; especially §§0–13.
- `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/SKILL.md` — current research workflow and contract.
- `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/references/mechanism_pool.md` — current active/deferred mechanism scope, subordinate to SKILL.md.
- `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/references/failed_mechanism_ledger.md` — historical mechanism evidence only.

## Primary experiment evidence to verify
- Iter26 contract: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/mechanism_contract_iter26.json`.
- Iter26 protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md`.
- Iter26 exact hypothesis and attribution: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/hypothesis_iter26.md`, `failure_attribution_iter26.md`.
- Iter26 exact Stage3 result: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`.
- Iter18 baseline result: `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`.
- Iter28 outcome (historical evidence; verify contract/protocol compatibility before comparison): `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/failure_attribution_iter28.md` and `results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/test_final.json`.

## Current repo observation supplied for verification
- Local branch `main`; local HEAD was `20e1fab8b067dd329ddba54ab1795815b2911ddb`; `origin` was `https://github.com/cpxjazz123/HiCurvRec.git`; remote main matched at last verified closure.
- The earlier untracked iter29 draft was created before deliberation and proposed a curvature-conditioned behavior-loss schedule. That proposal conflicts with FCCR-1 and is not canonical; its copied stale iter28 logs were removed. No iter29 GPU process was found in the process snapshot.
- Existing unrelated untracked paths must be preserved and excluded from any iter29 commit.

## User objective
Autonomously iterate on curvature mechanism experiments until `test_recall@10 >= 0.065`, without asking for a mechanism choice. Each experiment still requires a falsifiable one-factor registration, the active research contract, the mandated independent A/B/Judge workflow, and project artifact/Git closure. Do not promise the metric in advance.

## Candidate task
Write a concise, evidence-cited source-of-truth extraction for iter29. Identify controlling contract, project execution/Git constraints, the status and comparability limits of prior outcomes, unresolved conflicts, and mandatory next action. Treat lower-priority historical recommendations according to the current source hierarchy. Do not design a new mechanism, alter repository files outside this candidate artifact, or read another worker's candidate.
