ROLE=CANONICAL_SOURCE_PACKET
STAGE_ID=S13_GIT_CLOSURE
ROUND=1

# Goal
Independently audit required iter29 code, Stage2/Stage3 artifacts, ignore handling, Git branch/remote state, and intended single commit/push. Candidates must not stage, commit, push, or edit shared production files.

# Current result and S12 disposition
S12 canonical files are present: `logs/stage3_outcome_iter29.md`, `logs/failure_attribution_iter29.md`, and `logs/gate_decision_iter29.md`. Iter29 Stage3 completed once, supervisor exit 0, no restart; S12 classifies `ACTIVE_NEUTRAL + PROMOTION_FAIL`, R@10 `0.05921064085377531` vs iter26 control `0.057017009349048554`, observed delta `+0.002193631504726755`, and inclusive target shortfall `0.005789359146224693`. S14 is triggered for a direction review after S13, not as proof of family failure. See S12 primary records and `logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/judge.md`.

# Mandatory commit scope
1. Iter29 mechanism/audit tree: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/` including code, config, modules, scripts, configs, permitted input copies and audit logs/deliberation, excluding generated `__pycache__` and outputs (Stage2 products belong under `results/`).
2. Stage2 outputs under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`, at minimum:
   - `item_sids.json` (verified SHA256 `58665e08a97e122f47a77ed3616b67efdeb8eadba8253771a1498cb5e57487ff`)
   - `out/rqvae/instruments/rqvae_best.pth`
   - `out/rqvae/instruments/sids_raw.npy`
   - `out/sids_for_hgrec.npy` (verified SHA256 `a8efe6e1315166486aff162e1bc6a5722b5d5d9ac1b537b83bc003ec0ea7cde1`)
3. Stage3 shared launcher patch: `stage3_T5Train/train_HG-Rec.py` (the iter29-approved five-value child environment patch; code hash after final comment edit `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`).
4. Stage3 outputs under `results/stage3_T5Train/curvature_RQ-VAE_iter29/`:
   - `logs/_stage3_launcher.log` (SHA256 `fe9a49a8d7a4f46ffc0a91d20cf6a4768696bc54b778774714231f029f283806`)
   - timestamped log `logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/` containing `test_final.json` (SHA256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`), `training_metrics.jsonl` (SHA256 `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`) and `HG_Rec.log` (SHA256 `2527330613702fff9e18abe23902432f217720473cf890207cb8bae7bfc0773c`)
   - `ckpt/Amazon_2023_Instruments/Sep-27-2026_06-11-34/HG_Rec_best.pth` (SHA256 `70facbde7de7961bbe71e23be6f4f951f30e696b959d04be66cdf7d32dab7e94`)
5. `.gitignore` iter29 exceptions added in the final deliverable exception section so Stage2 `.pth`/`.npy`, Stage3 `ckpt`, launcher log and other required outputs are not ignored. Preserve existing excludes for transient `_ddp_sync`, `_stage3_run*.log`, `__pycache__`, and shared embedding.
6. S13 A/B/Judge artifacts and S12 documents are under the iter29 source tree and must be included.

The project rules in root `CLAUDE.md` §0, §10–13 govern these paths. Stage2 result folder is `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`; Stage3 result folder is `results/stage3_T5Train/curvature_RQ-VAE_iter29/`. Do not commit products inside the Stage2 mechanism source tree.

# Ignore check
Before adding the iter29 exceptions, `git check-ignore -v` showed global rules ignoring `.npy`, `.pth`, `ckpt`, and `results/**/_stage3_launcher.log`. The exact exception rules were added to `.gitignore`, and a second check of the required artifact paths resolved each through the iter29 negation whitelist; no critical listed artifact remained ignored. Candidate must independently verify current behavior and check any other required result artifacts. `git check-ignore -v` output of a `!pattern` is an unignore match, not an ignore failure.

# Git state before commit
Captured 2026-09-27:
- branch: `main`
- upstream: `origin/main`
- only remote: `origin https://github.com/cpxjazz123/HiCurvRec.git` (fetch/push)
- local `main`: `612a5a41dfe524205b6afa46370ad0d0ce377882`
- `git ls-remote origin refs/heads/main`: same `612a5a41dfe524205b6afa46370ad0d0ce377882`
- GitHub main was in sync before iter29 commit.
- `git status --short` showed intended changes: `.gitignore`, `stage3_T5Train/train_HG-Rec.py`, new iter29 source tree, and new iter29 Stage2/Stage3 results. It also showed unrelated user/untracked paths: root `scripts/` (contains `run_stage3_iter25.py`) and prior iter4/7/8/9/10/11/12/13 logs. These unrelated paths MUST remain unstaged and untouched.

# Commit contract
After independent A/B audit and Judge C approval, stage only `.gitignore`, `stage3_T5Train/train_HG-Rec.py`, `stage2_RQ-VAE/curvature_RQ-VAE_iter29/`, `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`, and `results/stage3_T5Train/curvature_RQ-VAE_iter29/`. Confirm `git status` and `git check-ignore -v` before commit. Use one semantic main-branch commit such as `iter29: bounded rational mapping + stage2/3 results`; push only via `git push origin main`, no force. After push, compare `git rev-parse main` with `git ls-remote origin refs/heads/main` and require exact equality. S14 review will occur after S13 and will need its own GitHub-main synchronization before iter30.

# Candidate requirements
Independently inspect exact repository state/artifacts. Record assumptions, evidence, risks, self-rejection conditions. Do not stage or mutate shared state. Write only your candidate file.