# Iter30 S00 Source Packet — Round 1

STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
ITERATION=30

## Objective

Independently extract controlling rules, verified current facts, and unresolved constraints for the next autonomous research iteration. Do not select or alter a mechanism, create production code, launch Stage2/Stage3, or infer current facts from historical status text. The only project-level success target is the user's explicit inclusive `test_recall@10 >= 0.065`; root/skill documents contain lower-priority strict `> 0.065` wording, which remains a documented conflict but does not override the user.

## Controlling sources

1. Current repository `CLAUDE.md` (re-read directly): FCCR-1-related operational requirements; no CLI arguments or environment overrides for project scripts; no worktrees; all work on `main`; sole permitted remote is `https://github.com/cpxjazz123/HiCurvRec.git`; Stage2 outputs only in `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/`; Stage3 outputs only in `results/stage3_T5Train/curvature_RQ-VAE_iter<N>/`; Stage2 gradient-path check before every actual Stage2 run; full Stage2 candidates proceed to Stage3 without SID-quality hard gates; each completed iteration's code and Stage2/Stage3 results must be pushed and local/remote main hashes compared.
2. Active `curvature-rqvae-iter` skill: independent A/B plus Judge C at every stage; only judge-approved canonical artifacts propagate; FCCR-1 remains fixed, closed-form, non-trainable and time-invariant curvature; no mechanism stacking or Stage1/Stage3 change; abort only for directly evidenced infeasibility; never await user direction.
3. Canonical iter29 S14 Judge and `logs/global_review_after_iter29.md`: S14 passed the GLOBAL_REVIEW gate, is synchronized to GitHub before iter30, preserves FCCR-1, and authorizes a finite prospective three-pair matched-seed replication of the two previously tested FCCR-1 mappings. It does not promise target attainment or establish statistical power.
4. Prior iter26 and iter29 protocol manifests, mechanism contracts/hypotheses, and exact primary Stage3 `test_final.json` files cited below.

## Current verified closure state

- Iter29 S13 code/results closure was pushed as `57b4a594d92435fd74fa4bba7c7c1439a33eb102`.
- Iter29 S14 artifacts were pushed as `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6`.
- Final iter29 closure-log amendment was pushed as `ecd01e4712a1badd38a0338255f4b2ec7b030aff`; direct post-push `git rev-parse main` and GitHub-only `git ls-remote origin refs/heads/main` matched this hash. Current branch was `main`, upstream `origin/main`, and `origin` pointed only to the permitted GitHub repository. Existing unrelated untracked user paths remain unstaged and must remain untouched.
- The required iter29 deliberation gate returned `DELIBERATION_GATE_PASS`, `iter=29`, `phase=GLOBAL_REVIEW` after S14 artifacts existed. Iter30 must still pass its own staged gates before any Stage2/Stage3 run.

## Protocol-compatible observations

- Iter26 exact primary result: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`; `n_eval=57439`, `test_recall@10=0.057017009349048554`.
- Iter29 exact primary result: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`; `n_eval=57439`, `test_recall@10=0.05921064085377531`.
- Iter29 S01 names iter26 as its sole direct control and marks iter18 `HISTORICAL_NONCOMPARABLE`. Do not rank iter29 against iter18. S14 classifies iter29 `ACTIVE_NEUTRAL + PROMOTION_FAIL`; its small single-run point delta does not establish causal benefit or FCCR-1 family failure.

## S14-authorized replication facts

- Compare only the two already registered fixed FCCR-1 mappings. Within each pair, use the same Stage2 seed and same Stage3 seed in both arms. Predeclared pairs: `43`, `44`, `45`; thus six complete Stage2/Stage3 pipelines, subject to all prelaunch hard gates.
- Iter26 mapping: `s_l=log1p(B_l)/log1p(m_l_raw/min(m_raw))`; `z_l=(s_l-mean(s))/(std(s)+1e-12)`; `c_l=clip(0.5*exp(0.2*z_l), 0.05, 1.5)`. Registered fixed values: `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`.
- Iter29 mapping: `c_l=0.05+1.45*0.5*(B_l/(B_l+2.0)+m_l_raw/(m_l_raw+0.1))`. Registered fixed values: `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.
- S14 specifies shared formula input values in L0/L1/L2 order: `B=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`; historical `raw_residual_medians=[1.0, 0.10941, 0.09331]`. S03 previously accepted only medium-confidence historical method/value provenance, not checkpoint-level reproduction or byte identity. Iter30 must independently re-establish raw-residual provenance; never substitute normalized layer scales.
- Within each seed pair hold Stage1 embedding and item identities/order, Stage0 splits and ordering, immutable Stage2 warm-start checkpoint, all Stage2 settings except the paired seed and mapping, Stage3 code/configuration and runtime settings, evaluation set, and metric computation identical. The shared warm-start recorded in iter29 S01 is `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`; S01 must reverify it and all data/Stage1 identities against current files and actual consumers.
- Seed-42 historical observations may be included in pooled analysis only if S01 establishes the exact necessary protocol and identity compatibility. They are not one of the three newly predeclared pairs and may not replace a failed new pair.

## Unresolved source/implementation constraints for later stages

- S01 must inspect current primary code/configuration to establish how Stage2 and Stage3 seeds are set, how a six-run block can be executed without CLI/environment overrides, and how six distinct outputs can be retained without collisions beneath the mandated short roots `curvature_RQ-VAE_iter30`.
- Stage2 results must remain beneath `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/`; Stage3 logs/checkpoints beneath `results/stage3_T5Train/curvature_RQ-VAE_iter30/`. Iter30's source subtree may contain code, configs, scripts, logs, and input copies only; no generated `.pth`, `.npy`, or `item_sids.json` there. Preserve the full `MECHANISM_NAME` naming contract while keeping output roots short. No project CLI args or environment override may select a run.
- S01 must record a protocol-compatible baseline/comparison plan, exact identity hashes, seed pairs, code/config identity, Stage2/Stage3 settings, and a prospective analysis/target rule. A declaration is not proof of runtime consumption.
- No Stage2/Stage3 run is authorized by this S00 packet. Before actual Stage2 training, complete S00–S09 A/B/Judge gates, FCCR-1 preflight, contract-aware MVG including the required model gradient-path checks, the deliberation gate, output-path checks, and the one-checkpoint/one-batch gradient verification required by `CLAUDE.md`.

## Candidate instructions

Agent A and Agent B must independently produce a source-of-truth snapshot from this same packet and cited primary repository evidence. Separate verified facts from unresolved claims; preserve source precedence and the threshold conflict resolution; identify any inconsistency or missing prerequisite that affects iter30. Do not read the other candidate. Do not write the canonical `logs/source_snapshot_iter30.md`; only the Judge-approved result may be materialized there. No Git, code, training, or output-path mutation in this stage.
