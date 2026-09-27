# Iter30 Source Snapshot — S00 canonical

## Governing authority and target

Apply the hierarchy in active skill §1: repository-root `CLAUDE.md` → active `curvature-rqvae-iter` skill → current canonically accepted iter30 mechanism contract → current protocol manifest → iteration-local records → historical material. Iter30 has not yet completed S01 protocol lock or later contract/hypothesis adjudications; historical records bound the next stage but do not replace it. Every stage requires independent A/B proposals on the same source packet, Judge C adjudication, and canonical-only propagation (skill §§2, 2.1–2.4, 2.7, 2.17–2.18).

The controlling project success threshold is the user's inclusive `test_recall@10 >= 0.065`, as also recorded in iter29's protocol manifest and S14 review. Root `CLAUDE.md` §2 and skill §§0,12 contain lower-priority strict `> 0.065` wording; preserve that wording conflict, but do not apply it over the user's inclusive threshold.

## Verified historical result facts

The exact primary Stage3 results are:

- Iter26: `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`; `n_eval=57439`, `test_recall@10=0.057017009349048554`, `test_recall@5=0.03788366789115409`, `test_ndcg@5=0.025169911931406087`, `test_ndcg@10=0.03133359761524377`.
- Iter29: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`; `n_eval=57439`, `test_recall@10=0.05921064085377531`, `test_recall@5=0.03953759640662268`, `test_ndcg@5=0.026252776900288842`, `test_ndcg@10=0.03257647953179143`.

Iter29's canonical S01 manifest identifies iter26 as its sole protocol-valid direct control and marks iter18 `HISTORICAL_NONCOMPARABLE`; do not directly rank iter29 against iter18. The iter29-minus-iter26 R@10 point difference is `+0.002193631504726755`; one observation per mapping does not estimate run-to-run variance or establish causal gain. Iter29 S14 classifies the result `ACTIVE_NEUTRAL + PROMOTION_FAIL`. Neither this point delta nor a below-target score establishes FCCR-1 family failure.

Primary references: the two exact `test_final.json` records above; `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md` §§ roles/comparison and provenance; `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/global_review_after_iter29.md` §§ protocol-compatible evidence, result classification, target.

## Active FCCR-1 constraints

FCCR-1 is the fixed closed-form mapping from behavior branching `B_l` and **raw residual magnitude/median** `m_l_raw` to three precomputed curvatures. Curvatures are non-trainable and invariant across time, optimizer updates, and train/eval mode. Do not introduce learned/scheduled/cyclic curvature, curvature regularization, new curvature-conditioned optimizer or auxiliary-loss mechanisms, mechanism stacking, a new Sinkhorn/behavior mechanism, or Stage1/Stage3 changes. Existing consumers of fixed curvature may remain only when held identical between compared arms. Stage2 SID/Gini/collision/entropy statistics are descriptive, not gates; a valid, non-aborted candidate proceeds to Stage3. See skill §§3, 9–10, 13, 16 and root `CLAUDE.md` §2.

## S14 design bounds (not an S00 mechanism choice)

Iter29 S14 authorizes only a future prospectively registered matched-seed replication of the **two previously tested FCCR-1 mappings**, with three pairs `43`, `44`, `45`; each pair uses matching Stage2 and Stage3 seeds in both arms. This implies six complete Stage2/Stage3 pipelines, subject to all gates. Within each pair, the mapping is the only conceptual difference; hold Stage1 and Stage0 identities/order, immutable warm start, other Stage2 settings, Stage3 code/config/runtime, evaluation set, and metric computation fixed. It does not guarantee target attainment or statistical power. Seed-42 historical results may enter pooled analysis only if S01 establishes required compatibility; they cannot replace a failed predeclared pair.

The S14-approved shared historical inputs in L0/L1/L2 order are `B=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and `raw_residual_medians=[1.0, 0.10941, 0.09331]`. The already-tested mapping equations and outputs are:

- Iter26: `s_l=log1p(B_l)/log1p(m_l_raw/min(m_raw))`; `z_l=(s_l-mean(s))/(std(s)+1e-12)`; `c_l=clip(0.5*exp(0.2*z_l),0.05,1.5)`, yielding `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.
- Iter29: `c_l=0.05+1.45*0.5*(B_l/(B_l+2.0)+m_l_raw/(m_l_raw+0.1))`, yielding `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.

These are boundaries for S01/S02 deliberation, not a mechanism selection or training authorization in S00. Primary references: `logs/deliberation/S14_GLOBAL_REVIEW/round_1/judge.md`, `logs/global_review_after_iter29.md` §§ strongest direction and equations, and `logs/protocol_manifest_iter29.md`.

## Raw-residual provenance limitation

The raw-residual values are approved historical method/value provenance at **medium confidence**, not checkpoint-level reproduction, current remeasurement, or demonstrated historic input-byte identity. Iter29 S03 records missing historical calibration checkpoint/raw SID inputs and absent historical hashes; it distinguishes raw residual medians from normalized layer scales `[0.001,0.932889,1.0]`. Do not substitute normalized scales for `m_l_raw`. The prior S01 lock expressly required a separate S03 provenance audit. Preserve this limitation; iter30 must independently re-establish/approve semantic provenance and source/value evidence before mapping approval or MVG (skill §7; iter29 `logs/deliberation/S03_PROVENANCE/round_1/judge.md`; iter29 `logs/protocol_manifest_iter29.md` §§72–78).

## Parent-side identity checks and runtime-use limit

The source packet reports that the parent directly rechecked SHA256 values for the Stage1 embedding, item-ID sidecar, all four Stage0 parquet inputs, and the iter8 warm-start checkpoint; each matched the identities recorded in the iter29 protocol manifest/S14 record at check time. This is parent-side direct evidence as reported in the packet, not a command run by this Judge. It establishes file identity at that check, not availability/identity at a future launch, resolved consumer paths, or actual runtime consumption. S01 must revalidate exact identities and trace actual input use. The warm-start identity recorded in iter29 S01 is `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.

## Iter29 closure hash: document record versus parent direct evidence

`stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md` records the S13 code/results closure push and local/GitHub hash match at `57b4a594d92435fd74fa4bba7c7c1439a33eb102`, then the separate S14 review synchronization at `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6`. That markdown does **not** record the later final closure-log amendment at `ecd01e4712a1badd38a0338255f4b2ec7b030aff`. Separately, the S00 source packet records parent-side direct post-push `git rev-parse main` and permitted GitHub-only `git ls-remote origin refs/heads/main` both matching `ecd01…`, with branch `main`, upstream `origin/main`, and the sole permitted `origin`. Thus the packet's explicit direct tool evidence corroborates hash parity for that amendment, while the closure document independently records only the earlier synchronization steps. Do not state that the markdown itself corroborates `ecd01…`, and do not present these historical checks as a fresh current Git check. No Git operation is part of this S00 task.

Primary references: iter29 `logs/git_closure_iter29.md` §§ post-commit/remote verification and S14 synchronization; iter30 S00 `source_packet.md` § current verified closure state.

## Mandatory repository and launch constraints

Current root `CLAUDE.md` controls:

- Project scripts have no CLI arguments or environment-variable overrides; work is on `main`, with no worktrees and only the permitted GitHub `origin` for Git operations.
- Stage2 artifacts go only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/`; Stage3 logs/checkpoints/results only under `results/stage3_T5Train/curvature_RQ-VAE_iter30/`. Keep the iter30 source subtree to code/configs/scripts/logs/input copies; generated `.pth`, `.npy`, and `item_sids.json` belong outside it. Preserve full `MECHANISM_NAME=iter30_<descriptive>` while output roots use the short mandated `curvature_RQ-VAE_iter30` name. Verify resolved `RQVAE_OUT_DIR`, Stage3 `LOG_PATH`, and `SAVE_PATH` before execution (`CLAUDE.md` §§0,1,10–11).
- Before **each actual Stage2 training run**, perform the required one-checkpoint/one-batch nonzero gradient-path check for total and applicable model/loss paths; fixed curvature itself remains non-trainable (`CLAUDE.md` §6).
- Stage2 proxy metrics do not gate Stage3 (`CLAUDE.md` §2; skill §10).
- An iter is not declared closed absent required artifacts and GitHub `main` push/hash verification under root §§8,13. No Git action is authorized by S00.

No Stage2 or Stage3 launch is authorized by this snapshot. Before any launch, complete the required S00–S09 independent A/B/Judge canonical gates, FCCR-1 contract preflight, contract-aware MVG, deliberation gate, resolved-path checks, and required per-run Stage2 gradient verification.

## Exact unresolved S01 verifications and next stage

S01_PROTOCOL_LOCK must inspect current primary source/configuration and:

1. Verify Stage1 embedding, item-ID/order sidecar, Stage0 train/valid/test/items, and warm-start SHA256 identities against the locked records; establish resolved paths and actual runtime consumers/use for every input, not merely declarations. Verify immutability and consumption of the warm start.
2. Establish Stage2 and Stage3 seed-setting locations/semantics, including how both seed values are actually consumed. Determine a no-CLI/no-environment-override way to execute all six predeclared pairs and retain six distinct outputs without collisions under the mandated short roots or changing experimental factors. If impossible within the registered protocol, stop before training under the governing gates; do not silently reduce the design or overwrite outputs.
3. Lock exact Stage2/Stage3 source and config identities, settings/runtime, evaluation code and metric, per-pair/per-arm path conventions, hashes, and the sole permitted within-pair mapping difference. Prove all resolved output roots comply with `CLAUDE.md`.
4. Re-establish and adjudicate raw-residual semantics/source/value provenance; preserve the medium-confidence historical limit and never use normalized scales as raw values. Confirm S14's two equations, values, and three seed pairs without inventing a third mapping in S00/S01.
5. Determine whether seed-42 history meets all identity/protocol requirements for pooled analysis; otherwise exclude it from pooled estimates and do not use it to replace any new pair.
6. Register a prospective paired analysis/reporting rule against the inclusive user target and complete the canonical iter30 protocol manifest. No Stage2/Stage3 run may begin until S01 and all later applicable gates pass.

`AUTONOMOUS_NEXT_ACTION=Advance to S01_PROTOCOL_LOCK using this canonical snapshot only; independently verify and adjudicate the protocol, current input identities and actual use, raw-residual provenance, seed semantics, six-run collision-free routing, and paired analysis plan, then materialize the canonical protocol manifest. Do not launch Stage2 or Stage3 from S00.`
