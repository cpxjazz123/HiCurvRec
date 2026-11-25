# Issue #260 RUN_LOG — long-term root-cause-driven mixed-curvature SID iteration

## 1. Issue identity (frozen 2026-11-26)

| field | value |
|---|---|
| number / state | 260 / OPEN |
| title | Long-term research: root-cause-driven mixed-curvature SID iteration until Recall@10 > 0.065 |
| body bytes / sha256 | 9255 / `69fed718a05514f752d06bfe604f3f44f12722244a85866f8745ce1d2a65f4a0` |
| source | `results/research/issue260/ISSUE.json` |

## 2. INFRA_MODE = FROZEN (explicit issue text)

The issue states verbatim: “单 Agent，使用最新 complete-open-issue；INFRA_MODE=FROZEN。禁止新增或修改
公共 Infra、数据资产、Skill、公共训练/评估规则。仅研究机制与专属测试允许改变，调用 #259 已有依赖计划、
持久化执行、收据、分析及回放工具。”

Consequences recorded: no new/changed public tool, asset, Skill, CLAUDE.md, launcher, evaluator or rule.
Only the candidate mechanism and its dedicated tests may change. Training only through
`stage2_start` / `stage3_start`; diagnostics use existing checkpoint/SID/analyse interfaces.

## 3. Training decision

The issue's **only success condition** is one legal, complete Stage3 test Recall@10 **strictly > 0.065**
(no rounding). The first round is explicitly “先诊断，再决定机制（不预先指定 R06 改法）”, so no Stage2
training starts before a preregistered R06 contract with root-cause evidence. After the diagnosis and
contract, one formal candidate must run the frozen Stage2/Stage3 protocols; a negative round leaves the
issue OPEN and starts the next round.

## 4. Frozen acceptance list (C1..C10)

| id | criterion |
|---|---|
| C1 | Use existing checkpoints/SID/gradients/valid-side assets to diagnose the five chain segments; at least two competing hypotheses including a null; no default full retraining |
| C2 | `DIAGNOSIS.md` + cross-round `HYPOTHESES.json` record phenomena, causes, evidence, conditions, counterexamples, status and reopen reasons |
| C3 | Preregister exactly one main mechanism change with a causal path distinct from R01–R05; `CONTRACT.md` has formula, I/O, prediction, tolerance, success/failure and validation gate, and answers the six questions |
| C4 | TDD: independent reference, deterministic tests, required mutation/gradient/entry/performance checks; reuse #259 RUN/REUSE and durable evidence where valid |
| C5 | Pre-training local checks verify the mechanism's predicted intermediate effect; a negative local prediction returns to diagnosis without burning formal training |
| C6 | Formal Stage2 = 3000 epochs, 4 cards, seed 42, global batch 4096, collision < 0.40; final checkpoint-consistent hard SID; report all preregistered structure/gate/gradient/curvature metrics |
| C7 | Stage3 = 150 epochs, beam 20, seed 42, frozen data/evaluator, validation-frozen candidate; one final test only |
| C8 | Recompute metrics with existing replay/report tools; update VERDICT/ROUND_RESULT/PROGRESS/HYPOTHESES; Infra hash unchanged; commit/push/comment/archive |
| C9 | Close only if the same legal complete candidate has test Recall@10 > 0.065; otherwise record the negative round and continue with the next executable round |
| C10 | No new Infra/assets/Skill/CLAUDE changes; no TIGER retraining; no seed sweep; no repeated test-driven candidate selection |

## 5. Round 1 scope (diagnosis)

Mechanism changes: none yet. Evidence only: read R05 checkpoint/SID and baseline SID, run freeze-variable
interventions, inspect gates/role mixtures/margins/prefix behaviour, write DIAGNOSIS/HYPOTHESES.
No Stage2/Stage3 launch in the diagnostic step.

## 6. Reused immutable inputs

| input | sha256 |
|---|---|
| assets/manifest.json | `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b` |
| train.parquet | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` |
| valid.parquet | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` |
| test.parquet | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` |
| sentence_t5.npy | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |
| R05 checkpoint | `f93da1a7466d62f07c6482c64cf1a8efdd9dd1fa7ccdf7891be98d6d37bc7f26` |
| R05 SID | `dc206dba7fbdadc28bbdedb582c7c755e412d70bdfec930b380a2bd5ea9d5ba6` |
| baseline TIGER SID | `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059` |

## 7. Start integrity checkpoint

`INFRA_BASELINE.json` (same protected list as #259) is recorded before any mechanism edit. It must match
at the close checkpoint except for authorized mechanism files and this issue's evidence directory.

## 8. R06 formal dispatch

- arm: `TIGER_topology_residual_l2_rq`
- dispatch: `stage2_f704f55a4fd04d85a9955a2ed02d20cf`
- smoke: `PASS` (entry one epoch + behavior graph; one-epoch collision 0.739171 is not the formal gate)
- formal launch: 4 cards, global batch 4096, seed 42, 3000 epochs, port 50202
- result: `results/stage2_RQ-VAE/_runs/stage2_f704f55a4fd04d85a9955a2ed02d20cf/TIGER_topology_residual_l2_rq/result`

## 9. R06 formal failure (epoch 1334)

- dispatch `stage2_f704f55a4fd04d85a9955a2ed02d20cf` stopped at epoch 1334.
- root cause: `assignment_floor.CodebookCollapse` in `verify_separation`; codeword spacing
  fell to `3.427e-05` from `4.535e-04` (x0.076) for 50 consecutive steps.
- diagnosis: the identity token mean has norm `2.423`; the level-2 codebook mean norm is
  `0.046`, so the offset is ~52x the codebook resolution and destabilises the level-2 target.
- no SID/checkpoint was exported; the failed run is retained under its immutable run id.

## 10. R06b restart: scaled topology offset

- scale probe (`diagnosis/token_scale_probe.json`): level-2 codebook mean norm 0.046; raw
  token mean 2.423; unit direction rescaled to that norm changes 83.1% of L2 codes, keeps L1
  fixed, and raises the L3 positive behavior-pair rate 0.000416 -> 0.000511.
- source change: `scaled_token_offset` unit-normalises the token mean and the projected
  direction, then multiplies by the current mean level-2 codeword norm.  It replaces the raw
  identity projection that collapsed the level-2 codebook.
- focused tests: 19 passed (`test_token_residual`, `test_layer_mechanism`, `test_l2_role_reference`).
- new dispatch: `stage2_127a3cca78dc49bf899ded006b2a11ca`; smoke PASS, one-epoch collision
  0.238215.

## 11. R06b Stage2 completed

- final: 3000 epochs, 2010.8 s, collision 0.10054093626713301.
- artifacts: `rqvae_best.pth`, `item_sids.json`, `sids_for_hgrec.npy`, `archive_binding.json`.
- layer record: `results/stage2_RQ-VAE/TIGER_topology_residual_l2_rq/stage2_metrics/c5f0e739b11b3ee79237e22c2354f18b2dc6f18b8b78b03a3978d77bfec0691b.json`
  - distinct L2 prefixes 7714; distinct triples 22115; L3 refinement gain 2.2908 bits.
  - L1 hard ER 8.2843; conditional log enrichment 1.9128 bits.
- Stage3 launched on the SID; collision gate 0.1005 < 0.40; quant_error 0.0086 (warning low), effective_rank 5.05.

## 12. R06b token-switch ablation

`token_ablation_r06b.json` switches the token residual off on the trained checkpoint:
- L1 changed 0 / 24,587; L2 changed 12,831 (52.19%); L3 changed 12,571 (51.13%).
- per-role L2 changed fraction spans 0.413 (P4:0) to 0.575 (K1_3:1);
  the offset norm is 0.03928, the mean level-2 codeword norm.
This is the causal graphlet-role evidence; the full retrain comparison (role_divergence_r05_r06b.json)
is observational because the codebooks differ.

## 13. R06b Stage3 result

- test_R@10 **0.049600** (R05 0.050349; delta -0.000749), n_eval 57439.
- The token residual changes 52.19% of L2 codes and its effect differs by graphlet role
  (0.413-0.575), but the downstream recall is lower than R05 and below 0.065.
- R2 archive: `r2:genrec/107/`.
- This is a negative graphlet-specific result; it does not support the claim that different
  graphlets require different geometric biases.
