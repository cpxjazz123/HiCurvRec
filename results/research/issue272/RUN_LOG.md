# Issue #272 run log

- issue: 272 — "Stage2-only: enforce active global distance constraint, local curvature fidelity and SID capacity before Stage3"
- body sha256: 8814eae7356ac36ba26c0812b51a1d4327b499171327eaa31be3b70bbeca84f7; frozen at 2026-11-27 (before any mechanism code).
- **INFRA_MODE = FROZEN.** Issue quote: "INFRA_MODE=FROZEN，不开发或修改任何公共infra，不重建已有资产；
  只改候选机制与必要专属测试/分析。" No BUILD authorization.
- **Stage3 is forbidden.** Issue quote: "本Issue绝对不启动Stage3，不运行T5训练/新test、不生成新的Recall
  结果；即使Stage2全部达到要求也只冻结候选并交付". Only Stage2 runs.
- **Training is required**: acceptance is a full 3000-epoch Stage2 candidate with J, local curvature, SID
  capacity and reconstruction measured from its own checkpoint/SID. Issue §3.
- **Mechanism scope**: one active candidate tree under `stage2_RQ-VAE/`, its mechanism tests, and
  issue-local analysis under `results/research/issue272/`. Public trees/assets are read-only.
- **A leftover Stage3 from the already-closed #271 is still finishing on the only GPU; it is not part of
  this issue and is not used for #272. Stage2 starts only after the card is free.**

## Acceptance checklist (frozen from the issue body)

| id | requirement | criterion |
|---|---|---|
| C1 | unified ruler inventory | recompute full J, per-hop J, neighbourhood fidelity and local curvature for frozen TIGER/#265/#269/#271 from the actual checkpoints/SIDs, one scale/reference/mask; no mixing training kappa, absolute Sigma and relative J |
| C2.1 | global J | full common-ruler J **<= 0.338336** (or the recomputed #269 value if different); 1-hop/2-hop not worse than #269 beyond a pre-frozen tolerance; training bound based on #269/#271, not the loose TIGER bound |
| C2.2 | local curvature | MAE **<= 1.893506**, Spearman **> 0.01585**, sign accuracy **> 43.25%** (independent ruler); report Pearson/RMSE/within-centre order/Graphlet-Orbit strata/pred-target variance; exclude scale/constant degeneracy |
| C2.3 | SID capacity | collision recorded only, no numeric gate; L2 distinct prefixes **>= 10,918**; codebook usage, L3 gain, effective rank **target >= 8**, pos/neg prefix ER, behaviour neighbourhood, Graphlet/Orbit alignment from the actual exported SID; prefix discrimination and behaviour neighbourhood not below frozen TIGER |
| C2.4 | reconstruction | <= 1.05 x #265 final reconstruction 1.23e-4; record quant error/diagnostics; existing non-collision guards pass; collision exit gate is recorded only and does not fail this Stage2-only study |
| C3 | active constraint proof | same L_RQVAE + lambda L_curv + mu (J - J_max); lambda/dual step/schedule/budget/conflict handling adjustable inside the mechanism; real-batch probe with both violated and satisfied cases; local/global gradients reach encoder/codebooks; report active fraction and contribution; if the constraint is inactive, do not attribute improvement to it |
| C4 | #271 failure diagnosis | explain MAE-improved but rank/sign worse, effective rank 4.50 and L2 6,097: scale cancellation, prediction variance, reconstruction/local gradient competition, per-level codeword changes; falsifiable fix; no blind parameter sweep |
| C5 | iterative Stage2 | each round: one root cause, competing explanation, single primary change, expected effect, criterion; low-cost probes; full 3000 epochs seed 42; no short-run acceptance; final re-exported SID; pass only when all pre-frozen criteria hold; otherwise record the failing criterion and next fix |
| C6 | persistence | commands/exit/log/config/checkpoint/SID/full metrics/hashes; existing necessary validation only; no new infra; four-card target ~0.4 s/epoch, single-card recorded |
| C7 | no Stage3 | no T5 training/new test/new Recall; freeze the candidate and deliver; archive Stage2 outputs without a fake recall R2 set; close only when at least one full candidate passes all frozen non-collision criteria + legal admission, with an explicit PASS table; never relax |

## Decisions

- Rounds are allowed within the same mechanism (issue §3). A candidate is only called PASS when every
  frozen non-collision criterion and the legal admission hold.
- The #271 Stage3 process is not used; its result must not enter any #272 criterion or verdict.

## Reference inventory (in progress)

- full J: TIGER **0.739634552674212**, #265 **0.18947582206068891**, #269 **0.338335882630658** (matches the frozen reference), #271 pending.
- #269 per-hop: 1-hop 0.7602806467884131, 2-hop 0.26486264828660244; the frozen 5% tolerance is 0.7983 / 0.2781.
- SID: TIGER L2 7,374 / rank 11.479; #265 10,918 / 3.049; #269 6,246 / 4.335; #271 6,096 / 4.501.
- local: TIGER 7.309 / Spearman 0.015851 / sign 0.432453; #265 2.246 / -0.071041 / 0.332350;
  #269 1.997 / -0.070083 / 0.363704; #271 1.894 / -0.005483 / 0.392646.
- The leftover #271 Stage3 was stopped on explicit user instruction at epoch 150 before its final test; it is
  not used for #272 and produced no test_final. The GPU is free for the #272 Stage2.
