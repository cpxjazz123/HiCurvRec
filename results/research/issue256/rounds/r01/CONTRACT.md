# Issue256 R01 frozen contract — global module-budget calibration

Frozen after the old active mechanism was safely archived and parent identities audited, before new reference, tests or candidate implementation. Issue body SHA256:32a2764070e8502926965d355a3ef6ff7619d45881abd0ffa575cb7cef73c5a8. Diagnosis and falsifiable predictions:DIAGNOSIS.md. Candidate:`TIGER_module_budget_prefix_rq`; one formal generation:`issue256_r01_formal`. Explicit Issue256 authority permits a new hypothesis version, not a same-mechanism score rerun. This is not a retrospective adoption/repeat of #255's invalid reconstruction-only coefficient.

## 1. Single principal change

Replace the parent encoder-only initial coefficient, measured on256 sorted training items, with a **single maximal initial global module-norm budget** measured on the real first shuffled training batch. Module coverage and population together define this one calibration policy; this round cannot isolate their individual causal effects. Every other mechanism computation and frozen recipe remains unchanged.

Let O be the actual original RQ-VAE objective (reconstruction+commitment+codebook, with the original coefficients), and C=(C1+C2+C3)/3 the actual cumulative hard-prefix curvature Huber objective. Each rank r uses exactly its1,024 first-epoch training items and the same structural sample the actual first step would use. No optimizer step or train-time random state advancement is permitted during calibration.

For every coupled parameter tensor p, compute o_r,p=dO_r/dp and c_r,p=dC_r/dp. Before taking norms, average the actual gradient vectors over the four ranks:

`o_p=(1/4) sum_r o_r,p`, `c_p=(1/4) sum_r c_r,p`.

Partition the curvature-coupled parameters into17 nonoverlapping natural groups:encoder; GNN.input; the six GNN.layers.{0,1}.{self_linear,incoming,outgoing}; sharedKey; sharedValue; sharedScore; Query0/1/2 (the respective rows of the real shared query parameter); codebook0/1/2. Every coupled parameter element appears exactly once; decoder is excluded because C is independent of it. Query rows are slices, not separate fictitious parameters.

`a_g=||o_g||_2`, `b_g=||c_g||_2`.

Choose the greatest finite positive scalar satisfying every active module constraint:

`lambda*=max{lambda>=0 : lambda*b_g<=a_g for all g with b_g>0}=min_(g:b_g>0) a_g/b_g`.

Then freeze the float32 buffer once, broadcast/verify agreement, and train `L=O+lambda* C` without further adaptive balancing. No manually chosen multiplier, temperature, schedule, new objective or parameter search. Returning merely the encoder ratio, the maximum ratio, the mean of group ratios, a reconstruction-only numerator, local gradient norms averaged after reduction, or omitting a query/codebook is incorrect.

Cauchy-Schwarz gives `o_g dot (o_g+lambda*c_g)>=a_g*(a_g-lambda*b_g)>=0` **at calibration for the raw Euclidean gradient**. It does not guarantee an AdamW descent direction, later-step norm balance, noncollapse or higher Recall. Those remain measured hypotheses, not mathematical consequences.

## 2. Inputs, numerics and boundary cases

- Full semantic catalogue:[24587,768] float32; IDs:int64, zero-based globally stable. Calibration batch per rank:[1024,768], IDs:[1024], exact first epoch DistributedSampler(seed42,epoch0) positions mapped through the unique legal training target catalogue. The reference sample is StructuralSampler.draw(step0,rank,world4,track=False). Deriving positions must not consume DataLoader/global RNG or a different epoch shuffle.
- Parameter and VJP tensors:float32 on the same device, exactly the actual loss graph. All coupled parameter gradients must be present and finite; absent coupling is a failure, not a silently skipped constraint. Global averaging is SUM then /world_size on VJP vectors, not on scalar norms. CPU unit tests may use a non-distributed one-population call; real verification must use all four ranks.
- Accumulate module norm/dot reductions in float64. Norm budgets are finite nonnegative Python numbers. b=0 imposes no constraint; a=b=0 is inactive. a=0,b>0 would force lambda=0 and disable the required structural coupling, so fail closed with a degenerate-calibration error. No active group, NaN, infinity, negative norms or nonfinite/nonpositive lambda fails closed. No arbitrary epsilon to hide degeneracy. Unconstrained decoder is not a zero-original coupled group.
- Store lambda asfloat32, positive finite. Reference agreement relative2e-6/absolute1e-12; active budget relative2e-6; global four-rank coefficient agreement exact after the final broadcast. Calibration must not modify any model parameter or optimizer state. Its stored coefficient/scales and diagnostics are persisted in the checkpoint/log.
- Two/rank cancellation is a real edge:average vectors before norms; opposite local gradients may cancel. If this makes an active global original norm zero, reject rather than inventing a safe coefficient. Unit tests cover normal, inactive, zero, degenerate and nonfinite cases.

## 3. Preserved inherited mathematics and routes

The original mathematical definitions are frozen in `results/research/issue255/CONTRACT.md` SHA256`a47efec99b0dff2e03adcfdbb113af925693927736382630e5a1d496de7eb7d5`; those sections1–4 and6–9 remain the definitions below except the calibration clause explicitly replaced above.

- Three spaces{-1,0,+1}; original768->32 encoder/decoder, three256-word residual codebooks and unchanged fourth SID collision counter. Full one-hop/two-hop/in-only/out-only induced directed ego tokens, root exactly once and explicit masks/null token, one shared directed GNN with jointly trained gates. Empty neighborhoods/root-only tokens, reciprocal and duplicate-edge behavior retain the verified parent semantics; no token or training data pruning.
- H_i:[N,4,32], three independent level queries over shared K/V/score, weights:[N,3,3] normalized over experts, attention:[N,3,3,5] including null. Gate row-sum absolute tolerance1e-6.
- At level l, residual r_l, cost_j=sum_k w_l,k*d_k^2(r_l,e_l,j) using the identical parent stable ambient geometry and initial frozen tangent scale. Actual hard assignment remains parent L1/L2 floor-constrained nearest and L3 Sinkhorn. This is the expressly inherited Issue255 actual protocol, not a claim of three-level baseline Sinkhorn compliance. Same-cost selection surrogate remains hard.detach()+(soft-soft.detach()); a plain direct straight-through/detached-cost replacement is forbidden.
- q_l=sum_(j<=l) selected-codeword_j, forward bitwise equal to the actual hard SID decoded prefix. Earlier-prefix losses have no future-codebook direct dependence. Each C_l is Huber(K_l/target_scale,K_graph/target_scale,delta1,mean), alpha_l=1/3. K_l=mean_a[d^2(a,m)+d^2(b,c)/4-(d^2(a,b)+d^2(a,c))/2], fixed gate-independent equal-three-expert distance, original full reference components and finite biased training sampler. All three levels share the same structural sample; no repeated learned GNN pass per level.
- Initial per-level distance scales and tangent/target scale calculation remain unchanged; no geometry normalization change is hidden inside this intervention. Coincident points, zero vectors, near-sphere/ball boundaries and degenerate triples keep inherited finite-domain checks/tests; unmeasurable frozen graph triples retain explicit masks. The estimator is a discrete proxy and training is a biased finite-population gradient, not intrinsic/unbiased sectional curvature.
- Structural supervision must reach encoder, each current/past codebook, each appropriate level query, K/V/score and shared GNN through legal selected-cost gradients. Disabling behavior, curvature and both-on switches retains the original cheap wiring checks; no ablation training.

## 4. Paper/official implementation cross-check

| Module | Paper position | Official implementation | Project choice and reason |
|---|---|---|---|
| Three-curvature experts | GraphMoRE arXiv2412.11085v1 Eqs4–6 | RingBDStack/GraphMoRE models.py, frozen blob5352bddb1b8a4eb6c32f6585af956b477eb836bd | Preserve inherited shared tangent/codeword metric mixture; not three separate Riemannian encoders |
| Directed GNN/attention | GraphMoRE representation module; Vaswani et al1706.03762 §3.2.1 | GraphMoRE official models.py supplies graph modules, not our three-level ego-query architecture | Preserve owner-specified full directed ego tokens and layer Cross-Attention; no claim this exact gate is official GraphMoRE |
| Curvature supervision | GraphMoRE AppendixA.1 proxy; Eqs9–10 discuss alignment | Official alignment differs from our frozen graph-triple Huber proxy | Preserve the explicitly frozen full-reference proxy, sampled biased gradient and independent common_fixed3 evaluator |
| RQ-VAE/SID | TIGER/RecBole semantic-ID residual quantization | Frozen repository TIGER tree | Preserve original capacity/losses/export; inherited actual floor/Sinkhorn protocol explicitly qualified |
| New norm budget | This contract §1, elementary norm inequalities | No GraphMoRE/TIGER official equivalent claimed | Single initial scalar derived from actual full original and curvature VJPs over global batch; no Recall-driven weight |

Primary sources:https://arxiv.org/html/2412.11085v1 ; https://github.com/RingBDStack/GraphMoRE/blob/main/models.py ; https://arxiv.org/abs/1706.03762 . Existing frozen comparisons in issue251/PAPER_COMPARISON.md and issue252/253 contracts remain the detailed historical audit, not a claim that a new paper was reproduced in this round.

## 5. Independent proof and execution gates

Reference.py must not import production modules. It solves a one-variable constrained linear program using SciPy HiGHS (`maximize lambda` subject to b_g/a_g*lambda<=1), independent of the production closed-form minimum. Hand answers hardcode the optimum, including a non-encoder bottleneck. Existing ambient-geometry and explicit-mean curvature oracles remain independent of production moment identities. Tests exercise the actual production calibrator on real training data and actual original+curvature losses, not just the new helper.

At least five designated-assertion mutations covering coefficient/norm reduction/group omission and global-gradient routing must be detected; crashes/import errors do not count. Full candidate tests and relevant inherited framework/asset tests run after implementation. Actual four-rank observer verifies per-level VJPs, original/structural directions, optimizer updates and hard prefix identity. A first-batch check independently recomputes the global group budget and chosen coefficient. Real machine entry plus behavior_graph must PASS before any formal dispatch.

R5 requires at least30 consecutive steady inclusive epochs on four L40S, complete data/loss/communication/optimizer and periodic work:mean<=0.8s; target0.4–0.5s. Initialization/export and per-rank/module costs/memory reported separately. Never satisfy timing by disabling the GNN or dropping2-hop/data.

One formal3000-epoch world4/global4096/seed42 run, port50202, sanctioned launcher and immutable source/result workspace. No early stopping or score rerun. Exact checkpoint-bound global hard SID re-export, first-three collision<0.4, persisted layer metrics before reading Stage3 score. Only then fixed150-epoch/beam20/full-test Stage3 via MCP, frozen protocol in ../../EVALUATION.md; external recommendation validation uses that exact final checkpoint. Record all test accesses and independently recompute metrics, then complete four-piece R2 archive/source binding. R7 continues with a new evidence-driven distinct hypothesis if the threshold is not reached.

## 6. Lineage and immutable records

The new tree begins as an exact source-only copy of the read-only baseline/TIGER_RQ-VAE. Apply the explicitly required inherited Issue255 framework as an inventoried mechanism patch from the archived parent, then this one calibration policy. Do not seed from historical trained weights, copy cache/results/binaries or modify the baseline/archive. Every inherited and new file difference must be recorded in LINEAGE.json. Candidate paths, diagnostic/evidence identities and provenance contract references must point to this round; historical evidence is never rewritten. Per-round ledgers carry parent SHA, prediction, data, source, complete runs, outcome and next decision.
