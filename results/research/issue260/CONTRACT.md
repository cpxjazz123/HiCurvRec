# R06 CONTRACT — topology token residual on the level-2 assignment target

Status: frozen before formal Stage2 training. One main mechanism, one candidate, one legal
Stage2/Stage3 run. Run id will be assigned by `stage2_start`; arm is
`TIGER_topology_residual_l2_rq`.

## 1. Root cause this change addresses

R05's probe (`results/research/issue260/diagnosis/r05_probe.json`) measured that the behavior and
curvature switches on the existing cost moved only **0.11% / 1.69%** of hard L2 codes; role mixtures
were near-uniform (entropy 1.508–1.584 vs max 1.585), the L3 gate collapsed (entropy 0.176), and the
quantizer reproduced the encoder almost exactly (per-level MSE 1.91e-5 / 5.39e-6 / 2.32e-6). The
existing role/curvature channel is assignment-inert: it reweights distances after the nearest codeword
is already fixed by reconstruction.

## 2. The single main change

For item `i`, let `t_i` be the masked mean of its four frozen GNN topology tokens, `P` a learned
`32 x 32` linear projection initialized to identity, and `r_i^2` the residual entering level 2:

    z_i = P t_i
    C_2(i, c) = sum_e m_{r(i), e} * d_e(r_i^2 + z_i, c)

Level 1 and level 3 costs are unchanged; level 3 continues from the original residual
`r_i^2 - q_i^2`, exactly as R05. The hard assignment and the Sinkhorn/floor rule are unchanged; only
the level-2 cost's target is shifted by the topology term. `P` is trained by the existing
reconstruction/commitment/codebook/curvature losses through the shared backward.

- Behavior signal: `t_i` from the frozen local GNN (training-visible graph only).
- Curve operation: the existing three-expert mixed-curvature `d_e` with the R05 gate/role mixtures.
- Coupling path: `topology tokens -> P -> level-2 target -> C_2 -> hard level-2 code -> SID`.
  Removing the token shift (switch off) or the curve operation changes this path.

The three switches exist in code and will be tested, not trained as extra arms:

| switch | behavior | curve | expected local effect |
|---|---|---|---|
| `TOKEN_RESIDUAL_OFF` | zero shift | on | L2/L3 codes revert to R05-like assignment |
| `CURVE_OFF` | identity shift | Euclidean | token shift remains, curvature removed |
| `BOTH_ON` | identity shift | on | candidate |

## 3. Pre-training local check already performed (fixed before the run)

With the frozen R05 checkpoint and the identity projection (the R06 initialization), adding the token
mean to the level-2 target changed **99.94% of L2 codes**, reduced L2 prefixes from 9609 to **4082**,
reduced triples from 23228 to **7448**, left L1 unchanged, and raised the validation behavior-pair
rates:

| metric | R05 | R06 local |
|---|---:|---:|
| L2 positive prefix rate | 0.003595 | **0.010709** |
| L2 negative prefix rate | 0.000316 | 0.001436 |
| L3 positive prefix rate | 0.000416 | **0.008779** |
| L3 negative prefix rate | 7.6e-6 | 0.001024 |

This is a mechanical signature, not a recommendation claim. The candidate is allowed to train only
because the local prediction below is met by the actual checkpoint at initialization.

## 4. Preregistered predictions and failure criteria

After the formal 3000-epoch Stage2 and checkpoint-consistent export:

1. **Signature**: L2 prefixes <= 6000 (local 4082); L2 positive behavior rate >= 0.006 (local 0.0107);
   L3 positive behavior rate >= 0.002 (local 0.0088). L1 codes and L1 rate must remain unchanged from
   R05 (the change is level-2 only).
2. **Technical gate**: three-code collision < 0.40; quantisation error and effective rank recorded as
   diagnostics. Checkpoint/SID export must match exactly.
3. **Candidate freeze**: the above validation-side signature is evaluated before Stage3. If any primary
   signature fails, do not launch Stage3 and no test is read; record the negative round.
4. **Stage3**: one frozen 150-epoch run, beam 20, seed 42, existing evaluator. One final test only.
5. **Success**: final test Recall@10 **strictly > 0.065** (no rounding) and all evidence/identity checks
   pass. Otherwise the round is a negative result and the issue stays OPEN.

## 5. Six answers required before Stage2

1. **Which problem?** The behavior/topology signal is inert in the hard assignment (<=1.7% code change).
2. **Evidence?** The R05 checkpoint probe: switch effects, near-uniform role mixtures, collapsed L3
   gate, near-zero quantisation error, and the token-vs-degree probe.
3. **Why does the mechanism act on it?** It adds the topology token mean to the level-2 assignment
   target before the cost; the local check changes 99.94% of L2 codes and raises behavior rates.
4. **Substantive difference from prior failures?** R03–R05 moved geometry weights on the same cost;
   R06 changes the target entering the assignment, so the nearest-codeword decision itself moves.
5. **Which metric decides?** L2 prefix/behavior signature on validation, then one final Stage3
   test Recall@10 > 0.065.
6. **What is learned if it fails?** That a topology target shift at initialization is not sufficient
   for recommendation; next diagnosis separates reconstruction/label quality from target shift and
   tests a different path without reusing test-driven tuning.

## 6. Frozen protocol and comparison

- Stage2: 3000 epochs, 4 cards, global batch 4096, seed 42, no early stop, port 50202, existing
  launcher; one formal candidate only.
- Stage3: 150 epochs, beam 20, seed 42, existing data split/evaluator, port 50201; one final test.
- Baseline: frozen TIGER RQ-VAE result (0.0557) reused, never retrained. Historical R01–R05 remain
  diagnostic records, not additional arms.
- No new Infra/assets/Skill/CLAUDE changes; no seed sweep; no tuning sweep; no test-driven selection.

## Dispatch (frozen)

- dispatch: `stage2_f704f55a4fd04d85a9955a2ed02d20cf`
- arm: `TIGER_topology_residual_l2_rq`
- source tree: `results/stage2_RQ-VAE/_runs/stage2_f704f55a4fd04d85a9955a2ed02d20cf/TIGER_topology_residual_l2_rq/src`
- result: `results/stage2_RQ-VAE/_runs/stage2_f704f55a4fd04d85a9955a2ed02d20cf/TIGER_topology_residual_l2_rq/result`
- smoke: `entry` PASS (one-epoch collision 0.739171, not the formal gate) and `behavior_graph` PASS.

## R06b dispatch

- dispatch: `stage2_127a3cca78dc49bf899ded006b2a11ca`
- the only mechanism change from R06: `scaled_token_offset` normalises the topology direction
  to the mean level-2 codeword norm; no path, hyperparameter, budget or Stage3 change.
