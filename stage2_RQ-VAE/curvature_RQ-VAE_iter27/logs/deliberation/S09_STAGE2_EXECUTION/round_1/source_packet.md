# S00 Source Packet — iter27

This packet is the canonical source-of-truth for the iter27 2+1
deliberation.  Every agent reads only this packet (plus primary
repository evidence) until after their own artifact is written.

## Source-of-truth documents (priority-ordered)

1. `CLAUDE.md` (root).
2. `.claude/skills/curvature-rqvae-iter/SKILL.md` §3 (active contract).
3. `mechanism_contract_iter27.json` (CAO-1 contract for iter27).
4. `iteration_bridge.md` (iter26 → iter27 direction).
5. `protocol_manifest_iter27.md`, `hypothesis_iter27.md`,
   `mechanism_manifest_iter27.md`, `one_factor_diff_iter27.md`,
   `implementation_plan_iter27.md` (all written before this packet).

## Active contract

CAO-1 (Curvature-Aware Optimization).  The user explicitly retired
FCCR-1 after iter26's NO-GO result; the contract for iter27 is the
post-step hyperbolic codebook trust-region mechanism described in
`mechanism_contract_iter27.json` and `hypothesis_iter27.md`.  Curvature
remains cyclic-learnable (inherited verbatim from iter18).

## Stage2/3 canonical artifact paths

- Stage2 source: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter27/`
- Stage2 outputs: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/`
- Stage3 outputs: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter27/`

## Iter18 baseline (canonical)

- Test `R@10 = 0.05988962203380978` (n_eval=57439).
- Source: `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`.

## Hard target

`test_R@10 > 0.065` (iter27 must beat the strict hard target AND
beat iter18's `0.0599` to qualify as a positive CAO-1 result).

## Stage2 protocol inheritance

- `MAX_GLOBAL_STEPS = 100_000`
- `BATCH_SIZE = 640` per GPU × 4 GPUs
- `COMMITMENT_WEIGHT = 1.0`, `BEHAVIOR_LOSS_WEIGHT = 0.20`,
  `BEHAVIOR_TEMPERATURE = 0.07`, `CURVATURE_REG_WEIGHT = 0.005`
- `C_CYCLIC_MIN = 0.05`, `C_CYCLIC_MAX = 1.5`, `C_CYCLIC_PERIOD = 100_000`
- `ADAMW_BASE_LR = 1e-3`, `ADAMW_BETA1 = 0.9`, `ADAMW_BASE_BETA2 = 0.999`,
  `ADAMW_BETA2_SPAN = 0.009`, `ADAMW_EPS = 1e-8`, `ADAMW_WEIGHT_DECAY = 1e-4`
- `MIDPOINT_LAYER_MASK = [False, False, False]`
- Warm-start from iter8's `rqvae_best.pth` (skip `c_layer_scale`).

## CAO-1 new constants

- `ITER27_TRUST_RADIUS_FRAC = 0.5`
- `ITER27_TRUST_RADIUS_MIN = 1e-3`
- `ITER27_TRUST_LOG_EVERY = 1000`

## Primary evidence to use (do not paraphrase)

| File | What to read |
|---|---|
| `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json` | iter18 baseline R@10 |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter18/curvature_RQ-VAE.py` (lines 90–120, 214–290, 600–650) | β₂-from-c_l(0) implementation |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter18/modules/quantize.py` | cyclic learnable curvature schedule |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter27/modules/quantize.py` | iter27 trust-region method |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter27/modules/rqvae.py` | iter27 RqVae snapshot/apply methods |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter27/curvature_RQ-VAE.py` (Step5 + Step10 + post-step block) | iter27 wiring |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/mvg_check_iter27.md` | MVG result + F-1 diagnosis |

## Independence rule

Each worker (Agent A, Agent B) writes their candidate artifact under
`logs/deliberation/<STAGE_ID>/round_1/agent_a.md` (or `agent_b.md`).
They MUST NOT read each other's draft before submitting.  Judge C
writes `logs/deliberation/<STAGE_ID>/round_1/judge.md`.