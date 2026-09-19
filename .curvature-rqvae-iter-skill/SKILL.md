---
name: curvature-rqvae-iter
description: Iterate and optimize the variable-curvature RQ-VAE at /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE for the Amazon-2023 Instruments dataset. Each iteration clones the working directory, modifies only the RQ-VAE mechanics (NOT stage1 embeddings or stage3 T5 trainer), trains the new variant, evaluates SID quality with FORGE-style metrics, and only promotes a variant to stage3 downstream training if its SID quality is at least as good as the baseline AND shows a credible path to beating the baseline downstream test metric. The hard success target is downstream test metric > 0.065, with valid-stage projected > 0.07. Innovation focus is variable curvature + hyperbolic geometry (Poincaré ball / Lorentz / projective hyperbolic).
---

# curvature-rqvae-iter

Iterate on the variable-curvature RQ-VAE pipeline. The deliverable is a promoted variant whose downstream GR (generative recommender) test metric exceeds **0.065** on the Amazon-2023 Instruments dataset, with valid-stage projected above **0.07**.

## Scope Boundaries (Hard Constraints)

- **Only edit** files inside `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE/`.
- **Never edit** `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/` (input embeddings).
- **Never edit** `/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/` (downstream trainer). Stage3 is invoked **read-only** after each SID promotion.
- **Never edit** the input `item_emb.parquet` or `Instruments.inter.json`. They are the immutable contract with stage1.

## Iteration Protocol (Loop Until Success or N Iterations)

For each iteration `i = 1, 2, 3, ...`:

1. **Clone the current best working directory** to a new sibling:
   ```bash
   SRC=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE
   NEXT=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter${i}
   cp -r "$SRC" "$NEXT"
   ```
   The previous iteration's directory is **read-only after the clone** — never edit it again.

2. **One novelty per iteration.** Pick exactly one new mechanism from the candidate pool (see Innovation Focus) and apply it to `curvature_RQ-VAE.py` + `curvature_config.py`. Do not stack multiple novelties in one iteration.

3. **Run the iteration in the cloned directory.** Use the `train_iter.sh` helper script (see `scripts/`). It pins:
   - Working directory = `$NEXT`
   - `CUDA_VISIBLE_DEVICES` from `nvidia-smi` (least-utilized free GPU)
   - Log to `$NEXT/logs/train_iter${i}.log`
   - Output ckpt to `$NEXT/out/rqvae/instruments_iter${i}/`
   - `MAX_GLOBAL_STEPS` from config (default 50_000, ~8 min on single GPU)



   - `MAX_GLOBAL_STEPS` from config (default 50_000, ~8 min on single GPU)

   **3a. Mid-training early-stop judgement (mandatory, time-saving).** The trainer is allowed — even *encouraged* — to terminate a step 3 run **before** `MAX_GLOBAL_STEPS` if any of the following signals appear in the log tail (`tail -50 $NEXT/logs/train_iter${i}.log`):

   - **(a) Codebook collapse observed.** Per-layer unique codes drops below `0.85 × codebook_size` for ≥ 2 layers in any 500-step window (e.g. for 256 codes, L1 < 218 AND L2 < 218 simultaneously). The run is dead, do NOT wait it out.
   - **(b) Convergence plateau.** Both `loss` and `vl` (vq loss) do not improve by more than 1% over a 2k-step sliding window. Compare last 2000 steps vs the previous 2000 steps.
   - **(c) Step budget already sufficient.** If `MAX_GLOBAL_STEPS = 50_000` and the iteration reaches `step >= 30_000` with `unique codes ≥ 240/256` on every layer and `loss` is monotonically decreasing, the trainer is free to declare "30k is enough, the SID quality won't improve past this" and skip the remaining 20k steps.
   - **(d) Numerical instability.** NaN/Inf in any reported metric, `c` collapsing to 0 or exploding > 5, `margin` becoming negative, or `loss` flipping sign twice within 1k steps.
   - **(e) Time-budget hard cap.** If the iteration has already consumed > 35 minutes wall-clock on a single GPU and the loss curve looks no better than the baseline v318 trajectory, stop. There is no value in running 50 more minutes for a 0.1% delta.

   **How to early-stop cleanly:**
   ```bash
   # Find the trainer PID
   pgrep -f "curvature_RQ-VAE.py" | head -1
   # SIGTERM first (let trainer save final ckpt if it has a signal handler); if not, SIGKILL
   kill -TERM <PID>; sleep 5; kill -KILL <PID> 2>/dev/null
   # Verify last partial ckpt exists
   ls -lt $NEXT/out/rqvae/instruments/rqvae_step*.pt | head -3
   ```
   Then export SIDs from the latest `rqvae_step*.pt` (the trainer saves every `CKPT_EVERY` steps so a partial ckpt is always available). If no partial ckpt exists (i.e. terminated before first `CKPT_EVERY`), fall back to `rqvae_final.pt` from the previous best iteration.

   **Mandatory log entry on early-stop:** append a one-line note to `$NEXT/logs/train_iter${i}.log` saying `EARLY-STOP TRIGGERED at step <N>: <reason>`, plus the wall-clock time and the saved ckpt that downstream will use. This goes into the gate decision file as the explanation for why `MAX_GLOBAL_STEPS` was not hit.

4. **Quick SID quality gate (FORGE metrics)** before any stage3 run:
   ```bash
   python3 /home/wlia0047/ar57/wenyu/GeneRec/sid_eval/eval_sids.py \
       --sid_npy "$NEXT/dataset/Instruments/sids_for_hgrec.npy" \
       --name "iter${i}" \
       --out "$NEXT/logs/sid_quality_iter${i}.json"
   ```
   Promotion requires **both** gates to pass:
   - `SID_occupancy_Gini <= baseline_Gini + 0.005`
   - `collision_rate <= baseline_collision * 1.10` (allow 10% slack)
   - `Embedding_HitRate@500 >= baseline_HR500 - 0.005`
   
   If any gate fails: **log NO-GO, archive logs to scratch, do NOT proceed to stage3**. Move to iteration `i+1`.

5. **Stage3 downstream evaluation** (only after SID gate passes):
   ```bash
   python3 /home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/train_t5.py \
       --sid_npy "$NEXT/dataset/Instruments/sids_for_hgrec.npy" \
       --product_dir "$NEXT" \
       --tag "iter${i}"
   ```
   Use the existing stage3 script with the SID path swapped. Do NOT modify the script.

6. **Decision**:
   - If `valid_ndcg@20 >= 0.07` AND projected `test_R@10 > 0.065` → **PROMOTE**: copy the iteration directory back to `curvature_RQ-VAE/` (overwrite), commit, log `Gate 3 PASS`. Stop the loop.
   - If `valid_ndcg@20 < 0.07` BUT SID quality was better than baseline → **archive iteration**, continue with `i+1`.
   - If SID gate failed → skip step 5 entirely, archive iteration, continue with `i+1`.

## Innovation Focus: Variable Curvature + Hyperbolic

The novelty MUST touch curvature geometry. Candidate mechanisms (rotate through these, do NOT stack):

- **Cyclic c(t) curriculum** — `c(t) = c_min + (c_max - c_min) * |sin(πt/T)|`, period T ∈ {25k, 50k, 100k}.
- **Layer-wise learned curvature** — each RQ layer has its own `c_l`, learned via reparameterization (sigmoid) with momentum-restricted drift.
- **Lorentz (hyperboloid) model instead of Poincaré ball** — replace `exp_0` / `log_0` with `exp_x^c` / `log_x^c` in the Lorentz formulation. Keep `c` learnable.
- **Projective hyperbolic (Hesse/Poincaré half-plane)** — alternative manifold model with Möbius addition.
- **Adaptive margin loss** with curvature-dependent target: `margin = α / c_l` (sharper separation at high c).
- **Negative c (spherical interpolation)** — partially spherical, partially hyperbolic for items that lie in low-density regions.
- **Mixed-curvature product manifold** — `M = H^{c1} × H^{c2} × S` (product of hyperbolic + spherical).
- **Curriculum on quantization difficulty** — anneal codebook temperature by `1/c`.
- **Riemannian Adam** instead of plain Adam (gradient rescaling by inverse metric `1/c`).
- **Sinkhorn-Knopp regularization on codebook with c-dependent epsilon** `ε = ε_0 / c`.

When picking, the priority is: (a) mechanisms with prior success in published papers, (b) mechanisms that interact with the existing curriculum rather than orthogonal axes. **Do not invent novel loss terms** unless they directly modulate curvature behavior.

## Literature Search (Mandatory Each Iteration)

Before designing the mechanism for iteration `i`:
1. Use `web_search` with queries combining: `"variable curvature RQ-VAE"`, `"hyperbolic quantization"`, `"Poincaré recommender"`, `"learned curvature product manifold"`, `"Möbius scalar multiplication recommender"`, `"Riemannian quantization"`, `"curriculum learning residual quantization"`, `"FORGE tokenizer recommender"`, `"LETTER recommendation"`.
2. Identify 1-2 recent (2023+) papers that motivate the mechanism.
3. Record the search summary in `$NEXT/logs/lit_search_iter${i}.md` with: query, top-3 hits, the chosen mechanism, the paper rationale.
4. The novelty must be defensible against these papers — if a paper already claims the exact combination, pick a different mechanism.

## Hard Targets

- **Final target**: downstream `test_R@10 > 0.065` on Amazon-2023 Instruments.
- **Valid-stage projection**: `valid_ndcg@20 >= 0.07` (typical valid→test drift ~−15%, so 0.07 → ~0.06-0.07 test).
- **SID quality floor**: Gini ≤ baseline + 0.005, collision_rate ≤ baseline × 1.10.
- **Iteration cap**: 12 iterations. If after 12 iterations the target is not met, halt and surface a NO-GO summary.

## Files Layout Per Iteration

```
/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/
├── curvature_RQ-VAE.py            (modified — exactly one new mechanism)
├── curvature_config.py            (modified — new mechanism constants)
├── configs/                        (unchanged unless mechanism requires gin update)
├── dataset/                        (input — unchanged, cloned)
├── logs/
│   ├── train_iter${i}.log
│   ├── lit_search_iter${i}.md     (literature search record)
│   ├── sid_quality_iter${i}.json  (FORGE metrics)
│   └── gate_decision_iter${i}.md  (GO / NO-GO)
└── out/rqvae/instruments_iter${i}/ (training output)
```

## Commit Discipline

After each iteration decision (whether GO or NO-GO), commit:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
git add stage2_RQ-VAE/curvature_RQ-VAE_iter${i}/logs/
git commit -m "iter${i}: <mechanism-name> | Gate 1 (SID Gini) PASS/FAIL | Gate 2 (Collision) PASS/FAIL | Gate 3 (Valid@20) PASS/FAIL | <reason>"
```

Commit hash MUST appear in the decision log. Never write "pending"/"TBD".

## Cleanup Between Iterations

Once an iteration is archived as NO-GO:
- Move the iteration directory to `/home/wlia0047/hj82_scratch2/wenyu/iter_archive/curvature_RQ-VAE_iter${i}_<timestamp>/`
- Never delete the archive; it is the audit trail.
- The active iteration directory under `stage2_RQ-VAE/` is always the **latest one**; previous best lives in `curvature_RQ-VAE/` until promoted.

## Pre-Flight Before Each Iteration

Before launching training:
1. `nvidia-smi` — confirm GPU util < 10% and mem < 5GB on the chosen GPU.
2. `git status` — confirm previous iteration is committed.
3. `cat $NEXT/curvature_config.py | grep MECHANISM_NAME` — confirm mechanism is switched.
4. Check that `MAX_GLOBAL_STEPS` is set and `CKPT_EVERY` is reasonable.

## Stopping Conditions

Stop the loop and surface a summary when:
- (a) Downstream `test_R@10 > 0.065` AND `valid_ndcg@20 >= 0.07` achieved → **SUCCESS**.
- (b) 12 iterations exhausted without success → **HARD STOP**, write summary of best 3 iterations by valid_ndcg@20.
- (c) Same mechanism tried with 3 different hyperparameter settings, all NO-GO → drop the mechanism from the candidate pool.



### Stop Triggers (Mid-Iteration, Skill-Local)

- (d) **Codebook collapse**: per-layer unique < 85% × codebook_size in ≥ 2 layers → SIGTERM, do NOT complete the step budget.
- (e) **Loss plateau**: last 2k steps Δloss < 1% → SIGTERM, ckpt at the plateau is enough.
- (f) **Step already sufficient**: ≥ 30k steps with healthy unique + monotone loss → SIGTERM, no need to drain to MAX_GLOBAL_STEPS.
- (g) **Time budget exceeded**: > 35 min wall-clock with no downstream-projected improvement → SIGTERM.

These triggers OVERRIDE the "run to MAX_GLOBAL_STEPS" default. The default `MAX_GLOBAL_STEPS` is an upper bound, not a target. Always prefer to ship a clean 30k-step run + 1 quality check over a 50k-step run with no quality check at all.

## Reference Files

- `references/baseline_metrics.md` — baseline numbers for promotion gates (Gini, collision, HitRate, valid_ndcg@20, test_R@10).
- `references/mechanism_pool.md` — the 10 candidate mechanisms with paper citations.
- `scripts/train_iter.sh` — helper to launch an iteration training run with all paths pinned.
- `scripts/eval_sids.py` — promoted copy of the FORGE-style SID quality evaluator.
