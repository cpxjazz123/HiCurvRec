# source_snapshot_iter27 (Agent G — 2026-09-26)

Iter27 is the **first iteration under the new CAO-1 (Curvature-Aware Optimization)
contract**, replacing FCCR-1 (Fixed Closed-Form Curvature).  The user explicitly
ended FCCR-1 after iter26's NO-GO result; iter27 must therefore satisfy CAO-1
process invariants, not FCCR-1's.

## 1. Repository rules (priority-ordered)

1. `CLAUDE.md` (root): hard-codes paths, no-CLI scripts, four-card DDP launchers,
   Stage2/Stage3 unified artifact directories, GitHub-only remote, single
   `main` branch, mandatory commit+push per iter (§8/§10/§11/§12/§13).
2. `.claude/skills/curvature-rqvae-iter/SKILL.md`: 2+1 deliberation, FCCR-1 by
   default, but the user explicitly switched the contract to CAO-1 in iter27's
   directive.  The skill's §22 (success definition) is kept; the FCCR-1-specific
   §3 invariants are retired for iter27 only.
3. `mechanism_contract_iter27.json`: `contract_version="CAO-1"` — pinned
   candidate mechanism and its hard invariants.  This contract supersedes the
   FCCR-1 fields in the iter27 preflight.
4. `iteration_bridge.md`: iter26→27 falsifiable objective and forbidden
   directions (no fixed-curvature repeat, no further `c → LR/β₂` tweaks).
5. Historical evidence: iter18 (current best, `R@10=0.0599`); iter26
   (`R@10=0.0570`); iter11 (`R@10=0.0598`); iter25 (`R@10=0.0588`).

## 2. Stage1/Stage2/Stage3 paths (already wired in `curvature_config.py`)

- `STAGE0_DIR`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet`
- `ITEM_EMB_NPY`: `…/stage1_GeneEmbedding/output/sentence_t5.npy`
- `RQVAE_OUT_DIR`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/out/rqvae/instruments`
- `ITEM_SIDS_JSON`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/item_sids.json`
- `SIDS_NPY`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/dataset/Instruments/sids_for_hgrec.npy`
- Stage3 `CODE_PATH`: `…/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/item_sids.json`
- Stage3 `LOG_PATH`: `…/results/stage3_T5Train/curvature_RQ-VAE_iter27/logs/`
- Stage3 `SAVE_PATH`: `…/results/stage3_T5Train/curvature_RQ-VAE_iter27/ckpt/`

## 3. iter26→iter27 carryover (verbatim from `iteration_bridge.md`)

- iter26 used FCCR-1 fixed `c_l = [0.6145, 0.5333, 0.3814]` for 100k steps; contract met; geometry aligned with iter18 on most proxies but `R@10 = 0.057017`.
- iter11/iter18/iter25/iter24 all above iter26 → bottleneck is **not** in `c` itself; it is in **how codebook parameters move under AdamW given any `c`**.
- Forbidden next directions: another fixed-curvature experiment, another `c → LR/β₂` tweak, any Stage1/Stage3 trainer change, any new codebook parameter, any new loss term.

## 4. New contract (CAO-1) hard invariants

- Curvature is **still cyclic learnable** (inherited verbatim from iter18).
- The new mechanism is a **post-optimizer-step hyperbolic codebook trust-region** on `layer.embedding.weight` only.
- No new `nn.Parameter`; no new loss term; no new assignment rule.
- Trust-region parameters: `τ_l = 0.5 · D_l` where `D_l = median_k min_{j≠k} d_{c_l(t)}(p_k, p_j)`, with a floor of `1e-3` for degenerate layers.

## 5. Unresolved conflicts

None at this snapshot.  The iter26→27 bridge explicitly retired FCCR-1; the
SKILL.md's FCCR-1 paragraphs are retained for historical evidence only and do
not bind iter27.

## 6. Verified repo facts (independent readback)

| Fact | Path | Result |
|---|---|---|
| `MECHANISM_NAME` | `stage2_RQ-VAE/curvature_RQ-VAE_iter27/curvature_config.py:13` | `iter27_hyperbolic_codebook_trust_region` |
| `apply_hyperbolic_trust_region` defined | `modules/quantize.py` | yes (post-step projection on `embedding.weight`) |
| `snapshot_codebooks` defined | `modules/rqvae.py` | yes (detach-clone every `layer.embedding.weight`) |
| `ITER27_TRUST_*` constants | `curvature_RQ-VAE.py` | `0.5`, `1e-3`, `1000` |
| git HEAD | `cbf38da` | clean |
| remote HEAD | `cbf38da` | matches |
| parent iter18 `R@10` | `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/…/test_final.json` | `0.05988962203380978` |