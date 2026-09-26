ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S05_ONE_FACTOR/round_1/source_packet.md
STAGE_ID=S05_ONE_FACTOR

# Iter29 One-Factor Proposal — Agent A

## Registration

```text
PARENT_ITER=iter26
PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
CANONICAL_BASELINE_ITER=iter26
EXPERIMENT_TYPE=single_factor
ACTIVE_MECHANISMS_BEFORE=FCCR-1 fixed closed-form curvature using iter26's standardized cross-layer mapping over behavior_branching and the recorded raw residual medians; unchanged inherited behavior loss and Sinkhorn quantization path
NEW_MECHANISM=Replace only the per-layer fixed-curvature mapping with the S02/S04 bounded rational/additive mapping, explicitly using behavior_branching and raw_residual_medians
ACTIVE_MECHANISMS_AFTER=FCCR-1 fixed closed-form curvature using the S02/S04 mapping and the same approved behavior_branching and raw_residual_medians; unchanged inherited behavior loss and Sinkhorn quantization path
```

Iter26 is both the immediate parent and the sole protocol-valid direct control under canonical S01. Its exact Stage3 control is `test_recall@10=0.057017009349048554`, `n_eval=57439`; this is below the inclusive user target `>=0.065`. Iter18 remains historical/noncomparable, not the control (S01 manifest; S00 snapshot).

## Exact conceptual delta

Inputs remain in `[L0,L1,L2]` order and are exactly:

```text
behavior_branching B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians m_raw = [1.0, 0.10941, 0.09331]
```

The new registered map is:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
```

with `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`. The canonical intermediates and output are:

```text
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
c = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Iter26 instead computes `m_min=min(m_raw)`, `s_l=log1p(B_l)/log1p(m_l_raw/m_min)`, `z_l=(s_l-mean(s))/(std(s)+1e-12)`, then `c_l=clip(0.5*exp(0.2*z_l),0.05,1.5)`. Its registered result is `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`; iter26's hypothesis records `m_min=0.09331`, `s=[1.2238118890466054,1.1604516044603779,1.010644112691917]`, and `z=[1.03129493309937,0.3223997103378134,-1.3536946434371857]`. Thus only the mapping, including its constants/intermediates/output, changes. No input vector, input semantics, layer ordering, or downstream code/rule changes. The inherited curvature-consuming quantization path remains unchanged; its realized `effective_eps` changes only as the deterministic consequence of the changed `c`.

The residual input must be loaded from the explicit plural JSON field `raw_residual_medians`, with fail-closed behavior if absent. Do not substitute `residual_norm`, normalized scales, or another fallback. S03 permits historical method/value provenance only; the values are not checkpoint-level replay or historic byte-identity evidence. Do not rerun the old behavior writer, which overwrites the JSON without the explicit raw key and requires missing iter8 raw SIDs (S03 manifest/Judge).

## Inherited mechanisms and unchanged comparison dimensions

- Curvature remains fixed, closed-form, non-trainable, and time-invariant; fixed values are consumed by the existing quantizer path. Cyclic/learnable branches remain inactive, `_fixed_c` buffer behavior is retained, and curvature regularization remains zero (iter26 source packet/contract evidence; FCCR-1 contract).
- Preserve inherited behavior loss weight `0.20` and temperature `0.07`; Sinkhorn configured `sk_eps=0.05`, `iters=3`, including the unchanged parent rule `effective_eps = sk_eps * (c / c_cyclic_max)`; three quantizer layers with codebook size 256; seed 42; 100,000 Stage2 global steps; iter8 warm-start checkpoint with curvature state excluded; AdamW `lr=1e-3`, `weight_decay=1e-4` (S05 packet, S01 manifest, iter26 records). Because `c` is the sole changed mapping output, realized `effective_eps` changes through this inherited formula as a mediated downstream consequence, not as a second code/mechanism delta.
- Optimizer differences: none. Loss differences: none. No new optimizer conditioning, auxiliary loss, behavior-loss change, Sinkhorn rule/parameter, schedule, regularizer, or second mechanism. The inherited `effective_eps` rule itself is not edited.
- Stage1 differences: none; use the S01-locked embeddings/sidecar and Stage0 inputs and identities.
- Stage3 differences: trainer, settings, seed 42, 150 epochs, beam size 20, and `n_eval=57439` are unchanged. Only the iteration-local wrapper's SID path, mechanism variant label, and output/results directory are different as required for iter29 isolation (S01 and S05 packet). Do not call those path/label changes scientific mechanisms.

## Expected implementation files and path-only hygiene

These are S06 implementation expectations from the canonical S05 packet, not edits performed by this proposal:

1. `scripts/compute_closed_form_curvature.py` — implement the exact registered map; explicitly read/validate `raw_residual_medians`; fail closed on absence/invalid values.
2. `scripts/computed_behavior_branching.json` — retain the approved branching vector and raw medians with historical source record; remove ambiguous legacy aliases/obsolete old-map fields; store canonical intermediates and fixed vector. Any writer touching this file must preserve and validate the plural raw key. Do not invoke the old behavior writer.
3. `curvature_RQ-VAE.py` — consume/validate the canonical fixed vector; remove unused legacy layer-scale fallback and old mapping constants/docs/labels. Preserve model architecture, losses, optimizer, Sinkhorn, warm-start, seed, steps, and data loading.
4. `curvature_config.py` — update only full iter29 mechanism label and iter29 result paths; keep upstream data paths, seeds, GPU, model, and runtime settings unchanged.
5. `scripts/run_stage3_iter29.py` — iteration-local wrapper with iter29 SID/output paths and full mechanism variant; Stage3 trainer/source/settings unchanged.
6. `scripts/mvg_check.py` — verify accepted vector, fixed/time-invariant curvature, and one real loss-backward path; compare same-checkpoint/same-batch curvature-consuming assignment/geometry/loss against iter26 vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]` for measurable change.
7. Remove only stale copied iter-specific Stage3 wrappers, project-local `sid_metrics_any.py`, unauthorized `calibrate_residual_scales.py`, and unused `grad_check.sh` if present; retain compatible no-argument `grad_check.py` and `export_sids_for_stage3.py`.

Changes to stored deterministic results, strict input/contract checks, iteration labels, result-directory isolation, and lightweight verification are implementation/path hygiene, not additional conceptual mechanisms. These hygiene changes must not alter any locked data input or runtime protocol dimension.

## Evidence and risks

- S01 directly locks iter26 as parent/control, its commit, protocol and exact control result; S00 establishes FCCR-1 and that the discarded draft proposing curvature-conditioned behavior-loss scheduling is not canonical and must not propagate.
- Canonical S02 hypothesis supplies the exact equation, constants, arithmetic intermediates, and output. S03 Judge/manifest approves the raw and branching values with stated historical provenance limitations and the explicit plural-key requirement. S04 Judge/contract confirms fixed-curvature FCCR-1 with `formula_inputs=["behavior_branching","raw_residual_median"]` and exact final vector; schema semantic singular does not relax the required plural data key.
- Iter26 hypothesis and one-factor record establish the previous equation, output, fixed-buffer behavior, and inherited mechanics. The S05 packet summarizes implementation paths and locked mechanisms.
- Key risk: a stale `residual_norm` alias may coincidentally equal the raw vector in the copied JSON, masking a semantic/key error. Require explicit plural-key consumption; fail closed rather than silently fallback. A second risk is accidental overwrite by the old behavior writer. Historical missing checkpoint/raw-SID evidence prevents claiming replay or byte-level reproduction. A third risk is leaking iter26 output/SID paths into iter29; isolate paths without changing the experiment.

## Self-rejection conditions

Reject this proposal or block propagation if the mapping is altered from S02/S04; either input vector/order or any constant changes; implementation uses `residual_norm`, normalized layer scales, or fallback instead of explicit `raw_residual_medians`; it claims checkpoint-level reproduction; an inherited optimizer/loss/Sinkhorn/Stage1/Stage3/protocol condition changes; or any additional scientific mechanism is introduced. A missing plural key is a hard implementation/provenance blocker, not permission to reconstruct or substitute data. No Stage2 launch is authorized by this S05 proposal.

## Autonomous next action

Judge C should adjudicate this proposal against the independent candidate and canonical S00–S04 plus primary iter26 evidence, then materialize exactly one canonical `logs/one_factor_diff_iter29.md` and the S05 Judge record. If accepted, advance to S06 implementation design with the exact mapping and key hygiene above; do not implement, validate, or train at S05.
