# Evaluation plan: the approved candidate's single cold run (issue #112)

Read-only preparation for the one authorised official GPU critical path. Nothing here starts a
job; the run needs the operator's explicit authorisation, and nothing is reserved until then.

## The candidate

| item | value |
|:---|:---|
| mechanism | `hyp_product_l1_quantiser` (path B: the level-0 assignment metric) |
| proposal | v2, digest `b5bcfa8a11a3152c…`, frozen `b5bcfa8a…` (`results/hyperbolic_research_runs/subspace_111/subspace_mechanism_proposals.json`) |
| tree | `stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser` (sealed; `model/` byte-identical to the host) |
| declaration | `results/hyperbolic_research_runs/issue112/spec_hyp_product_l1_quantiser.json` |
| CGV | **G1–G6 pass** on the identical tree (`cgv_hyp_product_l1_quantiser.json`); G5's pooled codebook gradient matches the independent pooled reference within 9.313e-10, the ranks bitwise identical |
| H gates | complete (`join_e_hyp_product_l1_quantiser.json`); H3 in its restated form (finding F4) |
| the mechanism's mathematics | the level-0 assignment minimises `d_{c_s}^2(x_s, A_j) + d_{c_b}^2(x_b, B_j)` with `c_s = 1`, `c_b = 2` and ONE common tangent scale `s = 1.4937731…` (measured on the frozen host) |


## 0.1 What an authorised run would ACTUALLY measure (read this first)

The implementation in this tree is a **reduced** version of the frozen proposal
(`implementation_proposal_mismatch.json`): the level-0 assignment is the product-geodesic one, and
there is **no factor-role supervision** -- the mechanism's branch returns a zero term and the
executor returns None. The proposal's distinctive hypothesis ("because the two factors are
separately supervised, the behaviour term can shape the L1 partition without moving the content
factor's geometry") is therefore **not implemented**, and a prior-art search has since shown that
the operator that *is* implemented (a product-metric hyperbolic level-0 assignment) sits in crowded
territory: HiHPQ, HyperVQ, the mixed-curvature VAEs, and the differentiable product-quantisation
machinery.

So an authorised run of this tree would measure the **reduced** mechanism, whose distinctness
rests on nothing the prior art does not already cover. The role-supervision re-implementation is
queued (`hyp_product_l1_quantiser:role_supervision`) and must pass a fresh join_R and join_E before
a run is worth spending.

**The recommendation is therefore: do not authorise a run of THIS tree yet.** Either authorise the
role-supervision re-implementation once it is sealed and joined, or accept that the reduced
mechanism is a prior-art variant and close the candidate on novelty.

## 1. The single cold Stage-2 (needs the authorisation)

```bash
cd stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 train_rqvae.py
```

The trainer forks the 4-GPU DDP itself (`_launch_via_torchrun`, nproc=4, master_port 50202) ✓ — the
same entry point the CGV's G6 gate ran (48 steps, mechanism_calls 48, optimizer_steps 48,
parameter_updates 48) ✓. Frozen: seed 42, 3000 epochs, cold init, the host's architecture and
capacity, the host's data and splits.

**The reservation comes first** (`tools/harness2_genrec/dag.py`):

```python
graph.reserve("hyp_product_l1_quantiser", "stage2", seed=42, protocol="frozen-3000-epochs")
```

The reservation is the only thing that counts as a round, it is atomic, and a second attempt for
the same key returns False — so a double launch is impossible by construction rather than by
care.

Outputs (the tree's own paths): `results/stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser/` with
`rqvae_best.pth`, `item_sids.json`, `sids_for_hgrec.npy` ✓.

**Guardrails that apply**: no warm start, no baseline or control re-run, no second seed, no
curvature or weight sweep, no re-run of any historical arm, and the tree is not touched after the
launch (the sealed tree's hash is recorded with the reservation).

## 2. The frozen ER and D2, in ONE evaluation

The campaign's own analysis reads the exported `item_sids.json` and the train-split pairs; the
records it must produce for the arm are: the Hard-L1 ER **with both shares** (the numerator
positive share and the denominator negative share) and their CI, the level-0 largest cell, the
number of used codes, the Gini, the prefix branching (L1→L2 and L2→L3), and the conditional
level-2 discrimination with its support and a `low_support` flag when the support is thin.

The frozen thresholds: ER **> 16.38953942232631**, and the D2 is diagnostic — never a veto.

**The L0 tail is the thing to read first**: the pre-check applied the product metric to the
host's Euclidean-trained codebook, so the trained arm is the first real evidence on whether the
largest cell falls and the distinct-code count holds. The pre-check's numbers (the host's largest
cell 41 → the product metric's 58 on the same codebook) are a geometry probe, not a prediction.

## 3. The frozen G_H1 probe (once, train data only)

```bash
python tools/harness2_genrec/history_probe.py \
    --mechanism-id hyp_product_l1_quantiser \
    --sid results/stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser/sids_for_hgrec.npy \
    --device cuda
```

**The probe reads the `.npy` export, not `item_sids.json`** — a rehearsal against the archived
control showed the JSON export cannot be loaded by the probe's loader, and the plan said
otherwise until the rehearsal caught it.

The probe is defined by the frozen configuration (history length 20, embedding and hidden 128, one
layer, 6 epochs, batch 512, Adam 1e-3, a user-disjoint train/dev split, seed 42) and it computes
`G_H1 = (CE_train_only_frequency_prior − CE_GRU) / CE_train_only_frequency_prior` on the held-out
development users. It never reads the frozen evaluation pairs, and it is **not** replaced by any
train-time auxiliary loss.

**G_H1 is only comparable at matched occupancy** (the Case C finding: a degenerate two-code
partition makes the probe maximal), so the L0 occupancy must be reported next to it.

## 4. Stage 3 — only if the ER gate is cleared

One official Stage-3 run, the original protocol, the original seed, on the arm's own exported SIDs.
Shards are recoverable artifacts: if rank 0..3 shards exist and are complete, the metrics are
recovered from them and no inference or training is repeated.

## 5. The stop condition

One predicate, over the SAME arm: ER strictly above `16.38953942232631` **and** Recall@10 strictly
above `0.06406796775709883` **and** NDCG@10 strictly above `0.03530651701940677`, with complete
audited artifacts. An ER-only win is not a win; a Stage-3-only win is not a win; the mission stays
open until the user stops it.

## 6. What is deliberately NOT done

* no baseline or control re-run — the archived control (seed 42) is the comparison;
* no second seed, no ensemble, no curvature sweep, no weight tuning after seeing the ER;
* no Stage-3 before the ER gate;
* no second cold run of this mechanism under any name — a failed arm is recorded and the portfolio
  moves on.

## 7. The open risk, stated before the run

The candidate's own reason for existing is that the product metric changes the L0 partition
*without* the behaviour term coarsening it. The pre-check shows the assignment does change (19.35%)
and that the two factors are not a rotation (they disagree on half the items) ✓ — but neither says
what the *trained* partition looks like. If the trained largest cell grows the way the cohort's
high-ER arms grew theirs (237 → 374 in Case A), the mechanism will have reproduced the campaign's
central failure: a better ER and a worse downstream. The L0 tail is therefore the first number to
read, before the ER.
