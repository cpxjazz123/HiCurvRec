"""The RQ chain's information structure at each level -- the baseline a candidate must beat.

The level-wise pre-check showed that swapping the metric on a codebook trained for another metric
cannot change the TREE. The tree's structure is therefore a property of the TRAINED codebook, and
the question is what in the RESIDUAL makes the deeper levels redundant.

This measures that, on the frozen host with the matched control checkpoint:

* the residual's norm at each level, and how much of it the level's codebook removes;
* the distinct codes each level uses (is the codebook the bottleneck, or the residual?);
* the conditional redundancy: the fraction of items whose level-2 code is DETERMINED by their
  (level-1, level-2) cell -- this is the number a mechanism that enriches the third token has to
  move, and the number the approved candidate's run will be read against.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/issue112/probe_residual_information.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools/cgv"))
import engine
TREE = REPO / "stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser"
CONTROL = REPO / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "residual_information_baseline.json"
SAMPLE = 4000

trainer, saved, mods = engine.load_trainer(TREE, module_name="residual_probe")
try:
    emb = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(emb.shape[1]))
    state = torch.load(CONTROL, map_location="cpu"); model.load_state_dict(state.get("state_dict") or state, strict=False); model.eval()
    rng = np.random.default_rng(20261023)
    rows = rng.choice(emb.shape[0], size=min(SAMPLE, emb.shape[0]), replace=False)
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(emb[rows])).double()
    codes = [layer.get_code_embs().double() for layer in model.rq.vq_layers]
    residual = z.clone()
    levels = []
    for level, codebook in enumerate(codes):
        dist = torch.cdist(residual, codebook)
        idx = dist.argmin(-1)
        best = dist.gather(1, idx[:, None]).squeeze(1)
        before = residual.norm(dim=-1)
        residual = residual - codebook[idx]
        after = residual.norm(dim=-1)
        levels.append({
            "level": level,
            "residual_norm_before": float(before.mean()),
            "residual_norm_after": float(after.mean()),
            "removed_fraction": float((1.0 - (after / before.clamp_min(1e-12)).mean())),
            "distinct_codes_used": int(len(set(idx.tolist()))),
            "codebook_size": int(codebook.shape[0]),
            "mean_distance_to_the_chosen_codeword": float(best.mean()),
        })
    l1, l2, l3 = [level["assignment"] for level in []] or (None, None, None)
    # the conditional redundancy of the third token
    residual = z.clone()
    assignments = []
    for level, codebook in enumerate(codes):
        idx = torch.cdist(residual, codebook).argmin(-1)
        assignments.append(idx)
        residual = residual - codebook[idx]
    cells = {}
    for a, b, c in zip(assignments[0].tolist(), assignments[1].tolist(), assignments[2].tolist()):
        cells.setdefault((a, b), set()).add(c)
    per_cell = [len(v) for v in cells.values()]
    determined = sum(1 for v in per_cell if v == 1)
    payload = {
        "schema_version": 1, "issue": 112,
        "purpose": ("the baseline a candidate that enriches the third token has to move: the RQ "
                    "chain's information structure at each level, measured on the frozen host"),
        "items": int(len(rows)), "checkpoint": str(CONTROL), "checkpoint_epoch": state.get("epoch"),
        "levels": levels,
        "third_token": {
            "distinct_l1_l2_cells": len(cells),
            "mean_l3_codes_per_cell": float(sum(per_cell) / max(len(per_cell), 1)),
            "fraction_of_cells_with_a_single_l3_code": determined / max(len(cells), 1),
            "note": ("a cell whose third token is already determined carries no information: that "
                     "fraction is the redundancy a mechanism must move, and the number the approved "
                     "candidate's run will be read against"),
        },
        "normalisation_caveat": ("the cohort's prefix_tree.l2_to_l3_branches near 1.47 is NOT the "
                                 "normalisation used here (the mean L3 codes per (L1,L2) cell); the "
                                 "two are not compared"),
    }
    OUT.write_text(json.dumps(payload, indent=2) + chr(10))
    print(json.dumps(payload, indent=1))
finally:
    engine.unload_trainer(saved, mods)
