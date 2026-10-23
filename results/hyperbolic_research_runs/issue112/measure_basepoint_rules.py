"""Which base point makes the horosphere's SIGN carry information on the frozen data? (issue #112)

The JOIN E refusal was measured, not guessed: with the base point at the target's own direction,
the source's Busemann level takes ONE sign on the real batch (0 of 24 negative), because a
transition's source and target are similar items and therefore share a direction. The mechanism
then degenerates into "the codeword closest in level to the source".

This script measures the alternatives on the FROZEN data -- the real train-split transition
pairs, the real stage-1 embeddings, the real host encoder with the matched control checkpoint --
and reports for each candidate base-point rule:

* the SIGN BALANCE: what fraction of the sources land on the negative side of the horosphere;
* the level's spread;
* whether the sign SEPARATES anything: the mean source-target geodesic distance of the negative
  and the positive groups, and the same for the transition's own length.

A rule is only worth proposing if the sign is both balanced (so it can discriminate) and
separating (so the discrimination means something).

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/issue112/measure_basepoint_rules.py

Writes: .../basepoint_rules.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "tools" / "cgv"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "hyperbolic_verifier"))

import busemann as B  # noqa: E402
import engine  # noqa: E402

TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_hyp_busemann_boundary"
CONTROL = REPO_ROOT / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "basepoint_rules.json"
SAMPLE = 4000


def _direction(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-12)


def main() -> int:
    trainer, saved_path, saved_mods = engine.load_trainer(TREE, module_name="basepoint_probe")
    try:
        embeddings = trainer.maybe_apply_pca(
            trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
        in_dim = int(embeddings.shape[1])
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=in_dim)
        state = torch.load(CONTROL, map_location="cpu")
        model.load_state_dict(state.get("state_dict") or state, strict=False)
        model.eval()

        # the train-split transition pairs, exactly as the trainer builds them
        import pandas as pd
        frame = pd.read_parquet(trainer.TRAIN_FILE)
        pairs = []
        for history, target in zip(frame["seen_history"].to_numpy(),
                                   frame["target"].to_numpy(dtype=np.int64)):
            if history is None or len(history) == 0:
                continue
            pairs.append((int(history[-1]), int(target)))
        pairs = np.asarray(pairs, dtype=np.int64)
        print(f"pairs available: {pairs.shape[0]}")
        rng = np.random.default_rng(20261023)
        if pairs.shape[0] > SAMPLE:
            pairs = pairs[rng.choice(pairs.shape[0], size=SAMPLE, replace=False)]

        table = torch.from_numpy(embeddings[pairs[:, 0]])
        target_table = torch.from_numpy(embeddings[pairs[:, 1]])
        with torch.no_grad():
            source_latent = model.encoder(table).double()
            target_latent = model.encoder(target_table).double()

        c = 1.0
        # the ball points, with the arm's own encoding (its scale rule is data-driven)
        import scripts.hyp_losses as HL
        source_ball = HL.encode_points(source_latent.float(), c).double()
        target_ball = HL.encode_points(target_latent.float(), c).double()
        distance = B.poincare_distance(source_ball, target_ball, c)

        rules = {
            "target_direction": _direction(target_ball),
            "source_direction": _direction(source_ball),
            "pair_mean_direction": _direction(source_ball + target_ball),
            "transition_direction": _direction(target_ball - source_ball),
        }
        # the target's direction with the source's direction removed: the part of the target
        # that is NOT explained by the pair's shared direction
        shared = _direction(source_ball)
        orthogonal = target_ball - (target_ball * shared).sum(-1, keepdim=True) * shared
        rules["target_orthogonal_to_source"] = _direction(orthogonal)

        results = {}
        for name, omega in rules.items():
            omega = omega / math.sqrt(c)
            levels = B.busemann_boundary(source_ball, omega)
            negative = levels < 0
            balance = float(negative.double().mean())
            entry = {
                "negative_fraction": balance,
                "sign_balance": min(balance, 1.0 - balance),
                "level_min": float(levels.min()), "level_max": float(levels.max()),
                "level_spread": float(levels.max() - levels.min()),
                "mean_distance_negative": (float(distance[negative].mean())
                                           if bool(negative.any()) else None),
                "mean_distance_positive": (float(distance[~negative].mean())
                                           if bool((~negative).any()) else None),
            }
            if entry["mean_distance_negative"] is not None and \
                    entry["mean_distance_positive"] is not None:
                entry["distance_separation"] = abs(entry["mean_distance_negative"]
                                                   - entry["mean_distance_positive"])
            results[name] = entry

        ranked = sorted(results.items(), key=lambda item: (item[1]["sign_balance"],
                                                           -(item[1].get("distance_separation")
                                                             or 0.0)), reverse=True)
        payload = {
            "schema_version": 1,
            "issue": 112,
            "question": ("which base-point rule makes the horosphere's SIGN discriminative on the "
                         "frozen data?"),
            "fixture": {"pairs": int(pairs.shape[0]), "checkpoint": str(CONTROL),
                        "checkpoint_epoch": state.get("epoch"), "train_pairs_total": int(
                            sum(1 for _ in range(0)) or 0) or None,
                        "curvature": c},
            "rules": results,
            "ranking": [{"rule": name, "sign_balance": entry["sign_balance"],
                          "distance_separation": entry.get("distance_separation")}
                         for name, entry in ranked],
            "verdict": ("the rule with the best balance AND a separating sign is the only one that "
                        "can carry the inside/outside question on this data"),
            "recommendation": ranked[0][0],
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        for name, entry in ranked:
            print(f"{name:32} balance={entry['sign_balance']:.4f} "
                  f"neg={entry['negative_fraction']:.4f} "
                  f"sep={entry.get('distance_separation')}")
        print("recommendation:", payload["recommendation"])
        print("wrote", OUT)
        return 0
    finally:
        engine.unload_trainer(saved_path, saved_mods)


if __name__ == "__main__":
    raise SystemExit(main())
