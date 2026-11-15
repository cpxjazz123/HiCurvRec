"""Frozen-checkpoint curvature sweep on L1, read per ORCA role.

No model is trained and no hyper-parameter is fitted: the encoder output (`TIGER_z.npy`) and the three
frozen codebooks are fixed, and only the geometry used to pick the **level-1** code changes:

    c = 0      Euclidean argmin on z                      (validated: reproduces the published SID 100%)
    c > 0      Poincare distance of curvature -c on z mapped into the ball of radius 1/sqrt(c)
               u = R * tanh(||x|| / 2) * x / ||x||,  R = 1 / sqrt(c)
               d_c(u, v) = (1/sqrt(c)) arcosh(1 + 2c ||u-v||^2 / ((1-c||u||^2)(1-c||v||^2)))

Levels 2 and 3 keep the original scheme: Euclidean argmin, on the residual recomputed after the new
level-1 code.  Everything is a counterfactual *readout* of the frozen model, not a retrained geometry.

Per curvature and per ORCA role (K = 6, the cached assignment):

    change share    share of the role's items whose three-token SID differs from the c = 0 SID
    P5 / T5 rate    the path and branch conditions of section 5.23, anchor-based, per-item first
    gain            the role's pattern-set satisfaction rate minus its degree x popularity null
    collision       distinct triple share over the whole catalogue (not per role)

    python3 results/research/issue215/curvature_role_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "analysis_sid_recall"))

import role_pattern_gain as rpg  # noqa: E402

VECTORS = HERE / "vectors"
CKPT = REPO / "analysis_sid_recall" / "r2" / "24" / "rqvae_best.pth"
OUT = HERE / "curvature_role_sweep.json"
CURVATURES = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0)
ROLES = range(6)
TARGET_PATTERNS = ("P5_centre", "P5_hops", "T5")
SEED = 42


def codebooks() -> list[np.ndarray]:
    state = torch.load(CKPT, map_location="cpu")["state_dict"]
    return [state[f"rq.vq_layers.{level}.embed.weight"].numpy().astype(np.float64)
            for level in range(3)]


#: Radial scale of the fixed embedding inside the unit ball.  It must not depend on c: scaling the
#: points by 1/sqrt(c) as well would make the whole construction isometric to curvature 1 and no
#: assignment could ever change.  With s = 0.05 the constraint c ||p||^2 < 1 holds up to c = 400.
BALL_SCALE = 0.05


def to_ball(vectors: np.ndarray, curvature: float | None = None) -> np.ndarray:
    """Map into the unit ball with a fixed tanh radial profile, independent of the curvature."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    direction = vectors / np.maximum(norms, 1e-12)
    return BALL_SCALE * np.tanh(norms / 2.0) * direction


def hyperbolic_distance(points: np.ndarray, codes: np.ndarray, curvature: float) -> np.ndarray:
    """Poincare distance between every point and every code, curvature -c.

    d_c(u, v) = (1/sqrt(c)) arcosh(1 + 2c ||u-v||^2 / ((1-c||u||^2)(1-c||v||^2))), with ||u||<1/sqrt(c).
    """
    squared_points = (points ** 2).sum(axis=1)[:, None]
    squared_codes = (codes ** 2).sum(axis=1)[None, :]
    cross = points @ codes.T
    euclid = np.maximum(squared_points + squared_codes - 2 * cross, 0.0)
    denominator = (1 - curvature * squared_points) * (1 - curvature * squared_codes)
    argument = 1 + 2 * curvature * euclid / np.maximum(denominator, 1e-12)
    return np.arccosh(np.maximum(argument, 1.0)) / np.sqrt(curvature)


def quantise(z: np.ndarray, books: list[np.ndarray], curvature: float) -> np.ndarray:
    tokens = np.zeros((z.shape[0], 3), dtype=np.int64)
    residual = z.copy()
    for level, book in enumerate(books):
        if level == 0 and curvature > 0:
            distances = hyperbolic_distance(to_ball(residual, curvature),
                                            to_ball(book, curvature), curvature)
        else:
            distances = ((residual[:, None, :] - book[None, :, :]) ** 2).sum(axis=2)
        choice = distances.argmin(axis=1)
        tokens[:, level] = choice
        residual = residual - book[choice]
    return tokens


def item_means(values: np.ndarray, incidence: dict[int, list[int]], n: int) -> np.ndarray:
    items = sorted(incidence)
    item_index = np.concatenate([np.full(len(incidence[item]), item, dtype=np.int64)
                                 for item in items])
    inst_index = np.concatenate([np.asarray(incidence[item], dtype=np.int64) for item in items])
    counts = np.bincount(item_index, minlength=n)
    sums = np.bincount(item_index, weights=values[inst_index], minlength=n)
    means = np.full(n, np.nan)
    present = counts > 0
    means[present] = sums[present] / counts[present]
    return means


def main() -> int:
    books = codebooks()
    z = np.load(VECTORS / "TIGER_z.npy").astype(np.float64)
    published = np.load(VECTORS / "TIGER_tokens.npy").astype(np.int64)
    n = z.shape[0]
    print(f"{n} items, codebooks {[book.shape for book in books]}", flush=True)

    role_of = np.full(n, -1, dtype=np.int64)
    cached = np.load(REPO / "analysis_sid_recall" / "structure_roles5.npz")
    for node, role in zip(cached["nodes"], cached["role"]):
        if 0 <= int(node) < n:
            role_of[int(node)] = int(role)
    sizes = {role: int((role_of == role).sum()) for role in ROLES}
    print("role sizes:", sizes, flush=True)

    prepared = rpg.prepare()
    subsets = prepared["subsets"]
    incidences = prepared["incidences"]
    patterns_by_role = prepared["patterns_by_role"]

    def lcp_distance(level0: np.ndarray, level1: np.ndarray, level2: np.ndarray):
        def distance(left: int, right: int) -> float:
            count = 0
            for table in (level0, level1, level2):
                if table[left] != table[right]:
                    break
                count += 1
            return 1.0 - count / 3.0
        return distance

    rows = []
    for curvature in CURVATURES:
        tokens = quantise(z, books, curvature)
        match = float(np.mean(np.all(tokens == published, axis=1)))
        triples = {tuple(int(value) for value in row) for row in tokens}
        collision = 1.0 - len(triples) / n
        changed = np.any(tokens != published, axis=1)
        distance = lcp_distance(tokens[:, 0], tokens[:, 1], tokens[:, 2])

        per_pattern = {}
        for name, (chosen, nulls) in subsets.items():
            real_values, null_values = [], []
            for instance, draws in zip(chosen, nulls):
                if not draws:
                    continue
                computed = rpg.per_instance(name, instance, draws, distance)
                if computed is None:
                    continue
                real_values.append(computed[0])
                null_values.append(computed[1])
            per_pattern[name] = (np.asarray(real_values), np.asarray(null_values), incidences[name])

        entry = {"curvature": curvature, "matches_published": match, "collision": collision,
                 "changed_share": float(changed.mean()), "roles": {}}
        for role in ROLES:
            member = role_of == role
            entry["roles"][str(role)] = {
                "items": int(member.sum()),
                "changed_share": float(changed[member].mean()) if member.any() else None}
            rates = {}
            for name in patterns_by_role[role]:
                real_values, null_values, incidence = per_pattern[name]
                if real_values.size == 0:
                    continue
                means_real = item_means(real_values, incidence, n)
                means_null = item_means(null_values, incidence, n)
                mask = member & np.isfinite(means_real)
                if not mask.any():
                    continue
                rates[name] = {"real": float(np.nanmean(means_real[mask])),
                               "null": float(np.nanmean(means_null[mask]))}
            entry["roles"][str(role)]["patterns"] = rates
            if rates:
                entry["roles"][str(role)]["gain"] = float(
                    np.mean([value["real"] - value["null"] for value in rates.values()]))
                entry["roles"][str(role)]["mean_rate"] = float(
                    np.mean([value["real"] for value in rates.values()]))
            else:
                entry["roles"][str(role)]["gain"] = None
                entry["roles"][str(role)]["mean_rate"] = None
        rows.append(entry)
        print(f"c={curvature:<6g} published-match {match:.4f}  collision {collision:.4f}  "
              f"changed {changed.mean():.4f}", flush=True)

    result = {"curvatures": list(CURVATURES), "role_sizes": sizes, "rows": rows,
              "note": "frozen encoder and codebooks; only the level-1 geometry changes; levels 2 and 3 "
                      "are Euclidean on recomputed residuals; c=0 reproduces the published SID"}
    OUT.write_text(json.dumps(result, indent=1, default=float))

    print("\n=== per role: change share / P5 rate / T5 rate / gain, by curvature ===")
    print(f"  {'role':<6}" + "".join(f"{f'c={curvature:g}':>22}" for curvature in CURVATURES))
    for role in ROLES:
        line = f"  R{role + 1:<5}"
        for entry in rows:
            block = entry["roles"][str(role)]
            p5 = block["patterns"].get("P5_centre", {}).get("real")
            t5 = block["patterns"].get("T5", {}).get("real")
            line += (f"{block['changed_share']:.3f}/"
                     f"{(p5 if p5 is not None else float('nan')):.3f}/"
                     f"{(t5 if t5 is not None else float('nan')):.3f}/"
                     f"{(block['gain'] if block['gain'] is not None else float('nan')):.3f}").rjust(22)
        print(line)
    print("\n  column = changed share / P5 rate / T5 rate / gain")
    print(f"\nwritten {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
