"""HYP-CERT-LEAFREPAIR: certificate-gated repair of the deepest SID code.

The first L-1 tokens of every item stay bit-identical; only the deepest code may
change, and only when a training-split certificate proves the move does not add a
behaviourally coherent cross-cell pair (a "split") nor a behaviourally disjoint
same-cell pair (a "merge"). The move is scored by a behaviour-displaced Poincare
point: the item's residual is lifted into the ball around its frozen prefix
centroid, displaced towards the weighted gyro-midpoint of its success
neighbourhood, and compared with the prefix-local deepest-cell centres.

Nothing is trained; the frozen Stage-2 checkpoint and its SID are read-only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
PARQUET = REPO / "results" / "stage0_build_parquet"
TEXT = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
OUTPUT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_HYP-CERT-LEAFREPAIR"
N_ITEMS = 24587
MASS = 0.90
MASS_CAP = 40
SCALE = 7.6


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = float(x @ y)
    xx = float(x @ x)
    yy = float(y @ y)
    return ((1 + 2 * xy + yy) * x + (1 - xx) * y) / (1 + 2 * xy + xx * yy)


def geodesic(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """The geodesic from ``x`` to ``y`` at parameter ``t`` in the unit ball."""
    return mobius_add(x, t * mobius_add(-x, y))


def gyro_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 2.0 * np.arctanh(min(float(np.linalg.norm(mobius_add(-x, y))), 1.0 - 1e-12))


def exponential(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(tangent))
    if length < 1e-15:
        return base
    lam = 1.0 / (1.0 - float(base @ base))
    return mobius_add(base, np.tanh(length * lam / 2.0) * tangent / length)


def interior(point: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    length = float(np.linalg.norm(point))
    if length < 1e-12:
        return point
    return (length / (1.0 + length)) * fraction ** 0.0 * (point / length) \
        if False else (length / (1.0 + length)) * (point / length)


def main() -> int:
    import train_rqvae as trainer
    from model.model import RQVAE

    text = np.load(TEXT).astype(np.float32)
    config = trainer._tokenizer_config()
    model = RQVAE(config, in_dim=text.shape[1])
    state = torch.load(BASELINE / "rqvae_best.pth", map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        latents = model.encoder(torch.from_numpy(text)).double().numpy()
    deepest = state["rq.vq_layers.2.embed.weight"].double().numpy()

    baseline = json.loads((BASELINE / "item_sids.json").read_text())
    items = sorted(int(key) for key in baseline)
    tokens = np.asarray([baseline[str(item)][:3] for item in items], dtype=np.int64)
    original = tokens.copy()
    row_of = {item: row for row, item in enumerate(items)}

    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    pairs: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    degree = defaultdict(float)
    total_pairs = 0.0
    negative = 0.0
    positive = 0.0
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for first, second in zip(sequence[:-1], sequence[1:]):
            pairs[first][second] += 1.0
            pairs[second][first] += 1.0
            degree[first] += 1.0
            degree[second] += 1.0
            total_pairs += 1.0
            if row_of.get(first) is None or row_of.get(second) is None:
                continue
            if tokens[row_of[first], 0] != tokens[row_of[second], 0]:
                negative += 1.0
            else:
                positive += 1.0
    r0 = negative / max(negative + positive, 1.0)
    bound = float(np.log((1.0 - r0) / max(r0, 1e-9)))
    n_pair = max(total_pairs, 1.0)

    def log_ratio(first: int, second: int) -> float:
        count = pairs[first].get(second, 0.0)
        return float(np.log(count * 2.0 * n_pair /
                            ((degree[first] + 1.0) * (degree[second] + 1.0))))

    sizes = []
    neighbourhood: dict[int, list[int]] = {}
    for item, table in pairs.items():
        ordered = sorted(table.items(), key=lambda pair: -pair[1])
        mass = sum(count for _, count in ordered)
        running = 0.0
        chosen = []
        for target, count in ordered:
            chosen.append(target)
            running += count
            if running >= MASS * mass:
                break
        neighbourhood[item] = chosen[:MASS_CAP]
        sizes.append(len(neighbourhood[item]))
    cap = int(np.percentile(sizes, 95)) if sizes else MASS_CAP
    neighbourhood = {item: targets[:cap] for item, targets in neighbourhood.items()}

    prefix_of = [tuple(int(value) for value in tokens[row, :2]) for row in range(len(items))]
    centroid: dict[tuple, np.ndarray] = {}
    for prefix in set(prefix_of):
        members = [items[row] for row, value in enumerate(prefix_of) if value == prefix]
        centroid[prefix] = latents[members].mean(axis=0)
    residual = np.stack([latents[items[row]] - centroid[prefix_of[row]]
                         for row in range(len(items))])
    alpha = 1.0 / max(float(np.linalg.norm(latents, axis=1).max()), 1e-12)
    scale = SCALE / max(float(np.linalg.norm(residual, axis=1).max()), 1e-12)

    def phi(point: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(point))
        return np.tanh(alpha * length) * point / max(length, 1e-12)

    def lift(row: int) -> np.ndarray:
        return exponential(phi(centroid[prefix_of[row]]), scale * residual[row])

    centres: dict[tuple, np.ndarray] = {}
    for prefix in set(prefix_of):
        base = phi(centroid[prefix])
        for code in range(deepest.shape[0]):
            centres[(prefix, code)] = interior(exponential(base, scale * deepest[code]))

    moved = 0
    decisions = []
    for row in sorted(range(len(items)), key=lambda r: -degree[items[r]]):
        item = items[row]
        others = [row_of[target] for target in neighbourhood.get(item, []) if target in row_of]
        if not others:
            continue
        current = int(tokens[row, 2])
        candidates = sorted({int(tokens[other, 2]) for other in others} - {current})
        if not candidates:
            continue
        weights = np.asarray([max(0.0, log_ratio(item, items[other])) for other in others])
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        points = [lift(row)] + [lift(other) for other in others]
        masses = [1.0] + weights.tolist()
        displaced = points[0]
        accumulated = masses[0]
        for point, mass in zip(points[1:], masses[1:]):
            displaced = geodesic(displaced, point, mass / (accumulated + mass))
            accumulated += mass
        scores = {}
        for code in candidates:
            mass = float(sum(weight for other, weight in zip(others, weights)
                             if int(tokens[other, 2]) == code))
            scores[code] = mass * float(np.exp(-gyro_distance(displaced, centres[(prefix_of[row], code)])))
        if not scores:
            continue
        best = max(scores, key=lambda code: scores[code])

        def split(code: int) -> int:
            return sum(1 for other in others
                       if log_ratio(item, items[other]) >= bound
                       and int(tokens[other, 2]) != code)

        def merge(code: int) -> int:
            return sum(1 for other in others
                       if log_ratio(item, items[other]) < bound
                       and int(tokens[other, 2]) == code)

        peers = [other for other in others if int(tokens[other, 2]) == best]
        radius = max((float(np.linalg.norm(residual[other] - deepest[best]))
                      for other in peers), default=0.0)
        inside = float(np.linalg.norm(residual[row] - deepest[best])) <= radius
        if (split(best) <= split(current) and merge(best) <= merge(current)
                and (split(best) < split(current) or merge(best) < merge(current))
                and inside):
            tokens[row, 2] = best
            moved += 1
            decisions.append({"item": item, "from": current, "to": best})

    assert np.array_equal(tokens[:, :2], original[:, :2]), "levels 1..L-1 changed"
    groups: dict[tuple, list[int]] = defaultdict(list)
    for row, item in enumerate(items):
        groups[tuple(int(value) for value in tokens[row])].append(item)
    width = len(next(iter(baseline.values()))) - 3
    token = {}
    for members in groups.values():
        for position, item in enumerate(sorted(members)):
            token[item] = [0] * (width - 1) + [position]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "item_sids.json").write_text(json.dumps(
        {str(item): [int(value) for value in tokens[row]] + token[item]
         for row, item in enumerate(items)}))
    report = {
        "moved": moved,
        "moved_fraction": moved / len(items),
        "train_negative_rate": r0,
        "coherence_bound": bound,
        "levels_1_to_L_minus_1_unchanged": True,
        "L3_prefixes": int(len(np.unique(tokens, axis=0))),
        "baseline_L3_prefixes": int(len(np.unique(original, axis=0))),
        "certificate": "dSplit and dMerge non-increasing with one strict decrease and the "
                       "item inside the target cell radius",
        "decisions": decisions[:50],
    }
    (OUTPUT / "leafrepair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("moved", "moved_fraction", "L3_prefixes",
                                             "baseline_L3_prefixes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
