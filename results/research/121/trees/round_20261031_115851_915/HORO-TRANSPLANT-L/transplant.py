"""HORO-TRANSPLANT-L: horosphere-gated transplant of the deepest RQ code.

Only the deepest token may change. A move is accepted when three tests agree:
the split/merge certificate, the Busemann (horosphere) height against the frozen
child anchor, and a decrease of the curved distance to that child's Frechet mean.
Everything is read-only: the frozen Stage-2 encoder, its codebooks and its SID are
inputs, and the training split is the only source of behaviour.
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
OUTPUT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_HORO-TRANSPLANT-L"
SAMPLE = 4096
PAIR_BUDGET = 200_000
STEPS = 8


def hyperbolic(x: np.ndarray, y: np.ndarray) -> float:
    difference = float(np.sum((x - y) ** 2))
    argument = 1.0 + 2.0 * difference / max((1.0 - float(x @ x)) * (1.0 - float(y @ y)), 1e-12)
    return float(np.arccosh(max(argument, 1.0 + 1e-8)))


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = float(x @ y)
    xx = float(x @ x)
    yy = float(y @ y)
    return ((1 + 2 * xy + yy) * x + (1 - xx) * y) / (1 + 2 * xy + xx * yy)


def log_map(mean: np.ndarray, point: np.ndarray) -> np.ndarray:
    delta = mobius_add(-mean, point)
    length = float(np.linalg.norm(delta))
    if length < 1e-15:
        return np.zeros_like(mean)
    return (2.0 / max(1.0 - float(mean @ mean), 1e-12)) * np.arctanh(min(length, 1 - 1e-12)) * delta / length


def exp_map(mean: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    scale = 1.0 / max(1.0 - float(mean @ mean), 1e-12)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-15:
        return mean
    return mobius_add(mean, np.tanh(norm * scale / 2.0) * tangent / norm)


def frechet_mean(points: list[np.ndarray], steps: int = STEPS) -> np.ndarray:
    mean = points[0]
    for _ in range(steps):
        gradient = np.mean([log_map(mean, point) for point in points], axis=0)
        mean = exp_map(mean, gradient)
        radius = float(np.linalg.norm(mean))
        if radius > 1 - 1e-6:
            mean = mean * ((1 - 1e-6) / radius)
    return mean


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

    baseline = json.loads((BASELINE / "item_sids.json").read_text())
    items = sorted(int(key) for key in baseline)
    tokens = np.asarray([baseline[str(item)][:3] for item in items], dtype=np.int64)
    original = tokens.copy()
    row_of = {item: row for row, item in enumerate(items)}
    child_code = {row: int(tokens[row, 2]) for row in range(len(items))}

    # (0) behaviour from the TRAINING split only, deduplicated per user
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    pairs: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    popularity: dict[int, set] = defaultdict(set)
    training_ids: set[int] = set()
    total_pairs = 0.0
    negative = 0.0
    positive = 0.0
    for user, (history, target) in enumerate(zip(train["history"].tolist(),
                                                 train["target"].tolist())):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        training_ids.update(values)
        for item in set(values):
            popularity[item].add(user)
        visited = set()
        for first, second in zip(values[:-1], values[1:]):
            if (first, second) in visited or first not in row_of or second not in row_of:
                continue
            visited.add((first, second))
            pairs[first][second] += 1.0
            total_pairs += 1.0
            if int(tokens[row_of[first], 0]) != int(tokens[row_of[second], 0]):
                negative += 1.0
            else:
                positive += 1.0
    pop = {item: float(len(users)) for item, users in popularity.items()}
    r0 = negative / max(negative + positive, 1.0)
    bound = float(np.log((1.0 - r0) / max(r0, 1e-9)))

    # (1) the ball embedding, scale from the 4096 smallest training ids
    anchors = sorted(training_ids)[:SAMPLE]
    anchor_points = latents[anchors]
    spread = np.sqrt(((anchor_points[:, None, :] - anchor_points[None, :, :]) ** 2).sum(-1))
    upper = spread[np.triu_indices(len(anchors), k=1)]
    if len(upper) > PAIR_BUDGET:
        upper = upper[:: max(1, len(upper) // PAIR_BUDGET)]
    scale = float(np.median(upper) / 2.0)
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    points = np.tanh(norms / scale) * latents / np.maximum(norms, 1e-12)
    radii = np.linalg.norm(points, axis=1)
    over = radii > 1 - 1e-3
    points[over] *= ((1 - 1e-3) / radii[over])[:, None]

    # (2) the frozen child anchors: Frechet mean, boundary direction, horosphere height
    prefix_of = [tuple(int(value) for value in tokens[row, :2]) for row in range(len(items))]
    children: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in range(len(items)):
        children[prefix_of[row]][int(tokens[row, 2])].append(row)

    mean_of: dict[tuple, np.ndarray] = {}
    direction: dict[tuple, np.ndarray] = {}
    for prefix, table in children.items():
        for code, rows in table.items():
            mean = frechet_mean([points[items[row]] for row in rows])
            mean_of[(prefix, code)] = mean
            direction[(prefix, code)] = mean / max(float(np.linalg.norm(mean)), 1e-12)

    def busemann(key: tuple, row: int) -> float:
        point = points[items[row]]
        east = direction[key]
        return float(np.log(max(1.0 - float(point @ point), 1e-12) /
                            max(2.0 * float(np.sum((point - east) ** 2)), 1e-12)))

    support: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for row, item in enumerate(items):
        prefix = prefix_of[row]
        for target, count in pairs.get(item, {}).items():
            other = row_of[target]
            if prefix_of[other] != prefix:
                continue  # only the siblings of the SAME prefix are children
            mass = count / max(pop.get(item, 1.0) * pop.get(target, 1.0), 1e-12)
            support[(prefix, int(tokens[other, 2]))][target] += mass
    height = {(prefix, code): (float(np.median([busemann((prefix, code), row_of[target])
                                                for target in targets]))
                               if targets else 0.0)
              for (prefix, code), targets in support.items()}

    # (3) the transplant: certificate AND horosphere AND curved decrease
    moved = 0
    decisions = []
    for row, item in enumerate(items):
        prefix = prefix_of[row]
        current = int(tokens[row, 2])
        scores = {}
        for code in children[prefix]:
            if code == current:
                continue
            total = 0.0
            for target in support.get((prefix, code), {}):
                other = row_of[target]
                total += (pairs.get(item, {}).get(target, 0.0)
                          + pairs.get(target, {}).get(item, 0.0)) \
                    / max(np.sqrt(pop.get(item, 1.0) * pop.get(target, 1.0)), 1e-12)
            scores[code] = total
        if not scores:
            continue
        best = max(scores, key=lambda code: scores[code])
        if scores[best] <= 0:
            continue

        neighbours = list(pairs.get(item, {}))

        def split(code: int) -> int:
            return sum(1 for target in neighbours
                       if np.log(max(pairs[item].get(target, 0.0) * 2.0 * max(total_pairs, 1.0), 1e-12)
                                 / ((len(neighbours) + 1.0) * (pop.get(target, 1.0) + 1.0))) >= bound
                       and int(tokens[row_of[target], 2]) != code)

        def merge(code: int) -> int:
            return sum(1 for target in neighbours
                       if np.log(max(pairs[item].get(target, 0.0) * 2.0 * max(total_pairs, 1.0), 1e-12)
                                 / ((len(neighbours) + 1.0) * (pop.get(target, 1.0) + 1.0))) < bound
                       and int(tokens[row_of[target], 2]) == code)

        certificate = (split(best) <= split(current) and merge(best) <= merge(current)
                       and (split(best) < split(current) or merge(best) < merge(current)))
        gain = busemann((prefix, best), row) - height.get((prefix, best), 0.0)
        delta = (hyperbolic(points[item], mean_of[(prefix, best)]) ** 2
                 - hyperbolic(points[item], mean_of[(prefix, current)]) ** 2)
        if certificate and gain >= 0.0 and delta <= 0.0:
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
        "scale": scale,
        "train_negative_rate": r0,
        "coherence_bound": bound,
        "L3_prefixes": int(len(np.unique(tokens, axis=0))),
        "baseline_L3_prefixes": int(len(np.unique(original, axis=0))),
        "gate": "split/merge certificate with one strict decrease, Busemann height "
                "gain >= 0, curved distance^2 decrease <= 0",
        "decisions": decisions[:50],
    }
    (OUTPUT / "transplant_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("moved", "moved_fraction", "L3_prefixes",
                                             "baseline_L3_prefixes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
