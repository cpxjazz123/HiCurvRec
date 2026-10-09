"""Faithful continuous-cone reproduction on the ICML-2018 reference data.

Implements Ganea, Becigneul & Hofmann (2018) as published:

* angle ``xi`` per Eqs. 28 (Poincare) and 31 (Euclidean), *without* the
  supplement flip that ``bench_hier_behavior.cones.poincare_xi_pairs`` applies;
* aperture ``psi(x) = arcsin(K * g(x))`` with the paper's constant ``K`` and
  ``g`` = ``(1-||x||^2)/||x||`` (Poincare) or ``1/||x||`` (Euclidean);
* loss ``E(u,v) + sum_i max(0, margin - E(u,v_i))`` with ``E = max(0, xi-psi)``;
* Riemannian gradient descent ``u <- u - lr*(1-||u||^2)^2/4 * grad`` for the
  Poincare arm, plain SGD for the Euclidean arm, both followed by the paper's
  projection (annulus for Poincare, lower norm bound for Euclidean);
* negatives drawn uniformly among non-children, ten per positive, refreshed
  every epoch; the near-universal-subsumer fallback matches the reference;
* F1 with the decision threshold tuned on the validation split and then applied
  unchanged to the test split.

Data are the authors' released splits, so the 0/10/25/50% supervision axis
matches the paper exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import semantic_config as cfg

EPS = 1e-7


@dataclass(frozen=True)
class PaperConeData:
    name: str
    ratio: str
    n_nodes: int
    train_u: np.ndarray
    train_v: np.ndarray
    adjacency_keys: np.ndarray  # sorted int64 keys u * n_nodes + c over adjacency
    adjacency_size: np.ndarray  # (n_nodes,) number of blocked partners
    uniform_nodes: np.ndarray  # nodes whose adjacency blocks the whole catalogue
    valid_u: np.ndarray
    valid_v: np.ndarray
    valid_neg_u: np.ndarray
    valid_neg_v: np.ndarray
    test_u: np.ndarray
    test_v: np.ndarray
    test_neg_u: np.ndarray
    test_neg_v: np.ndarray

    @property
    def n_train(self) -> int:
        return int(len(self.train_u))


def load_structure(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Longest-path depth and closure out-degree per node.

    The released closure file is the full transitive closure, so depth is the
    longest hypernym chain and out-degree is the number of descendants. Both
    come straight from the data the reference ships, with no extra tree file.
    """
    index = {}
    with (data_dir / f"{name}_closure.tsv.vocab").open() as handle:
        for line in handle:
            position, node = line.rstrip("\n").split("\t")
            index[node] = int(position)
    pairs = []
    with (data_dir / f"{name}_closure.tsv").open() as handle:
        for line in handle:
            parent, child = line.rstrip("\n").split("\t")
            # The closure file also holds nodes above the retained vocabulary
            # (the reference removes the top of the tree before splitting).
            if parent not in index or child not in index:
                continue
            pairs.append((index[parent], index[child]))
    table = np.asarray(pairs, dtype=np.int64)
    n_nodes = len(index)
    # The name-keyed closure file and the index-keyed splits were produced by
    # different tools, so the column order is settled against the index-keyed
    # transitive closure rather than assumed.
    reference_parent, reference_child = _load_pairs(
        data_dir / f"{name}_closure.tsv.full_transitive"
    )
    reference = set(
        (int(u), int(v))
        for u, v in zip(reference_parent[:400_000], reference_child[:400_000])
    )
    sample = table[:20_000]
    forward = sum(1 for u, v in sample if (int(u), int(v)) in reference)
    backward = sum(1 for u, v in sample if (int(v), int(u)) in reference)
    if backward > forward:
        table = table[:, ::-1]
    parents, children = table[:, 0], table[:, 1]
    out_degree = np.bincount(parents, minlength=n_nodes).astype(np.int64)
    # Longest hypernym chain. WordNet carries a handful of tiny cycles, so the
    # traversal drops intra-cycle edges and reports the nodes it had to skip.
    remaining = np.bincount(children, minlength=n_nodes).astype(np.int64)
    depth = np.zeros(n_nodes, dtype=np.int64)
    frontier = np.flatnonzero(remaining == 0)
    visited = 0
    while frontier.size:
        remaining[frontier] = -1
        mask = np.isin(parents, frontier)
        source = parents[mask]
        target = children[mask]
        depth[target] = np.maximum(depth[target], depth[source] + 1)
        np.subtract.at(remaining, target, 1)
        visited += int(frontier.size)
        frontier = np.flatnonzero(remaining == 0)
    cyclic = np.flatnonzero(remaining >= 0)
    depth[cyclic] = -1
    return depth + 1, out_degree, int(cyclic.size)


def _load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, dtype=np.int64, delimiter="\t")
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"Relation file {path} is not a pair table")
    return table[:, 0], table[:, 1]


def load_dataset(
    data_dir: Path, name: str, ratio: str
) -> PaperConeData:
    vocab_path = data_dir / f"{name}_closure.tsv.vocab"
    n_nodes = sum(1 for _ in vocab_path.open())
    base = data_dir / f"{name}_closure.tsv"
    train_u, train_v = _load_pairs(Path(str(base) + f".train_{ratio}"))
    valid_u, valid_v = _load_pairs(Path(str(base) + ".valid"))
    valid_neg_u, valid_neg_v = _load_pairs(Path(str(base) + ".valid_neg"))
    test_u, test_v = _load_pairs(Path(str(base) + ".test"))
    test_neg_u, test_neg_v = _load_pairs(Path(str(base) + ".test_neg"))
    for array in (train_u, train_v, valid_u, valid_v, test_u, test_v):
        if array.min() < 0 or array.max() >= n_nodes:
            raise ValueError(f"Node index outside the vocabulary of {name}")

    # where_not_to_sample='children': a parent blocks its training children and
    # itself, which is exactly the adjacency built by the reference loader.
    adjacency_keys = np.unique(
        np.concatenate(
            [
                train_u.astype(np.int64) * n_nodes + train_v.astype(np.int64),
                np.arange(n_nodes, dtype=np.int64) * n_nodes
                + np.arange(n_nodes, dtype=np.int64),
            ]
        )
    )
    adjacency_size = np.bincount(train_u, minlength=n_nodes).astype(np.int64) + 1
    uniform_nodes = np.flatnonzero(adjacency_size >= n_nodes)
    return PaperConeData(
        name=name,
        ratio=ratio,
        n_nodes=int(n_nodes),
        train_u=train_u,
        train_v=train_v,
        adjacency_keys=adjacency_keys,
        adjacency_size=adjacency_size,
        uniform_nodes=uniform_nodes,
        valid_u=valid_u,
        valid_v=valid_v,
        valid_neg_u=valid_neg_u,
        valid_neg_v=valid_neg_v,
        test_u=test_u,
        test_v=test_v,
        test_neg_u=test_neg_u,
        test_neg_v=test_neg_v,
    )


def _norms(points: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(points, dim=-1)


def aperture(geometry: str, apex: torch.Tensor, k: float) -> torch.Tensor:
    """``psi(x)`` of Eqs. 26 (Poincare) and the Euclidean constant-h analogue."""
    norm = _norms(apex).clamp_min(1e-8)
    if geometry == "poincare":
        factor = (1.0 - norm.square()).clamp_min(0.0) / norm
    elif geometry == "euclid":
        factor = 1.0 / norm
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    return torch.asin((k * factor).clamp(0.0, 1.0))


def angle_child(geometry: str, apex: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """``xi(u, v)`` per Eq. 28 / Eq. 31, i.e. the angle at the apex."""
    apex_sq = apex.square().sum(-1)
    point_sq = point.square().sum(-1)
    dot = (apex * point).sum(-1)
    delta = _norms(apex - point).clamp_min(1e-6)
    if geometry == "poincare":
        numerator = dot * (1.0 + apex_sq) - apex_sq * (1.0 + point_sq)
        denominator = (
            delta
            * _norms(apex).clamp_min(1e-8)
            * (1.0 + apex_sq * point_sq - 2.0 * dot).clamp_min(1e-12).sqrt()
        )
    elif geometry == "euclid":
        numerator = point_sq - apex_sq - delta.square()
        denominator = 2.0 * delta * _norms(apex).clamp_min(1e-8)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    cosine = (numerator / denominator).clamp(-1.0 + EPS, 1.0 - EPS)
    return torch.acos(cosine)


def energy(
    geometry: str, apex: torch.Tensor, point: torch.Tensor, k: float
) -> torch.Tensor:
    return torch.relu(angle_child(geometry, apex, point) - aperture(geometry, apex, k))


def inner_radius(k: float) -> float:
    return 2.0 * k / (1.0 + math.sqrt(1.0 + 4.0 * k * k))


def _project(points: torch.Tensor, low: float, high: float) -> None:
    """In-place radial projection into ``[low, high]``."""
    norm = _norms(points).clamp_min(1e-12)
    points.mul_((low / norm).clamp_min(1.0).unsqueeze(-1))
    norm = _norms(points).clamp_min(1e-12)
    points.mul_((high / norm).clamp_max(1.0).unsqueeze(-1))


def _clip(geometry: str, points: torch.Tensor, k: float, epsilon: float) -> None:
    """In-place projection used after every cone update."""
    if geometry == "poincare":
        _project(points, inner_radius(k) + epsilon, 1.0 - epsilon)
    else:
        _project(points, k + epsilon, float("inf"))


def _apply_update(
    embeddings: torch.Tensor, update: torch.Tensor, cap: float
) -> None:
    """Bounded in-place descent step: never move a node far in one chunk."""
    row_norm = _norms(update).clamp_min(1e-12)
    limit = cap * _norms(embeddings)
    factor = (limit / row_norm).clamp_max(1.0).unsqueeze(-1)
    embeddings.sub_(update * factor)


def _init_points(
    n_nodes: int, dim: int, device: str, generator: torch.Generator
) -> torch.Tensor:
    """One shared initialisation for both geometries, inside both domains."""
    direction = torch.randn(n_nodes, dim, device=device, generator=generator)
    direction = direction / _norms(direction).clamp_min(1e-12).unsqueeze(-1)
    radius = torch.empty(n_nodes, 1, device=device).uniform_(
        cfg.PAPER_INIT_LOW, cfg.PAPER_INIT_HIGH, generator=generator
    )
    return direction * radius


_ADJACENCY_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _adjacency_keys(data: PaperConeData, device: str) -> torch.Tensor:
    """Device copy of the blocked-pair keys, built once per dataset and device."""
    key = (id(data), device)
    cached = _ADJACENCY_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(data.adjacency_keys, dtype=torch.int64, device=device)
        _ADJACENCY_CACHE[key] = cached
    return cached


def _sample_negatives(
    data: PaperConeData,
    apex: torch.Tensor,
    generator: torch.Generator,
    device: str,
    probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    n_nodes = data.n_nodes
    count = cfg.PAPER_NEGATIVES
    apex = apex.to(torch.int64)
    if probabilities is None:
        negatives = torch.randint(
            0, n_nodes, (apex.shape[0], count), device=device, generator=generator
        )
    else:
        negatives = _sample_from(probabilities, (apex.shape[0], count), generator)
    keys = _adjacency_keys(data, device)
    uniform_rows = torch.zeros(apex.shape[0], dtype=torch.bool, device=device)
    if len(data.uniform_nodes):
        blocked = torch.as_tensor(data.uniform_nodes, dtype=torch.int64, device=device)
        uniform_rows = (apex.unsqueeze(1) == blocked.unsqueeze(0)).any(dim=1)
    for _ in range(8):
        flat_apex = apex.unsqueeze(1).expand_as(negatives)
        query = flat_apex * n_nodes + negatives
        position = torch.searchsorted(keys, query.reshape(-1))
        position = position.clamp_max(keys.numel() - 1)
        member = (keys[position] == query.reshape(-1)).reshape(negatives.shape)
        member &= ~uniform_rows.unsqueeze(1)
        if not bool(member.any()):
            break
        if probabilities is None:
            negatives[member] = torch.randint(
                0, n_nodes, (int(member.sum()),), device=device, generator=generator
            )
        else:
            negatives[member] = _sample_from(
                probabilities, (int(member.sum()),), generator
            )
    return negatives


def _best_threshold_f1(
    positive: np.ndarray, negative: np.ndarray
) -> tuple[float, float]:
    """Reference threshold sweep: best F1 over the sorted validation scores."""
    scores = np.concatenate([negative, positive])
    labels = np.concatenate(
        [np.zeros(len(negative)), np.ones(len(positive))]
    )
    order = np.argsort(scores, kind="stable")
    scores = scores[order]
    labels = labels[order]
    n_positive = float(len(positive))
    true_positive = np.cumsum(labels)
    false_positive = np.cumsum(1.0 - labels)
    precision = 100.0 * true_positive / (true_positive + false_positive + 1e-6)
    recall = 100.0 * true_positive / n_positive
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    best = int(np.argmax(f1))
    return float(scores[best]), float(f1[best])


def _classification(
    positive: np.ndarray, negative: np.ndarray, threshold: float
) -> dict:
    true_positive = float((positive <= threshold).sum())
    false_negative = float((positive > threshold).sum())
    false_positive = float((negative <= threshold).sum())
    precision = 100.0 * true_positive / (true_positive + false_positive + 1e-6)
    recall = 100.0 * true_positive / (true_positive + false_negative + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
    }


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC for the cone energy: a *lower* score means a more likely edge."""
    order = np.argsort(np.concatenate([negative, positive]), kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    negative_rank_sum = ranks[: len(negative)].sum()
    n_positive = float(len(positive))
    n_negative = float(len(negative))
    return float(
        (negative_rank_sum - n_negative * (n_negative + 1.0) / 2.0)
        / (n_positive * n_negative)
    )


def poincare_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Nickel & Kiela distance, Eq. 4 of the reference implementation."""
    left_sq = left.square().sum(-1)
    right_sq = right.square().sum(-1)
    delta_sq = (left - right).square().sum(-1).clamp_min(1e-12)
    gamma = 1.0 + 2.0 * delta_sq / (
        (1.0 - left_sq).clamp_min(1e-12) * (1.0 - right_sq).clamp_min(1e-12)
    )
    return torch.acosh(gamma.clamp_min(1.0 + 1e-9))


def _power_probabilities(data: PaperConeData, power: float) -> torch.Tensor:
    counts = np.bincount(data.train_u, minlength=data.n_nodes).astype(np.float64) + 1.0
    weights = np.power(counts, power)
    return torch.as_tensor(weights / weights.sum(), dtype=torch.float32)


def _sample_from(
    probabilities: torch.Tensor,
    shape: tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    flat = torch.multinomial(
        probabilities, int(np.prod(shape)), replacement=True, generator=generator
    )
    return flat.reshape(shape)


def train_initializer(
    data: PaperConeData,
    dim: int,
    seed: int,
    device: str,
    *,
    batch: int | None = None,
    learning_rate: float | None = None,
    update_cap: float | None = None,
) -> torch.Tensor:
    """Poincare NLL warm start shared by both cone arms.

    Reproduces the reference's init stage: hyperbolic distance, NLL objective
    ``d(u,v) + log sum_i exp(-d(u,v_i))``, RSGD with a burn-in decade, negatives
    drawn with a 0.75 unigram exponent. Both cone geometries start from this
    same point set, so the geometry stays the only difference between them.
    """
    # The reference seeds this stage essentially at the origin and relies on a
    # zero-gradient guard for coincident points. Starting from the same
    # well-conditioned radius band the cone stage uses keeps the distance law
    # finite without that guard, and is applied to both arms identically.
    batch = cfg.PAPER_BATCH if batch is None else batch
    learning_rate = cfg.PAPER_INIT_LR if learning_rate is None else learning_rate
    update_cap = cfg.PAPER_UPDATE_CAP if update_cap is None else update_cap
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    embeddings = torch.nn.Parameter(
        _init_points(data.n_nodes, dim, device, generator)
    )
    probabilities = _power_probabilities(data, cfg.PAPER_INIT_NEG_POWER).to(device)
    train_u = torch.as_tensor(data.train_u, dtype=torch.long, device=device)
    train_v = torch.as_tensor(data.train_v, dtype=torch.long, device=device)
    order = torch.Generator(device="cpu").manual_seed(seed + 13)
    for epoch in range(1, cfg.PAPER_INIT_EPOCHS + 1):
        epoch_lr = (
            learning_rate
            if epoch > cfg.PAPER_INIT_BURN_IN
            else learning_rate / 10.0
        )
        permutation = torch.randperm(len(train_u), generator=order)
        for start in range(0, len(train_u), batch):
            index = permutation[start : start + batch]
            apex = train_u[index]
            child = train_v[index]
            negatives = _sample_negatives(data, apex, generator, device, probabilities)
            parent_points = embeddings[apex]
            distance_positive = poincare_distance(parent_points, embeddings[child])
            flat_parent = parent_points.unsqueeze(1).expand(-1, negatives.shape[1], -1)
            distance_negative = poincare_distance(
                flat_parent.reshape(-1, dim), embeddings[negatives.reshape(-1)]
            ).reshape(negatives.shape)
            loss = distance_positive.sum() + torch.logsumexp(
                -distance_negative, dim=1
            ).sum()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"Non-finite warm-start loss dim={dim} seed={seed} epoch={epoch}"
                )
            with torch.no_grad():
                scale = (
                    (1.0 - _norms(embeddings).square()).clamp_min(1e-12).square() / 4.0
                )
            gradient = torch.autograd.grad(loss, embeddings)[0]
            # The distance law is singular for coincident points, exactly as
            # in the reference, which zeroes those gradients explicitly.
            gradient = torch.nan_to_num(
                gradient, nan=0.0, posinf=0.0, neginf=0.0
            )
            with torch.no_grad():
                _apply_update(
                    embeddings,
                    epoch_lr * scale.unsqueeze(-1) * gradient,
                    update_cap,
                )
                _project(embeddings, cfg.PAPER_INIT_MIN_RADIUS, 1.0 - 1e-4)
            del loss, gradient
    return embeddings.detach()


def prepare_cone_init(init_points: torch.Tensor) -> torch.Tensor:
    """Paper's ``resc_vecs`` rescale applied before the cone stage."""
    return init_points * cfg.PAPER_RESC_VECS


def _group_negatives(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """The released negative files list a fixed block of negatives per positive."""
    if len(negative) % len(positive):
        raise ValueError("Negative count is not a multiple of the positive count")
    return negative.reshape(len(positive), -1)


def _per_positive_auc(
    positive: np.ndarray, negative: np.ndarray
) -> np.ndarray:
    """Per-positive AUC under the energy convention (lower is a better edge)."""
    less = (positive[:, None] < negative).sum(axis=1)
    tied = (positive[:, None] == negative).sum(axis=1)
    return (less + 0.5 * tied) / negative.shape[1]


def _score_pairs(
    geometry: str,
    embeddings: torch.Tensor,
    parent: np.ndarray,
    child: np.ndarray,
    k: float,
    device: str,
    chunk: int = 200_000,
) -> np.ndarray:
    parent_tensor = torch.as_tensor(parent, dtype=torch.long, device=device)
    child_tensor = torch.as_tensor(child, dtype=torch.long, device=device)
    out = np.empty(len(parent), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(parent), chunk):
            stop = min(start + chunk, len(parent))
            out[start:stop] = (
                energy(
                    geometry,
                    embeddings[parent_tensor[start:stop]],
                    embeddings[child_tensor[start:stop]],
                    k,
                )
                .double()
                .cpu()
                .numpy()
            )
    return out


def train_one(
    geometry: str,
    dim: int,
    data: PaperConeData,
    seed: int,
    device: str,
    init_points: torch.Tensor | None = None,
    *,
    batch: int | None = None,
    learning_rate: float | None = None,
    update_cap: float | None = None,
    steps_cap: int | None = None,
) -> dict:
    """Train one cone arm.

    Only the rows a chunk touches are gathered into a differentiable leaf, so a
    step costs the touched rows rather than the whole 82k-node table. That is
    what makes a near-reference batch size affordable: the reference's own
    10-pair step count is out of reach, but 64-pair steps are not.
    """
    k = cfg.PAPER_K
    batch = cfg.PAPER_BATCH if batch is None else batch
    learning_rate = (
        cfg.PAPER_LR_BY_GEOMETRY[geometry]
        if learning_rate is None
        else learning_rate
    )
    update_cap = cfg.PAPER_UPDATE_CAP if update_cap is None else update_cap
    started = time.time()
    generator = torch.Generator(device=device).manual_seed(seed)
    table = (
        _init_points(data.n_nodes, dim, device, generator)
        if init_points is None
        else prepare_cone_init(init_points).clone()
    )
    _clip(geometry, table, k, cfg.PAPER_EPSILON)
    train_u = torch.as_tensor(data.train_u, dtype=torch.long, device=device)
    train_v = torch.as_tensor(data.train_v, dtype=torch.long, device=device)
    epoch_order = torch.Generator(device="cpu").manual_seed(seed + 17)

    steps_per_epoch = math.ceil(len(train_u) / batch)
    epochs = cfg.PAPER_EPOCHS
    if steps_cap is not None:
        epochs = max(1, min(epochs, math.ceil(steps_cap / steps_per_epoch)))
    curve: list[dict] = []
    step = 0
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_u), generator=epoch_order)
        epoch_loss = 0.0
        for start in range(0, len(train_u), batch):
            index = permutation[start : start + batch]
            apex_index = train_u[index]
            child_index = train_v[index]
            negative_index = _sample_negatives(data, apex_index, generator, device)
            touched = torch.cat(
                [apex_index, child_index, negative_index.reshape(-1)]
            )
            unique, inverse = torch.unique(touched, return_inverse=True)
            sub = table[unique].detach().clone().requires_grad_(True)
            positions = inverse.reshape(-1)
            count = apex_index.shape[0]
            negative_count = negative_index.shape[1]
            apex_row = positions[:count]
            child_row = positions[count : count + count]
            negative_row = positions[count + count :].reshape(count, negative_count)
            parent_points = sub[apex_row]
            positive_energy = energy(geometry, parent_points, sub[child_row], k)
            flat_parent = parent_points.unsqueeze(1).expand(-1, negative_count, -1)
            negative_energy = energy(
                geometry,
                flat_parent.reshape(-1, dim),
                sub[negative_row.reshape(-1)],
                k,
            ).reshape(negative_row.shape)
            loss = positive_energy.sum() + torch.relu(
                cfg.PAPER_MARGIN - negative_energy
            ).sum()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"Non-finite cone loss geometry={geometry} dim={dim} "
                    f"ratio={data.ratio} seed={seed} step={step}"
                )
            gradient = torch.autograd.grad(loss, sub)[0]
            # The distance law is singular for coincident points, exactly as in
            # the reference, which zeroes those gradients explicitly.
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                if geometry == "poincare":
                    scale = (
                        (1.0 - _norms(sub).square()).clamp_min(1e-12).square() / 4.0
                    )
                else:
                    scale = torch.ones(sub.shape[0], device=device)
                update = learning_rate * scale.unsqueeze(-1) * gradient
                if update_cap is not None:
                    row_norm = _norms(update).clamp_min(1e-12)
                    limit = update_cap * _norms(sub)
                    update = update * (limit / row_norm).clamp_max(1.0).unsqueeze(-1)
                # Advanced indexing returns a copy, so the projection has to
                # run on the fresh tensor that is written back; clipping
                # ``table[unique]`` in place would silently be a no-op.
                updated = sub.detach() - update
                _clip(geometry, updated, k, cfg.PAPER_EPSILON)
                table[unique] = updated
            epoch_loss += float(loss.detach())
            step += 1
            del loss, gradient, sub
        if epoch % 20 == 0 or epoch == epochs:
            validation = _evaluate(
                geometry, table, data, "valid", device
            )
            curve.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(1, data.n_train),
                    "valid_f1": validation["f1"],
                    "valid_auc": validation["auc"],
                }
            )

    valid_scores = _evaluate(geometry, table, data, "valid", device)
    test_scores = _evaluate(geometry, table, data, "test", device)
    threshold = valid_scores["best_threshold"]
    test_classification = _classification(
        test_scores["positive"], test_scores["negative"], threshold
    )
    per_positive = _per_positive_auc(
        test_scores["positive"],
        _group_negatives(test_scores["positive"], test_scores["negative"]),
    )
    metrics = {
        "geometry": geometry,
        "dim": int(dim),
        "ratio": data.ratio,
        "seed": int(seed),
        "dataset": data.name,
        "k": float(k),
        "margin": float(cfg.PAPER_MARGIN),
        "n_nodes": int(data.n_nodes),
        "training": {
            "epochs": int(epochs),
            "batch": int(batch),
            "learning_rate": float(learning_rate),
            "update_cap": None if update_cap is None else float(update_cap),
            "negatives": int(cfg.PAPER_NEGATIVES),
            "n_train_pairs": int(data.n_train),
            "steps": int(step),
            "elapsed_seconds": float(time.time() - started),
        },
        "valid": {
            "f1": valid_scores["f1"],
            "auc": valid_scores["auc"],
            "threshold": valid_scores["best_threshold"],
        },
        "test": {
            "precision": test_classification["precision"],
            "recall": test_classification["recall"],
            "f1": test_classification["f1"],
            "auc": test_scores["auc"],
            "threshold": float(threshold),
        },
        "loss_curve": curve,
    }
    return {"metrics": metrics, "test_per_positive_auc": per_positive}


def _evaluate(
    geometry: str,
    embeddings: torch.Tensor,
    data: PaperConeData,
    split: str,
    device: str,
) -> dict:
    if split == "valid":
        positive = _score_pairs(
            geometry, embeddings, data.valid_u, data.valid_v, cfg.PAPER_K, device
        )
        negative = _score_pairs(
            geometry,
            embeddings,
            data.valid_neg_u,
            data.valid_neg_v,
            cfg.PAPER_K,
            device,
        )
    else:
        positive = _score_pairs(
            geometry, embeddings, data.test_u, data.test_v, cfg.PAPER_K, device
        )
        negative = _score_pairs(
            geometry,
            embeddings,
            data.test_neg_u,
            data.test_neg_v,
            cfg.PAPER_K,
            device,
        )
    threshold, f1 = _best_threshold_f1(positive, negative)
    return {
        "positive": positive,
        "negative": negative,
        "best_threshold": threshold,
        "f1": f1,
        "auc": _auc(positive, negative),
    }
