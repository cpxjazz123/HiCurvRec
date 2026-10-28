"""The hard hyperbolic prefix score Q_H of Issue #126, and its record.

Q_H asks one thing of a hard SID: does its first-level prefix keep enough
behavioural information to order a real next item above the fixed negatives of a
validation query? The prefix is read through the hyperbolic Frechet barycentre of
the target representations of the items in each bucket, and the query is the
source representation of the item that conditions it.

The score is deterministic for a given SID, a given frozen reference and a given
probe. It is not a probability and it is not bounded below by zero: a SID whose
prefixes order the negatives above the positive scores below zero, which is the
point of measuring it. It never replaces a Stage 2 -> Stage 3 admission rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import hyperbolic_behavior_reference as reference_module

REPO = reference_module.REPO

#: The frozen reading rule. The solver is fixed before any SID is scored: the
#: barycentre is the exact Karcher mean up to a stated tolerance, not a Euclidean
#: average renamed as one.
SPEC = {
    "version": "issue126-hard-v1",
    "curvature": 1.0,
    "barycentre": {
        "init": "exp_0(mean_i log_0(v_i))",
        "iteration": "mu <- exp_mu(mean_i log_mu(v_i))",
        "steps": 300,
        "step_size": 1.0,
        "tol": 1e-10,
    },
    "score": {
        "delta": "d(u_a, mu_b(j-))^2 - d(u_a, mu_b(i+))^2",
        "per_query": "mean over the fixed negatives of softplus(-delta/tau)",
        "aggregate": "1 - mean over queries / ln 2",
        "temperature": "the frozen reference's temperature",
    },
    "ignores": "tokens after the first level, and the disambiguation token",
}

OUTPUT_DIR = REPO / "results" / "research" / "issue126" / "metrics"


def _geometry():
    return reference_module._geometry()


class BallGeometry:
    """Thin numpy wrapper over the frozen baseline's hyperbolic helpers."""

    def __init__(self, curvature: float = None):
        self.module = _geometry()
        self.c = float(curvature if curvature is not None else SPEC["curvature"])

    def _t(self, array):
        import torch
        return torch.as_tensor(np.asarray(array, dtype=np.float64))

    def expmap0(self, tangent):
        return self.module.expmap0(self._t(tangent), self.c).numpy()

    def logmap0(self, point):
        return self.module.logmap0(self._t(point), self.c).numpy()

    def expmap(self, base, tangent):
        """exp_base(v) through the canonical Mobius sum."""
        import torch
        module, c = self.module, self.c
        base_t, tangent_t = self._t(base), self._t(tangent)
        norm = tangent_t.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scaled = torch.tanh(c * norm / 2) / (c * norm) * tangent_t
        moved = module.mobius_add(base_t, scaled, c)
        return module.proj_to_ball(moved, c).numpy()

    def logmap(self, base, point):
        import torch
        module, c = self.module, self.c
        base_t, point_t = self._t(base), self._t(point)
        difference = module.mobius_add(-base_t, point_t, c)
        norm = difference.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scale = (2 / c) * module.artanh((c * norm).clamp(max=1 - 1e-9)) / norm
        return (scale * difference).numpy()

    def distance(self, x, y):
        return self.module.poincare_distance(self._t(x), self._t(y), self.c).numpy()


def frechet_barycentre(points: np.ndarray, geometry: BallGeometry = None,
                       spec: dict = None) -> np.ndarray:
    """The Karcher mean of the points, by the frozen fixed-point iteration.

    The tolerance is on the size of the step, so the answer is reproducible: the
    same points and the same rule give the same centre to the stated precision.
    """
    geometry = geometry or BallGeometry()
    rule = (spec or SPEC)["barycentre"]
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 1:
        return points[0].copy()
    centre = geometry.expmap0(geometry.logmap0(points).mean(axis=0))
    for _ in range(int(rule["steps"])):
        step = geometry.logmap(np.broadcast_to(centre, points.shape), points).mean(axis=0)
        moved = geometry.expmap(centre, float(rule["step_size"]) * step)
        shift = float(np.linalg.norm(geometry.logmap(centre, moved)))
        centre = moved
        if shift < float(rule["tol"]):
            break
    return centre


def first_level_buckets(sid_rows: np.ndarray, item_ids: np.ndarray) -> dict:
    """The item ids of every non-empty first-level bucket."""
    rows = np.asarray(sid_rows)
    buckets: dict = {}
    for position, item in enumerate(np.asarray(item_ids)):
        key = int(rows[position, 0])
        buckets.setdefault(key, []).append(int(item))
    return buckets


def bucket_centres(targets: np.ndarray, sid_rows: np.ndarray, item_ids: np.ndarray,
                   geometry: BallGeometry = None) -> dict:
    """One hyperbolic barycentre per non-empty first-level bucket."""
    geometry = geometry or BallGeometry()
    buckets = first_level_buckets(sid_rows, item_ids)
    return {key: frechet_barycentre(targets[members], geometry)
            for key, members in buckets.items()}, {k: len(v) for k, v in buckets.items()}


def score(sid_path: Path, *, reference: dict = None, probe: Path = None,
          temperature: float = None, geometry: BallGeometry = None,
          popularity: dict = None) -> dict:
    """Score one exported hard SID on the fixed validation probe."""
    reference = reference or reference_module.load()
    arrays = np.load(probe or reference_module.VALID_PAIRS)
    anchors = arrays["anchors"].astype(np.int64)
    positives = arrays["positives"].astype(np.int64)
    negatives = arrays["negatives"].astype(np.int64)
    tau = float(temperature if temperature is not None
                else reference["record"]["temperature"])
    geometry = geometry or BallGeometry()

    sid = json.loads(Path(sid_path).read_text(encoding="utf-8"))
    items = sorted(int(key) for key in sid)
    rows = np.array([sid[str(item)] for item in items], dtype=np.int64)
    item_ids = np.array(items, dtype=np.int64)
    if rows.ndim != 2:
        raise ValueError("the SID rows are not rectangular")
    centres, sizes = bucket_centres(reference["target"], rows, item_ids, geometry)

    first = {item: int(rows[position, 0]) for position, item in enumerate(items)}
    source = reference["source"][anchors]

    def centre_of(item_ids_array):
        keys = np.array([first[int(item)] for item in item_ids_array.ravel()])
        unknown = sorted({key for key in keys if key not in centres})
        if unknown:
            raise ValueError("the probe names items outside the SID: " + str(unknown[:5]))
        return np.stack([centres[key] for key in keys]).reshape(
            item_ids_array.shape + (-1,))

    mu_pos = centre_of(positives)
    mu_neg = centre_of(negatives)
    d_pos = geometry.distance(source, mu_pos) ** 2
    d_neg = geometry.distance(source[:, None, :], mu_neg) ** 2
    delta = d_neg - d_pos[:, None]
    positive_bucket = np.array([first[int(item)] for item in positives])[:, None]
    negative_bucket = np.array([[first[int(item)] for item in row] for row in negatives])
    same_bucket = positive_bucket == negative_bucket
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    q_h = 1.0 - float(per_query.mean()) / float(np.log(2.0))
    return {
        "spec": SPEC,
        "sid": {"path": str(Path(sid_path)),
                "sha256": hashlib.sha256(Path(sid_path).read_bytes()).hexdigest(),
                "items": int(len(items)),
                "levels": int(rows.shape[1])},
        "reference": {"digest": reference["record"]["spec"]["digest"],
                      "arrays_sha256": reference["record"]["arrays"]["sha256"],
                      "temperature": tau},
        "probe": {"path": str(Path(probe or reference_module.VALID_PAIRS)),
                  "queries": int(anchors.shape[0]),
                  "negatives_per_query": int(negatives.shape[1])},
        "q_h": q_h,
        "same_bucket_ratio": float(same_bucket.mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "negatives_ranked_above_positive": float((delta < 0).mean()),
        "buckets": {"count": int(len(centres)),
                    "mean_size": float(np.mean(list(sizes.values()))),
                    "max_size": int(max(sizes.values())),
                    "min_size": int(min(sizes.values()))},
        "popularity_groups": _popularity_groups(delta, anchors, popularity),
    }


def _popularity_groups(delta: np.ndarray, anchors: np.ndarray, popularity: dict) -> dict:
    """Q_H per baseline popularity tercile, when the counts are supplied.

    The issue asks whether an association is driven by popularity alone: the
    grouped numbers answer that question directly instead of by argument.
    """
    if not popularity:
        return {}
    counts = np.array([popularity.get(int(item), 0) for item in anchors], dtype=np.float64)
    if counts.size == 0:
        return {}
    cuts = np.quantile(counts, [1 / 3, 2 / 3])
    groups = {}
    for name, mask in (("cold", counts <= cuts[0]), ("mid", (counts > cuts[0]) & (counts <= cuts[1])),
                       ("warm", counts > cuts[1])):
        if not mask.any():
            continue
        per_query = np.log1p(np.exp(-delta[mask] / 1.0)).mean(axis=1)
        groups[name] = {"queries": int(mask.sum()),
                        "q_h_raw": float(1.0 - per_query.mean() / np.log(2.0))}
    return groups


def record(sid_path: Path, **kwargs) -> Path:
    """Score one SID and persist the record next to the other Issue #126 files."""
    report = score(Path(sid_path), **kwargs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / (report["sid"]["sha256"] + ".json")
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target
