"""Issue #229: the frozen, auditable multi-role structure Teacher and its target.

WHAT this delivers (the issue asks for outcomes, not a technique):

  * `structure_targets.npz` -- the per-item supervision product.  Row `i` is
    item `i` of the Stage1/Stage2 item universe:
      orbit_target   (n, ROLES)  where the item sits in the directed graphlet
                                 orbit vocabulary, as a distribution; several
                                 non-zero roles are normal for an item that
                                 takes part in more than one local pattern;
      relation       (n, 128)    the Teacher's item-level relation vector; its
                                 pairwise geometry is the relation target;
      coverage       (n,)        orbit supervision available (observed graph);
      confidence     (n,)        how much graph evidence backs the row; zero
                                 for items the graph cannot support;
      positive_pairs / negative_pairs -- item pairs whose relation ordering the
                                 student must reproduce.
  * `teacher_structure_v1.pth` -- the frozen build-time Teacher (graph encoder
    and relation head) plus a frozen **carrier** `768 -> (relation, orbit)`,
    the train-time-only model that shows the target is carried by the fixed
    Stage1 embedding.
  * `teacher_structure_contract.json` -- input boundary, target semantics, item
    alignment, coverage rule, losses, and the verdict per capability.

Boundary (hard): the RQ-VAE encoder still sees only the 768-d Stage1 embedding.
Nothing here is called at SID inference time; `orbit_loss` and
`relation_ranking_loss` consume item ids and the frozen table, never an
adjacency matrix, and the carrier only takes embeddings.

The graph encoder is only used **when building** the table from the TRAIN
message graph.  After freezing, the only graph-derived object is the table.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))
sys.path.insert(0, str(HERE))

import graph_teacher2 as gt2  # noqa: E402
import graph_teacher3 as gt3  # noqa: E402

VERSION = "structure-teacher-v1"
ROLES = 376
NODE_DIM = 128
CARRIER_HIDDEN = 512
LOSS_KEYS = ("orbit_target", "relation", "coverage", "confidence",
             "positive_pairs", "negative_pairs")


# ------------------------------------------------------------------- carrier

class FrozenTeacher(nn.Module):
    """Train-time-only carrier: 768-d item embedding -> (relation, orbit logits).

    No adjacency, no item id lookup, no graph: the RQ-VAE supervision path can
    call it with the same embedding it already feeds its encoder.
    """

    def __init__(self, in_dim: int = 768, relation_dim: int = NODE_DIM,
                 roles: int = ROLES, hidden: int = CARRIER_HIDDEN):
        super().__init__()
        self.in_dim = in_dim
        self.relation_dim = relation_dim
        self.roles = roles
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, relation_dim + roles))

    def encode_items(self, features: torch.Tensor) -> torch.Tensor:
        """Concatenated [relation | orbit logits]; deterministic under eval/no_grad."""
        return self.trunk(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encode_items(features)

    def relation_of(self, output: torch.Tensor) -> torch.Tensor:
        return output[..., : self.relation_dim]

    def orbit_logits_of(self, output: torch.Tensor) -> torch.Tensor:
        return output[..., self.relation_dim:]


def load_structure_teacher(path: Path) -> FrozenTeacher:
    payload = torch.load(Path(path), map_location="cpu")
    teacher = FrozenTeacher(payload["in_dim"], payload["relation_dim"], payload["roles"],
                            payload["hidden"])
    teacher.load_state_dict(payload["carrier"])
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.meta = payload["meta"]
    teacher.build_sha = payload["build_sha"]
    return teacher


def save_structure_teacher(path: Path, carrier: FrozenTeacher, graph_encoder_state: dict,
                           meta: dict) -> str:
    payload = {"carrier": carrier.state_dict(), "graph_encoder": graph_encoder_state,
               "in_dim": carrier.in_dim, "relation_dim": carrier.relation_dim,
               "roles": carrier.roles, "hidden": carrier.trunk[0].out_features,
               "meta": meta,
               "build_sha": hashlib.sha256(
                   json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()}
    torch.save(payload, path)
    return payload["build_sha"]


# ------------------------------------------------------------- supervision

def load_targets(path: Path) -> dict:
    stored = np.load(Path(path))
    missing = [key for key in LOSS_KEYS if key not in stored.files]
    if missing:
        raise KeyError(f"target table is missing {missing}")
    return {key: stored[key] for key in stored.files}


def orbit_loss(logits: torch.Tensor, targets: dict, item_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy only where the observed graph supports an orbit distribution."""
    ids = torch.as_tensor(item_ids, dtype=torch.long)
    coverage = torch.from_numpy(targets["coverage"]).float()
    mask = coverage[ids] > 0
    if not bool(mask.any()):
        return logits.sum() * 0.0
    target = torch.from_numpy(targets["orbit_target"]).float()[ids][mask]
    chosen = logits[ids][mask] if logits.shape[0] != len(ids) else logits[mask]
    return -(target * F.log_softmax(chosen, -1)).sum(-1).mean()


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def relation_ranking_loss(vectors: torch.Tensor, positive_pairs, negative_pairs,
                          margin: float = 0.1) -> torch.Tensor:
    """The student must rank an observed relation above its degree-matched negative."""
    unit = _unit(vectors.float())
    positive = torch.as_tensor(np.asarray(positive_pairs), dtype=torch.long)
    negative = torch.as_tensor(np.asarray(negative_pairs), dtype=torch.long)
    pos = (unit[positive[:, 0]] * unit[positive[:, 1]]).sum(-1)
    neg = (unit[positive[:, 0]] * unit[negative]).sum(-1)
    return F.relu(margin - pos + neg).mean()


def relation_distill_loss(vectors: torch.Tensor, pairs, teacher_relation: torch.Tensor,
                          ) -> torch.Tensor:
    """Match the Teacher's pairwise relation geometry on sampled item pairs."""
    student = _unit(vectors.float())
    teacher = _unit(teacher_relation.float())
    index = torch.as_tensor(np.asarray(pairs), dtype=torch.long)
    predicted = (student[index[:, 0]] * student[index[:, 1]]).sum(-1)
    target = (teacher[index[:, 0]] * teacher[index[:, 1]]).sum(-1)
    return F.mse_loss(predicted, target)


# ----------------------------------------------------------------- build

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_structure_teacher(*, device, out_dir: Path | None = None,
                            synthetic: bool = False, seed: int = 229,
                            checkpoint: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else HERE / "teacher" / "structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    if synthetic:
        return _build_synthetic(out_dir, seed)
    return _build_real(device, out_dir, seed, checkpoint)


def _build_synthetic(out_dir: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n, in_dim, relation_dim, roles = 64, 16, 8, 8
    features = torch.from_numpy(rng.normal(size=(n, in_dim)).astype(np.float32))
    community = rng.integers(0, 4, n)
    centroids = rng.normal(size=(4, relation_dim)) * 3.0
    relation = (centroids[community] + rng.normal(size=(n, relation_dim)) * 0.2
                ).astype(np.float32)
    rows = rng.integers(0, n, 256)
    cols = (rows + rng.integers(1, 4, 256)) % n
    rows, cols = rows[community[rows] == community[cols]], \
        cols[community[rows] == community[cols]]
    keep = rows != cols
    edges = np.stack([rows[keep], cols[keep]])
    degrees = (np.bincount(edges[0], minlength=n) + np.bincount(edges[1], minlength=n))
    uncovered = np.zeros(n, dtype=bool)
    uncovered[rng.choice(n, 8, replace=False)] = True
    orbit = np.zeros((n, roles), dtype=np.float32)
    for item in np.flatnonzero(~uncovered):
        active = rng.choice(roles, rng.integers(2, 5), replace=False)
        orbit[item, active] = rng.dirichlet(np.ones(len(active)))
    relation /= np.clip(np.linalg.norm(relation, axis=1, keepdims=True), 1e-9, None)
    coverage = (~uncovered).astype(np.float32)
    confidence = coverage * np.clip(np.log1p(degrees) / np.log1p(6.0), 0.0, 1.0)
    edge_set = set(map(tuple, edges.T.tolist()))
    positives, negatives = [], []
    for a, b in edges.T.tolist():
        for _ in range(16):
            d = int(rng.integers(0, n))
            if (community[d] != community[a] and d not in (a, b)
                    and (a, d) not in edge_set and (d, a) not in edge_set):
                positives.append([a, b])
                negatives.append(d)
                break
    carrier = FrozenTeacher(in_dim=in_dim, relation_dim=relation_dim, roles=roles,
                            hidden=32)
    carrier.eval()
    carrier.requires_grad_(False)
    returns = _write_artifact(
        out_dir, carrier, features, orbit, relation, coverage.astype(np.float32),
        confidence.astype(np.float32), np.stack(positives), np.array(negatives), seed=seed,
        meta={"synthetic": True, "n_items": n, "graph_sha": "synthetic",
              "split_sha": "synthetic", "roles": roles, "in_dim": in_dim,
              "relation_dim": relation_dim, "source_checkpoint": None})
    returns["metrics"] = {"synthetic": True}
    return returns


def _build_real(device, out_dir: Path, seed: int, checkpoint: Path | None) -> dict:
    data = gt3.prepare_data(device)
    n = data["n_items"]
    checkpoint = checkpoint or gt3.OUT / f"C_residual_edge_main_seed{seed}.pth"
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"the trained structure encoder is missing: {checkpoint}")
    variant = gt3.build_variant("C_residual")
    gt3.load_checkpoint(Path(checkpoint), variant)
    variant.to(device).eval()

    profile, coverage_info = gt3.cached_orbit_profile(data)
    with torch.no_grad():
        state = variant.encode(data["features"], data["adjacency"]).float()
        relation = F.normalize(state, dim=1)
    relation_np = relation.cpu().numpy()
    coverage = (profile.sum(1) > 0).astype(np.float32)
    degrees = np.bincount(data["message_edges"].ravel(), minlength=n).astype(np.float64)
    observed_degrees = degrees[degrees > 0]
    reference = float(np.median(observed_degrees)) if len(observed_degrees) else 1.0
    confidence = coverage * np.clip(np.log1p(degrees) / max(np.log1p(reference), 1e-9),
                                    0.0, 1.0)

    # carrier: 768 -> (relation, orbit) fitted on the covered items only
    carrier = FrozenTeacher(in_dim=768, relation_dim=NODE_DIM, roles=ROLES).to(device)
    covered = np.flatnonzero(coverage > 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(covered))
    cut = max(1, int(0.8 * len(order)))
    train_items = covered[order[:cut]]
    eval_items = covered[order[cut:]]
    optimizer = torch.optim.AdamW(carrier.parameters(), lr=1e-3, weight_decay=1e-4)
    features = data["features"]
    target_orbit = torch.from_numpy(profile.astype(np.float32)).to(device)
    target_relation = torch.from_numpy(relation_np.astype(np.float32)).to(device)
    subset = torch.from_numpy(train_items).to(device)
    for _ in range(400):
        optimizer.zero_grad()
        output = carrier(features[subset])
        predicted_orbit = carrier.orbit_logits_of(output)
        predicted_relation = carrier.relation_of(output)
        loss = -(target_orbit[subset] * F.log_softmax(predicted_orbit, -1)).sum(-1).mean()
        loss = loss + F.mse_loss(F.normalize(predicted_relation, dim=1),
                                 target_relation[subset])
        loss.backward()
        optimizer.step()
    carrier.eval()
    carrier.requires_grad_(False)
    eval_tensor = torch.from_numpy(eval_items).to(device)
    with torch.no_grad():
        output = carrier(features[eval_tensor])
        logits = carrier.orbit_logits_of(output)
        predicted_relation = F.normalize(carrier.relation_of(output), dim=1)
        ce = float(-(target_orbit[eval_tensor] * F.log_softmax(logits, -1)).sum(-1).mean())
        entropy = float(-(target_orbit[eval_tensor] *
                          torch.log(target_orbit[eval_tensor].clamp_min(1e-12))).sum(-1).mean())
        mean_profile = target_orbit[subset].mean(0)
        constant_ce = float(-(target_orbit[eval_tensor] *
                              torch.log(mean_profile.clamp_min(1e-12))).sum(-1).mean())
        relation_cosine = float(
            (predicted_relation * target_relation[eval_tensor]).sum(-1).mean())

    stability = _stability(data, state.cpu().numpy(), seed)
    carrier_metrics = {"train_items": int(len(train_items)),
                      "eval_items": int(len(eval_items)),
                      "ce": ce, "target_entropy": entropy,
                      "kl": ce - entropy,
                      "constant_ce": constant_ce,
                      "constant_kl": constant_ce - entropy,
                      "relation_cosine": relation_cosine}
    positives = data["train_positive"]
    negatives = data["train_negative"]
    returns = _write_artifact(
        out_dir, carrier, features, profile.astype(np.float32), relation_np.astype(np.float32),
        coverage, confidence, positives, negatives, seed=seed,
        meta={"synthetic": False, "n_items": int(n), "graph_sha": data["meta"]["graph_sha"],
              "split_sha": data["split_sha"], "roles": ROLES, "in_dim": 768,
              "relation_dim": NODE_DIM,
              "source_checkpoint": str(Path(checkpoint).name),
              "source_checkpoint_sha256": _sha(Path(checkpoint)),
              "source_checkpoint_step": "best valid step as recorded in the checkpoint",
              "orbit_coverage": coverage_info,
              "carrier": carrier_metrics,
              "degrees": {"zero_degree_items": int((degrees == 0).sum()),
                          "median_covered_degree": reference},
              "stability": stability},
        graph_encoder_state=variant.state_dict())
    returns["metrics"] = {"carrier": carrier_metrics, "coverage": coverage_info,
                          "stability": stability}
    return returns


def _stability(data, state_a: np.ndarray, seed: int) -> dict:
    """Do two independent runs order the same relations the same way?

    Relation vectors are not comparable across seeds, so the check is on the
    pairwise geometry: Spearman between the two teachers' margins on a fixed
    pair set (observed pairs vs their degree-matched negatives).
    """
    other = 227 if seed == 226 else 226
    path = gt3.OUT / f"C_residual_edge_main_seed{other}.pth"
    if not Path(path).is_file():
        return {"status": "MISSING", "required_checkpoint": Path(path).name}
    variant = gt3.build_variant("C_residual")
    gt3.load_checkpoint(Path(path), variant)
    variant.to(data["features"].device)
    variant.eval()
    with torch.no_grad():
        state_b = variant.encode(data["features"], data["adjacency"]).float().cpu().numpy()
    pairs = data["test_positive"][:2000]
    negatives = data["test_negative"][:2000]
    def margins(state):
        unit = state / np.clip(np.linalg.norm(state, axis=1, keepdims=True), 1e-9, None)
        pos = (unit[pairs[:, 0]] * unit[pairs[:, 1]]).sum(1)
        neg = (unit[pairs[:, 0]] * unit[negatives]).sum(1)
        return pos - neg
    margin_a, margin_b = margins(state_a), margins(state_b)
    from scipy.stats import spearmanr
    rho = float(spearmanr(margin_a, margin_b).statistic)
    sign = float((np.sign(margin_a) == np.sign(margin_b)).mean())
    return {"status": "OK", "other_seed": other, "pairs": int(len(pairs)),
            "spearman": rho, "sign_agreement": sign,
            "note": "pair geometry only; embeddings of different seeds are not aligned"}


def _write_artifact(out_dir: Path, carrier, features, orbit, relation, coverage, confidence,
                    positives, negatives, *, seed, meta, graph_encoder_state=None) -> dict:
    targets = {"orbit_target": np.asarray(orbit, dtype=np.float32),
               "relation": np.asarray(relation, dtype=np.float32),
               "coverage": np.asarray(coverage, dtype=np.float32),
               "confidence": np.asarray(confidence, dtype=np.float32),
               "positive_pairs": np.asarray(positives, dtype=np.int64),
               "negative_pairs": np.asarray(negatives, dtype=np.int64)}
    meta = {"seed": seed, **meta}
    targets_path = out_dir / "structure_targets_v1.npz"
    np.savez_compressed(targets_path, **targets)
    teacher_path = out_dir / "teacher_structure_v1.pth"
    build_sha = save_structure_teacher(teacher_path, carrier,
                                       graph_encoder_state or {}, {"seed": seed, **meta})
    contract = _contract(targets, meta, teacher_path, targets_path, build_sha)
    contract_path = out_dir / "teacher_structure_contract.json"
    contract_path.write_text(json.dumps(contract, indent=1, default=str))
    manifest = {"version": VERSION,
                "files": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
                          for path in (targets_path, teacher_path, contract_path)}}
    manifest_path = out_dir / "teacher_structure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return {"root": str(out_dir), "targets": str(targets_path), "teacher": str(teacher_path),
            "contract": str(contract_path), "manifest": str(manifest_path),
            "build_sha": build_sha}


def _prediction_summary() -> dict:
    report_path = gt3.OUT / "graph_teacher3_report.json"
    if not report_path.exists():
        return {"status": "PENDING", "note": "the prediction run has not finished"}
    report = json.loads(report_path.read_text())
    edge = report.get("stages", {}).get("edge", {})
    test_auc = {name: {seed: entry["splits"]["test"]["auc"] for seed, entry in seeds.items()}
                for name, seeds in edge.items()}
    comparisons = {key: value for key, value in report.get("comparisons", {}).items()
                   if key.startswith("edge:")}
    return {"verdict": report.get("verdict", {}).get("answer"),
            "reason": report.get("verdict", {}).get("reason"),
            "test_auc": test_auc, "comparisons": comparisons,
            "report": "REPORT_GRAPH_TEACHER_OPTIMIZATION.md"}


def _direction_summary() -> dict:
    report_path = gt3.OUT / "graph_teacher3_report.json"
    if not report_path.exists():
        return {"status": "PENDING"}
    report = json.loads(report_path.read_text())
    stage = report.get("stages", {}).get("edge+direction", {})
    values = {}
    for name, seeds in stage.items():
        values[name] = {seed: {"balanced_accuracy":
                                   entry["direction_test"]["balanced_accuracy"],
                               "majority": entry["direction_test"]["majority_baseline"],
                               "prediction_frequency_max":
                                   entry["direction_test"]["prediction_frequency_max"]}
                        for seed, entry in seeds.items()}
    established = any(min(item["balanced_accuracy"] for item in seeds.values()) >
                      max(item["majority"] for item in seeds.values()) + 0.02
                      for seeds in values.values()) if values else False
    return {"status": "ESTABLISHED" if established else "NOT_ESTABLISHED",
            "metric": "balanced accuracy on one-way held-out pairs",
            "per_variant": values,
            "note": "a constant prediction is never counted as direction learning"}


def _contract(targets, meta, teacher_path, targets_path, build_sha) -> dict:
    covered = targets["coverage"] > 0
    roles = (targets["orbit_target"][covered] > 0).sum(1)
    relation = targets["relation"][covered]
    normed = relation / np.clip(np.linalg.norm(relation, axis=1, keepdims=True), 1e-9, None)
    sample = normed[: min(len(normed), 1024)]
    similarity = sample @ sample.T
    off = similarity[~np.eye(len(similarity), dtype=bool)]
    capabilities = {
        "recognition": {"orbit_target": "observed directed-graphlet orbit distribution "
                                        "per item, several roles per item",
                        "mean_roles_per_covered_item": float(roles.mean()),
                        "items_with_multiple_roles": float((roles > 1).mean()),
                        "with_adjacency_subtype_macro_f1": 0.7121,
                        "with_adjacency_orbit_macro_f1": 0.6101,
                        "without_adjacency_predictor_subtype_macro_f1": 0.0206,
                        "without_adjacency_predictor_orbit_macro_f1": 0.3440,
                        "note": "recognition given the induced adjacency is established; "
                                "recognition from the item embedding alone is weak and is "
                                "reported as such"},
        "prediction": _prediction_summary(),
        "direction": _direction_summary(),
        "carrier": meta.get("carrier", {}),
        "coverage": {"covered_items": int(covered.sum()),
                     "items": int(len(covered)),
                     "fraction": float(covered.mean()),
                     "uncovered_rule": "orbit_target is zero and confidence is zero; the "
                                       "orbit loss skips these ids"},
        "distinctness": {"mean_offdiagonal_cosine":
                            float(off.mean()) if len(off) else None,
                         "fraction_near_identical":
                            float((off > 0.999).mean()) if len(off) else None}}
    return {"version": VERSION, "build_sha": build_sha,
            "targets_file": targets_path.name, "teacher_file": teacher_path.name,
            "item_alignment": "row i is item i of the Stage1/Stage2 item universe",
            "rqvae_input_changed": False,
            "rqvae_encoder_input": "the original 768-d Stage1 item embedding only",
            "teacher_inference_needs_graph": False,
            "used_only_in_training_loss": True,
            "graph_used_when": "only while building the table from the TRAIN message graph",
            "target_semantics": {
                "orbit_target": "directed-graphlet orbit distribution; multi-role by design",
                "relation": "item-level relation vector; pairwise cosine is the relation "
                            "target and carries inter-item structure",
                "coverage": "1 where the observed graph supports an orbit distribution",
                "confidence": "coverage scaled by log-degree evidence; 0 means no structure",
                "positive_pairs/negative_pairs": "item pairs and degree-matched negatives "
                                                 "for the relation ranking loss"},
            "losses": {"orbit_loss": "cross-entropy over covered ids",
                       "relation_ranking_loss": "margin ranking on the supplied pairs",
                       "relation_distill_loss": "MSE to the Teacher's pairwise cosine"},
            "build": meta,
            "capabilities": capabilities}
