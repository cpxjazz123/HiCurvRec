"""Issue #229 deliverable: one frozen Teacher bundle with a documented API.

The research artifacts (v3 target table, carrier, contract) are useful but a
consumer still has to know which file is which.  This module packages them into
a single loadable object:

    bundle = load_structure_teacher("teacher/structure/bundle")
    bundle.recognise(matrix)                  # induced adjacency -> family/key/orbits
    bundle.targets_for([12, 900, 24001])      # graph-derived supervision rows
    bundle.predict_targets(embeddings)        # carrier fallback for any item
    bundle.structure_loss(student_vectors, item_ids, role_logits=None)
    bundle.verify()                           # every SHA in the bundle

Boundary: the RQ-VAE encoder never receives the graph, the teacher hidden or an
instance label; the bundle is consumed only inside the training loss.  The
`capabilities` block is copied from the v3 contract, so a consumer cannot use a
component whose evidence says it is not carried without seeing that flag.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import structure_teacher as st  # noqa: E402
import graph_teacher3 as gt3  # noqa: E402

BUNDLE_VERSION = "structure-teacher-bundle-v1"
BUNDLE_DIR = HERE / "teacher" / "structure" / "bundle"
V3_DIR = HERE / "teacher" / "structure" / "v3"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StructureTeacherBundle:
    """Frozen teacher: recognition, graph-derived targets, carrier and losses."""

    def __init__(self, carrier, target_table: dict, capabilities: dict, meta: dict,
                 weights: dict):
        self.carrier = carrier
        self.targets = target_table
        self.capabilities = capabilities
        self.meta = meta
        self.weights = weights
        self.carrier.eval()
        self.carrier.requires_grad_(False)

    # -- recognition ------------------------------------------------------
    def recognise(self, matrix) -> dict:
        return st.recognise_subgraph(matrix)

    # -- targets ----------------------------------------------------------
    def _rows(self, item_ids) -> np.ndarray:
        ids = np.asarray(item_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("item_ids must be one-dimensional")
        if len(ids) and (ids.min() < 0 or ids.max() >= len(self.targets["coverage"])):
            raise IndexError("item id outside the target table")
        return ids

    def targets_for(self, item_ids) -> dict:
        ids = self._rows(item_ids)
        keys = ("relation", "neighbour_relation", "neighbour_orbit", "neighbour_family",
                "own_orbit", "own_subtype", "coverage", "coverage_neighbour",
                "coverage_family", "confidence")
        return {key: self.targets[key][ids].copy() for key in keys}

    def predict_targets(self, embeddings: torch.Tensor) -> dict:
        """Carrier predictions; for items the graph cannot support, mask by confidence."""
        with torch.no_grad():
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            parts = self.carrier.components(self.carrier(embeddings.float()))
        return parts

    # -- supervision ------------------------------------------------------
    def structure_loss(self, student_vectors: torch.Tensor, item_ids=None,
                       role_logits=None, weights: dict | None = None):
        """Losses ready to add to an RQ-VAE training step, full table or one batch.

        Two accepted shapes, so the same function works in a full-batch probe and
        in a mini-batch training loop:

          * `student_vectors` (n_items, d) with `item_ids=None`: the whole universe,
            the pair tables index it directly;
          * `student_vectors` (len(item_ids), d): a batch; pairs whose endpoints are
            not all in the batch are dropped and the rest are re-indexed locally.

        `role_logits` accepts the same two shapes as `student_vectors` (the full
        table or `len(item_ids)` batch rows); without it the neighbourhood-orbit
        term is skipped and reported as `None`.
        """
        total_items = len(self.targets["relation"])
        ids = np.arange(total_items) if item_ids is None else self._rows(item_ids)
        if student_vectors.shape[0] == total_items and item_ids is None:
            local = None
        elif student_vectors.shape[0] == len(ids):
            local = {int(item): row for row, item in enumerate(ids)}
        else:
            raise ValueError(f"student_vectors must have {total_items} rows or "
                             f"{len(ids)} rows aligned with item_ids")
        weights = weights or self.weights
        pairs = self.targets["positive_pairs"]
        negatives = self.targets["negative_pairs"]
        relation = torch.from_numpy(self.targets["relation"]).float()
        neighbour = torch.from_numpy(self.targets["neighbour_relation"]).float()
        if local is None:
            local_pairs, local_negatives = pairs, negatives
            teacher, teacher_neighbour = relation, neighbour
        else:
            keep = np.asarray([index for index, (a, b, d) in
                               enumerate(zip(pairs[:, 0], pairs[:, 1], negatives))
                               if int(a) in local and int(b) in local and int(d) in local],
                              dtype=np.int64)
            local_pairs = np.stack([[local[int(a)], local[int(b)]]
                                    for a, b in pairs[keep]]).astype(np.int64) \
                if len(keep) else np.zeros((0, 2), dtype=np.int64)
            local_negatives = np.asarray([local[int(d)] for d in negatives[keep]],
                                         dtype=np.int64)
            teacher = relation[torch.from_numpy(ids)]
            teacher_neighbour = neighbour[torch.from_numpy(ids)]
        if len(local_pairs):
            rank = st.relation_ranking_loss(student_vectors, local_pairs,
                                            local_negatives)
            distill = st.relation_distill_loss(student_vectors, local_pairs, teacher)
            n_distill = st.relation_distill_loss(student_vectors, local_pairs,
                                                 teacher_neighbour)
            total = (weights.get("relation", 1.0) * rank + 0.1 * distill
                     + weights.get("neighbour_relation", 0.0) * n_distill)
            result = {"relation_ranking": float(rank.detach()),
                      "relation_distill": float(distill.detach()),
                      "neighbour_relation_distill": float(n_distill.detach()),
                      "pairs_used": int(len(local_pairs)), "orbit": None}
        else:
            # no observed pair survives inside this batch: report the absence
            # instead of the NaN that mean()/mse_loss() over an empty set gives
            total = student_vectors.sum() * 0.0
            result = {"relation_ranking": None, "relation_distill": None,
                      "neighbour_relation_distill": None,
                      "pairs_used": 0, "orbit": None}
        if role_logits is not None and weights.get("neighbour_orbit", 0.0) > 0:
            target = torch.from_numpy(self.targets["neighbour_orbit"]).float()
            coverage = torch.from_numpy(self.targets["coverage_neighbour"]).float()
            if role_logits.shape[0] == len(ids):
                logit_rows = torch.arange(len(ids))
            elif role_logits.shape[0] == total_items:
                logit_rows = torch.from_numpy(ids)
            else:
                raise ValueError(f"role_logits must have {total_items} rows or "
                                 f"{len(ids)} rows aligned with item_ids")
            mask = coverage[torch.from_numpy(ids)] > 0
            if bool(mask.any()):
                chosen = role_logits[logit_rows][mask]
                orbit = -(target[torch.from_numpy(ids)][mask]
                          * F.log_softmax(chosen, -1)).sum(-1).mean()
                total = total + weights["neighbour_orbit"] * orbit
                result["orbit"] = float(orbit.detach())
        result["total"] = float(total.detach())
        return result, total

    # -- integrity --------------------------------------------------------
    def verify(self) -> dict:
        checks = {}
        for name, expected in self.meta["shas"].items():
            path = HERE.parents[2] / name
            checks[name] = {"ok": path.is_file() and _sha(path) == expected,
                            "expected": expected}
        checks["carrier_sha"] = {"ok": gt3.param_sha_from_state(self.carrier.state_dict())
                                 == self.meta["carrier_sha"],
                                 "expected": self.meta["carrier_sha"]}
        healthy = all(entry["ok"] for entry in checks.values())
        return {"healthy": healthy, "checks": checks}


def build_bundle(out_dir: Path | None = None, v3_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else BUNDLE_DIR
    v3_dir = Path(v3_dir) if v3_dir is not None else V3_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = v3_dir / "teacher_structure_v3.pth"
    targets_path = v3_dir / "structure_targets_v3.npz"
    contract_path = v3_dir / "teacher_structure_contract_v3.json"
    for path in (teacher_path, targets_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    carrier = st.load_carrier_v3(teacher_path)
    payload = torch.load(teacher_path, map_location="cpu")
    contract = json.loads(contract_path.read_text())
    targets = st.load_targets_v3(targets_path)
    weights = contract["supervision_recommendation"]["weights"]
    meta = {"version": BUNDLE_VERSION,
            "carrier_sha": gt3.param_sha_from_state(carrier.state_dict()),
            "teacher_source": str(teacher_path.relative_to(HERE.parents[2])),
            "targets_source": str(targets_path.relative_to(HERE.parents[2])),
            "contract_source": str(contract_path.relative_to(HERE.parents[2])),
            "shas": {str(teacher_path.relative_to(HERE.parents[2])): _sha(teacher_path),
                     str(targets_path.relative_to(HERE.parents[2])): _sha(targets_path),
                     str(contract_path.relative_to(HERE.parents[2])): _sha(contract_path)},
            "source_build_sha": payload["build_sha"],
            "counts": {"items": int(len(targets["coverage"])),
                       "neighbourhood_items": int((targets["coverage_neighbour"] > 0).sum()),
                       "own_orbit_items": int((targets["coverage"] > 0).sum())}}
    bundle_path = out_dir / "structure_teacher_bundle.pth"
    torch.save({"carrier": carrier.state_dict(), "in_dim": carrier.in_dim,
                "hidden": carrier.hidden, "widths": list(carrier.widths),
                "capabilities": contract["carrier"],
                "weights": weights, "meta": meta}, bundle_path)
    bundle_contract = {
        "version": BUNDLE_VERSION,
        "bundle_file": bundle_path.name,
        "targets_source": meta["targets_source"],
        "teacher_source": meta["teacher_source"],
        "source_contract": meta["contract_source"],
        "shas": meta["shas"],
        "api": {
            "load": "structure_teacher_bundle.load_structure_teacher(path)",
            "recognise": "induced adjacency matrix -> {family, key, orbits}",
            "targets_for": "item ids -> graph-derived supervision rows",
            "predict_targets": "768-d embeddings -> carrier prediction (portable fallback)",
            "structure_loss": "student vectors (+ optional role logits) -> losses"},
        "capabilities": contract["carrier"],
        "weights": weights,
        "boundary": {"rqvae_input_changed": False,
                     "rqvae_encoder_input": "the original 768-d Stage1 item embedding only",
                     "teacher_inference_needs_graph": False,
                     "used_only_in_training_loss": True},
        "not_established": ["direction prediction",
                            "per-role identification of minority roles",
                            "any SID collision or Recall effect"],
        "coverage": contract["coverage"]}
    contract_out = out_dir / "structure_teacher_bundle_contract.json"
    contract_out.write_text(json.dumps(bundle_contract, indent=1, default=str))
    manifest = {"version": BUNDLE_VERSION,
                "files": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
                          for path in (bundle_path, contract_out)}}
    manifest_out = out_dir / "structure_teacher_bundle_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=1))
    return {"bundle": str(bundle_path), "contract": str(contract_out),
            "manifest": str(manifest_out), "meta": meta, "weights": weights}


def load_structure_teacher(path) -> StructureTeacherBundle:
    path = Path(path)
    if path.is_dir():
        path = path / "structure_teacher_bundle.pth"
    payload = torch.load(path, map_location="cpu")
    carrier = st.CarrierV3(payload["in_dim"], payload["hidden"], tuple(payload["widths"]))
    carrier.load_state_dict(payload["carrier"])
    carrier.eval()
    carrier.requires_grad_(False)
    meta = payload["meta"]
    targets_path = HERE.parents[2] / meta["targets_source"]
    targets = st.load_targets_v3(targets_path)
    return StructureTeacherBundle(carrier, targets, payload["capabilities"], meta,
                                  payload["weights"])


def main() -> None:
    print(json.dumps(build_bundle(), indent=1, default=str))


if __name__ == "__main__":
    main()
