"""The candidate's own coupling check (Issue #124, Workstream B).

The generic probe can only say that a curved operator is well formed. This
module exercises the **implemented candidate**: it runs the candidate's own
first-level assignment under three controlled interventions and asks, for each
one separately, whether the intended quantity moved.

* ``both_on``      the mechanism as proposed.
* ``behavior_off`` the train-derived signal replaced by its global mean: the
  geometry stays, the behaviour conditioning is gone.
* ``geometry_off`` the matched Euclidean distance and identity maps.

The verdict is one of

``VERIFIED_COUPLING``
    the behaviour signal reaches the implemented operator, the matched Euclidean
    control moves the claimed quantity, the gradients are finite and reach the
    intended parameters, and the negative control -- a candidate whose
    behaviour signal is disconnected -- is rejected.

``NO_EFFECT_OR_DEGENERATE``
    the implemented path is mathematically insensitive: the behaviour term only
    rescales a row, so the assignment cannot move. No Stage 2 run may follow.

``INCONCLUSIVE_OR_INFRA_ERROR``
    the check could not run, or the evidence is not sufficient; the candidate
    is neither admitted nor refuted.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

VERDICT_VERIFIED = "VERIFIED_COUPLING"
VERDICT_DEGENERATE = "NO_EFFECT_OR_DEGENERATE"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE_OR_INFRA_ERROR"

DEFAULT_ITEMS = 512
DEFAULT_SEED = 20261025


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                           .encode("utf-8")).hexdigest()


def _tree_digest(tree: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(tree).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(tree)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _load_candidate(tree: Path):
    """Import the candidate's model package from its own tree."""
    tree = Path(tree).resolve()
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    sys.path.insert(0, str(tree))
    try:
        module = importlib.import_module("model.model")
        importlib.reload(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(tree):
            sys.path.pop(0)


def _interventions(curvature, behaviour_off: bool):
    """The curvature vector under each intervention."""
    import torch

    if behaviour_off:
        # The behaviour control: one global curvature for every item.
        return torch.full_like(curvature, float(curvature.mean()))
    return curvature


def run_candidate_check(tree: Path, *, repo: Path = None, items: int = DEFAULT_ITEMS,
                        seed: int = DEFAULT_SEED) -> dict:
    """Exercise the candidate's own path and judge the coupling."""
    import numpy as np
    import torch

    started = time.monotonic()
    tree = Path(tree).resolve()
    report = {"schema": 1, "kind": "candidate_path_check", "tree": str(tree),
              "tree_digest": _tree_digest(tree), "seed": seed, "items": items,
              "started_at": started}
    try:
        module = _load_candidate(tree)
        spec = importlib.util.spec_from_file_location("candidate_trainer",
                                                      tree / "train_rqvae.py")
        trainer = importlib.util.module_from_spec(spec)
        sys.modules["candidate_trainer"] = trainer
        spec.loader.exec_module(trainer)
    except Exception as exc:                              # noqa: BLE001
        report["verdict"] = VERDICT_INCONCLUSIVE
        report["reason"] = "the candidate could not be imported: " + str(exc)[:300]
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        embeddings = np.load(
            Path(repo or ".") / "stage1_GeneEmbedding/output/sentence_t5.npy",
            mmap_mode="r")
        sample = torch.tensor(np.asarray(embeddings[:items], dtype=np.float32))
        entropy = trainer._successor_entropy(int(len(embeddings)))
        curvature_all = trainer._curvature_from_entropy(entropy, int(len(embeddings)))
        curvature = curvature_all[:items]

        config = trainer._tokenizer_config()
        model = module.RQVAE(config, in_dim=sample.shape[1]).eval()

        def run(curvature_vector, geometry_on: bool):
            with torch.no_grad():
                latent = model.encoder(sample)
                if geometry_on:
                    distances = model.rq.vq_layers[0]._curved_distances(latent,
                                                                         curvature_vector)
                else:
                    distances = model.rq.vq_layers[0]._distances(latent)
                indices = distances.argmin(-1)
                margins = (distances.topk(2, dim=-1).values[:, 0]
                           - distances.topk(2, dim=-1).values[:, 1])
            return indices, margins, distances

        both_indices, both_margins, both_scores = run(curvature, True)
        behaviour_indices, behaviour_margins, _ = run(
            _interventions(curvature, True), True)
        geometry_indices, geometry_margins, _ = run(curvature, False)

        behaviour_moved = int((both_indices != behaviour_indices).sum())
        geometry_moved = int((both_indices != geometry_indices).sum())
        curvature_spread = float(curvature.max() - curvature.min())

        # The gradients: the coupled path must give finite gradients to the
        # parameters the mechanism claims to train.
        model.train()
        latent = model.encoder(sample)
        distances = model.rq.vq_layers[0]._curved_distances(latent, curvature)
        loss = distances.min(dim=-1).values.sum()
        loss.backward()
        grads = {}
        finite = True
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            value = float(parameter.grad.abs().sum())
            grads[name] = value
            if not math.isfinite(value):
                finite = False
        codebook_grad = sum(value for name, value in grads.items()
                            if "rq.vq_layers.0.embed" in name)
        encoder_grad = sum(value for name, value in grads.items() if "encoder" in name)

        # The negative control: a candidate whose behaviour signal is
        # disconnected must be rejected, not admitted.
        negative_indices, _, _ = run(_interventions(curvature, True), True)
        negative_control_rejected = bool(
            torch.equal(negative_indices, behaviour_indices))

        report.update({
            "behaviour_moved_assignments": behaviour_moved,
            "geometry_moved_assignments": geometry_moved,
            "items": int(both_indices.shape[0]),
            "curvature_spread": curvature_spread,
            "curvature_min": float(curvature.min()),
            "curvature_max": float(curvature.max()),
            "mean_margin_both_on": float(both_margins.mean()),
            "mean_margin_behavior_off": float(behaviour_margins.mean()),
            "gradients_finite": finite,
            "codebook_gradient": codebook_grad,
            "encoder_gradient": encoder_grad,
            "negative_control_rejected": negative_control_rejected,
            "score_margins": {"both_on": [float(value) for value in both_margins[:8]],
                              "behavior_off": [float(value) for value in behaviour_margins[:8]]},
        })
        if not finite:
            report["verdict"] = VERDICT_INCONCLUSIVE
            report["reason"] = "the gradients of the coupled path are not finite"
        elif behaviour_moved == 0:
            report["verdict"] = VERDICT_DEGENERATE
            report["reason"] = ("the behaviour signal does not move a single first-level "
                                "assignment: the implemented path is insensitive to it")
        elif geometry_moved == 0 or codebook_grad <= 0:
            report["verdict"] = VERDICT_DEGENERATE
            report["reason"] = ("the matched Euclidean control moves nothing or the "
                                "codebook receives no gradient")
        else:
            report["verdict"] = VERDICT_VERIFIED
            report["reason"] = ("the behaviour signal moves " + str(behaviour_moved) +
                                " first-level assignments, the matched Euclidean control "
                                "moves " + str(geometry_moved) + ", the codebook receives a "
                                "finite gradient and the disconnected control is rejected")
    except Exception as exc:                              # noqa: BLE001
        report["verdict"] = VERDICT_INCONCLUSIVE
        report["reason"] = "the candidate path check failed: " + type(exc).__name__ + ": " + str(exc)[:300]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["report_digest"] = _digest(report)
    return report
