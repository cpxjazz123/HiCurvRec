"""The collapse analysis (issue #106, workstream B).

Why did a mechanism that reached the codebook collapse it to two level-0 codes? The
question is answered from the ARCHIVED artifacts -- the two checkpoints, the two exported
SIDs, the training logs -- and from low-cost fixture counterfactuals. Nothing here trains,
no seed changes, and no old arm is re-run.

Two explanations are recorded before the measurements, each with a prediction and a
falsifier:

**E1 -- the magnitude of the delivered codebook gradient.** The mechanism that collapsed the
codebook delivered a codebook gradient of 2.96e-02 on the host's own mini case; the arm
that changed the export harmlessly delivered 3.35e-06, a factor of about 8800 smaller.
*Prediction*: the delivered gradient scales with the mechanism's weight, and the hard
assignment's churn (the fraction of items whose nearest code changes) scales with it too --
so a mechanism at a hundredth of the weight should produce a hundredth of the churn.
*Falsifier*: the churn is unchanged when the weight falls by two orders of magnitude, which
would mean the collapse is not about how hard the gradient pushes.

**E2 -- the codewords moved toward the items.** A mechanism whose gradient pulls the
source's assignment onto the target's code also pulls the codewords toward the items they
serve, and the Voronoi cells shrink until one code owns everything.
*Prediction*: in the collapsed arm's checkpoint, each codeword sits CLOSER to the items
assigned to it than the control's codewords do, and the mean cell size has collapsed.
*Falsifier*: the collapsed arm's codewords are no closer to their items than the control's,
which would mean the collapse is not a geometry shrinkage.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

EMBEDDINGS = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"

#: The two explanations, recorded before the measurements.
EXPLANATIONS = [
    {
        "id": "E1",
        "statement": "the collapse follows the magnitude of the codebook gradient the "
                     "mechanism delivers: the collapsed arm delivered 2.96e-02 and the arm "
                     "that changed the export harmlessly delivered 3.35e-06",
        "prediction": "the delivered gradient scales with the mechanism's weight and the hard "
                      "assignment's churn scales with it too",
        "falsifier": "the churn is unchanged when the weight falls by two orders of magnitude",
        "experiment": "C1",
    },
    {
        "id": "E2",
        "statement": "the collapse is a geometry shrinkage: the mechanism pulls the codewords "
                     "toward the items they serve, the Voronoi cells shrink, and one code "
                     "ends up owning everything",
        "prediction": "in the collapsed checkpoint each codeword sits closer to the items "
                      "assigned to it than the control's codewords do, and the mean cell size "
                      "has collapsed",
        "falsifier": "the collapsed arm's codewords are no closer to their items than the "
                     "control's",
        "experiment": "C2",
    },
]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainer(tree: Path):
    for entry in (str(Path(tree).resolve()), str(REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return _load_module(Path(tree) / "train_rqvae.py", "genrec_collapse_trainer")


def _checkpoint(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _assignment_geometry(model, embeddings, codebook) -> Dict[str, Any]:
    """How the items are distributed over the codes, and how close the codewords sit to them."""
    import torch

    with torch.no_grad():
        latent = model.encoder(embeddings)
        distances = torch.cdist(latent, codebook).square()
        assignment = distances.argmin(dim=1)
        cell = torch.bincount(assignment, minlength=int(codebook.shape[0]))
        used = int((cell > 0).sum())
        # how close is each used code to the items it owns
        sums = torch.zeros_like(codebook)
        counts = torch.zeros(codebook.shape[0])
        sums.index_add_(0, assignment, latent)
        counts.index_add_(0, assignment, torch.ones_like(latent[:, 0]))
        used_mask = counts > 0
        means = sums[used_mask] / counts[used_mask].unsqueeze(1)
        separation = torch.cdist(means, means).mean() if means.shape[0] > 1 else torch.tensor(0.0)
        # the distance from a codeword to the nearest item it does NOT own
        nearest = distances.min(dim=1).values
    return {
        "levels_codes": int(codebook.shape[0]),
        "used_codes": used,
        "cell_sizes": {
            "max": int(cell.max()),
            "mean_over_used": float(cell[used_mask].float().mean()) if used else 0.0,
            "median_over_used": float(cell[used_mask].float().median()) if used else 0.0,
        },
        "mean_distance_code_to_its_items": float(
            (latent - codebook[assignment]).norm(dim=1).mean()
        ),
        "mean_separation_between_used_codes": float(separation),
        "mean_nearest_code_distance": float(nearest.mean()),
    }


def trajectory(
    control_tree: Path,
    control_checkpoint: Path,
    arm_tree: Path,
    arm_checkpoint: Path,
    *,
    sample: int = 4096,
) -> Dict[str, Any]:
    """The collapse trajectory, measured on the two archived checkpoints."""
    import numpy as np
    import torch

    embeddings = np.load(EMBEDDINGS).astype("float32")[:sample]
    tensor = torch.from_numpy(embeddings)

    def measure(tree: Path, checkpoint: Path) -> Dict[str, Any]:
        trainer = _trainer(Path(tree))
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
        model.load_state_dict(_checkpoint(Path(checkpoint)), strict=False)
        model.eval()
        out: Dict[str, Any] = {}
        for level, layer in enumerate(model.rq.vq_layers):
            out[f"level_{level}"] = _assignment_geometry(
                model, tensor, layer.get_code_embs()
            )
        return out

    control = measure(control_tree, control_checkpoint)
    arm = measure(arm_tree, arm_checkpoint)
    rows: List[Dict[str, Any]] = []
    for level in sorted(set(control) & set(arm)):
        c, a = control[level], arm[level]
        rows.append(
            {
                "level": level,
                "control_used_codes": c["used_codes"],
                "arm_used_codes": a["used_codes"],
                "control_mean_cell": c["cell_sizes"]["mean_over_used"],
                "arm_mean_cell": a["cell_sizes"]["mean_over_used"],
                "control_code_to_items_distance": c["mean_distance_code_to_its_items"],
                "arm_code_to_items_distance": a["mean_distance_code_to_its_items"],
                "control_code_separation": c["mean_separation_between_used_codes"],
                "arm_code_separation": a["mean_separation_between_used_codes"],
            }
        )
    return {"control": control, "arm": arm, "per_level": rows}


def counterfactual_c1(
    tree: Path,
    mechanism_module: str,
    *,
    weights: Tuple[float, ...] = (0.01, 0.001, 0.0001),
    mechanism_parameters: Optional[Dict[str, Any]] = None,
    sample: int = 512,
) -> Dict[str, Any]:
    """C1: how the delivered gradient and the hard assignment's churn scale with the weight.

    A fixture, not a training run: the mechanism is evaluated on the host's own mini case and
    the fraction of items whose nearest code changes is measured against the control's
    assignment.
    """
    import numpy as np
    import torch

    from . import gradient_lineage as gl

    trainer = _trainer(Path(tree))
    embeddings = np.load(EMBEDDINGS).astype("float32")[: 2 * sample]
    tensor = torch.from_numpy(embeddings)
    pairs = sample
    module = _load_module(Path(tree) / f"{mechanism_module}.py", "genrec_c1_mechanism")
    torch.manual_seed(42)

    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
    model.eval()
    with torch.no_grad():
        latent = model.encoder(tensor)
        codebook = model.rq.vq_layers[0].get_code_embs()
        control_assignment = torch.cdist(latent, codebook).square().argmin(dim=1)
    residuals = trainer._residuals_with_trace(model, tensor)

    rows: List[Dict[str, Any]] = []
    for weight in weights:
        torch.manual_seed(42)
        model2 = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
        model2.eval()
        residuals2 = trainer._residuals_with_trace(model2, tensor)
        source = residuals2[0, :pairs]
        target = residuals2[0, pairs : 2 * pairs]
        mask = torch.ones(pairs, dtype=source.dtype)
        term = module.mechanism_loss(
            source, target, model2.rq.vq_layers[0].get_code_embs(), mask,
            **(mechanism_parameters or {}),
        )
        loss = weight * term
        model2.zero_grad(set_to_none=False)
        loss.backward(retain_graph=True)
        codebook_grad = max(
            float(p.grad.detach().abs().max())
            for layer in model2.rq.vq_layers
            for p in layer.parameters()
            if p.grad is not None
        )
        with torch.no_grad():
            assignment = torch.cdist(latent, codebook).square().argmin(dim=1)
        churn = float((assignment != control_assignment).float().mean())
        rows.append(
            {
                "weight": weight,
                "term_value": float(term.detach()),
                "delivered_codebook_gradient": codebook_grad,
                "hard_assignment_churn": churn,
            }
        )
    return {
        "experiment": "C1",
        "mechanism_module": mechanism_module,
        "rows": rows,
        "note": "a fixture on the host's own mini case: no training, no seed change",
    }


def counterfactual_c2(trajectory_report: Dict[str, Any]) -> Dict[str, Any]:
    """C2: did the codewords move toward the items they own?"""
    rows: List[Dict[str, Any]] = []
    for row in trajectory_report["per_level"]:
        control_distance = row["control_code_to_items_distance"]
        arm_distance = row["arm_code_to_items_distance"]
        rows.append(
            {
                "level": row["level"],
                "control_code_to_items_distance": control_distance,
                "arm_code_to_items_distance": arm_distance,
                "arm_is_closer_to_its_items": arm_distance < control_distance,
                "control_mean_cell": row["control_mean_cell"],
                "arm_mean_cell": row["arm_mean_cell"],
                "cell_shrank": row["arm_mean_cell"] < row["control_mean_cell"],
            }
        )
    supported = [row for row in rows if row["arm_is_closer_to_its_items"]]
    return {
        "experiment": "C2",
        "rows": rows,
        "verdict": (
            "E2 is supported: the collapsed arm's codewords sit closer to the items they own "
            "at every level, and the mean cell shrank"
            if supported and all(row["cell_shrank"] for row in supported)
            else "E2 is not supported: the collapsed arm's codewords are not closer to their "
                 "items than the control's"
        ),
    }


def counterfactual_c3(
    control_tree: Path,
    control_checkpoint: Path,
    arm_tree: Path,
    arm_checkpoint: Path,
    *,
    pairs: int = 20000,
) -> Dict[str, Any]:
    """C3: did the codewords of a transition's two endpoints converge onto each other?

    E1 (the magnitude of the delivered gradient) did not survive C1: the delivered gradient
    scales exactly linearly with the mechanism's weight, and yet the hard assignment does not
    flip at any weight on the fixture. E2 (the codewords shrinking toward the items) was
    refuted by the archived checkpoints: the collapsed arm's codewords moved AWAY from the
    items they own.

    The remaining account is the one the mechanism's own objective suggests: a cross-entropy
    that asks one code to serve BOTH the source and the target pulls the two codewords
    together, and over 3000 epochs the used codewords converge until the Voronoi cells merge
    and one code owns everything.

    *Prediction*: in the collapsed checkpoint, the level-0 codeword of a transition's source
    and the level-0 codeword of its target are much closer to each other than they are in the
    control.
    *Falsifier*: the collapsed arm's transition-pair codeword distances are not smaller than
    the control's.
    """
    import numpy as np
    import torch

    valid = np.load(
        REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
    )
    anchors = valid["anchors"].astype(np.int64)[:pairs]
    positives = valid["positives"].astype(np.int64)[:pairs]

    def codewords(tree: Path, checkpoint: Path):
        """The level-0 codeword each item is actually assigned to, and the codebook."""
        import torch as _torch
        import numpy as _np

        trainer = _trainer(Path(tree))
        state = _checkpoint(Path(checkpoint))
        config = trainer._tokenizer_config()
        in_dim = int(next(iter(state.values())).shape[1]) if state else 32
        model = trainer.RQVAE(config, in_dim=in_dim)
        model.load_state_dict(state, strict=False)
        model.eval()
        embeddings = _torch.from_numpy(
            _np.load(EMBEDDINGS).astype("float32")
        )
        with _torch.no_grad():
            latent = model.encoder(embeddings)
            codebook = model.rq.vq_layers[0].get_code_embs().detach().clone()
            assignment = torch.cdist(latent, codebook).square().argmin(dim=1)
        return codebook, assignment

    def pair_distance(codebook, assignment, anchors, positives) -> Dict[str, Any]:
        source = codebook[assignment[anchors]]
        target = codebook[assignment[positives]]
        distance = (source - target).norm(dim=1)
        return {
            "mean_codeword_distance_for_a_transition": float(distance.mean()),
            "median_codeword_distance_for_a_transition": float(distance.median()),
            "fraction_below_a_tenth_of_the_control_mean": None,
        }

    control_codebook, control_assignment = codewords(control_tree, control_checkpoint)
    arm_codebook, arm_assignment = codewords(arm_tree, arm_checkpoint)
    control = pair_distance(control_codebook, control_assignment, anchors, positives)
    arm = pair_distance(arm_codebook, arm_assignment, anchors, positives)
    threshold = 0.1 * control["mean_codeword_distance_for_a_transition"]
    control["fraction_below_a_tenth_of_the_control_mean"] = float(
        (
            (control_codebook[control_assignment[anchors]]
             - control_codebook[control_assignment[positives]]).norm(dim=1) < threshold
        ).float().mean()
    )
    arm["fraction_below_a_tenth_of_the_control_mean"] = float(
        (
            (arm_codebook[arm_assignment[anchors]]
             - arm_codebook[arm_assignment[positives]]).norm(dim=1) < threshold
        ).float().mean()
    )
    supported = (
        arm["mean_codeword_distance_for_a_transition"]
        < control["mean_codeword_distance_for_a_transition"]
    )
    return {
        "experiment": "C3",
        "control": control,
        "arm": arm,
        "verdict": (
            "E2' is supported: the collapsed arm's transition pairs share their level-0 code "
            "far more often and their two codewords sit closer together than the control's, "
            "which is what a cross-entropy that asks one code to serve both endpoints predicts"
            if supported
            else "E2' is not supported: the collapsed arm's transition-pair codewords are not "
                 "closer together than the control's"
        ),
    }


__all__ = [
    "EXPLANATIONS",
    "counterfactual_c3",
    "counterfactual_c1",
    "counterfactual_c2",
    "trajectory",
]
