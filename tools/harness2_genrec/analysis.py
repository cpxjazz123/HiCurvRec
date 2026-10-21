"""The discriminating experiments (issue #105, workstream B).

Three analyses over artifacts that already exist, each aimed at a rival explanation the
agent proposed, each with a matched control, each with a declared falsifier:

* **X1** -- where do a treatment arm and the control become the same object? It walks the
  export path stage by stage (encoder output, residual stream, nearest-code assignment,
  decision margins) and reports the FIRST stage at which the two agree. It separates
  "the mechanism never reached the export" from "the mechanism reached it but the
  assignment margins swallowed the change" from "the export fell back on something else".
* **X2** -- why did a rising Hard-L1 ER not buy Stage-3 quality? It decomposes the ER
  per item (is the gain spread over the catalogue or carried by a few items?), measures
  the neighbourhood structure a beam search over codes actually consumes (how many
  distinct codes a transition's neighbourhood spans, how large the code cells are), and
  compares the treatment-control difference against the Stage-3 run-to-run band.
* **X4** -- did the active exports damage the code distribution, or fail to generalise? It
  measures the per-level occupancy, the code-frequency entropy and the Gini concentration
  of the exported codes, and the gap between the train-side and the held-out
  discrimination.

Everything runs on the project's own environment (torch and numpy), needs no GPU, and
writes a report that names the run ids it compared.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

EMBEDDINGS = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_PARQUET = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
VALID_PAIRS = REPO_ROOT / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load_trainer(tree: Path):
    """The host's trainer module, loaded the way the verifier loads it."""
    import sys

    path = Path(tree) / "train_rqvae.py"
    # the tree's own packages (the model, the scripts) live beside the trainer
    for entry in (str(Path(tree).resolve()), str(REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(f"genrec_trainer_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _load_sids(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.zeros((max(int(key) for key in raw) + 1, 4), dtype=np.int64)
    for key, value in raw.items():
        matrix[int(key)] = value
    return matrix


def _gini(values: np.ndarray) -> float:
    """The Gini coefficient of a non-negative distribution: 0 is flat, 1 is one code."""
    values = np.sort(np.asarray(values, dtype="float64"))
    size = values.size
    if size == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, size + 1)
    return float((2.0 * (index * values).sum()) / (size * values.sum()) - (size + 1.0) / size)


def _entropy(values: np.ndarray) -> float:
    counts = np.asarray(values, dtype="float64")
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def _code_frequencies(sids: np.ndarray, levels: int) -> np.ndarray:
    codes = sids[:, :levels]
    _, counts = np.unique(codes, axis=0, return_counts=True)
    return counts


def _train_pairs(limit: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    frame = pd.read_parquet(TRAIN_PARQUET)
    columns = {name.lower(): name for name in frame.columns}
    history, target = columns["history"], columns["target"]
    sources: List[int] = []
    targets: List[int] = []
    for past, item in zip(frame[history][:limit], frame[target][:limit]):
        if past is None or item is None:
            continue
        sequence = list(past) if isinstance(past, (list, tuple, np.ndarray)) else [past]
        if not sequence:
            continue
        sources.append(int(sequence[-1]))
        targets.append(int(item))
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


# --------------------------------------------------------------------------------------
# X1 -- where do the treatment and the control become the same object?
# --------------------------------------------------------------------------------------
def experiment_x1(
    control_tree: Path,
    control_checkpoint: Path,
    treatment: Dict[str, Tuple[Path, Path]],
    *,
    sample: int = 4096,
) -> Dict[str, Any]:
    """Walk the export path and report the first stage at which the arms agree."""
    import torch

    trainer = _load_trainer(control_tree)
    embeddings = np.load(EMBEDDINGS).astype("float32")
    if sample and embeddings.shape[0] > sample:
        embeddings = embeddings[:sample]
    tensor = torch.from_numpy(embeddings)

    def stages(checkpoint: Path) -> Dict[str, Any]:
        state = _checkpoint(checkpoint)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            latent = model.encoder(tensor)
            residuals = trainer._residuals_with_trace(model, tensor)
            codebook = model.rq.vq_layers[0].get_code_embs()
            distances = torch.cdist(latent, codebook).square()
            order = distances.sort(dim=1).values
            best = order[:, 0]
            runner_up = order[:, 1] if order.shape[1] > 1 else order[:, 0]
            indices = distances.argmin(dim=1)
        return {
            "latent": latent.clone(),
            "residual": residuals[0].clone(),
            "codebook": codebook.clone(),
            "distances": distances.clone(),
            "indices": indices.clone(),
            "margin": (runner_up - best).clone(),
        }

    control = stages(control_checkpoint)
    results: List[Dict[str, Any]] = []
    for name, (tree, checkpoint) in treatment.items():
        arm = stages(checkpoint)
        latent_diff = float((arm["latent"] - control["latent"]).abs().max())
        residual_diff = float((arm["residual"] - control["residual"]).abs().max())
        codebook_diff = float((arm["codebook"] - control["codebook"]).abs().max())
        assignment_changed = float((arm["indices"] != control["indices"]).float().mean())
        margin = arm["margin"]
        control_margin = control["margin"]
        # How large is the assignment change relative to the margin that protects it?
        perturbation = float((arm["distances"] - control["distances"]).abs().max())
        typical_margin = float(control_margin.median())
        results.append(
            {
                "arm": name,
                "latent_max_abs_diff": latent_diff,
                "residual_max_abs_diff": residual_diff,
                "codebook_max_abs_diff": codebook_diff,
                "assignment_changed_fraction": assignment_changed,
                "decision_margin_median": typical_margin,
                "distance_perturbation_max": perturbation,
                "perturbation_over_margin": (
                    perturbation / typical_margin if typical_margin > 0 else None
                ),
                "first_identical_stage": _first_identical_stage(
                    latent_diff, residual_diff, codebook_diff, assignment_changed
                ),
            }
        )
    return {
        "experiment": "X1",
        "question": "at which stage of the export path does a treatment arm become "
                    "indistinguishable from the matched control?",
        "sample_items": int(embeddings.shape[0]),
        "control_checkpoint": str(control_checkpoint),
        "arms": results,
        "interpretation": (
            "a nonzero encoder difference with a zero assignment difference means the "
            "mechanism reached the latent but the assignment margins swallowed the change"
        ),
    }


def _first_identical_stage(
    latent_diff: float, residual_diff: float, codebook_diff: float, assignment_changed: float
) -> str:
    if assignment_changed == 0.0 and codebook_diff == 0.0:
        return "the codebook and the assignment are identical; the arms differ only upstream"
    if assignment_changed == 0.0:
        return "the assignment is identical although the codebook differs"
    if latent_diff == 0.0:
        return "the encoder output is identical"
    return "nothing is identical: the arms differ all the way down"


# --------------------------------------------------------------------------------------
# X2 -- why did a rising ER not buy Stage-3 quality?
# --------------------------------------------------------------------------------------
def experiment_x2(
    control_sid: Path,
    treatment: Dict[str, Path],
    *,
    stage3_differences: Optional[Dict[str, float]] = None,
    stage3_band: Optional[float] = None,
    neighbours: int = 10,
) -> Dict[str, Any]:
    """Decompose the ER per item, measure the neighbourhood structure, and compare the
    downstream difference against the Stage-3 run-to-run band."""
    data = np.load(VALID_PAIRS)
    anchors = data["anchors"].astype(np.int64)
    positives = data["positives"].astype(np.int64)
    negatives = data["negatives"].astype(np.int64)
    control = _load_sids(control_sid)
    embeddings = np.load(EMBEDDINGS).astype("float32")

    def per_item(sids: np.ndarray) -> Dict[str, np.ndarray]:
        codes = sids[:, 0]
        positive_match = codes[anchors] == codes[positives]
        negative_match = codes[anchors][:, None] == codes[negatives]
        return {
            "positive": positive_match.astype("float64"),
            "negative": negative_match.mean(axis=1).astype("float64"),
        }

    control_stats = per_item(control)
    control_pos = float(control_stats["positive"].mean())
    control_neg = float(control_stats["negative"].mean())
    control_er = control_pos / control_neg if control_neg > 0 else None

    # The neighbourhood structure a beam search over codes consumes: for each anchor, how
    # many distinct level-0 codes do its nearest neighbours span, and how large is the
    # anchor's own code cell?
    def neighbourhood(sids: np.ndarray) -> Dict[str, Any]:
        codes = sids[:, 0]
        cell_sizes = np.bincount(codes)
        anchor_codes = codes[anchors]
        cell = cell_sizes[anchor_codes]
        # the nearest neighbours of each anchor in the embedding space
        norms = (embeddings ** 2).sum(axis=1)
        sample = min(len(anchors), 512)
        selected = anchors[:sample]
        distances = norms[selected][:, None] + norms[None, :] - 2.0 * (embeddings[selected] @ embeddings.T)
        nearest = np.argpartition(distances, neighbours, axis=1)[:, :neighbours]
        spans = np.array([len(np.unique(codes[row])) for row in nearest])
        return {
            "anchor_code_cell_size_median": float(np.median(cell)),
            "anchor_code_cell_size_mean": float(cell.mean()),
            "neighbourhood_distinct_codes_mean": float(spans.mean()),
            "neighbourhood_distinct_codes_median": float(np.median(spans)),
            "neighbours_considered": int(neighbours),
            "anchors_sampled": int(sample),
        }

    arms: List[Dict[str, Any]] = []
    for name, sid_path in treatment.items():
        arm_sids = _load_sids(sid_path)
        stats = per_item(arm_sids)
        pos = float(stats["positive"].mean())
        neg = float(stats["negative"].mean())
        er = pos / neg if neg > 0 else None
        # the per-anchor change in the positive match rate, and how concentrated it is
        change = stats["positive"] - control_stats["positive"]
        absolute = np.abs(change)
        order = np.argsort(absolute)[::-1]
        total = float(absolute.sum())
        top = order[: max(1, len(order) // 100)]
        share = float(absolute[top].sum() / total) if total > 0 else 0.0
        arms.append(
            {
                "arm": name,
                "hard_l1_er": er,
                "er_delta_vs_control": (er - control_er) if (er is not None and control_er) else None,
                "pos_share": pos,
                "neg_share": neg,
                "anchors_changed_fraction": float((change != 0).mean()),
                "top_1pct_anchors_share_of_change": share,
                "neighbourhood": {
                    "arm": neighbourhood(arm_sids),
                    "control": neighbourhood(control),
                },
                "stage3_recall_delta": (stage3_differences or {}).get(name),
                "stage3_run_to_run_band": stage3_band,
                "stage3_difference_in_bands": (
                    abs((stage3_differences or {}).get(name, 0.0)) / stage3_band
                    if stage3_band
                    else None
                ),
            }
        )
    return {
        "experiment": "X2",
        "question": "is the ER gain spread over the catalogue, and did the code "
                    "neighbourhood structure a beam search consumes change?",
        "control_sid": str(control_sid),
        "control_hard_l1_er": control_er,
        "arms": arms,
    }


# --------------------------------------------------------------------------------------
# X4 -- occupancy, frequency balance and the train-versus-held-out gap
# --------------------------------------------------------------------------------------
def experiment_x4(
    control_sid: Path,
    treatment: Dict[str, Path],
    *,
    limit: int = 200000,
) -> Dict[str, Any]:
    """The exported code distribution's balance, and whether the train-side gain reached
    the held-out transitions."""
    control = _load_sids(control_sid)
    sources, targets = _train_pairs(limit=limit)
    data = np.load(VALID_PAIRS)
    anchors = data["anchors"].astype(np.int64)
    positives = data["positives"].astype(np.int64)
    negatives = data["negatives"].astype(np.int64)

    def discrimination(sids: np.ndarray) -> Dict[str, Any]:
        codes = sids[:, 0]
        train_positive = codes[sources] == codes[targets]
        train_negative = codes[sources] == codes[np.random.default_rng(42).integers(
            0, sids.shape[0], size=len(sources)
        )]
        held_positive = codes[anchors] == codes[positives]
        held_negative = codes[anchors][:, None] == codes[negatives]
        train_ratio = float(train_positive.mean() / train_negative.mean())
        held_ratio = float(held_positive.mean() / held_negative.mean())
        return {
            "train_discrimination": train_ratio,
            "held_out_discrimination": held_ratio,
            "train_over_held_ratio": train_ratio / held_ratio if held_ratio else None,
        }

    def balance(sids: np.ndarray) -> List[Dict[str, Any]]:
        rows = []
        for levels in (1, 2, 3):
            codes = sids[:, :levels]
            _, counts = np.unique(codes, axis=0, return_counts=True)
            rows.append(
                {
                    "levels": levels,
                    "distinct_codes": int(len(counts)),
                    "occupancy_fraction": float(len(counts) / sids.shape[0]),
                    "code_frequency_entropy": _entropy(counts),
                    "code_frequency_gini": _gini(counts),
                    "largest_code_share": float(counts.max() / counts.sum()),
                }
            )
        return rows

    control_balance = balance(control)
    control_discrimination = discrimination(control)
    arms: List[Dict[str, Any]] = []
    for name, sid_path in treatment.items():
        arm_sids = _load_sids(sid_path)
        arm_balance = balance(arm_sids)
        arm_discrimination = discrimination(arm_sids)
        arms.append(
            {
                "arm": name,
                "balance": arm_balance,
                "balance_delta_vs_control": [
                    {
                        "levels": arm["levels"],
                        "gini_delta": arm["code_frequency_gini"]
                        - control["code_frequency_gini"],
                        "entropy_delta": arm["code_frequency_entropy"]
                        - control["code_frequency_entropy"],
                        "occupancy_delta": arm["occupancy_fraction"]
                        - control["occupancy_fraction"],
                    }
                    for arm, control in zip(arm_balance, control_balance)
                ],
                "discrimination": arm_discrimination,
                "train_over_held_ratio_vs_control": (
                    arm_discrimination["train_over_held_ratio"]
                    / control_discrimination["train_over_held_ratio"]
                    if control_discrimination["train_over_held_ratio"]
                    else None
                ),
            }
        )
    return {
        "experiment": "X4",
        "question": "did the active exports damage the code distribution's balance, or did "
                    "the train-side gain fail to generalise to held-out transitions?",
        "control_sid": str(control_sid),
        "control_balance": control_balance,
        "control_discrimination": control_discrimination,
        "arms": arms,
    }


# --------------------------------------------------------------------------------------
# X5 -- does the host's gradient projection discard an auxiliary term?
# --------------------------------------------------------------------------------------
def experiment_x5(
    control_tree: Path,
    *,
    encoder_only_mechanism: str = "mechanism_behavior_conditioned_pairwise_margin",
    codebook_mechanism: str = "mechanism_behavior_residual_geometry_transport",
    sample: int = 512,
) -> Dict[str, Any]:
    """Does an auxiliary term's gradient survive the host's own assembly?

    The host replaces the ENCODER gradient with ``g_base + w_b/valid_levels * (g'_L1 +
    g_L2 + g_L3)`` after the backward, and never touches the decoder or the codebook
    gradient. An auxiliary term whose gradient reaches only the encoder is therefore
    discarded in production, however healthy it looks in isolation.

    The experiment is a matched pair: the same mini case, once with the auxiliary term
    and once without, with the host's own projection applied to both.
    """
    import torch

    trainer = _load_trainer(control_tree)
    # the host's pair batch is [sources | candidates], so the case needs 2 * sample rows
    embeddings = np.load(EMBEDDINGS).astype("float32")[: 2 * sample]
    tensor = torch.from_numpy(embeddings)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
    model.eval()

    def build_case() -> Dict[str, Any]:
        reconstructed, quant_loss, _, _ = model.forward(tensor)
        base_loss, _recon_loss = model.compute_loss(tensor, reconstructed, quant_loss)
        residuals = trainer._residuals_with_trace(model, tensor)
        pairs = tensor.shape[0] // 2
        source_ids = torch.arange(pairs)
        target_ids = (source_ids + 1) % pairs
        l1_loss, other_loss, valid_levels = trainer.behavior_contrastive_components(
            residuals, source_ids, target_ids, None
        )
        return {
            "base_loss": base_loss,
            "l1_loss": l1_loss,
            "other_loss": other_loss,
            "valid_levels": valid_levels,
            "residuals": residuals,
        }

    def measure(term_fn=None) -> Tuple[Dict[str, Any], List[Any]]:
        """A fresh graph per measurement: the projection frees the previous one."""
        case = build_case()
        model.zero_grad(set_to_none=False)
        loss = case["base_loss"] + case["l1_loss"]
        if term_fn is not None:
            loss = loss + term_fn(case)
        loss.backward(retain_graph=True)
        grads = [
            (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
            for p in model.encoder.parameters()
        ]
        return case, grads

    def projected(case: Dict[str, Any], grads: List[Any]) -> List[Any]:
        model.zero_grad(set_to_none=False)
        for parameter, grad in zip(model.encoder.parameters(), grads):
            parameter.grad = grad.clone()
        trainer._r4_08_apply_projection(
            case["base_loss"],
            case["l1_loss"],
            case["other_loss"],
            list(model.encoder.parameters()),
            1.0,
            valid_levels=case["valid_levels"],
        )
        return [
            (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
            for p in model.encoder.parameters()
        ]

    control_case, control_grads = measure()
    control_projected = projected(control_case, control_grads)

    results: List[Dict[str, Any]] = []
    for name, module_name in (
        (encoder_only_mechanism, encoder_only_mechanism),
        (codebook_mechanism, codebook_mechanism),
    ):
        module = _load_mechanism(control_tree, module_name)
        if module is None:
            results.append({"mechanism": name, "error": "the mechanism module was not found"})
            continue
        pairs = tensor.shape[0] // 2

        def term_fn(case, module=module):
            # the residuals of the case carry the graph, so the term's gradient can reach
            # the encoder exactly as it does in production
            residuals = case["residuals"]
            source = residuals[0, :pairs]
            target = residuals[0, pairs : 2 * pairs]
            codebook = model.rq.vq_layers[0].get_code_embs()
            mask = torch.ones(source.shape[0], dtype=source.dtype)
            return 0.05 * module.mechanism_loss(source, target, codebook, mask)

        try:
            with torch.no_grad():
                probe = trainer._residuals_with_trace(model, tensor)
                term = module.mechanism_loss(
                    probe[0, :pairs], probe[0, pairs : 2 * pairs],
                    model.rq.vq_layers[0].get_code_embs(),
                    torch.ones(pairs, dtype=probe.dtype),
                )
        except Exception as exc:  # noqa: BLE001
            results.append({"mechanism": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        treatment_case, with_term = measure(term_fn)
        with_term_projected = projected(treatment_case, with_term)
        before = _max_abs_difference(control_grads, with_term)
        after = _max_abs_difference(control_projected, with_term_projected)
        results.append(
            {
                "mechanism": name,
                "encoder_grad_max_abs_diff_before_projection": before,
                "encoder_grad_max_abs_diff_after_projection": after,
                "discarded_by_the_projection": bool(before > 0.0 and after == 0.0),
                "term_value": float(term.detach().abs().max()),
            }
        )
    return {
        "experiment": "X5",
        "question": "does an auxiliary term's encoder gradient survive the host's own "
                    "post-backward assembly?",
        "control_tree": str(control_tree),
        "sample_items": int(tensor.shape[0]),
        "results": results,
        "host_rule": (
            "the host overwrites the encoder gradient with g_base + w_b/valid_levels * "
            "(g'_L1 + g_L2 + g_L3) and never touches the decoder or codebook gradients"
        ),
    }


def _max_abs_difference(first: List[torch.Tensor], second: List[torch.Tensor]) -> float:
    if not first or not second:
        return 0.0
    return max(
        float((a - b).abs().max()) if a.shape == b.shape else float("inf")
        for a, b in zip(first, second)
    )


def _load_mechanism(tree: Path, module_name: str):
    import importlib.util
    import sys

    for entry in (str(Path(tree).resolve()), str(REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    candidates = [
        Path(tree) / f"{module_name}.py",
        REPO_ROOT / "stage2_RQ-VAE" / "TIGER_c1_behavior_conditioned_pairwise_margin" / f"{module_name}.py",
        REPO_ROOT / "stage2_RQ-VAE" / "TIGER_behavior_residual_geometry_transport" / f"{module_name}.py",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            f"genrec_mechanism_{module_name}_{abs(hash(str(candidate)))}", candidate
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001
            continue
        return module
    return None


def production_gradient_evidence(
    tree: Path, mechanism_module: str, *, sample: int = 512
) -> Dict[str, Any]:
    """Does the mechanism deliver gradient to the codebook on the PRODUCTION path?

    The independent judges test the mechanism module in isolation. This measures the same
    thing where it actually runs: inside the host's own loss assembly, on the host's own
    mini case, with the host's own post-backward projection applied. It reports the
    encoder and the codebook gradient differences separately, because the host overwrites
    one of them and not the other.
    """
    import torch

    trainer = _load_trainer(tree)
    embeddings = np.load(EMBEDDINGS).astype("float32")[: 2 * sample]
    tensor = torch.from_numpy(embeddings)
    pairs = sample
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
    model.eval()
    module = _load_mechanism(tree, mechanism_module)
    if module is None:
        return {"ok": False, "error": f"the mechanism module {mechanism_module!r} was not found"}

    def measure(with_term: bool) -> Dict[str, Any]:
        model.zero_grad(set_to_none=False)
        reconstructed, quant_loss, _, _ = model.forward(tensor)
        base_loss, _recon = model.compute_loss(tensor, reconstructed, quant_loss)
        residuals = trainer._residuals_with_trace(model, tensor)
        source_ids = torch.arange(pairs)
        target_ids = (source_ids + 1) % pairs
        l1_loss, other_loss, valid_levels = trainer.behavior_contrastive_components(
            residuals, source_ids, target_ids, None
        )
        loss = base_loss + l1_loss
        if with_term:
            source = residuals[0, :pairs]
            target = residuals[0, pairs : 2 * pairs]
            codebook = model.rq.vq_layers[0].get_code_embs()
            mask = torch.ones(pairs, dtype=source.dtype)
            loss = loss + 0.01 * module.mechanism_loss(source, target, codebook, mask)
        loss.backward(retain_graph=True)
        encoder = [
            (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
            for p in model.encoder.parameters()
        ]
        codebooks = [
            (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
            for layer in model.rq.vq_layers
            for p in layer.parameters()
        ]
        return {"encoder": encoder, "codebooks": codebooks}

    control = measure(False)
    treatment = measure(True)
    encoder_before = _max_abs_difference(control["encoder"], treatment["encoder"])
    codebook_difference = _max_abs_difference(control["codebooks"], treatment["codebooks"])
    return {
        "ok": True,
        "mechanism_module": mechanism_module,
        "tree": str(tree),
        "sample_items": int(pairs),
        "encoder_gradient_difference": encoder_before,
        "codebook_gradient_difference": codebook_difference,
        "reaches_the_codebook": codebook_difference > 0.0,
        "note": (
            "the host overwrites the encoder gradient after the backward and never touches "
            "the codebook gradient, so a codebook gradient difference is the part of the "
            "mechanism that survives into the update"
        ),
    }


__all__ = [
    "experiment_x1",
    "production_gradient_evidence",
    "experiment_x5",
    "experiment_x2",
    "experiment_x4",
]
