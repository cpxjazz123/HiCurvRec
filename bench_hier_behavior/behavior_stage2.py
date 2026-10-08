"""Train and evaluate matched Stage2 RQ-VAE geometry arms on real behavior."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import behavior_config as cfg
from .behavior_data import BehaviorHierarchy, HierarchyEdge, build_behavior_hierarchy
from .behavior_metrics import (
    _node_representations,
    behavior_hierarchy_stability,
    evaluate_next_item_ranking,
    paired_bootstrap_delta,
    sid_hierarchy_metrics,
)
from .cones import FACTOR, entailment_margin_loss, fit_k, xi_pairs
from .models import RQVAE


def _members(hierarchy: BehaviorHierarchy, node: int, count: int,
             rng: np.random.Generator) -> np.ndarray:
    items = hierarchy.node_items[int(node)]
    if len(items) == 0:
        raise ValueError(f"Hierarchy node {node} has no item members")
    return rng.choice(items, size=count, replace=len(items) < count).astype(np.int64)


def _sample_roles(
    hierarchy: BehaviorHierarchy,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_level = {
        level: [edge for edge in hierarchy.train_edges if edge.level == level]
        for level in range(3)
    }
    roles = []
    for level in range(3):
        edges = by_level[level]
        if not edges:
            raise ValueError(f"Behavior hierarchy has no training edges at L{level + 1}")
        picks = rng.integers(0, len(edges), size=cfg.HIERARCHY_EDGES_PER_LEVEL)
        for pick in picks:
            edge = edges[int(pick)]
            anchor_items = None
            child_item = None
            negative_item = None
            if level < 2:
                anchor_items = _members(
                    hierarchy, edge.child, cfg.PROTOTYPE_ITEMS, rng
                )
            else:
                child_item = int(hierarchy.node_items[edge.child][0])
                negative_item = int(hierarchy.node_items[edge.negative][0])
            roles.append(
                {
                    "level": level,
                    "parent_node": edge.parent,
                    "child_node": edge.child,
                    "negative_node": edge.negative if level == 1 else None,
                    "anchor_items": anchor_items,
                    "child_item": child_item,
                    "negative_item": negative_item,
                }
            )
    return roles


def _represent_hierarchy_sample(
    model: RQVAE,
    features: torch.Tensor,
    hierarchy: BehaviorHierarchy,
    roles: list[dict],
    k_by_level: tuple[float, float, float],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    item_arrays = []
    for role in roles:
        if role["anchor_items"] is not None:
            item_arrays.append(role["anchor_items"])
        if role["child_item"] is not None:
            item_arrays.append(np.asarray([role["child_item"]], dtype=np.int64))
            item_arrays.append(np.asarray([role["negative_item"]], dtype=np.int64))
    item_ids = np.unique(np.concatenate(item_arrays))
    positions = {int(item): index for index, item in enumerate(item_ids)}
    selected = torch.as_tensor(item_ids, dtype=torch.long, device=features.device)
    latent = model._training_latent(features[selected])
    _, _, _, prefixes = model.rq(latent, return_prefixes=True)
    losses = []
    for level in range(3):
        level_roles = [role for role in roles if role["level"] == level]
        parents = []
        children = []
        negatives = []
        anchors = []
        for role in level_roles:
            if level == 0:
                parent = model.root_direction
                child = model.coarse_node_prototypes[role["child_node"] - 1]
            elif level == 1:
                parent = model.coarse_node_prototypes[role["parent_node"] - 1]
                child = model.fine_node_prototypes[
                    role["child_node"] - 1 - hierarchy.n_coarse
                ]
                negatives.append(
                    model.fine_node_prototypes[
                        role["negative_node"] - 1 - hierarchy.n_coarse
                    ]
                )
            else:
                parent = model.fine_node_prototypes[
                    role["parent_node"] - 1 - hierarchy.n_coarse
                ]
                child = prefixes[2][positions[role["child_item"]]]
                negatives.append(prefixes[2][positions[role["negative_item"]]])
            parents.append(parent)
            children.append(child)
            if level < 2:
                anchor_positions = torch.as_tensor(
                    [positions[int(item)] for item in role["anchor_items"]],
                    dtype=torch.long,
                    device=features.device,
                )
                anchor = prefixes[level][anchor_positions].mean(dim=0)
                anchors.append(model.geometry.pair_distance(child, anchor).square())
        negative_tangent = torch.stack(negatives) if negatives else None
        cone_loss, _, _, _ = entailment_margin_loss(
            model.geometry_name,
            model.geometry,
            torch.stack(parents),
            torch.stack(children),
            negative_tangent,
            k=k_by_level[level],
            angle_margin=cfg.CONE_ANGLE_MARGIN,
            radial_margin=cfg.CONE_RADIAL_MARGIN,
            radial_weight=cfg.CONE_RADIAL_WEIGHT,
        )
        if anchors:
            cone_loss = cone_loss + torch.stack(anchors).mean()
        losses.append(cone_loss)
    return torch.stack(losses).mean(), losses


def _all_item_representations(
    model: RQVAE,
    features: torch.Tensor,
) -> tuple[np.ndarray, list[np.ndarray]]:
    model.eval()
    with torch.no_grad():
        latent = model._training_latent(features)
        _, _, tokens, prefixes = model.rq(latent, return_prefixes=True)
    return (
        tokens.cpu().numpy().astype(np.int16, copy=False),
        [prefix.detach().cpu().numpy().astype(np.float32, copy=False) for prefix in prefixes],
    )

def _initialize_behavior_node_prototypes(
    model: RQVAE,
    features: torch.Tensor,
    hierarchy: BehaviorHierarchy,
) -> None:
    _, prefixes = _all_item_representations(model, features)
    coarse_nodes = [
        hierarchy.node_items[1 + index] for index in range(hierarchy.n_coarse)
    ]
    fine_nodes = [
        hierarchy.node_items[1 + hierarchy.n_coarse + index]
        for index in range(hierarchy.n_fine)
    ]
    coarse = np.stack(
        [prefixes[0][items].mean(axis=0) for items in coarse_nodes]
    )
    fine = np.stack(
        [prefixes[1][items].mean(axis=0) for items in fine_nodes]
    )
    model.register_parameter(
        "coarse_node_prototypes",
        torch.nn.Parameter(torch.as_tensor(coarse, device=features.device)),
    )
    model.register_parameter(
        "fine_node_prototypes",
        torch.nn.Parameter(torch.as_tensor(fine, device=features.device)),
    )


def _node_tangent_overrides(
    model: RQVAE, hierarchy: BehaviorHierarchy
) -> dict[int, np.ndarray]:
    if not hasattr(model, "coarse_node_prototypes"):
        return {}
    coarse = model.coarse_node_prototypes.detach().cpu().numpy()
    fine = model.fine_node_prototypes.detach().cpu().numpy()
    overrides = {
        1 + index: tangent for index, tangent in enumerate(coarse)
    }
    fine_offset = 1 + hierarchy.n_coarse
    overrides.update(
        {
            fine_offset + index: tangent
            for index, tangent in enumerate(fine)
        }
    )
    return overrides


def _calibrate_cone_k(
    model: RQVAE,
    hierarchy: BehaviorHierarchy,
    prefix_tangents: list[np.ndarray],
    *,
    target_coverage: float,
    device: str,
    node_tangent_overrides: dict[int, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    root = model.root_direction.detach().cpu().numpy().astype(np.float32)
    values = []
    for level in range(3):
        edges = [edge for edge in hierarchy.train_edges if edge.level == level]
        reps = _node_representations(hierarchy, prefix_tangents[level], root)
        if node_tangent_overrides:
            reps.update(node_tangent_overrides)
        apex = np.stack([
            root if edge.parent == 0 else reps[edge.parent] for edge in edges
        ])
        child = np.stack([reps[edge.child] for edge in edges])
        apex_t = torch.as_tensor(apex, dtype=torch.float32, device=device)
        child_t = torch.as_tensor(child, dtype=torch.float32, device=device)
        with torch.no_grad():
            apex_point = model.geometry.to_point(apex_t)
            child_point = model.geometry.to_point(child_t)
            factor = FACTOR[model.geometry_name](apex_point)
            xi = xi_pairs(model.geometry_name, apex_point, child_point)
        values.append(
            fit_k(
                factor.detach().cpu().numpy(),
                xi.detach().cpu().numpy(),
                target=target_coverage,
            )
        )
    return tuple(float(value) for value in values)


def _hierarchy_edge_arrays(
    edges: tuple[HierarchyEdge, ...],
    hierarchy: BehaviorHierarchy,
    prefix_tangents: list[np.ndarray],
    root_tangent: np.ndarray,
    model: RQVAE,
    k_by_level: tuple[float, float, float],
    device: str,
    node_items: dict[int, np.ndarray] | None = None,
    node_tangent_overrides: dict[int, np.ndarray] | None = None,
) -> dict:
    by_level = {}
    all_correct = []
    all_coverage = []
    all_radial = []
    for level in range(3):
        level_edges = [edge for edge in edges if edge.level == level]
        if not level_edges:
            by_level[f"L{level + 1}"] = {"n_edges": 0}
            continue
        reps = _node_representations(
            hierarchy,
            prefix_tangents[level],
            root_tangent,
            node_items=node_items,
        )
        if node_tangent_overrides:
            reps.update(node_tangent_overrides)
        parent = np.stack(
            [
                root_tangent if edge.parent == 0 else reps[edge.parent]
                for edge in level_edges
            ]
        )
        child = np.stack([reps[edge.child] for edge in level_edges])
        parent_t = torch.as_tensor(parent, dtype=torch.float32, device=device)
        child_t = torch.as_tensor(child, dtype=torch.float32, device=device)
        if level > 0:
            negative = np.stack([reps[edge.negative] for edge in level_edges])
            negative_t = torch.as_tensor(negative, dtype=torch.float32, device=device)
        with torch.no_grad():
            parent_point = model.geometry.to_point(parent_t)
            child_point = model.geometry.to_point(child_t)
            aperture = torch.asin(
                (
                    k_by_level[level]
                    * FACTOR[model.geometry_name](parent_point)
                ).clamp(0.0, 1.0)
            )
            positive_xi = xi_pairs(model.geometry_name, parent_point, child_point)
            parent_radius = torch.linalg.vector_norm(parent_point, dim=-1)
            child_radius = torch.linalg.vector_norm(child_point, dim=-1)
            coverage = (positive_xi <= aperture).float().cpu().numpy()
            radial = (child_radius > parent_radius).float().cpu().numpy()
            if level > 0:
                negative_point = model.geometry.to_point(negative_t)
                negative_xi = xi_pairs(model.geometry_name, parent_point, negative_point)
                correct = (
                    (positive_xi < negative_xi).float()
                    + 0.5 * (positive_xi == negative_xi).float()
                ).cpu().numpy()
            else:
                correct = np.empty(0, dtype=np.float32)
        by_level[f"L{level + 1}"] = {
            "n_edges": int(len(level_edges)),
            "coverage": float(np.mean(coverage)),
            "pair_auc": float(np.mean(correct)) if len(correct) else None,
            "radial_order_rate": float(np.mean(radial)),
            "edge_correct": correct,
            "edge_coverage": coverage,
            "edge_radial": radial,
        }
        all_correct.extend(correct.tolist())
        all_coverage.extend(coverage.tolist())
        all_radial.extend(radial.tolist())
    return {
        "edge_auc": float(np.mean(all_correct)) if all_correct else 0.0,
        "coverage": float(np.mean(all_coverage)) if all_coverage else 0.0,
        "radial_order_rate": float(np.mean(all_radial)) if all_radial else 0.0,
        "by_level": by_level,
        "edge_correct": np.asarray(all_correct, dtype=np.float32),
        "edge_coverage": np.asarray(all_coverage, dtype=np.float32),
        "edge_radial": np.asarray(all_radial, dtype=np.float32),
    }




def _json_hierarchy(result: dict) -> dict:
    return {
        "edge_auc": result["edge_auc"],
        "coverage": result["coverage"],
        "radial_order_rate": result["radial_order_rate"],
        "by_level": {
            level: {
                key: value
                for key, value in metrics.items()
                if not key.startswith("edge_")
            }
            for level, metrics in result["by_level"].items()
        },
    }


def _save_run(
    run_dir: Path,
    model: RQVAE,
    tokens: np.ndarray,
    metrics: dict,
    curve: list[dict],
    *,
    arm: str,
    seed: int,
    k_by_level: tuple[float, float, float],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "arm": arm,
            "seed": seed,
            "k_by_level": list(k_by_level),
            "hidden_sizes": list(cfg.HIDDEN_SIZES),
            "codebook_sizes": list(model.rq.codebook_sizes),
            "codebook_dim": cfg.CODEBOOK_DIM,
        },
        run_dir / "rqvae.pth",
    )
    np.save(run_dir / "sids.npy", tokens)
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with (run_dir / "training_metrics.jsonl").open("w") as handle:
        for row in curve:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _train_one(
    arm: str,
    seed: int,
    all_features: torch.Tensor,
    train_ids: np.ndarray,
    hierarchy: BehaviorHierarchy,
    heldout_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RQVAE(
        "poincare" if arm.startswith("hyp_") else "euclid",
        in_dim=all_features.shape[1],
        codebook_dim=cfg.CODEBOOK_DIM,
        normalize_latent=True,
        include_root=True,
        hidden_sizes=cfg.HIDDEN_SIZES,
    ).to(device)
    if model.rq.codebook_sizes != cfg.CODEBOOK_SIZES:
        raise RuntimeError(
            f"Unexpected codebook sizes: {model.rq.codebook_sizes}"
        )
    with torch.no_grad():
        model.root_direction.zero_()
        model.root_direction[0] = 0.2
    train_tensor = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    model.init_codebooks(all_features[train_tensor], seed)
    if arm.endswith("cone"):
        _initialize_behavior_node_prototypes(model, all_features, hierarchy)
    model.eval()
    _, initial_prefixes = _all_item_representations(model, all_features)
    node_tangent_overrides = _node_tangent_overrides(model, hierarchy)
    k_by_level = _calibrate_cone_k(
        model,
        hierarchy,
        initial_prefixes,
        target_coverage=cfg.CONE_TRAIN_COVERAGE,
        device=device,
        node_tangent_overrides=node_tangent_overrides,
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    train_index = train_tensor
    steps_per_epoch = math.ceil(len(train_ids) / cfg.BATCH_SIZE)
    epochs = math.ceil(cfg.TRAIN_STEPS / steps_per_epoch)
    generator = torch.Generator(device=device).manual_seed(seed + 701)
    hierarchy_rng = seed + 1709
    curve = []
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    step = 0
    for epoch in range(epochs):
        order = train_index[torch.randperm(len(train_index), generator=generator, device=device)]
        for start in range(0, len(order), cfg.BATCH_SIZE):
            if step >= cfg.TRAIN_STEPS:
                break
            item_ids = order[start:start + cfg.BATCH_SIZE]
            batch = all_features[item_ids]
            reconstructed, quant_loss, _, _ = model(batch)
            recon_loss = F.mse_loss(reconstructed, batch)
            loss = recon_loss + quant_loss
            hierarchy_loss_value = loss.new_zeros(())
            if arm.endswith("cone"):
                roles = _sample_roles(hierarchy, hierarchy_rng + step)
                hierarchy_loss_value, _ = _represent_hierarchy_sample(
                    model, all_features, hierarchy, roles, k_by_level
                )
                loss = loss + cfg.HIERARCHY_LOSS_WEIGHT * hierarchy_loss_value
            if (step % 50 == 0) and not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite Stage2 loss in {arm}, seed={seed}, step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            optimizer.step()
            sums += torch.stack(
                [
                    loss.detach(),
                    recon_loss.detach(),
                    quant_loss.detach(),
                    hierarchy_loss_value.detach(),
                ]
            ).double()
            step += 1
            if step % 50 == 0 or step == cfg.TRAIN_STEPS:
                denominator = min(50, step % 50 or 50)
                values = (sums / denominator).cpu().tolist()
                curve.append(
                    {
                        "step": step,
                        "loss": values[0],
                        "reconstruction_loss": values[1],
                        "quantization_loss": values[2],
                        "hierarchy_loss": values[3],
                    }
                )
                sums.zero_()
        if step >= cfg.TRAIN_STEPS:
            break

    tokens, prefix_tangents = _all_item_representations(model, all_features)
    node_tangent_overrides = _node_tangent_overrides(model, hierarchy)
    hierarchy_train = _hierarchy_edge_arrays(
        hierarchy.train_edges,
        hierarchy,
        prefix_tangents,
        model.root_direction.detach().cpu().numpy(),
        model,
        k_by_level,
        device,
        node_tangent_overrides=node_tangent_overrides,
    )
    hierarchy_fit_edge_test = _hierarchy_edge_arrays(
        hierarchy.test_edges,
        hierarchy,
        prefix_tangents,
        model.root_direction.detach().cpu().numpy(),
        model,
        k_by_level,
        device,
        node_tangent_overrides=node_tangent_overrides,
    )
    hierarchy_test = _hierarchy_edge_arrays(
        hierarchy.heldout_edges,
        hierarchy,
        prefix_tangents,
        model.root_direction.detach().cpu().numpy(),
        model,
        k_by_level,
        device,
        node_items=hierarchy.heldout_node_items,
        node_tangent_overrides=node_tangent_overrides,
    )
    sid_metrics = sid_hierarchy_metrics(tokens, hierarchy)
    valid_ranking = evaluate_next_item_ranking(
        valid_frame,
        tokens,
        hierarchy.fit_sources,
        hierarchy.fit_targets,
        alpha=cfg.BEHAVIOR_SCORE_ALPHA,
        seed=seed + 11,
    )
    test_ranking = evaluate_next_item_ranking(
        test_frame,
        tokens,
        hierarchy.fit_sources,
        hierarchy.fit_targets,
        alpha=cfg.BEHAVIOR_SCORE_ALPHA,
        seed=seed + 17,
    )
    heldout_ranking = evaluate_next_item_ranking(
        heldout_frame,
        tokens,
        hierarchy.fit_sources,
        hierarchy.fit_targets,
        alpha=cfg.BEHAVIOR_SCORE_ALPHA,
        seed=seed + 23,
    )
    train_embeddings = all_features[train_tensor]
    with torch.no_grad():
        recon = model(train_embeddings)[0]
        recon_mse = float(F.mse_loss(recon, train_embeddings).item())
    output_metrics = {
        "arm": arm,
        "seed": seed,
        "geometry": model.geometry_name,
        "hierarchy_supervision": arm.endswith("cone"),
        "hidden_sizes": list(cfg.HIDDEN_SIZES),
        "codebook_sizes": list(model.rq.codebook_sizes),
        "codebook_dim": cfg.CODEBOOK_DIM,
        "n_train_items": int(len(train_ids)),
        "n_items": int(len(all_features)),
        "n_fit_users": hierarchy.n_fit_users,
        "n_heldout_behavior_users": hierarchy.n_heldout_users,
        "n_fit_transitions": int(len(hierarchy.fit_sources)),
        "n_heldout_behavior_transitions": int(len(hierarchy.heldout_sources)),
        "behavior_item_coverage": float(np.mean(hierarchy.behavior_covered)),
        "heldout_behavior_item_coverage": float(
            np.mean(hierarchy.heldout_behavior_covered)
        ),
        "n_behavior_coarse_groups": hierarchy.n_coarse,
        "n_behavior_fine_groups": hierarchy.n_fine,
        "behavior_coarse_group_sizes": np.bincount(hierarchy.item_coarse).tolist(),
        "behavior_fine_group_sizes": np.bincount(hierarchy.item_fine).tolist(),
        "heldout_behavior_coarse_group_sizes": np.bincount(
            hierarchy.heldout_item_coarse[hierarchy.heldout_behavior_covered]
        ).tolist(),
        "heldout_behavior_fine_group_sizes": np.bincount(
            hierarchy.heldout_item_fine[hierarchy.heldout_behavior_covered]
        ).tolist(),
        "n_hierarchy_train_edges": len(hierarchy.train_edges),
        "n_fit_hierarchy_test_edges": len(hierarchy.test_edges),
        "n_heldout_hierarchy_edges": len(hierarchy.heldout_edges),
        "train_steps": step,
        "train_reconstruction_mse": recon_mse,
        "k_by_level": list(k_by_level),
        "hierarchy_train": _json_hierarchy(hierarchy_train),
        "fit_hierarchy_test": _json_hierarchy(hierarchy_fit_edge_test),
        "hierarchy_test": _json_hierarchy(hierarchy_test),
        **sid_metrics,
        "valid_ranking": valid_ranking["metrics"],
        "test_ranking": test_ranking["metrics"],
        "heldout_behavior_ranking": heldout_ranking["metrics"],
        "behavior_node_prototypes": arm.endswith("cone"),
        "n_behavior_node_prototypes": (
            hierarchy.n_coarse + hierarchy.n_fine if arm.endswith("cone") else 0
        ),
        "behavior_node_prototype_parameters": (
            cfg.CODEBOOK_DIM * (hierarchy.n_coarse + hierarchy.n_fine)
            if arm.endswith("cone")
            else 0
        ),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    run_dir = cfg.RESULT_DIR / f"{arm}_seed{seed}"
    _save_run(
        run_dir,
        model,
        tokens,
        output_metrics,
        curve,
        arm=arm,
        seed=seed,
        k_by_level=k_by_level,
    )
    return {
        "metrics": output_metrics,
        "test_per_user": test_ranking["per_user"],
        "valid_per_user": valid_ranking["per_user"],
        "heldout_per_user": heldout_ranking["per_user"],
        "test_edge_correct": hierarchy_test["edge_correct"],
        "test_edge_coverage": hierarchy_test["edge_coverage"],
        "test_edge_radial": hierarchy_test["edge_radial"],
    }


def _load_completed_run(
    arm: str,
    seed: int,
    all_features: torch.Tensor,
    hierarchy: BehaviorHierarchy,
    heldout_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    device: str,
) -> dict:
    run_dir = cfg.RESULT_DIR / f"{arm}_seed{seed}"
    with (run_dir / "metrics.json").open() as handle:
        metrics = json.load(handle)
    if (
        metrics["arm"] != arm
        or int(metrics["seed"]) != seed
        or int(metrics["train_steps"]) != cfg.TRAIN_STEPS
    ):
        raise ValueError(f"Saved metrics are incomplete or mismatched in {run_dir}")
    checkpoint = torch.load(run_dir / "rqvae.pth", map_location=device)
    if checkpoint["arm"] != arm or checkpoint["seed"] != seed:
        raise ValueError(f"Checkpoint identity mismatch in {run_dir}")
    model = RQVAE(
        "poincare" if arm.startswith("hyp_") else "euclid",
        in_dim=all_features.shape[1],
        codebook_dim=cfg.CODEBOOK_DIM,
        normalize_latent=True,
        include_root=True,
        hidden_sizes=cfg.HIDDEN_SIZES,
    ).to(device)
    if arm.endswith("cone"):
        model.register_parameter(
            "coarse_node_prototypes",
            torch.nn.Parameter(
                torch.zeros(hierarchy.n_coarse, cfg.CODEBOOK_DIM, device=device)
            ),
        )
        model.register_parameter(
            "fine_node_prototypes",
            torch.nn.Parameter(
                torch.zeros(hierarchy.n_fine, cfg.CODEBOOK_DIM, device=device)
            ),
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    tokens = np.load(run_dir / "sids.npy")
    _, prefix_tangents = _all_item_representations(model, all_features)
    hierarchy_test = _hierarchy_edge_arrays(
        hierarchy.heldout_edges,
        hierarchy,
        prefix_tangents,
        model.root_direction.detach().cpu().numpy(),
        model,
        tuple(checkpoint["k_by_level"]),
        device,
        node_items=hierarchy.heldout_node_items,
        node_tangent_overrides=_node_tangent_overrides(model, hierarchy),
    )
    valid_ranking = evaluate_next_item_ranking(
        valid_frame,
        tokens,
        hierarchy.fit_sources,
        hierarchy.fit_targets,
        alpha=cfg.BEHAVIOR_SCORE_ALPHA,
        seed=seed + 11,
    )
    test_ranking = evaluate_next_item_ranking(
        test_frame,
        tokens,
        hierarchy.fit_sources,
        hierarchy.fit_targets,
        alpha=cfg.BEHAVIOR_SCORE_ALPHA,
        seed=seed + 17,
    )
    heldout_ranking = evaluate_next_item_ranking(
        heldout_frame,
        tokens,
        hierarchy.fit_sources,
        hierarchy.fit_targets,
        alpha=cfg.BEHAVIOR_SCORE_ALPHA,
        seed=seed + 23,
    )
    return {
        "metrics": metrics,
        "test_per_user": test_ranking["per_user"],
        "valid_per_user": valid_ranking["per_user"],
        "heldout_per_user": heldout_ranking["per_user"],
        "test_edge_correct": hierarchy_test["edge_correct"],
        "test_edge_coverage": hierarchy_test["edge_coverage"],
        "test_edge_radial": hierarchy_test["edge_radial"],
    }


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
    }


def _paired_user_vectors(
    results: dict,
    arm: str,
    level: int,
    metric: str,
    *,
    split: str = "test",
) -> np.ndarray:
    vectors = []
    result_key = "test_per_user" if split == "test" else "heldout_per_user"
    for seed in cfg.SEEDS:
        per_user = results[(arm, seed)][result_key][f"sid_prefix_L{level}"]
        vectors.append(np.asarray(per_user[metric], dtype=np.float64))
    return np.stack(vectors).mean(axis=0)


def _paired_seed_delta(
    results: dict,
    left_arm: str,
    right_arm: str,
    metric: str,
) -> dict:
    seed_deltas = {
        seed: float(
            results[(left_arm, seed)]["metrics"][metric]
            - results[(right_arm, seed)]["metrics"][metric]
        )
        for seed in cfg.SEEDS
    }
    values = np.asarray(list(seed_deltas.values()), dtype=np.float64)
    return {
        "mean_delta": float(values.mean()),
        "seed_deltas": {str(seed): value for seed, value in seed_deltas.items()},
        "n_seeds": int(len(values)),
        "all_seeds_positive": bool(np.all(values > 0.0)),
    }


def _bootstrap_comparison(
    results: dict,
    left_arm: str,
    right_arm: str,
) -> dict:
    left_mrr = _paired_user_vectors(results, left_arm, 3, "mrr")
    right_mrr = _paired_user_vectors(results, right_arm, 3, "mrr")
    left_hit = _paired_user_vectors(results, left_arm, 3, "hit10")
    right_hit = _paired_user_vectors(results, right_arm, 3, "hit10")
    left_auc = _paired_user_vectors(results, left_arm, 3, "matched_auc")
    right_auc = _paired_user_vectors(results, right_arm, 3, "matched_auc")
    left_heldout_mrr = _paired_user_vectors(
        results, left_arm, 3, "mrr", split="heldout"
    )
    right_heldout_mrr = _paired_user_vectors(
        results, right_arm, 3, "mrr", split="heldout"
    )
    left_heldout_auc = _paired_user_vectors(
        results, left_arm, 3, "matched_auc", split="heldout"
    )
    right_heldout_auc = _paired_user_vectors(
        results, right_arm, 3, "matched_auc", split="heldout"
    )
    left_edge = np.stack(
        [results[(left_arm, seed)]["test_edge_correct"] for seed in cfg.SEEDS]
    ).mean(axis=0)
    right_edge = np.stack(
        [results[(right_arm, seed)]["test_edge_correct"] for seed in cfg.SEEDS]
    ).mean(axis=0)
    left_radial = np.stack(
        [results[(left_arm, seed)]["test_edge_radial"] for seed in cfg.SEEDS]
    ).mean(axis=0)
    right_radial = np.stack(
        [results[(right_arm, seed)]["test_edge_radial"] for seed in cfg.SEEDS]
    ).mean(axis=0)
    return {
        "full_mrr_L3": paired_bootstrap_delta(
            left_mrr, right_mrr, seed=cfg.BOOTSTRAP_SEED, samples=cfg.BOOTSTRAP_SAMPLES
        ),
        "full_hit@10_L3": paired_bootstrap_delta(
            left_hit, right_hit, seed=cfg.BOOTSTRAP_SEED + 1, samples=cfg.BOOTSTRAP_SAMPLES
        ),
        "matched_auc_L3": paired_bootstrap_delta(
            left_auc, right_auc, seed=cfg.BOOTSTRAP_SEED + 2, samples=cfg.BOOTSTRAP_SAMPLES
        ),
        "heldout_user_mrr_L3": paired_bootstrap_delta(
            left_heldout_mrr,
            right_heldout_mrr,
            seed=cfg.BOOTSTRAP_SEED + 3,
            samples=cfg.BOOTSTRAP_SAMPLES,
        ),
        "heldout_user_matched_auc_L3": paired_bootstrap_delta(
            left_heldout_auc,
            right_heldout_auc,
            seed=cfg.BOOTSTRAP_SEED + 4,
            samples=cfg.BOOTSTRAP_SAMPLES,
        ),
        "heldout_hierarchy_edge_auc": paired_bootstrap_delta(
            left_edge, right_edge, seed=cfg.BOOTSTRAP_SEED + 5, samples=cfg.BOOTSTRAP_SAMPLES
        ),
        "heldout_radial_order": paired_bootstrap_delta(
            left_radial, right_radial, seed=cfg.BOOTSTRAP_SEED + 6, samples=cfg.BOOTSTRAP_SAMPLES
        ),
        "heldout_sid_ami_L1": _paired_seed_delta(
            results, left_arm, right_arm, "sid_ami_L1_coarse_heldout"
        ),
        "heldout_sid_ami_L2": _paired_seed_delta(
            results, left_arm, right_arm, "sid_ami_L2_fine_heldout"
        ),
    }


def _aggregate(results: dict) -> dict:
    summary = {"runs": {}, "comparisons": {}, "stage3_entry_gate": {}}
    metric_paths = (
        ("test_ranking", "full_mrr_L3"),
        ("test_ranking", "full_hit@10_L3"),
        ("test_ranking", "matched_auc_L3"),
        ("heldout_behavior_ranking", "full_mrr_L3"),
        ("heldout_behavior_ranking", "matched_auc_L3"),
        ("hierarchy_test", "edge_auc"),
        ("hierarchy_test", "coverage"),
        ("hierarchy_test", "radial_order_rate"),
        (None, "sid_ami_L1_coarse"),
        (None, "sid_ami_L2_fine"),
        (None, "sid_purity_L1_coarse"),
        (None, "sid_purity_L2_fine"),
        (None, "sid_ami_L1_coarse_heldout"),
        (None, "sid_ami_L2_fine_heldout"),
        (None, "sid_purity_L1_coarse_heldout"),
        (None, "sid_purity_L2_fine_heldout"),
        (None, "sid_collision_rate"),
    )
    for arm in cfg.GEOMETRY_ARMS:
        summary["runs"][arm] = {}
        for section, metric in metric_paths:
            values = []
            for seed in cfg.SEEDS:
                row = results[(arm, seed)]["metrics"]
                values.append(
                    row[metric] if section is None else row[section][metric]
                )
            summary["runs"][arm][metric] = _mean_std(values)
    for baseline in ("euclid_rq", "euclid_cone"):
        comparisons = _bootstrap_comparison(results, "hyp_cone", baseline)
        summary["comparisons"][f"hyp_cone_vs_{baseline}"] = comparisons
        summary["stage3_entry_gate"][f"vs_{baseline}"] = {
            "behavior_pass": (
                comparisons["full_mrr_L3"]["ci95_low"] > 0.0
                and comparisons["full_hit@10_L3"]["ci95_low"] > 0.0
                and comparisons["matched_auc_L3"]["ci95_low"] > 0.0
                and comparisons["heldout_user_mrr_L3"]["ci95_low"] > 0.0
                and comparisons["heldout_user_matched_auc_L3"]["ci95_low"] > 0.0
            ),
            "hierarchy_pass": (
                comparisons["heldout_hierarchy_edge_auc"]["ci95_low"] > 0.0
            ),
            "structure_pass": (
                comparisons["heldout_sid_ami_L1"]["all_seeds_positive"]
                and comparisons["heldout_sid_ami_L2"]["all_seeds_positive"]
                and comparisons["heldout_radial_order"]["ci95_low"] > 0.0
            ),
        }
        summary["stage3_entry_gate"][f"vs_{baseline}"]["all_pass"] = all(
            summary["stage3_entry_gate"][f"vs_{baseline}"][key]
            for key in ("behavior_pass", "hierarchy_pass", "structure_pass")
        )
    summary["stage3_entry_gate"]["eligible"] = all(
        summary["stage3_entry_gate"][f"vs_{baseline}"]["all_pass"]
        for baseline in ("euclid_rq", "euclid_cone")
    )
    return summary


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("The Stage2 behavior-cone experiment requires a CUDA device")
    device = "cuda:0"
    torch.set_num_threads(8)
    embeddings = np.asarray(np.load(cfg.EMBEDDING_FILE), dtype=np.float32)
    train_frame = pd.read_parquet(cfg.TRAIN_FILE)
    valid_frame = pd.read_parquet(cfg.VALID_FILE)
    test_frame = pd.read_parquet(cfg.TEST_FILE)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError(f"Invalid item embedding matrix: {embeddings.shape}")
    hierarchy = build_behavior_hierarchy(
        train_frame,
        len(embeddings),
        seed=cfg.BOOTSTRAP_SEED,
        n_coarse=cfg.COARSE_INTERESTS,
        children_per_coarse=cfg.FINE_INTERESTS_PER_COARSE,
        behavior_user_fraction=cfg.BEHAVIOR_USER_FRACTION,
        svd_dim=cfg.BEHAVIOR_SVD_DIM,
    )
    hierarchy_stability = behavior_hierarchy_stability(hierarchy)
    print(
        f"[Stage2] hierarchy split stability={hierarchy_stability}",
        flush=True,
    )
    heldout_frame = pd.DataFrame(
        {
            "user": hierarchy.heldout_edge_users,
            "seen_history": [
                np.asarray([source], dtype=np.int64)
                for source in hierarchy.heldout_sources
            ],
            "target": hierarchy.heldout_targets,
        }
    )
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.RESULT_DIR / "behavior_hierarchy.npz",
        item_coarse=hierarchy.item_coarse,
        item_fine=hierarchy.item_fine,
        behavior_covered=hierarchy.behavior_covered,
        heldout_item_coarse=hierarchy.heldout_item_coarse,
        heldout_item_fine=hierarchy.heldout_item_fine,
        heldout_behavior_covered=hierarchy.heldout_behavior_covered,
        fit_sources=hierarchy.fit_sources,
        fit_targets=hierarchy.fit_targets,
        fit_edge_users=hierarchy.fit_edge_users,
        heldout_sources=hierarchy.heldout_sources,
        heldout_targets=hierarchy.heldout_targets,
        heldout_edge_users=hierarchy.heldout_edge_users,
        train_hierarchy_edges=np.asarray(
            [[edge.level, edge.parent, edge.child, edge.negative]
             for edge in hierarchy.train_edges],
            dtype=np.int64,
        ).reshape(-1, 4),
        fit_test_hierarchy_edges=np.asarray(
            [[edge.level, edge.parent, edge.child, edge.negative]
             for edge in hierarchy.test_edges],
            dtype=np.int64,
        ).reshape(-1, 4),
        heldout_hierarchy_edges=np.asarray(
            [[edge.level, edge.parent, edge.child, edge.negative]
             for edge in hierarchy.heldout_edges],
            dtype=np.int64,
        ).reshape(-1, 4),
    )
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    if train_ids.min() < 0 or train_ids.max() >= len(embeddings):
        raise ValueError("Training target IDs do not match item embedding rows")
    all_features = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    results = {}
    reused_completed_runs = []
    for seed in cfg.SEEDS:
        for arm in cfg.GEOMETRY_ARMS:
            run_dir = cfg.RESULT_DIR / f"{arm}_seed{seed}"
            complete = all(
                (run_dir / name).is_file()
                for name in ("metrics.json", "rqvae.pth", "sids.npy")
            )
            if complete:
                reused_completed_runs.append(f"{arm}_seed{seed}")
                print(f"[Stage2] reuse arm={arm} seed={seed}", flush=True)
                result = _load_completed_run(
                    arm,
                    seed,
                    all_features,
                    hierarchy,
                    heldout_frame,
                    valid_frame,
                    test_frame,
                    device,
                )
            else:
                print(f"[Stage2] start arm={arm} seed={seed}", flush=True)
                result = _train_one(
                    arm,
                    seed,
                    all_features,
                    train_ids,
                    hierarchy,
                    heldout_frame,
                    valid_frame,
                    test_frame,
                    device,
                )
            results[(arm, seed)] = result
            print(
                f"[Stage2] done arm={arm} seed={seed} "
                f"test_mrr={result['metrics']['test_ranking']['full_mrr_L3']:.8f} "
                f"edge_auc={result['metrics']['hierarchy_test']['edge_auc']:.6f}",
                flush=True,
            )
    summary = _aggregate(results)
    summary["experiment"] = {
        "n_items": int(len(embeddings)),
        "train_rows": int(len(train_frame)),
        "valid_rows": int(len(valid_frame)),
        "test_rows": int(len(test_frame)),
        "train_steps_per_run": cfg.TRAIN_STEPS,
        "batch_size": cfg.BATCH_SIZE,
        "hidden_sizes": list(cfg.HIDDEN_SIZES),
        "codebook_sizes": list(cfg.CODEBOOK_SIZES),
        "codebook_dim": cfg.CODEBOOK_DIM,
        "seeds": list(cfg.SEEDS),
        "arms": list(cfg.GEOMETRY_ARMS),
        "behavior_user_fraction": cfg.BEHAVIOR_USER_FRACTION,
        "fit_behavior_users": hierarchy.n_fit_users,
        "heldout_behavior_users": hierarchy.n_heldout_users,
        "fit_behavior_transitions": int(len(hierarchy.fit_sources)),
        "heldout_behavior_transitions": int(len(hierarchy.heldout_sources)),
        "coarse_interests": cfg.COARSE_INTERESTS,
        "fine_interests_per_coarse": cfg.FINE_INTERESTS_PER_COARSE,
        "hierarchy_coarse_lag": 2,
        "hierarchy_fine_lag": 1,
        "coarse_profile_fallback": "lag1",
        "behavior_item_coverage": float(np.mean(hierarchy.behavior_covered)),
        "heldout_behavior_item_coverage": float(
            np.mean(hierarchy.heldout_behavior_covered)
        ),
        "behavior_split_seed": cfg.BOOTSTRAP_SEED,
        "behavior_hierarchy_stability": hierarchy_stability,
        "structural_ami_uncertainty": (
            "paired deltas over three fixed seeds; no item bootstrap"
        ),
        "elapsed_seconds": time.time() - started,
        "resumed_completed_runs": reused_completed_runs,
        "elapsed_seconds_scope": (
            "this invocation; prior timed-out training time is excluded"
            if reused_completed_runs
            else "full invocation"
        ),
        "stage3_started": False,
    }
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with (cfg.RESULT_DIR / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(
        f"[Stage2] complete stage3_eligible="
        f"{summary['stage3_entry_gate']['eligible']} "
        f"elapsed_seconds={summary['experiment']['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
