"""Run experiment 1: does a mixed free + hyperbolic encoder learn better communities?

Two models of the same capacity are trained once each on the train split's AF3
curvature ranking pairs, and both are measured on an independent sample: the
ranking pairs of the **validation** split's behaviour graph, which is built from
held-out interactions and never seen during training.

    model A (control)  frozen 768 -> free 128          -> 64 centres
    model B (experiment) frozen 768 -> free 64 + hyperbolic 64 -> 64 centres

The judgement is the validation curvature separation: if the mixed geometry does
not separate high-AF3 from low-AF3 pairs better than the Euclidean control on
data the model never saw, the mechanism is refuted and nothing is carried into
the RQ-VAE.

Run (no arguments):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/run_experiment.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import os
import numpy as np
import torch

# The cuBLAS GEMMs are not deterministic on CUDA without this, and the arms with
# a sparse propagation drift by several percent between runs -- more than the
# differences between the arms, which hides every real effect.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.science.community_curvature import data, metrics, model  # noqa: E402

OUT = data.REPO / "results" / "research" / "issue121" / "community_curvature"

SEED = 42
N_COMMUNITIES = 64
EPOCHS = 60
ANCHOR_BATCH = 1024
ITEM_BATCH = 4096
LEARNING_RATE = 1e-3
FREE_DIM_CONTROL = 128
FREE_DIM_EXPERIMENT = 64
HYPER_DIM_EXPERIMENT = 64
CURVATURE = 1.0
PERMUTATIONS = 20000


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_curvature(graph: data.Graph, curvature: dict) -> np.ndarray:
    """The mean AF3 curvature of every item's incident edges, standardised.

    It is the target the curvature head predicts: the local curvature of an item
    is a property of the behaviour graph, and a representation that can predict
    it has encoded the local structure the partition follows.
    """
    rows = np.asarray(graph.nodes, dtype=np.int64)
    values = np.asarray(curvature["af3"], dtype=np.float64)
    totals = np.zeros(rows.size, dtype=np.float64)
    counts = np.zeros(rows.size, dtype=np.float64)
    np.add.at(totals, graph.edges[0], values)
    np.add.at(counts, graph.edges[0], 1.0)
    np.add.at(totals, graph.edges[1], values)
    np.add.at(counts, graph.edges[1], 1.0)
    means = totals / np.maximum(counts, 1.0)
    spread = float(means.std()) if means.size else 0.0
    standardised = (means - float(means.mean())) / (spread if spread > 0 else 1.0)
    out = np.zeros(int(rows.max()) + 1, dtype=np.float32)
    out[rows] = standardised
    return out


def item_ids(graph: data.Graph, positions: np.ndarray) -> np.ndarray:
    """Graph positions -> item ids."""
    return np.asarray(graph.nodes, dtype=np.int64)[positions]


def curvature_graph(graph: data.Graph, curvature: dict, n_items: int,
                    curvature_aware: bool = True) -> model.CurvatureGraphConvolution:
    """The propagation weights of the behaviour graph, in the item id space.

    The AF3 curvature of every edge is standardised over the graph's edges, so
    the weight says how much higher than average this edge's curvature is, and
    the propagation mixes the neighbours into the representation. The control
    drops the curvature factor and keeps the plain GCN mean.
    """
    rows = item_ids(graph, graph.edges[0])
    cols = item_ids(graph, graph.edges[1])
    values = np.asarray(curvature["af3"], dtype=np.float64)
    spread = float(values.std())
    z = (values - float(values.mean())) / (spread if spread > 0 else 1.0)
    degrees = np.zeros(n_items, dtype=np.float64)
    np.add.at(degrees, rows, 1.0)
    return model.CurvatureGraphConvolution(rows, cols, z, degrees,
                                           curvature_aware=curvature_aware)


def soft_assignment(net: model.CommunityModel, embeddings: np.ndarray,
                    device: torch.device, batch: int = 8192) -> np.ndarray:
    """The soft community distribution of every item, in one pass."""
    out = np.empty((embeddings.shape[0], net.n_communities), dtype=np.float32)
    every_item = torch.from_numpy(embeddings).to(device)
    with torch.no_grad():
        free_all, ball_all = net.represent(every_item)
        for start in range(0, embeddings.shape[0], batch):
            rows = torch.arange(start, min(start + batch, embeddings.shape[0]), device=device)
            out[start:start + batch] = net.assignment(
                free_all[rows], ball_all[rows] if ball_all is not None else None).cpu().numpy()
    return out


def train(net: model.CommunityModel, embeddings: np.ndarray, anchor_items: np.ndarray,
          high_items: np.ndarray, low_items: np.ndarray, device: torch.device,
          seed: int = SEED, node_to_cluster: bool = False,
          curvature_target: np.ndarray | None = None,
          cross_space: bool = False) -> dict:
    """Train once: the curvature ranking loss plus the usage constraint.

    With ``node_to_cluster`` the model additionally keeps every item close to
    the centre it leans towards and every centre at least as far from its
    nearest neighbour as the centres started. Both extra terms are zero at
    initialisation and dimensionless, so they carry a weight of one.
    """
    rng = np.random.default_rng(seed)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    margin = model.margin_for(net.n_communities)
    # the representations are computed for every item, because the propagation
    # mixes in neighbours that need not be in the batch
    every_item = torch.from_numpy(embeddings).to(device)
    scales = None
    if node_to_cluster:
        sample = rng.integers(0, embeddings.shape[0], size=ITEM_BATCH)
        free_all, ball_all = net.represent(every_item)
        scales = net.freeze_scales(free_all[sample],
                                   ball_all[sample] if ball_all is not None else None)
    # Every weight is derived and the same for every model, so the arms stay
    # comparable: the ranking hinge lives on the scale of its margin (1/K), the
    # commitment on the scale of the frozen branch distances, the usage term is
    # bounded by one, and the separation hinge on the scale of the threshold.
    weights = {"ranking": 1.0 / margin, "usage": 1.0}
    curvature_loss = None
    if curvature_target is not None:
        curvature_loss = torch.nn.MSELoss()
        target = torch.from_numpy(curvature_target).to(device)
    if node_to_cluster:
        weights["commitment"] = 1.0
        weights["separation"] = 1.0 / max(float(scales["separation_threshold"]), 1e-8)
    n_anchors = anchor_items.size
    steps_per_epoch = max(1, n_anchors // ANCHOR_BATCH)
    history = []
    net.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(n_anchors)
        totals = {"loss": 0.0, "ranking": 0.0, "usage": 0.0, "commitment": 0.0,
                  "separation": 0.0, "consistency": 0.0, "cross_space": 0.0,
                  "s_high": 0.0, "s_low": 0.0}
        for step in range(steps_per_epoch):
            index = order[step * ANCHOR_BATCH:(step + 1) * ANCHOR_BATCH]
            free_all, ball_all = net.represent(every_item)
            rows_anchor = torch.from_numpy(anchor_items[index]).to(device)
            rows_high = torch.from_numpy(high_items[index]).to(device)
            rows_low = torch.from_numpy(low_items[index]).to(device)
            pi_anchor = net.assignment(free_all[rows_anchor],
                                       ball_all[rows_anchor] if ball_all is not None else None)
            pi_high = net.assignment(free_all[rows_high],
                                     ball_all[rows_high] if ball_all is not None else None)
            pi_low = net.assignment(free_all[rows_low],
                                    ball_all[rows_low] if ball_all is not None else None)

            sample = torch.from_numpy(
                rng.integers(0, embeddings.shape[0], size=ITEM_BATCH)).to(device)
            if node_to_cluster:
                pi_items, absolute_items = net.assignment_and_absolute(
                    free_all[sample], ball_all[sample] if ball_all is not None else None)
            else:
                pi_items = net.assignment(free_all[sample],
                                          ball_all[sample] if ball_all is not None else None)
                absolute_items = None

            ranking = model.ranking_loss(pi_anchor, pi_high, pi_low, margin)
            usage = model.usage_loss(pi_items.mean(dim=0), net.n_communities)
            consistency = torch.zeros((), device=device)
            cross = torch.zeros((), device=device)
            if cross_space:
                free_logit, ball_logit = net.branch_logits(free_all, ball_all)
                cross = model.cross_space_divergence(free_logit, ball_logit)
            if curvature_loss is not None:
                # the head reads the item's community point and predicts its
                # local curvature; the loss makes the two agree
                sample_items = torch.from_numpy(
                    rng.integers(0, embeddings.shape[0], size=ITEM_BATCH)).to(device)
                _, ball_sample = net.represent(
                    torch.from_numpy(embeddings).to(device))
                predicted = net.curvature_head(ball_sample[sample_items])
                consistency = curvature_loss(predicted, target[sample_items])
            commitment = separation = torch.zeros((), device=device)
            if node_to_cluster:
                commitment = model.commitment_loss(pi_items, absolute_items)
                separation = model.separation_loss(
                    net.centre_distances(), float(net.separation_threshold))
            terms = {"ranking": ranking, "usage": usage}
            if node_to_cluster:
                terms["commitment"] = commitment
                terms["separation"] = separation
            terms["consistency"] = consistency
            weights["consistency"] = 1.0
            terms["cross_space"] = cross
            weights["cross_space"] = 1.0
            loss = sum(weights[name] * value for name, value in terms.items())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            totals["loss"] += float(loss.detach())
            totals["ranking"] += float(ranking.detach())
            totals["usage"] += float(usage.detach())
            totals["commitment"] += float(commitment.detach())
            totals["consistency"] += float(consistency.detach())
            totals["cross_space"] += float(cross.detach())
            totals["separation"] += float(separation.detach())
            totals["s_high"] += float(model.overlap(pi_anchor, pi_high).mean().detach())
            totals["s_low"] += float(model.overlap(pi_anchor, pi_low).mean().detach())
        count = max(1, steps_per_epoch)
        history.append({key: value / count for key, value in totals.items()})
    net.eval()
    return {"epochs": EPOCHS, "steps_per_epoch": steps_per_epoch, "margin": margin,
            "scales": scales, "weights": weights, "history": history}


def evaluate(net: model.CommunityModel, name: str, embeddings: np.ndarray,
             graphs: dict, curvatures: dict, pair_sets: dict, device: torch.device,
             seed: int = SEED) -> dict:
    """Every measurement of one trained model."""
    pi = soft_assignment(net, embeddings, device)
    report = {"model": name,
              "free_dim": net.free_dim, "hyper_dim": net.hyper_dim,
              "n_communities": net.n_communities,
              "contract_sha256": net.contract_sha256}
    for split in ("train", "valid"):
        pairs = pair_sets[split]
        report[f"separation_{split}"] = metrics.separation(pi, pairs, seed=seed)
    report["occupancy"] = metrics.occupancy(pi)
    for split in ("train", "valid"):
        graph = graphs[split]
        labels = pi[graph.nodes].argmax(axis=1).astype(np.int64)
        report[f"curvature_gap_{split}"] = metrics.curvature_gap(graph, curvatures[split],
                                                                  labels)
        report[f"community_transitivity_{split}"] = metrics.community_transitivity(graph, labels)
        report[f"group_transitivity_{split}"] = metrics.group_transitivity(
            graph, labels, net.n_communities)
    return report


def paired_comparison(pi_control: np.ndarray, pi_experiment: np.ndarray, pairs,
                       seed: int = SEED) -> dict:
    """The paired difference of the validation separation, B - A, over the anchors."""
    def differences(pi):
        anchor = pi[pairs.anchor]
        return ((anchor * pi[pairs.high]).sum(axis=1)
                - (anchor * pi[pairs.low]).sum(axis=1))
    control = differences(pi_control)
    experiment = differences(pi_experiment)
    delta = experiment - control
    rng = np.random.default_rng(seed)
    n = delta.size
    draws = rng.integers(0, n, size=(5000, n))
    means = delta[draws].mean(axis=1)
    signs = rng.integers(0, 2, size=(PERMUTATIONS, n)).astype(bool)
    null = np.where(signs, delta, -delta).mean(axis=1)
    return {
        "anchors": int(n),
        "control_mean": float(control.mean()),
        "experiment_mean": float(experiment.mean()),
        "difference": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "permutation_p": float((np.abs(null) >= abs(delta.mean())).mean()),
        "permutations": PERMUTATIONS,
        "anchors_improved": int((delta > 0).sum()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    print("[data] embeddings", flush=True)
    embeddings = data.load_embeddings()
    graphs, curvatures, pair_sets = {}, {}, {}
    for split in ("train", "valid"):
        print(f"[data] {split} graph", flush=True)
        graphs[split] = data.build_graph(split)
        curvatures[split] = data.curvature_of(graphs[split])
        pairs = data.ranking_pairs(graphs[split], curvatures[split])
        pair_sets[split] = data.RankingPairs(
            anchor=item_ids(graphs[split], pairs.anchor),
            high=item_ids(graphs[split], pairs.high),
            low=item_ids(graphs[split], pairs.low),
            af3_high=pairs.af3_high, af3_low=pairs.af3_low)
        print(f"[data] {split}: nodes={graphs[split].n_nodes} "
              f"edges={graphs[split].edges[0].size} ranking_pairs={len(pairs)}", flush=True)

    # the local curvature target of the consistency arm
    curvature_target = local_curvature(graphs["train"], curvatures["train"])
    random_left, random_right = data.random_pairs(graphs["train"], 200000, SEED)

    # the propagation weights of the train graph: the curvature-aware one and
    # the plain GCN mean, which is the control for the curvature weighting
    curv_graph = curvature_graph_for = None
    specifications = [
        ("A_free_128", dict(free_dim=FREE_DIM_CONTROL, hyper_dim=0)),
        ("B_free_64_hyper_64", dict(free_dim=FREE_DIM_EXPERIMENT,
                                    hyper_dim=HYPER_DIM_EXPERIMENT)),
        ("C_free_64_hyper_64_ntc", dict(free_dim=FREE_DIM_EXPERIMENT,
                                        hyper_dim=HYPER_DIM_EXPERIMENT,
                                        node_to_cluster=True)),
        ("D_free_64_hyper_64_gcn", dict(free_dim=FREE_DIM_EXPERIMENT,
                                        hyper_dim=HYPER_DIM_EXPERIMENT,
                                        graph="plain")),
        ("E_free_64_hyper_64_curvgcn", dict(free_dim=FREE_DIM_EXPERIMENT,
                                             hyper_dim=HYPER_DIM_EXPERIMENT,
                                             graph="curvature")),
        ("F_free_64_hyper_64_learncurv", dict(free_dim=FREE_DIM_EXPERIMENT,
                                                hyper_dim=HYPER_DIM_EXPERIMENT,
                                                graph="curvature",
                                                learn_curvature=True)),
        # The resolution question: the gate for any use of a partition in the
        # identifiers is that it must discriminate the held-out behaviour pairs
        # better than the codes do (6.16), and a finer partition has more
        # behaviour information per community but also mixes more negatives.
        ("G_free_64_hyper_64_learncurv_k1024", dict(free_dim=FREE_DIM_EXPERIMENT,
                                                     hyper_dim=HYPER_DIM_EXPERIMENT,
                                                     graph="curvature",
                                                     learn_curvature=True,
                                                     n_communities=1024)),
        # The plan's last deferred item: an MLP head that predicts an item's
        # local AF3 curvature from its community point. A representation that can
        # predict the curvature has encoded the local graph structure the
        # partition is supposed to follow, so the curvature ordering the model
        # optimises should get better, not worse.
        ("H_free_64_hyper_64_learncurv_consistency", dict(free_dim=FREE_DIM_EXPERIMENT,
                                                          hyper_dim=HYPER_DIM_EXPERIMENT,
                                                          graph="curvature",
                                                          learn_curvature=True,
                                                          curvature_head=True)),
        # The plan's last missing ingredient: the cross-space contrast. The free
        # branch and the hyperbolic branch each score every community; the
        # symmetric Jensen-Shannon divergence between the two views is zero when
        # they agree, and the two subspaces then describe the same partition.
        ("I_free_64_hyper_64_crossspace", dict(free_dim=FREE_DIM_EXPERIMENT,
                                               hyper_dim=HYPER_DIM_EXPERIMENT,
                                               graph="curvature",
                                               learn_curvature=True,
                                               cross_space=True)),
    ]
    plain_graph = curvature_graph_for = curvature_graph(
        graphs["train"], curvatures["train"], embeddings.shape[0], curvature_aware=False)
    curvature_graph_for = curvature_graph(
        graphs["train"], curvatures["train"], embeddings.shape[0], curvature_aware=True)
    assignments, reports = {}, {}
    for name, spec in specifications:
        print(f"[train] {name}", flush=True)
        set_seed(SEED)
        spec = dict(spec)
        node_to_cluster = bool(spec.pop("node_to_cluster", False))
        graph_kind = spec.pop("graph", None)
        learn_curvature = bool(spec.pop("learn_curvature", False))
        graph = {"plain": plain_graph, "curvature": curvature_graph_for}.get(graph_kind)
        spec.setdefault("n_communities", N_COMMUNITIES)
        use_consistency = bool(spec.pop("curvature_head", False))
        use_cross_space = bool(spec.pop("cross_space", False))
        net = model.CommunityModel(input_dim=embeddings.shape[1],
                                   curvature=CURVATURE, seed=SEED, graph=graph,
                                   learn_curvature=learn_curvature,
                                   curvature_head=use_consistency, **spec).to(device)
        training = train(net, embeddings, pair_sets["train"].anchor, pair_sets["train"].high,
                         pair_sets["train"].low, device,
                         node_to_cluster=node_to_cluster)
        report = evaluate(net, name, embeddings, graphs, curvatures, pair_sets, device)
        report["training"] = {"epochs": training["epochs"], "margin": training["margin"],
                               "scales": training["scales"],
                               "weights": training["weights"],
                               "learned_curvature": (float(net.curvature_value.detach())
                                                     if net.learn_curvature else None),
                               "first_epoch": training["history"][0],
                               "last_epoch": training["history"][-1]}
        reports[name] = report
        assignments[name] = soft_assignment(net, embeddings, device)
        np.save(OUT / f"pi_{name}.npy", assignments[name])

    # the scale of a random pair's overlap, the null the separation is read against
    # the graph positions are item ids before they can index the assignment
    random_left = item_ids(graphs["train"], random_left)
    random_right = item_ids(graphs["train"], random_right)
    control_pi = assignments["A_free_128"]
    null_overlap = float((control_pi[random_left] * control_pi[random_right]).sum(axis=1).mean())

    comparisons = {
        "B_minus_A_valid": paired_comparison(assignments["A_free_128"],
                                             assignments["B_free_64_hyper_64"],
                                             pair_sets["valid"]),
        "C_minus_B_valid": paired_comparison(assignments["B_free_64_hyper_64"],
                                             assignments["C_free_64_hyper_64_ntc"],
                                             pair_sets["valid"]),
        "D_minus_B_valid": paired_comparison(assignments["B_free_64_hyper_64"],
                                             assignments["D_free_64_hyper_64_gcn"],
                                             pair_sets["valid"]),
        "E_minus_B_valid": paired_comparison(assignments["B_free_64_hyper_64"],
                                             assignments["E_free_64_hyper_64_curvgcn"],
                                             pair_sets["valid"]),
        "E_minus_D_valid": paired_comparison(assignments["D_free_64_hyper_64_gcn"],
                                             assignments["E_free_64_hyper_64_curvgcn"],
                                             pair_sets["valid"]),
    }
    # the hard communities, the thing the node-to-cluster terms are judged on
    hard_community = {}
    for name, report in reports.items():
        hard_community[name] = {
            split: {
                "af3_gap": report[f"curvature_gap_{split}"]["gap"],
                "af3_within_mean": report[f"curvature_gap_{split}"]["within_mean"],
                "af3_cross_mean": report[f"curvature_gap_{split}"]["cross_mean"],
                "community_transitivity": report[f"community_transitivity_{split}"]["mean"],
                "community_transitivity_edge_weighted":
                    report[f"community_transitivity_{split}"]["edge_weighted_mean"],
                "group_transitivity": report[f"group_transitivity_{split}"]["transitivity"],
            }
            for split in ("train", "valid")
        }
        hard_community[name]["occupancy"] = report["occupancy"]
    nulls = {split: metrics.null_measurements(graphs[split], curvatures[split],
                                               N_COMMUNITIES, seed=SEED)
             for split in ("train", "valid")}

    payload = {
        "experiment": "curvature-guided soft community learning",
        "question": ("does a mixed free + hyperbolic encoder learn item communities that "
                     "agree with the AF3 curvature of the behaviour graph better than a "
                     "pure Euclidean encoder of the same capacity?"),
        "protocol": {
            "seed": SEED, "communities": N_COMMUNITIES, "epochs": EPOCHS,
            "anchor_batch": ANCHOR_BATCH, "item_batch": ITEM_BATCH,
            "learning_rate": LEARNING_RATE, "curvature": CURVATURE,
            "margin": model.margin_for(N_COMMUNITIES),
            "margin_derivation": "1/K: the overlap a uniform soft assignment gives any pair",
            "usage_weight": 1.0,
            "usage_derivation": "1 - H(usage)/log K is dimensionless and bounded by 1, "
                                "the same order as the ranking hinge",
            "distance_combination": "per-item standardised branch distances, equal weights",
            "device": str(device), "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "data": {
            "embeddings": str(data.EMBEDDINGS),
            "embedding_shape": list(embeddings.shape),
            "train": {"nodes": graphs["train"].n_nodes,
                      "edges": int(graphs["train"].edges[0].size),
                      "ranking_pairs": len(pair_sets["train"]),
                      "af3_high_mean": float(pair_sets["train"].af3_high.mean()),
                      "af3_low_mean": float(pair_sets["train"].af3_low.mean())},
            "valid": {"nodes": graphs["valid"].n_nodes,
                      "edges": int(graphs["valid"].edges[0].size),
                      "ranking_pairs": len(pair_sets["valid"]),
                      "af3_high_mean": float(pair_sets["valid"].af3_high.mean()),
                      "af3_low_mean": float(pair_sets["valid"].af3_low.mean())},
            "random_pair_overlap": null_overlap,
        },
        "models": reports,
        "comparisons": comparisons,
        "hard_community": hard_community,
        "null_models": nulls,
        "wall_time_s": time.time() - started,
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparisons": comparisons,
                      "hard_community": hard_community,
                      "separations": {name: {"train": reports[name]["separation_train"],
                                             "valid": reports[name]["separation_valid"]}
                                      for name in reports}}, indent=2), flush=True)
    print(f"[done] {OUT / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
