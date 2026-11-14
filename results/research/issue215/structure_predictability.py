"""Can the item's own vector still say which structural role it has?

The three-graph comparison asked whether behaviour-connected items sit close together in the vector
space.  That question has the wrong premise: a behaviour edge does not have to mean semantic
similarity, so a model that never made behaviour neighbours close was not necessarily losing the
behaviour relation.

This asks the question the research goal actually wants: **does the item's vector still carry its
structural role from the behaviour graph, and does quantization make that harder to recover?**

For every arm, the item's pre-quantization vector ``z`` and the model's own quantized readout
``z_q`` are used to predict, from that item alone:

  * the 73 ORCA node-orbit counts of the item in the behaviour graph (how finely the structure is
    recoverable), plus its degree and clustering;
  * the item's ORCA role (six classes) -- the interpretable version of the same question.

Both predictors are the same architecture, the same folds, the same seed and the same budget; the
only thing that changes between the two is the vector.  The 768-dimensional stage-1 sentence
embedding is fitted the same way as the ceiling: what the semantics alone can say about structure.

    python3 results/research/issue215/structure_predictability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
ROLE_TABLE = OUT / "orca_roles_frozen.csv"
ROLE_SHA = "c7a29e6559aed1385c5ce514e506be22d88bf6e85eb61120b6dbf69375ea251c"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
JOURNAL = OUT / "structure_predictability.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
FOLDS = 5
SEED = 42
EPOCHS = 400
#: Full-batch steps: the matrices are small and the item count is modest, so one gradient step per
#: epoch over every training row is both faster and easier to keep identical across predictors.
BATCH = 0
HIDDEN = 128
LR = 3e-3        # full-batch steps want a larger step than mini-batch ones


def load_roles() -> dict[int, int]:
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("router", REPO / "stage2_RQ-VAE" / "TIGER_role_geom_rq" / "model" / "role_geometry.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_role_table(ROLE_TABLE, ROLE_SHA)


def behaviour_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 73 orbit counts of every item in the behaviour graph, with its degree and clustering."""
    orbits = np.loadtxt(GRAPHS / "behaviour5.out", dtype=np.float64)
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    degrees = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    triangles = orbits[:, 3]
    clustering = np.where(degrees > 1, 2.0 * triangles / np.maximum(degrees * (degrees - 1.0), 1e-12), 0.0)
    return orbits, degrees, clustering


def fit_mlp(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
            device: torch.device) -> np.ndarray:
    """The same small regressor for every predictor: fit on the train fold, predict the held-out one."""
    scaler = StandardScaler().fit(features[train_idx])
    x = torch.tensor(scaler.transform(features), dtype=torch.float32, device=device)
    y_scaler = StandardScaler().fit(targets[train_idx])
    y = torch.tensor(y_scaler.transform(targets), dtype=torch.float32, device=device)
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, y.shape[1]),
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    n_train = train_idx.size
    train_tensor = torch.tensor(train_idx, device=device)
    x_train, y_train = x[train_tensor], y[train_tensor]
    for _ in range(EPOCHS):
        loss = torch.nn.functional.mse_loss(model(x_train), y_train)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x[torch.tensor(test_idx, device=device)]).cpu().numpy()
    return prediction * y_scaler.scale_ + y_scaler.mean_


def evaluate(prediction: np.ndarray, truth: np.ndarray) -> dict:
    rho, r2 = [], []
    for column in range(truth.shape[1]):
        if np.ptp(truth[:, column]) == 0:
            continue
        rho.append(float(spearmanr(prediction[:, column], truth[:, column]).statistic))
        residual = ((truth[:, column] - prediction[:, column]) ** 2).sum()
        total = ((truth[:, column] - truth[:, column].mean()) ** 2).sum()
        r2.append(float(1.0 - residual / max(total, 1e-12)))
    return {"spearman_mean": float(np.nanmean(rho)), "r2_mean": float(np.nanmean(r2))}


def role_accuracy(features: np.ndarray, labels: np.ndarray, folds) -> dict:
    """Can the six ORCA roles be read off the vector at all?"""
    accuracy, f1 = [], []
    for train_idx, test_idx in folds:
        scaler = StandardScaler().fit(features[train_idx])
        model = LogisticRegression(max_iter=2000, multi_class="multinomial")
        model.fit(scaler.transform(features[train_idx]), labels[train_idx])
        predicted = model.predict(scaler.transform(features[test_idx]))
        accuracy.append(float((predicted == labels[test_idx]).mean()))
        for role in np.unique(labels):
            true_positive = int(((predicted == role) & (labels[test_idx] == role)).sum())
            false_positive = int(((predicted == role) & (labels[test_idx] != role)).sum())
            false_negative = int(((predicted != role) & (labels[test_idx] == role)).sum())
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {"accuracy_mean": float(np.mean(accuracy)), "macro_f1_mean": float(np.mean(f1))}


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    orbits, degrees, clustering = behaviour_targets()
    roles = load_roles()
    labels = np.asarray([roles.get(item, -1) for item in range(len(orbits))], dtype=np.int64)
    keep = labels >= 0
    print(f"items with a role and orbits: {int(keep.sum())}", flush=True)

    targets = np.concatenate([orbits, np.log1p(degrees)[:, None], clustering[:, None]], axis=1)
    n_orbits = orbits.shape[1]
    # the folds are drawn over the items that have both a role and orbit counts, and their indices
    # are positions in the masked arrays the predictors see
    masked = np.flatnonzero(keep)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(masked))

    report = {"items": int(keep.sum()), "folds": FOLDS, "seed": SEED, "epochs": EPOCHS,
              "targets": {"orbits": n_orbits, "log_degree": 1, "clustering": 1}, "predictors": {}}

    def run(name: str, features: np.ndarray) -> dict:
        features = features[keep]
        orbit_rho, orbit_r2, degree_rho, clustering_rho = [], [], [], []
        for train_idx, test_idx in folds:
            prediction = fit_mlp(features, targets, train_idx, test_idx, device)
            block = evaluate(prediction[:, :n_orbits], targets[test_idx][:, :n_orbits])
            orbit_rho.append(block["spearman_mean"])
            orbit_r2.append(block["r2_mean"])
            degree_rho.append(float(spearmanr(prediction[:, n_orbits], targets[test_idx][:, n_orbits]).statistic))
            clustering_rho.append(float(spearmanr(prediction[:, n_orbits + 1], targets[test_idx][:, n_orbits + 1]).statistic))
        role = role_accuracy(features, labels[keep], folds)
        entry = {"orbit_spearman_mean": float(np.mean(orbit_rho)), "orbit_r2_mean": float(np.mean(orbit_r2)),
                 "degree_spearman": float(np.mean(degree_rho)), "clustering_spearman": float(np.mean(clustering_rho)),
                 "role_accuracy": role["accuracy_mean"], "role_macro_f1": role["macro_f1_mean"],
                 "n_features": int(features.shape[1])}
        report["predictors"][name] = entry
        print(f"{name:<16} orbit rho {entry['orbit_spearman_mean']:+.4f} | r2 {entry['orbit_r2_mean']:+.4f} | "
              f"degree rho {entry['degree_spearman']:+.3f} | clustering rho {entry['clustering_spearman']:+.3f} | "
              f"role acc {entry['role_accuracy']:.4f} f1 {entry['role_macro_f1']:.4f}", flush=True)
        return entry

    embedding = np.load(EMBEDDING).astype(np.float64)
    run("semantic 768d", embedding)
    for arm in ARMS:
        z_path, q_path = VECTORS / f"{arm}_z.npy", VECTORS / f"{arm}_q.npy"
        if not z_path.is_file() or not q_path.is_file():
            continue
        run(f"{arm} z (pre)", np.load(z_path).astype(np.float64))
        run(f"{arm} q (post)", np.load(q_path).astype(np.float64))

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print("\nquantization effect on recoverable structure (post minus pre):")
    for arm in ARMS:
        pre = report["predictors"].get(f"{arm} z (pre)")
        post = report["predictors"].get(f"{arm} q (post)")
        if not pre or not post:
            continue
        print(f"  {arm:<6} orbit rho {pre['orbit_spearman_mean']:+.4f} -> {post['orbit_spearman_mean']:+.4f} "
              f"({post['orbit_spearman_mean'] - pre['orbit_spearman_mean']:+.4f}) | "
              f"role acc {pre['role_accuracy']:.4f} -> {post['role_accuracy']:.4f} "
              f"({post['role_accuracy'] - pre['role_accuracy']:+.4f})")
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
