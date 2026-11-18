"""QDA analysis: structure, CKA, codeword stability, optimization dynamics.

Reads the observation snapshots a diagnostic run writes and the frozen sample, and
answers the four hypotheses of the experiment without training anything.

Scales are kept apart on purpose: continuous representations are scored by the
AUC of a real behaviour edge against its matched non-edge (both from the frozen
sample), discrete SIDs by the prefix-sharing lift over the same non-edges.  The
two are reported side by side and never subtracted from each other; the only
deltas computed are between levels of the same kind.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/qda/analyze.py [--arm QDA_F0_seed42]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SAMPLE = HERE / "sample" / "sample.npz"
TREE = HERE / "trees" / "F0_seed42"
LEVELS = ("encoder", "l1", "l2", "l3")
ALPHA = 0.5


def load_geometry():
    spec = importlib.util.spec_from_file_location(
        "qda_model", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")])
    package = importlib.util.module_from_spec(spec)
    sys.modules["qda_model"] = package
    spec.loader.exec_module(package)
    from qda_model.poincare_contract import expmap0, squared_distance
    return expmap0, squared_distance


def pair_distance(vectors: np.ndarray, left: np.ndarray, right: np.ndarray,
                  expmap0, squared_distance) -> np.ndarray:
    """The F0 mixed distance, alpha = 1/2, for the given index pairs."""
    import torch
    tensor = torch.from_numpy(vectors.astype(np.float32))
    left_index = torch.from_numpy(np.asarray(left, dtype=np.int64))
    right_index = torch.from_numpy(np.asarray(right, dtype=np.int64))
    euclidean = tensor[:, :16]
    ball = expmap0(tensor[:, 16:], 1.0)
    d_e = (euclidean[left_index] - euclidean[right_index]).pow(2).sum(-1)
    d_h = squared_distance(ball[left_index], ball[right_index], 1.0)
    return (ALPHA * d_e + (1 - ALPHA) * d_h / 4.0).numpy()


def edge_auc(vectors, sample, mask, expmap0, squared_distance) -> float:
    left = sample["edges"][mask][:, 0].astype(np.int64)
    right = sample["edges"][mask][:, 1].astype(np.int64)
    neg_left = sample["negatives"][mask][:, 0].astype(np.int64)
    neg_right = sample["negatives"][mask][:, 1].astype(np.int64)
    positive = pair_distance(vectors, left, right, expmap0, squared_distance)
    negative = pair_distance(vectors, neg_left, neg_right, expmap0, squared_distance)
    return float((positive < negative).mean() + 0.5 * (positive == negative).mean())


def prefix_lift(codes, sample, mask, level: int) -> dict:
    """Share of real edges vs matched non-edges that agree on the first `level` tokens."""
    depth = level
    pos = sample["edges"][mask]
    neg = sample["negatives"][mask]
    share_pos = np.all(codes[pos[:, 0], :depth] == codes[pos[:, 1], :depth], axis=1)
    share_neg = np.all(codes[neg[:, 0], :depth] == codes[neg[:, 1], :depth], axis=1)
    return {"positive_share": float(share_pos.mean()),
            "negative_share": float(share_neg.mean()),
            "lift": float(share_pos.mean() - share_neg.mean())}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y) ** 2
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(numerator / denominator) if denominator > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="QDA_F0_seed42")
    args = parser.parse_args()
    run = HERE / "runs" / args.arm
    observations = sorted((run / "observations").glob("epoch_*.npz"))
    if not observations:
        raise SystemExit(f"no observations under {run}")
    sample = dict(np.load(SAMPLE))
    expmap0, squared_distance = load_geometry()
    epochs = [int(path.stem.split("_")[1]) for path in observations]
    report = {"arm": args.arm, "epochs": epochs, "structure": {}, "prefix": {},
              "cka": {}, "stability": {}, "dynamics": {}}

    selection = sample["split"] == 0
    final = sample["split"] == 1

    # --- A: structural alignment per level, per epoch; selection and final halves ---
    table = []
    for path, epoch in zip(observations, epochs):
        arrays = np.load(path)
        row = {"epoch": epoch, "levels": {}, "prefix": {}}
        for level in LEVELS:
            vectors = arrays["z"] if level == "encoder" else arrays[f"readout_{level[-1]}"]
            row["levels"][level] = {
                "auc_selection": edge_auc(vectors, sample, selection, expmap0, squared_distance),
                "auc_final": edge_auc(vectors, sample, final, expmap0, squared_distance),
            }
        for depth in (1, 2, 3):
            row["prefix"][f"l{depth}"] = prefix_lift(arrays["codes"], sample, final, depth)
        table.append(row)
        print(f"epoch {epoch:4d}  " + "  ".join(
            f"{level}={row['levels'][level]['auc_final']:.4f}" for level in LEVELS)
            + "  | lift " + " ".join(
                f"L{depth}={row['prefix'][f'l{depth}']['lift']:+.4f}" for depth in (1, 2, 3)),
            flush=True)
    report["structure"]["per_epoch"] = table

    # --- B: CKA between consecutive observation points, per level, plus E/H halves ---
    cka = {level: [] for level in LEVELS}
    cka_halves = {level: {"euclidean": [], "hyperbolic": []} for level in LEVELS}
    previous = None
    for path in observations:
        arrays = np.load(path)
        current = {level: (arrays["z"] if level == "encoder" else arrays[f"readout_{level[-1]}"])
                   for level in LEVELS}
        if previous is not None:
            for level in LEVELS:
                cka[level].append(linear_cka(current[level], previous[level]))
                for name, block in (("euclidean", slice(0, 16)), ("hyperbolic", slice(16, 32))):
                    cka_halves[level][name].append(
                        linear_cka(current[level][:, block], previous[level][:, block]))
        previous = current
    report["cka"] = {"consecutive": cka, "halves": cka_halves}

    # --- C: codeword stability across consecutive observation points ---
    codes = [np.load(path)["codes"] for path in observations]
    flips = np.zeros(codes[0].shape, dtype=np.int32)
    for before, after in zip(codes[:-1], codes[1:]):
        flips += (before != after).astype(np.int32)
    margins = np.mean([np.load(path)["margin_3"].astype(np.float32) for path in observations], axis=0)
    errors = np.mean([np.load(path)["error_3"].astype(np.float32) for path in observations], axis=0)
    degrees = sample["degree"].astype(np.float64)
    triangles = np.zeros(len(degrees))
    counts = np.zeros(len(degrees))
    np.add.at(triangles, sample["edges"][:, 0], sample["triangles"])
    np.add.at(triangles, sample["edges"][:, 1], sample["triangles"])
    np.add.at(counts, sample["edges"][:, 0], 1.0)
    np.add.at(counts, sample["edges"][:, 1], 1.0)
    triangle_density = np.divide(triangles, np.maximum(counts, 1))
    report["stability"] = {
        "flips_per_level_mean": [float(flips[:, level].mean()) for level in range(3)],
        "flips_per_level_max": [int(flips[:, level].max()) for level in range(3)],
        "items_never_flipped": [int((flips[:, level] == 0).sum()) for level in range(3)],
        "margin_mean_by_degree_quintile": _by_quintile(margins, degrees),
        "flips_by_triangle_quintile": _by_quintile(flips[:, 0].astype(np.float64), triangle_density),
        "flips_by_role": _by_role(flips[:, 0].astype(np.float64), sample["role"]),
        "correlation": {
            "flips_vs_degree": _spearman(flips[:, 0], degrees),
            "flips_vs_triangle_density": _spearman(flips[:, 0], triangle_density),
            "flips_vs_margin": _spearman(flips[:, 0], margins),
            "flips_vs_error": _spearman(flips[:, 0], errors),
        },
        "per_item": {"flips": flips.tolist(), "margin": margins.tolist(), "error": errors.tolist()},
    }
    inside_share = _item_inside_share(sample)
    clustering = _item_clustering(sample, degrees)
    association = graphlet_association(
        {"triangle_density": triangle_density, "inside_share": inside_share,
         "clustering": clustering},
        {"log_degree": np.log1p(degrees), "log_community_size": np.log1p(
            _community_size(sample))},
        flips[:, 0].astype(np.float64), sample["role"])
    report["stability"]["graphlet_association"] = association
    print("\ngraphlet association with L1 flips (partial Spearman, cluster bootstrap):")
    for name, entry in association.items():
        if "p_fdr" in entry:
            print(f"  {name:18s} rho={entry['partial_spearman']:+.4f} "
                  f"CI=[{entry['ci95'][0]:+.4f},{entry['ci95'][1]:+.4f}] "
                  f"p_fdr={entry['p_fdr']:.3g}")
        else:
            print(f"  {name:18s} flips={entry['mean_flips']:.4f} "
                  f"CI=[{entry['ci95'][0]:.4f},{entry['ci95'][1]:.4f}] "
                  f"items={entry['items']} (overall {entry['overall_mean']:.4f})")
    np.savez_compressed(HERE / "stability.npz", flips=flips, margin=margins, error=errors,
                        degree=degrees, triangle_density=triangle_density,
                        inside_share=inside_share, clustering=clustering,
                        role=sample["role"], community=sample["community"])

    # --- D: optimization dynamics, straight from the recorded summaries ---
    summaries = [json.loads(path.read_text()) for path in
                 sorted((run / "observations").glob("epoch_*.json"))]
    report["dynamics"] = {
        "epochs": [entry["epoch"] for entry in summaries],
        "used_codes": [entry["used_codes"] for entry in summaries],
        "entropy": [entry["entropy"] for entry in summaries],
        "mean_error": [entry["mean_error"] for entry in summaries],
        "mean_margin": [entry["mean_margin"] for entry in summaries],
        "effective_rank": [entry["effective_rank"] for entry in summaries],
    }
    metrics = run / "logs" / "training_metrics.jsonl"
    if metrics.is_file():
        steps = []
        for line in metrics.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") == "qda_step":
                steps.append(record)
        report["dynamics"]["steps"] = steps

    out = HERE / f"analysis_{args.arm}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out}")
    print("CKA (consecutive):", {level: [round(v, 4) for v in cka[level]] for level in LEVELS})


def _community_size(sample) -> np.ndarray:
    community = sample["community"].astype(np.int64)
    counts = np.bincount(community - community.min())
    return counts[community - community.min()].astype(np.float64)


def _item_inside_share(sample) -> np.ndarray:
    """Share of an item's behaviour edges that stay inside its community."""
    inside = sample["inside"].astype(np.float64)
    out = np.zeros(len(sample["degree"]))
    count = np.zeros(len(sample["degree"]))
    np.add.at(out, sample["edges"][:, 0], inside)
    np.add.at(out, sample["edges"][:, 1], inside)
    np.add.at(count, sample["edges"][:, 0], 1.0)
    np.add.at(count, sample["edges"][:, 1], 1.0)
    return np.divide(out, np.maximum(count, 1))


def _item_clustering(sample, degree: np.ndarray) -> np.ndarray:
    """Local clustering coefficient from the triangle counts already frozen."""
    triangles = np.zeros(len(degree))
    np.add.at(triangles, sample["edges"][:, 0], sample["triangles"])
    np.add.at(triangles, sample["edges"][:, 1], sample["triangles"])
    possible = np.maximum(degree, 1) * np.maximum(degree - 1, 1) / 2.0
    return triangles / possible


def _residualise(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Rank-residualise a variable on the control set."""
    from scipy.stats import rankdata
    ranked = rankdata(values)
    design = np.column_stack([np.ones(len(ranked))] +
                             [rankdata(controls[:, index]) for index in range(controls.shape[1])])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def graphlet_association(features: dict, controls: dict, outcome: np.ndarray,
                         role: np.ndarray, reps: int = 500, seed: int = 42) -> dict:
    """Partial Spearman of each graphlet feature with the outcome, plus role means.

    Degrees and community sizes are controlled because they are the obvious
    confounds: a hub flips codes more often than a leaf no matter its graphlets.
    The interval resamples *items*, which is the independent unit even though the
    edges are not.
    """
    from scipy.stats import spearmanr
    control_matrix = np.column_stack(list(controls.values()))
    out_residual = _residualise(outcome, control_matrix)
    rng = np.random.default_rng(seed)
    result = {}
    for name, values in features.items():
        residual = _residualise(values, control_matrix)
        rho = float(np.corrcoef(residual, out_residual)[0, 1])
        draws = np.array([
            np.corrcoef(residual[pick], out_residual[pick])[0, 1]
            for pick in (rng.integers(0, len(residual), len(residual)) for _ in range(reps))])
        result[name] = {"partial_spearman": rho,
                        "ci95": [float(np.quantile(draws, 0.025)),
                                 float(np.quantile(draws, 0.975))],
                        "raw_spearman": float(spearmanr(values, outcome).statistic)}
    if (role >= 0).any():
        # Roles are a partition, not a variable: report the mean outcome per role
        # with an item-level bootstrap, never a correlation coefficient.
        for label in sorted(set(int(r) for r in role if r >= 0)):
            mask = role == label
            values = outcome[mask]
            draws = np.array([values[rng.integers(0, len(values), len(values))].mean()
                              for _ in range(reps)])
            result[f"role_{label}"] = {
                "items": int(mask.sum()),
                "mean_flips": float(values.mean()),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "overall_mean": float(outcome.mean())}
    p_values = []
    names = []
    for name, entry in result.items():
        if "items" in entry:
            continue
        # Two-sided normal approximation on the residual correlation.
        from math import sqrt, erfc
        z = entry["partial_spearman"] * sqrt(max(len(outcome) - 3, 1))
        entry["p"] = float(erfc(abs(z) / sqrt(2)))
        p_values.append(entry["p"])
        names.append(name)
    order = np.argsort(p_values)
    adjusted = {}
    previous = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        index = order[rank_index]
        value = min(previous, p_values[index] * len(p_values) / (rank_index + 1))
        adjusted[names[index]] = value
        previous = value
    for name in names:
        result[name]["p_fdr"] = float(adjusted[name])
    return result


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _by_quintile(values: np.ndarray, axis: np.ndarray) -> list[float]:
    cuts = np.quantile(axis, [0.2, 0.4, 0.6, 0.8])
    bins = np.searchsorted(cuts, axis, side="right")
    return [float(values[bins == b].mean()) if (bins == b).any() else float("nan")
            for b in range(5)]


def _by_role(values: np.ndarray, role: np.ndarray) -> dict:
    out = {}
    for label in sorted(set(int(r) for r in role if r >= 0)):
        mask = role == label
        out[str(label)] = {"items": int(mask.sum()), "mean_value": float(values[mask].mean())}
    return out


if __name__ == "__main__":
    main()
