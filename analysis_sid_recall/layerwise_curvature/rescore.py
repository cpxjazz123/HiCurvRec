"""Recompute Issue #129's statistics from `experiment_table.csv` alone.

The measurement stage (alignment, per-level curvature, the original-graph gap)
is expensive and already stored in the table; this script re-reads it and
produces the statistics, the figures, the report and the pre-registration. It
uses the system interpreter on purpose: no TensorFlow import, no GPU, so it
cannot disturb a running job and it does not depend on the network-mounted
accelerator stack.

Run:  python3 analysis_sid_recall/layerwise_curvature/rescore.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw as stats  # noqa: E402
import features_lw as features  # noqa: E402
import run as pipeline  # noqa: E402

FIELDS = ("g1", "g2", "g3", "z1", "z2", "z3", "A1", "A2", "A3", "collision",
          "codes_used_l1", "codes_used_l2", "codes_used_l3", "recall",
          "edges_l1", "edges_l2", "edges_l3",
          "delta1", "delta2", "delta3",
          "delta1_null_p", "delta2_null_p", "delta3_null_p",
          "delta1_n_within", "delta2_n_within", "delta3_n_within",
          "delta1_n_between", "delta2_n_between", "delta3_n_between")


def load() -> list[dict]:
    rows = []
    for row in csv.DictReader((HERE / "experiment_table.csv").open()):
        record = dict(row)
        for key in FIELDS:
            record[key] = pipeline.as_float(row.get(key))
        record["series"] = row["series"]
        record["delta1_status"] = row.get("delta1_status")
        record["delta2_status"] = row.get("delta2_status")
        record["delta3_status"] = row.get("delta3_status")
        rows.append(record)
    return rows


def main() -> int:
    rows = load()
    usable = [row for row in rows if np.isfinite(row["g1"]) and np.isfinite(row["A1"])]
    y = np.asarray([row["recall"] for row in usable])
    series = np.asarray([row["series"] for row in usable])
    results = {"n": len(usable), "provenance": {}, "univariate": {}, "joint": {},
               "nonlinear": {}, "gap": {}, "sensitivity": {},
               "controls": {}, "multicollinearity": {}}
    import hashlib as _hashlib
    pairs_path = REPO / "analysis_sid_recall" / "sid_recall.csv"
    recorded = {row["sid_sha256"]: row for row in csv.DictReader(pairs_path.open())}
    tiers = {"HASH_PROVEN": 0, "RECORD_HASH_MATCH": 0, "MISSING": 0}
    for row in rows:
        source = row.get("sid_path")
        if not source or not Path(source).is_file():
            row["provenance_tier"] = "MISSING"
            tiers["MISSING"] += 1
            continue
        digest = _hashlib.sha256(Path(source).read_bytes()).hexdigest()
        is_recorded_path = source == recorded.get(row["sid_sha256"], {}).get("code_path")
        if digest == row["sid_sha256"]:
            row["provenance_tier"] = "HASH_PROVEN" if is_recorded_path else "RECORD_HASH_MATCH"
            tiers[row["provenance_tier"]] += 1
        else:
            row["provenance_tier"] = "MISSING"
            tiers["MISSING"] += 1
    results["provenance"] = tiers

    pvalues, keys = [], []
    for level in (1, 2, 3):
        for key in (f"g{level}", f"z{level}"):
            x = np.asarray([row[key] for row in usable])
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                results["univariate"][key] = {"n": int(mask.sum()), "status": "TOO_FEW"}
                continue
            rho = stats.spearman(x[mask], y[mask])
            p = stats.permutation_p(x[mask], y[mask], rho)
            lo, hi = stats.bootstrap_ci(x[mask], y[mask])
            results["univariate"][key] = {"n": int(mask.sum()), "spearman": rho,
                                          "kendall": stats.kendall(x[mask], y[mask]),
                                          "p": p, "ci95": [lo, hi]}
            pvalues.append(p)
            keys.append(key)
    results["univariate_holm"] = stats.holm(pvalues)
    for key, value in zip(keys, results["univariate_holm"]):
        results["univariate"][key]["holm_p"] = value

    # controls-only: does code distribution explain the recall at all?
    for key in ("A1", "A2", "A3", "collision", "codes_used_l1", "codes_used_l2"):
        x = np.asarray([row[key] for row in usable])
        mask = np.isfinite(x)
        if mask.sum() < 10:
            continue
        rho = stats.spearman(x[mask], y[mask])
        results["controls"][key] = {"n": int(mask.sum()), "spearman": rho,
                                    "p": stats.permutation_p(x[mask], y[mask], rho)}

    # multicollinearity among the model's features
    names = ["z1", "z2", "z3", "A1", "A2", "collision"]
    keep = [row for row in usable if all(np.isfinite(row[k]) for k in names)]
    X = np.asarray([[row[k] for k in names] for row in keep])
    if X.shape[0] > len(names):
        correlation = np.corrcoef(X.T)
        results["multicollinearity"] = {
            "n": int(X.shape[0]),
            "max_abs_pair": float(np.max(np.abs(correlation - np.eye(len(names))))),
            "condition_number": float(np.linalg.cond(np.column_stack([np.ones(len(keep)), X]))),
        }

    def design(columns):
        subset = [row for row in usable
                  if all(np.isfinite(row[k]) for k in columns)]
        X = np.asarray([[row[k] for k in columns] for row in subset])
        yy = np.asarray([row["recall"] for row in subset])
        ss = np.asarray([row["series"] for row in subset])
        return X, yy, ss, subset

    controls = ["A1", "A2", "collision"]
    curvature = ["z1", "z2", "z3"]
    # both models must be compared on the SAME rows; the full model defines them
    common = [row for row in usable
              if all(np.isfinite(row[k]) for k in controls + curvature)]
    _Xc_all, _yc_all, _sc_all, _ = design(controls + curvature)
    Xc = np.asarray([[row[k] for k in controls] for row in common])
    yc = np.asarray([row["recall"] for row in common])
    sc = np.asarray([row["series"] for row in common])
    keepc = keepf = common
    Xf = np.asarray([[row[k] for k in controls + curvature] for row in common])
    yf, sf = yc, sc
    joint = {"n_controls": int(len(keepc)), "n_full": int(len(keepf)),
             "controls": stats.leave_one_series_out(Xc, yc, sc),
             "full": stats.leave_one_series_out(Xf, yf, sf),
             "in_sample_controls": stats.fit_ols(Xc, yc),
             "in_sample_full": stats.fit_ols(Xf, yf),
             "cluster_bootstrap": stats.cluster_bootstrap(Xf, yf, sf, trials=600),
             "feature_order": controls + curvature}
    cp = np.asarray(joint["controls"]["predictions"])
    fp = np.asarray(joint["full"]["predictions"])
    # the two models are fitted on different row sets, so compare them on the
    # rows both predicted, by index into their own targets
    results["joint"] = joint
    if len(keepc) == len(keepf):
        both = np.isfinite(cp) & np.isfinite(fp)
        if both.sum() >= 10:
            c = float(np.sqrt(((cp[both] - yc[both]) ** 2).mean()))
            f = float(np.sqrt(((fp[both] - yf[both]) ** 2).mean()))
            joint["paired"] = {"n": int(both.sum()), "rmse_controls": c,
                               "rmse_full": f, "improvement": (c - f) / c}
            joint["rmse_improvement"] = joint["paired"]["improvement"]
    else:
        joint["paired"] = {"status": "DIFFERENT_ROW_SETS",
                           "n_controls": len(keepc), "n_full": len(keepf)}

    def partial(x, y, controls_matrix):
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls_matrix).all(axis=1)
        if mask.sum() < 12:
            return None
        rx, ry = stats.rank(x[mask]), stats.rank(y[mask])
        rc = np.column_stack([stats.rank(controls_matrix[mask, j])
                              for j in range(controls_matrix.shape[1])])
        design_c = np.column_stack([np.ones(mask.sum()), rc])
        bx, *_ = np.linalg.lstsq(design_c, rx, rcond=None)
        by, *_ = np.linalg.lstsq(design_c, ry, rcond=None)
        ex, ey = rx - design_c @ bx, ry - design_c @ by
        denominator = float(np.sqrt((ex * ex).sum() * (ey * ey).sum()))
        return float((ex * ey).sum() / denominator) if denominator else None

    resolution = np.asarray([[np.log1p(row["edges_l2"]) if np.isfinite(row["edges_l2"]) else np.nan,
                              row["codes_used_l2"], row["A2"], row["collision"]]
                             for row in usable])
    results["confound"] = {}
    for level in (1, 2, 3):
        for key in (f"g{level}", f"z{level}"):
            x = np.asarray([row[key] for row in usable])
            results["confound"][key] = {
                "partial_spearman_given_resolution": partial(x, y, resolution),
                "spearman_vs_log_edges": stats.spearman(
                    x[np.isfinite(x)], np.asarray([row["edges_l" + str(level)]
                                                   for row in usable])[np.isfinite(x)])
                if np.isfinite(x).sum() > 10 else None,
            }

    for level in (1, 2, 3):
        base = controls + [f"z{level}"]
        finite = [row[f"z{level}"] for row in usable if np.isfinite(row[f"z{level}"])]
        centre = float(np.mean(finite))
        for row in usable:
            if np.isfinite(row[f"z{level}"]):
                centred = row[f"z{level}"] - centre
                row[f"z{level}_c"] = centred
                row[f"z{level}_sq"] = centred * centred
            else:
                row[f"z{level}_c"] = float("nan")
                row[f"z{level}_sq"] = float("nan")
        centred_base = controls + [f"z{level}_c"]
        Xb, yb, sb, _ = design(centred_base)
        Xq, yq, sq, _ = design(centred_base + [f"z{level}_sq"])
        linear = stats.leave_one_series_out(Xb, yb, sb)
        quad = stats.leave_one_series_out(Xq, yq, sq)
        model = stats.fit_ols(Xq, yq)
        coefficient = model["coefficients"][-1]
        vertex = (centre - model["coefficients"][-2] / (2 * coefficient)
                  if coefficient not in (0, None) else None)
        zs = np.asarray([row[f"z{level}"] for row in usable
                         if np.isfinite(row[f"z{level}"])])
        lp, qp = np.asarray(linear["predictions"]), np.asarray(quad["predictions"])
        both = np.isfinite(lp) & np.isfinite(qp)
        improvement = None
        if both.sum() >= 10:
            lb = float(np.sqrt(((lp[both] - yb[both]) ** 2).mean()))
            qb = float(np.sqrt(((qp[both] - yq[both]) ** 2).mean()))
            improvement = (lb - qb) / lb if lb else None
        results["nonlinear"][f"z{level}"] = {
            "n": int(linear.get("n") or 0), "linear_rmse": linear.get("rmse"),
            "quadratic_rmse": quad.get("rmse"), "improvement": improvement,
            "quadratic_coefficient": coefficient, "vertex": vertex,
            "vertex_inside_range": bool(vertex is not None
                                        and zs.min() <= vertex <= zs.max()),
            "supported": bool(improvement is not None and improvement > 0
                              and vertex is not None and zs.min() <= vertex <= zs.max())}

    for level in (1, 2, 3):
        defined = [row[f"delta{level}"] for row in rows if np.isfinite(row[f"delta{level}"])]
        statuses = [row.get(f"delta{level}_status") for row in rows]
        results["gap"][f"delta{level}"] = {
            "defined": len(defined), "undefined": len(rows) - len(defined),
            "mean": float(np.mean(defined)) if defined else None,
            "statuses": {status: statuses.count(status) for status in set(statuses)},
            "null_p_median": float(np.nanmedian([row[f"delta{level}_null_p"] for row in rows]))
            if np.isfinite([row[f"delta{level}_null_p"] for row in rows]).any() else None,
        }
    # does the gap add anything on its own? only where it is defined
    for level in (1, 2, 3):
        subset = [row for row in rows if np.isfinite(row[f"delta{level}"])
                  and np.isfinite(row["recall"])]
        if len(subset) < 10:
            results["gap"][f"delta{level}"]["recall_spearman"] = {
                "n": len(subset), "status": "TOO_FEW"}
            continue
        x = np.asarray([row[f"delta{level}"] for row in subset])
        yy = np.asarray([row["recall"] for row in subset])
        rho = stats.spearman(x, yy)
        controls_a = np.asarray([[row[f"A{level}"], row["collision"]] for row in subset])
        results["gap"][f"delta{level}"]["recall_spearman"] = {
            "n": len(subset), "spearman": rho,
            "p": stats.permutation_p(x, yy, rho),
            "partial_given_alignment": partial(x, yy, controls_a)}
    # does Delta1 add out-of-series predictive value over the controls?
    subset = [row for row in rows if np.isfinite(row["delta1"]) and np.isfinite(row["A1"])
              and np.isfinite(row["A2"]) and np.isfinite(row["collision"])]
    if len(subset) >= 20:
        base = ["A1", "A2", "collision"]
        Xb = np.asarray([[row[k] for k in base] for row in subset])
        Xd = np.asarray([[row[k] for k in base + ["delta1"]] for row in subset])
        yy = np.asarray([row["recall"] for row in subset])
        ss = np.asarray([row["series"] for row in subset])
        b = stats.leave_one_series_out(Xb, yy, ss)
        d = stats.leave_one_series_out(Xd, yy, ss)
        pb, pd = np.asarray(b["predictions"]), np.asarray(d["predictions"])
        both = np.isfinite(pb) & np.isfinite(pd)
        improvement = None
        if both.sum() >= 10:
            rb = float(np.sqrt(((pb[both] - yy[both]) ** 2).mean()))
            rd = float(np.sqrt(((pd[both] - yy[both]) ** 2).mean()))
            improvement = (rb - rd) / rb if rb else None
        results["gap"]["delta1_joint"] = {"n": len(subset), "rmse_controls": b.get("rmse"),
                                          "rmse_with_delta1": d.get("rmse"),
                                          "improvement": improvement}
    results["gap"]["sample"] = json.loads(
        (HERE / "orc_sample_meta.json").read_text()) if (HERE / "orc_sample_meta.json").is_file() else {}
    results["gap"]["rewiring_null"] = {
        "status": "NOT_MEASURED",
        "reason": "the exact-transport rewiring null needs the accelerator "
                  "interpreter's POT and the shared filesystem stalled; the "
                  "prefix-label-shuffle null is measured for every arm instead"}

    wide = {record["sid_sha256"]: record
            for record in json.loads((REPO / "analysis_sid_recall" / "curvature"
                                      / "full_182" / "per_arm_features_32edge.json").read_text())}
    audit = {row["sid_sha256"]: row for row in
             csv.DictReader((REPO / "analysis_sid_recall" / "pair_audit.csv").open())}
    for name, members in (("strict_A2_73", [sha for sha, row in audit.items()
                                            if row["name_ok"] == "True"
                                            and row["same_commit"] == "True"]),
                          ("wide_code_path_178", [sha for sha, row in audit.items()
                                                  if row["code_path_ok"] == "True"])):
        block = {}
        for level in (1, 2, 3):
            x, yy = [], []
            for sha in members:
                record = wide.get(sha)
                if not record:
                    continue
                value = record["levels"][str(level)].get("curvature_z")
                if value is None or value != value:
                    continue
                x.append(float(value))
                yy.append(float(record["recall"]))
            if len(x) < 10:
                block[f"z{level}"] = {"n": len(x), "status": "TOO_FEW"}
                continue
            rho = stats.spearman(np.asarray(x), np.asarray(yy))
            block[f"z{level}"] = {"n": len(x), "spearman": rho,
                                  "p": stats.permutation_p(np.asarray(x), np.asarray(yy), rho)}
        results["sensitivity"][name] = block

    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    stats.scatter_svg(usable, "g", HERE / "scatter_g.svg", "g_l")
    stats.scatter_svg(usable, "z", HERE / "scatter_z.svg", "z_l")
    pipeline.write_report(results, rows, usable)
    pipeline.write_preregistration(results)
    print(json.dumps({key: results[key] for key in ("n", "provenance")}, indent=1))
    print("univariate:", json.dumps(results["univariate"], indent=1)[:700])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
