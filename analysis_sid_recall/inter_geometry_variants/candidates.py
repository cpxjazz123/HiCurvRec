"""The frozen candidate registry for Issue #135. Written before the scan runs.

Every entry states its family, the levels it applies to, its formula, its exact
denominator, whether it is exact or undefined-prone, and its priority class.
A/B are the primary family, C the null-relative family, D an explicitly
secondary sensitivity. Baselines (#133) are listed for comparison only and are
not part of the exploratory family.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAMILIES = {
    "A": "local-to-global aggregation on the complete quotient graph",
    "B": "parent-prefix-conditioned sibling quotient geometry (L2/L3 only)",
    "C": "null-relative versions of the same structural measures",
    "D": "SECONDARY sensitivity: multi-item-prefix-only induced subgraph",
}

BASELINE = [
    ("density", "2E/[N(N-1)] on Q_l", "all occupied prefixes, singletons included"),
    ("transitivity", "3T/wedges, wedge-weighted", "nodes with degree >= 2"),
    ("triangles", "exact sparse triangle count", "every edge"),
    ("cycle_rank_per_node", "(E-N+components)/N", "all quotient nodes"),
    ("degree_median", "median quotient degree", "all quotient nodes"),
    ("degree_p90", "p90 quotient degree", "all quotient nodes"),
    ("forman_mean", "mean of 4-deg(u)-deg(v) over every quotient edge", "every quotient edge"),
    ("forman_negative_fraction", "share of edges with negative Forman curvature", "every quotient edge"),
]

CANDIDATES = [
    # A - local aggregation
    ("A", "local_transitivity_mean", (1, 2, 3),
     "mean over nodes with deg>=2 of 2t_v/[d_v(d_v-1)], t_v = edges among N(v)",
     "nodes with deg>=2; degree<2 nodes counted in coverage only"),
    ("A", "local_transitivity_median", (1, 2, 3), "median of the local clustering coefficient",
     "nodes with deg>=2"),
    ("A", "local_transitivity_p10", (1, 2, 3), "p10 of the local clustering coefficient",
     "nodes with deg>=2"),
    ("A", "local_transitivity_p90", (1, 2, 3), "p90 of the local clustering coefficient",
     "nodes with deg>=2"),
    ("A", "local_transitivity_iqr", (1, 2, 3), "p90-p10 of the local clustering coefficient",
     "nodes with deg>=2"),
    ("A", "triangles_per_node_mean", (1, 2, 3), "mean t_v over all quotient nodes",
     "all quotient nodes; absent triangles count as 0, not undefined"),
    ("A", "ego_density_mean", (1, 2, 3),
     "mean of 2(d_v+t_v)/[d_v(d_v+1)], closed ego including the centre", "nodes with deg>=1"),
    ("A", "ego_density_median", (1, 2, 3), "median closed-ego density", "nodes with deg>=1"),
    ("A", "ego_cycle_surplus_mean", (1, 2, 3),
     "mean t_v: the closed ego is connected, so its cycle rank is t_v", "all quotient nodes"),
    ("A", "cycle_rank_per_edge", (1, 2, 3), "(E-N+components)/E", "quotient edges > 0"),
    ("A", "cycle_surplus_per_component_mean", (1, 2, 3),
     "mean E_c-N_c+1 over components with >=2 nodes", "components with >=2 nodes"),
    ("A", "degree_mean", (1, 2, 3), "mean quotient degree", "all quotient nodes", ),
    ("A", "degree_p10", (1, 2, 3), "p10 quotient degree", "all quotient nodes"),
    ("A", "degree_iqr", (1, 2, 3), "p90-p10 quotient degree", "all quotient nodes"),
    ("A", "degree_cv", (1, 2, 3), "sd/mean of quotient degree", "mean > 0"),
    ("A", "degree_p90_over_mean", (1, 2, 3), "p90/mean", "mean > 0"),
    ("A", "degree_median_over_mean", (1, 2, 3), "median/mean", "mean > 0"),
    ("A", "forman_p10", (1, 2, 3), "p10 of 4-deg(u)-deg(v)", "every quotient edge"),
    ("A", "forman_p90", (1, 2, 3), "p90 of 4-deg(u)-deg(v)", "every quotient edge"),
    ("A", "forman_node_incident_mean", (1, 2, 3),
     "mean over nodes with edges of the mean incident-edge Forman curvature",
     "nodes with degree >= 1"),
    ("A", "forman_node_incident_median", (1, 2, 3),
     "median over nodes with edges of the mean incident-edge Forman curvature",
     "nodes with degree >= 1"),
    # B - parent-conditioned sibling geometry
    ("B", "sibling_density_mean", (2, 3), "unweighted mean of 2E_P/[N_P(N_P-1)] over parents",
     "parents with >=2 children"),
    ("B", "sibling_density_median", (2, 3), "median sibling density", "parents with >=2 children"),
    ("B", "sibling_density_child_weighted", (2, 3), "child-count-weighted mean sibling density",
     "parents with >=1 child; weights are child counts"),
    ("B", "sibling_density_item_weighted", (2, 3),
     "parent item-count-weighted mean sibling density (declared sensitivity)",
     "parents with >=1 child; weights are item counts"),
    ("B", "sibling_transitivity_mean", (2, 3), "unweighted mean of parent transitivity",
     "parents with wedges > 0 and exact triangles"),
    ("B", "sibling_transitivity_child_weighted", (2, 3),
     "child-count-weighted mean of parent transitivity", "parents with wedges > 0"),
    ("B", "sibling_cycle_rank_per_node_mean", (2, 3),
     "mean over parents of (E_P-N_P+components_P)/N_P", "parents with >=2 children"),
    ("B", "sibling_degree_median_mean", (2, 3), "mean over parents of median child degree",
     "parents with >=2 children"),
    ("B", "sibling_degree_mean_mean", (2, 3), "mean over parents of mean child degree",
     "parents with >=2 children"),
    ("B", "sibling_degree_p90_mean", (2, 3), "mean over parents of p90 child degree",
     "parents with >=2 children"),
    ("B", "sibling_forman_mean", (2, 3), "mean over parents of parent Forman mean",
     "parents with >=1 sibling edge"),
    ("B", "sibling_forman_negative_fraction", (2, 3),
     "mean over parents of parent Forman negative fraction", "parents with >=1 sibling edge"),
    ("B", "sibling_edge_share", (2, 3), "E_sibling/E_quotient", "E_quotient > 0"),
    # C - null-relative (delta, z and ratio per declared feature and design)
    ("C", "null_delta", (1, 2, 3), "f_real - mean(f_null)", ">=20 nulls, metric defined"),
    ("C", "null_z", (1, 2, 3), "(f_real-mean)/sd", "sd > 0 and metric defined"),
    ("C", "null_ratio", (1, 2, 3), "f_real/mean_null", "mean_null bounded away from zero"),
    # D - secondary sensitivity
    ("D", "multi_only_density", (1, 2, 3), "density of Q_l restricted to multi-item prefixes",
     "induced subgraph; lost nodes/edges/items reported"),
    ("D", "multi_only_transitivity", (1, 2, 3), "transitivity of the induced subgraph",
     "wedges > 0"),
    ("D", "multi_only_cycle_rank_per_node", (1, 2, 3), "(E-N+c)/N on the induced subgraph",
     "N > 0"),
    ("D", "multi_only_degree_median", (1, 2, 3), "median degree on the induced subgraph",
     "N > 0"),
    ("D", "multi_only_forman_mean", (1, 2, 3), "Forman mean on the induced subgraph",
     "every induced edge"),
    ("D", "multi_only_forman_negative_fraction", (1, 2, 3),
     "negative-edge fraction on the induced subgraph", "every induced edge"),
]

NULL_FEATURES = ("density", "transitivity", "cycle_rank_per_node", "degree_median",
                 "forman_mean", "triangles_per_node_mean", "local_transitivity_mean",
                 "sibling_density_mean", "sibling_transitivity_mean")
NULL_DESIGNS = ("global", "degree_stratified", "parent_conditioned")
NULL_REPLICATES = 20
SEED = 20261102

#: candidate -> the baseline feature it is compared to for the paired |rho| test
BASELINE_PAIRS = {
    "local_transitivity_mean": "transitivity",
    "local_transitivity_median": "transitivity",
    "local_transitivity_p10": "transitivity",
    "local_transitivity_p90": "transitivity",
    "local_transitivity_iqr": "transitivity",
    "triangles_per_node_mean": "transitivity",
    "ego_density_mean": "density",
    "ego_density_median": "density",
    "ego_cycle_surplus_mean": "cycle_rank_per_node",
    "cycle_rank_per_edge": "cycle_rank_per_node",
    "cycle_surplus_per_component_mean": "cycle_rank_per_node",
    "degree_mean": "degree_median",
    "degree_p10": "degree_median",
    "degree_iqr": "degree_median",
    "degree_cv": "degree_median",
    "degree_p90_over_mean": "degree_p90",
    "degree_median_over_mean": "degree_median",
    "forman_p10": "forman_mean",
    "forman_p90": "forman_mean",
    "forman_node_incident_mean": "forman_mean",
    "forman_node_incident_median": "forman_mean",
    "sibling_density_mean": "density",
    "sibling_density_median": "density",
    "sibling_density_child_weighted": "density",
    "sibling_density_item_weighted": "density",
    "sibling_transitivity_mean": "transitivity",
    "sibling_transitivity_child_weighted": "transitivity",
    "sibling_cycle_rank_per_node_mean": "cycle_rank_per_node",
    "sibling_degree_median_mean": "degree_median",
    "sibling_degree_mean_mean": "degree_median",
    "sibling_degree_p90_mean": "degree_p90",
    "sibling_forman_mean": "forman_mean",
    "sibling_forman_negative_fraction": "forman_negative_fraction",
    "sibling_edge_share": "density",
    "multi_only_density": "density",
    "multi_only_transitivity": "transitivity",
    "multi_only_cycle_rank_per_node": "cycle_rank_per_node",
    "multi_only_degree_median": "degree_median",
    "multi_only_forman_mean": "forman_mean",
    "multi_only_forman_negative_fraction": "forman_negative_fraction",
}


def registry() -> dict:
    rows = [{"family": family, "name": name, "levels": list(levels),
             "formula": formula, "denominator": denominator, "status": "frozen"}
            for family, name, levels, formula, denominator in CANDIDATES]
    nulls = [{"family": "C", "name": f"null_{stat}_{feature}",
              "levels": [1, 2, 3] if feature not in ("sibling_density_mean",
                                                      "sibling_transitivity_mean") else [2, 3],
              "formula": f"{stat} of f_real against f_null for {feature}",
              "denominator": f"{NULL_REPLICATES} deterministic nulls; sd>0 required for z",
              "status": "frozen"}
             for stat in ("delta", "z", "ratio") for feature in NULL_FEATURES]
    return {"version": 1, "families": FAMILIES, "baseline": [
                {"name": name, "formula": formula, "denominator": denominator}
                for name, formula, denominator in BASELINE],
            "candidates": rows, "null_derived": nulls, "null_designs": list(NULL_DESIGNS),
            "null_replicates": NULL_REPLICATES, "seed": SEED,
            "baseline_pairs": BASELINE_PAIRS,
            "selection_rule": "GAIN only if a candidate has a valid geometry definition, "
                              "coverage >= 0.95, the paired-bootstrap CI for its |rho| gain "
                              "over its declared baseline excludes 0, and its direction is "
                              "stable within the 76-arm curvature_RQ-VAE cohort; otherwise "
                              "NO STABLE GAIN"}


def freeze(path: Path = HERE / "candidate_registry.json") -> dict:
    data = registry()
    blob = json.dumps(data, indent=1, sort_keys=True)
    path.write_text(blob + "\n", encoding="utf-8")
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    (HERE / "candidate_registry.sha256").write_text(digest + "\n", encoding="utf-8")
    data["sha256"] = digest
    return data


if __name__ == "__main__":
    print(json.dumps({"frozen": str(HERE / "candidate_registry.json"),
                      "sha256": freeze()["sha256"],
                      "candidates": len(CANDIDATES) + 3 * len(NULL_FEATURES)},
                     indent=1))
