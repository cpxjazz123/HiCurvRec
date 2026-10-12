"""Run Slice B: relation-level bridge between residual geometry and SIDs."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C
from distances import auc_neg_farther_than_pos, load_pairs, ordering_error_rate as slice_ordering_error_rate
from quant_bridge import anchor_bootstrap_ci, anchor_bootstrap_er_ci, cross_tabulate, decision_verdict, safe_er

BEHAVIOR_METRICS = C.ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/eval_behavior_metrics.py"
PREFIX_HELPERS = C.ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/analyze_prefix_behavior.py"

_PREFIX_MODULE = None


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _prefix_ids(levels, depth):
    if _PREFIX_MODULE is not None:
        return _PREFIX_MODULE.prefix_ids(levels, depth)
    return np.unique(levels[:, :depth], axis=0, return_inverse=True)[1]


def _slice_a_distance_records(path, model, pair_set):
    records = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["model"] != model or row["pair_set"] != pair_set or row["distribution"] != "comparison":
                continue
            for field in ("auc_neg_farther_than_pos", "ordering_error_rate",
                          "fraction_negative_closer", "overlapping_coefficient"):
                if row.get(field):
                    row[field] = float(row[field])
            records[(row["level"], row["variant"])] = row
    return records


def _flatten_comparisons(pairs, residual_vectors, levels, depth):
    anchors, positives, negatives = (pairs[k] for k in ("anchors", "positives", "negatives"))
    pos_dist = np.linalg.norm(residual_vectors[anchors] - residual_vectors[positives], axis=1)
    neg_dist = np.linalg.norm(residual_vectors[anchors, None, :] - residual_vectors[negatives], axis=2)
    min_pos = {}
    for anchor, dist in zip(anchors, pos_dist):
        anchor = int(anchor)
        min_pos[anchor] = min(min_pos.get(anchor, np.inf), float(dist))
    positive_correct = np.asarray([dist > min_pos[int(anchor)] for anchor, dist in zip(anchors, pos_dist)])
    negative_correct = np.asarray([[dist > min_pos[int(anchor)] for dist in row]
                                   for anchor, row in zip(anchors, neg_dist)], dtype=bool)
    correct = np.concatenate((positive_correct, negative_correct.reshape(-1)))
    other_ids = np.concatenate((positives, negatives.reshape(-1)))
    anchor_ids = np.concatenate((anchors, np.repeat(anchors, negatives.shape[1])))
    prefix = _prefix_ids(levels, depth)
    shared = prefix[anchor_ids] == prefix[other_ids]
    return {
        "pos_distances": pos_dist,
        "neg_distances": neg_dist,
        "continuous_correct": correct,
        "prefix_shared": shared,
        "comparison_anchors": anchor_ids,
        "positive_correct": positive_correct,
        "negative_correct": negative_correct,
    }


def _pair_distance_statistics(pair_data):
    pos, neg = pair_data["pos_distances"], pair_data["neg_distances"]
    metrics = _independent_distance_metrics(pos, neg)
    return {
        **metrics,
        "hard_negative_intrusion_rate": float(np.mean(~pair_data["negative_correct"])),
    }




def _independent_distance_metrics(pos, neg):
    positive = np.asarray(pos).reshape(-1, 1)
    negative = np.asarray(neg)
    if negative.ndim == 1:
        negative = negative.reshape(len(positive), -1)
    greater = negative > positive
    ties = negative == positive
    less = negative < positive
    return {
        "auc": float(np.mean(greater) + 0.5 * np.mean(ties)),
        "ordering_error_rate": float(np.mean(less) + 0.5 * np.mean(ties)),
    }
def _cluster_distance_counts(pos, neg, anchors):
    positive = np.asarray(pos).reshape(-1, 1)
    negative = np.asarray(neg)
    greater = negative > positive
    ties = negative == positive
    less = negative < positive
    unique, inverse = np.unique(np.asarray(anchors), return_inverse=True)
    denominator = np.bincount(inverse, weights=np.full(len(inverse), negative.shape[1]),
                              minlength=len(unique))
    auc_sums = np.bincount(inverse, weights=(greater + 0.5 * ties).sum(axis=1),
                           minlength=len(unique))
    error_sums = np.bincount(inverse, weights=(less + 0.5 * ties).sum(axis=1),
                             minlength=len(unique))
    return unique, auc_sums, error_sums, denominator


def _cluster_bootstrap_values(cluster_data, replicates, seed):
    unique, auc_sums, error_sums, denominators = cluster_data
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 2), dtype=float)
    for i in range(replicates):
        draws = rng.integers(0, len(unique), size=len(unique))
        multiplicities = np.bincount(draws, minlength=len(unique))
        denom = np.dot(multiplicities, denominators)
        values[i, 0] = np.dot(multiplicities, auc_sums) / denom
        values[i, 1] = np.dot(multiplicities, error_sums) / denom
    return values


def _bootstrap_pair_metrics(pair_data, pairs, reps, seed):
    clusters = _cluster_distance_counts(pair_data["pos_distances"],
                                        pair_data["neg_distances"], pairs["anchors"])
    values = _cluster_bootstrap_values(clusters, reps, seed)
    return {
        "auc": [float(x) for x in np.quantile(values[:, 0], [0.025, 0.975])],
        "ordering_error_rate": [float(x) for x in np.quantile(values[:, 1], [0.025, 0.975])],
    }


def _bootstrap_depth_changes(depth_data, pairs, replicates, seed):
    clusters = {
        label: _cluster_distance_counts(depth_data[label][0], depth_data[label][1], pairs["anchors"])
        for label in ("L1", "L2", "L3")
    }
    unique = clusters["L1"][0]
    rng = np.random.default_rng(seed)
    samples = {"L2": [], "L3": []}
    for _ in range(replicates):
        draws = rng.integers(0, len(unique), size=len(unique))
        multiplicities = np.bincount(draws, minlength=len(unique))
        metrics = {}
        for label, (_, auc_sums, error_sums, denominators) in clusters.items():
            denom = np.dot(multiplicities, denominators)
            metrics[label] = (np.dot(multiplicities, auc_sums) / denom,
                              np.dot(multiplicities, error_sums) / denom)
        for idx, label in ((1, "L2"), (2, "L3")):
            prev = ("L1", "L2")[idx-1]
            samples[label].append((metrics[label][0] - metrics[prev][0],
                                   metrics[label][1] - metrics[prev][1]))
    output = {"L1": {"delta_auc_from_previous": None, "delta_auc_ci": None,
                     "delta_ordering_error_from_previous": None, "delta_ordering_error_ci": None}}
    for idx, label in ((1, "L2"), (2, "L3")):
        values = np.asarray(samples[label])
        prev = ("L1", "L2")[idx-1]
        before = _independent_distance_metrics(*depth_data[prev])
        after = _independent_distance_metrics(*depth_data[label])
        output[label] = {
            "delta_auc_from_previous": float(after["auc"] - before["auc"]),
            "delta_auc_ci": [float(x) for x in np.quantile(values[:, 0], [0.025, 0.975])],
            "delta_ordering_error_from_previous": float(after["ordering_error_rate"] - before["ordering_error_rate"]),
            "delta_ordering_error_ci": [float(x) for x in np.quantile(values[:, 1], [0.025, 0.975])],
        }
    return output


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run():
    OUT = C.OUT_DIR / "quant_bridge"
    OUT.mkdir(parents=True, exist_ok=True)
    for model in ("Exp1a", "TIGER"):
        path = C.OUT_DIR / f"residuals_{model}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Slice A residual input not available: {path}")

    residuals = {}
    for model in ("Exp1a", "TIGER"):
        with np.load(C.OUT_DIR / f"residuals_{model}.npz") as data:
            residuals[model] = np.asarray(data["residuals"], dtype=np.float32)
    pairs_by_set = {
        "primary_valid": load_pairs(C.PRIMARY_PAIRS),
        "secondary_train": load_pairs(C.SECONDARY_PAIRS),
    }
    metrics_module = _module("crowding_behavior_metrics", BEHAVIOR_METRICS)
    prefix_module = _module("crowding_prefix_helpers", PREFIX_HELPERS)
    global _PREFIX_MODULE
    _PREFIX_MODULE = prefix_module
    sid_codes = {model: prefix_module.load_levels(C.ARMS[model]["sid_json"])[0]
                 for model in ("Exp1a", "TIGER")}

    step5, step6, cross_rows, er_rows = {}, {}, [], []
    distance_rows = []
    for pair_set, pairs in pairs_by_set.items():
        step5[pair_set] = {}
        step6[pair_set] = {}
        for model in ("Exp1a", "TIGER"):
            levels = sid_codes[model]
            behavior = metrics_module.metrics_for(levels, pairs)
            behavior_levels = {}
            for depth, key in enumerate(("L1", "L1_L2", "L1_L2_L3"), start=1):
                entry = behavior[key]
                flat_anchors = np.concatenate((pairs["anchors"], np.repeat(pairs["anchors"], pairs["negatives"].shape[1])))
                prefix = _prefix_ids(levels, depth)
                others = np.concatenate((pairs["positives"], pairs["negatives"].reshape(-1)))
                anchors = flat_anchors
                sharing = (prefix[anchors] == prefix[others]).astype(float)
                ci = anchor_bootstrap_ci(flat_anchors, sharing, C.BOOTSTRAP_REPLICATES, C.SEED + depth)
                entry = dict(entry)
                entry["p_share_pos_anchor_bootstrap_95ci"] = anchor_bootstrap_ci(
                    pairs["anchors"], (prefix[pairs["anchors"]] == prefix[pairs["positives"]]).astype(float),
                    C.BOOTSTRAP_REPLICATES, C.SEED + depth)
                neg_sharing_matrix = (prefix[pairs["anchors"][:, None]] == prefix[pairs["negatives"]]).astype(float)
                neg_sharing = neg_sharing_matrix.mean(axis=1)
                entry["p_share_neg_anchor_bootstrap_95ci"] = anchor_bootstrap_ci(
                    pairs["anchors"], neg_sharing, C.BOOTSTRAP_REPLICATES, C.SEED + 100 + depth)
                entry["er"] = safe_er(entry["p_share_pos"], entry["p_share_neg"])
                entry["er_undefined_reason"] = "negative sharing rate is zero" if entry["er"] is None else None
                entry["er_anchor_bootstrap_95ci"] = anchor_bootstrap_er_ci(
                    pairs["anchors"], (prefix[pairs["anchors"]] == prefix[pairs["positives"]]).astype(float),
                    neg_sharing_matrix, C.BOOTSTRAP_REPLICATES, C.SEED + 200 + depth)
                entry["p_share_all_comparisons_anchor_bootstrap_95ci"] = ci
                behavior_levels[key] = entry
                er_rows.append({"model": model, "pair_set": pair_set, "level": key,
                                "p_share_pos": entry["p_share_pos"], "p_share_neg": entry["p_share_neg"],
                                "gap": entry["gap"], "er": entry["er"],
                                "er_ci_low": entry["er_anchor_bootstrap_95ci"]["low"],
                                "er_ci_high": entry["er_anchor_bootstrap_95ci"]["high"],
                                "er_undefined_reason": entry["er_undefined_reason"],
                                "p_share_pos_ci_low": entry["p_share_pos_anchor_bootstrap_95ci"]["low"],
                                "p_share_pos_ci_high": entry["p_share_pos_anchor_bootstrap_95ci"]["high"],
                                "p_share_neg_ci_low": entry["p_share_neg_anchor_bootstrap_95ci"]["low"],
                                "p_share_neg_ci_high": entry["p_share_neg_anchor_bootstrap_95ci"]["high"]})
            level_tables = {}
            loss_by_level = {}
            for depth, label in enumerate(("L1", "L2", "L3"), start=1):
                p = _flatten_comparisons(pairs, residuals[model][:, depth-1, :], levels, depth)
                tab = cross_tabulate(p["continuous_correct"], p["prefix_shared"])
                tab["quantization_loss_given_continuous_correct"] = {
                    "count": tab["cells"]["quantization_loss"]["count"],
                    "denominator_continuous_correct": int(p["continuous_correct"].sum()),
                    "rate": (float(np.mean(p["prefix_shared"][p["continuous_correct"]]))
                             if p["continuous_correct"].any() else None),
                }
                tab["continuous_order_definition"] = "comparison distance strictly greater than this anchor's nearest positive-successor distance"
                level_tables[label] = tab
                loss_by_level[label] = tab["quantization_loss_given_continuous_correct"]["rate"]
                for cell, cell_data in tab["cells"].items():
                    cross_rows.append({"model": model, "pair_set": pair_set, "level": label,
                                       "cell": cell, **cell_data})
                cross_rows.append({"model": model, "pair_set": pair_set, "level": label,
                                   "cell": "quantization_loss_given_continuous_correct",
                                   "count": tab["quantization_loss_given_continuous_correct"]["count"],
                                   "rate": tab["quantization_loss_given_continuous_correct"]["rate"]})
            usable = {k: v for k, v in loss_by_level.items() if v is not None}
            max_loss_level = max(usable, key=lambda key: level_tables[key]["quantization_loss_given_continuous_correct"]["count"]) if usable else None
            positive_rates = {level: behavior_levels[key]["p_share_pos"]
                              for level, key in zip(("L1", "L2", "L3"), ("L1", "L1_L2", "L1_L2_L3"))}
            step5[pair_set][model] = {
                "prefix_metrics": behavior_levels,
                "codebook": behavior["codebook"],
                "conditional_retention": {k: v for k, v in behavior.items() if k.startswith("L2_given") or k.startswith("L3_given")},
                "over_dispersion": {
                    "phenomenon": "positive prefix-sharing rate collapses so true successors stop sharing",
                    "positive_prefix_sharing_by_level": positive_rates,
                    "change_from_L1": {level: float(rate - positive_rates["L1"])
                                       for level, rate in positive_rates.items()},
                },
                "cross_tabulation": level_tables,
                "most_quantization_loss_level": max_loss_level,
                "code_collision_occupancy_phenomenon": behavior["codebook"],
            }
            # Slice A's completed distance CSV supplies its canonical estimates.
            by = _slice_a_distance_records(C.OUT_DIR / "distances.csv", model, pair_set)
            scale_stats, aucs, errors, depth_report, normalized_data = [], [], [], {}, {}
            for depth, label in enumerate(("L1", "L2", "L3"), start=1):
                raw = _flatten_comparisons(pairs, residuals[model][:, depth-1, :], levels, depth)
                raw_stats = _pair_distance_statistics(raw)
                rms = float(np.sqrt(np.mean(residuals[model][:, depth-1, :].astype(np.float64) ** 2)))
                norm = _flatten_comparisons(pairs, residuals[model][:, depth-1, :] / rms, levels, depth)
                norm_stats = _pair_distance_statistics(norm)
                normalized_data[label] = (norm["pos_distances"], norm["neg_distances"])
                bootstrap = _bootstrap_pair_metrics(norm, pairs, C.BOOTSTRAP_REPLICATES, C.SEED + depth)
                a_rec = by.get((label, "rms_normalized"))
                if a_rec is None:
                    # Slice A's canonical metric functions are applied directly for the
                    # secondary pair file, which is not in Slice A's emitted CSV.
                    slice_auc = auc_neg_farther_than_pos(norm["pos_distances"], norm["neg_distances"])
                    slice_error = slice_ordering_error_rate(norm["pos_distances"], norm["neg_distances"])
                    slice_strict_error = float(np.mean(norm["neg_distances"] < norm["pos_distances"][:, None]))
                else:
                    slice_auc = a_rec["auc_neg_farther_than_pos"]
                    slice_error = a_rec["ordering_error_rate"]
                    slice_strict_error = a_rec["fraction_negative_closer"]
                if not np.isclose(norm_stats["auc"], slice_auc, rtol=0.0, atol=2e-6):
                    raise AssertionError(f"independent AUC mismatch beyond tie tolerance {model}/{pair_set}/{label}: own={norm_stats['auc']} SliceA={slice_auc} delta={norm_stats['auc'] - slice_auc}")
                if not np.isclose(norm_stats["ordering_error_rate"], slice_error, rtol=0.0, atol=2e-6):
                    raise AssertionError(f"independent ordering-error mismatch beyond tie tolerance {model}/{pair_set}/{label}: own={norm_stats['ordering_error_rate']} SliceA={slice_error} delta={norm_stats['ordering_error_rate'] - slice_error}")
                depth_report[label] = {
                    "distance_scale_rms": rms,
                    "slice_a_normalized_auc": slice_auc,
                    "slice_a_normalized_ordering_error_rate": slice_error,
                    "slice_a_strict_ordering_error_rate": slice_strict_error,
                    "recomputed_normalized_auc": norm_stats["auc"],
                    "recomputed_normalized_ordering_error_rate": norm_stats["ordering_error_rate"],
                    "auc_slice_a_minus_independent": float(slice_auc - norm_stats["auc"]),
                    "ordering_error_slice_a_minus_independent": float(slice_error - norm_stats["ordering_error_rate"]),
                    "recomputed_auc_anchor_bootstrap_95ci": bootstrap["auc"],
                    "recomputed_ordering_error_anchor_bootstrap_95ci": bootstrap["ordering_error_rate"],
                    "hard_negative_intrusion_rate": raw_stats["hard_negative_intrusion_rate"],
                    "quantized_behavior_loss": loss_by_level[label],
                    "distance_scale_shrinks_from_previous": None,
                }
                scale_stats.append(rms)
                aucs.append(norm_stats["auc"])
                errors.append(norm_stats["ordering_error_rate"])
            changes = _bootstrap_depth_changes(normalized_data, pairs, C.BOOTSTRAP_REPLICATES, C.SEED + 500)
            for idx, label in enumerate(("L1", "L2", "L3")):
                depth_report[label]["distance_scale_shrinks_from_previous"] = (
                    None if idx == 0 else bool(scale_stats[idx] < scale_stats[idx-1]))
                depth_report[label]["discrimination_change_from_previous"] = changes[label]
            reference_check = None
            for label in ("L1", "L2", "L3"):
                distance_rows.append({"model": model, "pair_set": pair_set,
                                      "level": label, **depth_report[label]})
            reference_check = None
            if model == "Exp1a" and pair_set == "primary_valid":
                references = np.asarray([0.7767, 0.5219, 0.5057])
                differences = np.asarray(aucs) - references
                reference_check = {
                    "reference_auc_by_level": references.tolist(),
                    "independent_auc_by_level": aucs,
                    "difference_by_level": differences.tolist(),
                    "within_0.0005": bool(np.all(np.abs(differences) <= 0.0005)),
                }
                if not reference_check["within_0.0005"]:
                    raise AssertionError(f"independent Exp1a primary AUC disagrees with supplied reference: {reference_check}")
            verdict = decision_verdict(scale_stats, aucs, errors)
            step6[pair_set][model] = {
                "distance_scale_source": "Slice A component-RMS definition, computed directly from each residual level; discrimination fields are read from Slice A distances.csv.",
                "auc_definition": "Slice A canonical AUC=P(d_neg>d_pos)+0.5*P(tie), informative direction high; ordering error=P(d_neg<d_pos)+0.5*P(tie), exact complement and low is good",
                "independent_slice_a_assertions": "Independent pairwise recomputation is asserted against Slice A's canonical fields within absolute tolerance 2e-6 to accommodate floating-point tie changes.",
                "reference_check": reference_check,
                "depth_metrics": depth_report,
                "decision_rule_inputs": {"scales": scale_stats, "normalized_aucs": aucs,
                                          "normalized_ordering_error_rates": errors},
                "verdict": verdict,
                "hypotheses": {
                    "effective_capacity": "Not tested; hypothesis only, not a finding.",
                    "behaviour_supervision": "Not isolated by this analysis; hypothesis only, not a finding.",
                    "quantization_decision": "Quantization-loss cross-tabs are descriptive; causal attribution is a hypothesis, not a finding.",
                },
            }
        for model in ("Exp1a", "TIGER"):
            other = "TIGER" if model == "Exp1a" else "Exp1a"
            current_book = step5[pair_set][model]["codebook"]
            other_book = step5[pair_set][other]["codebook"]
            supervision_contrast = {}
            for key in ("L1", "L1_L2", "L1_L2_L3"):
                left = step5[pair_set]["Exp1a"]["prefix_metrics"][key]
                right = step5[pair_set]["TIGER"]["prefix_metrics"][key]
                supervision_contrast[key] = {
                    "Exp1a_minus_TIGER_p_share_pos": left["p_share_pos"] - right["p_share_pos"],
                    "Exp1a_minus_TIGER_er": (None if left["er"] is None or right["er"] is None
                                              else left["er"] - right["er"]),
                }
            loss = step5[pair_set][model]["cross_tabulation"]
            step6[pair_set][model]["hypotheses"] = {
                "effective_capacity": {
                    "label": "HYPOTHESIS, not finding",
                    "own_codebook": current_book,
                    "control_codebook": other_book,
                    "plausibility_evidence": "Occupancy/collision values are reported for descriptive comparison; they do not identify capacity as a cause.",
                },
                "residual_noise": {
                    "label": "HYPOTHESIS, not finding",
                    "plausibility_evidence": "Deeper residuals may be dominated by quantization noise; this diagnostic does not separate useful residual signal from quantization error.",
                },
                "behaviour_supervision": {
                    "label": "HYPOTHESIS, not finding",
                    "Exp1a_minus_TIGER_sharing_and_ER": supervision_contrast,
                    "plausibility_evidence": "The matched-pair contrast indicates whether behaviour-related prefix outcomes differ across the behavior-trained and control arms, but does not isolate supervision causally.",
                },
                "quantization_decision": {
                    "label": "HYPOTHESIS, not finding",
                    "most_loss_level": step5[pair_set][model]["most_quantization_loss_level"],
                    "quantization_loss_by_level": {
                        level: value["quantization_loss_given_continuous_correct"]
                        for level, value in loss.items()
                    },
                    "plausibility_evidence": "The continuous-correct/shared-prefix cell measures relations lost at quantization; it does not establish why the quantizer made that decision.",
                },
            }
    undefined_metrics = []
    degenerate_metrics = []
    for pair_set, model_data in step5.items():
        for model, analysis in model_data.items():
            for level, metric in analysis["prefix_metrics"].items():
                if metric["er"] is None:
                    undefined_metrics.append({"scope": f"step5.{pair_set}.{model}.{level}.er",
                                              "reason": metric["er_undefined_reason"]})
                if metric["sampled_precision_given_share"] is None:
                    undefined_metrics.append({"scope": f"step5.{pair_set}.{model}.{level}.sampled_precision_given_share",
                                              "reason": "no positive or negative comparisons share this prefix"})
                for rate_key in ("p_share_pos", "p_share_neg", "macro_p_share_pos", "macro_p_share_neg",
                                 "sampled_precision_given_share"):
                    if metric[rate_key] in (0.0, 1.0):
                        degenerate_metrics.append({"scope": f"step5.{pair_set}.{model}.{level}.{rate_key}",
                                                   "value": metric[rate_key]})
                if metric["er"] == 0.0:
                    degenerate_metrics.append({"scope": f"step5.{pair_set}.{model}.{level}.er", "value": 0.0})
            for block, values in analysis["conditional_retention"].items():
                for side in ("pos", "neg"):
                    field = f"p_retain_{side}"
                    if values[field] is None:
                        undefined_metrics.append({
                            "scope": f"step5.{pair_set}.{model}.{block}.{field}",
                            "reason": f"effective comparison count {values[f'n_effective_{side}']} is below MIN_CELL=30",
                        })
                    elif values[field] in (0.0, 1.0):
                        degenerate_metrics.append({"scope": f"step5.{pair_set}.{model}.{block}.{field}",
                                                   "value": values[field]})
            for level, metrics in step6[pair_set][model]["depth_metrics"].items():
                for name in ("slice_a_normalized_auc", "slice_a_normalized_ordering_error_rate",
                             "recomputed_normalized_auc", "recomputed_normalized_ordering_error_rate",
                             "hard_negative_intrusion_rate"):
                    if metrics[name] in (0.0, 1.0):
                        degenerate_metrics.append({"scope": f"step6.{pair_set}.{model}.{level}.{name}",
                                                   "value": metrics[name]})
            for block, values in analysis["codebook"].items():
                for field, value in values.items():
                    if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
                        degenerate_metrics.append({
                            "scope": f"step5.{pair_set}.{model}.codebook.{block}.{field}",
                            "value": value,
                        })
            for level, table in analysis["cross_tabulation"].items():
                for cell, values in table["cells"].items():
                    if values["rate"] in (0.0, 1.0):
                        degenerate_metrics.append({"scope": f"step5.{pair_set}.{model}.{level}.{cell}",
                                                   "value": values["rate"]})
                if table["quantization_loss_given_continuous_correct"]["rate"] is None:
                    undefined_metrics.append({"scope": f"step5.{pair_set}.{model}.{level}.quantization_loss_given_continuous_correct",
                                              "reason": "no comparisons were continuously correct"})
    result = {
        "step5": step5,
        "step6": step6,
        "undefined_metrics": undefined_metrics,
        "degenerate_metrics": degenerate_metrics,
        "inputs": {
            "pair_sets": {k: {name: int(len(v[name])) for name in ("anchors", "positives", "negatives")}
                          for k, v in pairs_by_set.items()},
            "residual_files": {m: str(C.OUT_DIR / f"residuals_{m}.npz") for m in residuals},
        },
        "decision_rule": "If distance scale shrinks with depth and normalized discrimination does not worsen: NO EVIDENCE FOR INCREASED CROWDING; otherwise report deterioration with paired anchor-bootstrap confidence intervals, without attributing it to Euclidean geometry.",
    }
    (OUT / "quant_bridge.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    for name, rows in (("er_table.csv", er_rows), ("cross_tabulation.csv", cross_rows),
                       ("depth_metrics.csv", distance_rows)):
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (OUT / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print("ER_TABLE=model,pair_set,level,p_share_pos,p_share_neg,ER,ER_CI_low,ER_CI_high")
    for row in er_rows:
        print(f"{row['model']},{row['pair_set']},{row['level']},{row['p_share_pos']:.12g},"
              f"{row['p_share_neg']:.12g},{row['er'] if row['er'] is not None else 'undefined'},"
              f"{row['er_ci_low'] if row['er_ci_low'] is not None else 'undefined'},"
              f"{row['er_ci_high'] if row['er_ci_high'] is not None else 'undefined'}")
    print("CROSS_TAB=model,pair_set,level,cell,count,rate")
    for row in cross_rows:
        print(f"{row['model']},{row['pair_set']},{row['level']},{row['cell']},"
              f"{row['count']},{row['rate'] if row['rate'] is not None else 'undefined'}")
    print("STEP6=model,pair_set,level,scale,AUC,ordering_error,hard_negative_intrusion,quantized_loss,delta_AUC,delta_AUC_CI,delta_error,delta_error_CI")
    for pair_set, models in step6.items():
        for model, report in models.items():
            for level, metrics in report["depth_metrics"].items():
                change = metrics["discrimination_change_from_previous"]
                print(f"{model},{pair_set},{level},{metrics['distance_scale_rms']:.12g},"
                      f"{metrics['recomputed_normalized_auc']:.12g},"
                      f"{metrics['recomputed_normalized_ordering_error_rate']:.12g},"
                      f"{metrics['hard_negative_intrusion_rate']:.12g},"
                      f"{metrics['quantized_behavior_loss']:.12g},"
                      f"{change['delta_auc_from_previous']},"
                      f"{change['delta_auc_ci']},"
                      f"{change['delta_ordering_error_from_previous']},"
                      f"{change['delta_ordering_error_ci']}")
            print(f"VERDICT={model},{pair_set},{report['verdict']}")
            if report["reference_check"] is not None:
                print(f"REFERENCE_CHECK=Exp1a,primary_valid,{report['reference_check']}")
    print(f"UNDEFINED_METRICS={json.dumps(undefined_metrics, separators=(',', ':'))}")
    print(f"DEGENERATE_METRICS={json.dumps(degenerate_metrics, separators=(',', ':'))}")
    print(f"OUTPUT_JSON={OUT / 'quant_bridge.json'}")
    print(f"OUTPUT_CSV={OUT / 'er_table.csv'},{OUT / 'cross_tabulation.csv'},{OUT / 'depth_metrics.csv'}")


if __name__ == "__main__":
    run()
