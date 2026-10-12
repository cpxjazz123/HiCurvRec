"""Refresh completed primary metrics and chunked pairwise/max LCP ordering."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C
from distances import load_pairs
from pair_prefix import longest_common_prefix
from quant_bridge import anchor_bootstrap_ci

MODELS = ("Exp1a", "TIGER")
LEVELS = ("L1", "L1_L2", "L1_L2_L3")
PRIMARY = "primary_valid"
SECONDARY_STATUS = {
    "status": "blocked",
    "reason": "Full evaluator terminated by SIGKILL (exit 137) before secondary rows were written.",
    "pair_count": 323633,
    "negatives_per_anchor": 10,
    "peak_rss_kib": 128463488,
    "peak_rss_bytes": 131546611712,
}


def _ci(anchors, values, seed):
    return anchor_bootstrap_ci(anchors, np.asarray(values, dtype=float), C.BOOTSTRAP_REPLICATES, seed)


def _lcp_metrics(levels, pairs):
    """Accumulate pairwise and hardest-negative LCP outcomes by anchor chunks."""
    anchors, positives, negatives = (pairs[key] for key in ("anchors", "positives", "negatives"))
    n, k = len(anchors), negatives.shape[1]
    pos_lcp = longest_common_prefix(levels[anchors, :3], levels[positives, :3])
    pair_pos = np.empty(n, dtype=np.float32)
    pair_neg = np.empty(n, dtype=np.float32)
    pair_tie = np.empty(n, dtype=np.float32)
    max_pos = np.empty(n, dtype=bool)
    max_neg = np.empty(n, dtype=bool)
    max_tie = np.empty(n, dtype=bool)
    for start in range(0, n, 32768):
        stop = min(start + 32768, n)
        anchor_codes = np.broadcast_to(
            levels[anchors[start:stop], None, :3], (stop - start, k, 3)
        ).reshape(-1, 3)
        negative_codes = levels[negatives[start:stop]].reshape(-1, 3)
        neg_lcp = longest_common_prefix(anchor_codes, negative_codes).reshape(stop - start, k)
        pos = pos_lcp[start:stop, None]
        pair_pos[start:stop] = (pos > neg_lcp).mean(axis=1)
        pair_neg[start:stop] = (pos < neg_lcp).mean(axis=1)
        pair_tie[start:stop] = (pos == neg_lcp).mean(axis=1)
        hardest = neg_lcp.max(axis=1)
        pos = pos_lcp[start:stop]
        max_pos[start:stop] = pos > hardest
        max_neg[start:stop] = pos < hardest
        max_tie[start:stop] = pos == hardest

    pair_auc_by_anchor = pair_pos + 0.5 * pair_tie
    decisive = pair_pos + pair_neg
    pair_acc_by_anchor = np.divide(
        pair_pos, decisive, out=np.full(n, np.nan), where=decisive > 0
    )
    valid_acc = np.isfinite(pair_acc_by_anchor)
    pairwise = {
        "positive_rate": float(pair_pos.mean()),
        "negative_rate": float(pair_neg.mean()),
        "tie_rate": float(pair_tie.mean()),
        "auc": float(pair_auc_by_anchor.mean()),
        "auc_ci": _ci(anchors, pair_auc_by_anchor, C.SEED + 600),
        "ranking_accuracy": float(pair_pos.sum() / decisive.sum()) if decisive.sum() else None,
        "ranking_accuracy_ci": (_ci(anchors[valid_acc], pair_acc_by_anchor[valid_acc], C.SEED + 601)
                                 if valid_acc.any() else {"estimate": None, "low": None, "high": None}),
        "positive_rate_ci": _ci(anchors, pair_pos, C.SEED + 602),
        "negative_rate_ci": _ci(anchors, pair_neg, C.SEED + 603),
        "tie_rate_ci": _ci(anchors, pair_tie, C.SEED + 604),
        "n_pairwise_comparisons": int(n * k),
    }
    max_both = max_tie & (pos_lcp > 0)
    max_neither = max_tie & (pos_lcp == 0)
    max_auc_by_anchor = max_pos.astype(float) + 0.5 * max_tie
    max_decisive = max_pos | max_neg
    max_based = {
        "positive_rate": float(max_pos.mean()),
        "negative_rate": float(max_neg.mean()),
        "both_share_tie_rate": float(max_both.mean()),
        "neither_shares_tie_rate": float(max_neither.mean()),
        "tie_rate": float(max_tie.mean()),
        "auc": float(max_auc_by_anchor.mean()),
        "auc_ci": _ci(anchors, max_auc_by_anchor, C.SEED + 610),
        "preference_accuracy": (float(max_pos.sum() / max_decisive.sum())
                                if max_decisive.any() else None),
        "positive_rate_ci": _ci(anchors, max_pos, C.SEED + 611),
        "negative_rate_ci": _ci(anchors, max_neg, C.SEED + 612),
        "tie_rate_ci": _ci(anchors, max_tie, C.SEED + 613),
        "both_share_tie_rate_ci": _ci(anchors, max_both, C.SEED + 614),
        "neither_shares_tie_rate_ci": _ci(anchors, max_neither, C.SEED + 615),
    }
    return {"pairwise": pairwise, "max_based": max_based}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run():
    out = C.OUT_DIR / "quant_bridge_v2"
    result_path = out / "quant_bridge_v2.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Completed primary metrics are missing: {result_path}")
    result = json.loads(result_path.read_text())
    if PRIMARY not in result.get("results", {}):
        raise ValueError("Completed primary_valid metric results are unavailable")
    pairs = load_pairs(C.PRIMARY_PAIRS)

    for model in MODELS:
        sid_data = json.loads(Path(C.ARMS[model]["sid_json"]).read_text())
        levels = np.asarray([sid_data[key] for key in sorted(sid_data, key=int)], dtype=np.int64)[:, :3]
        lcp = _lcp_metrics(levels, pairs)
        model_result = result["results"][PRIMARY][model]
        model_result["depth_independent_metrics"] = {
            "cumulative_reconstruction_auc": model_result["levels"]["L1"]["cumulative_reconstruction"]["auc"],
            "lcp_first_three_tokens_pairwise": lcp["pairwise"],
            "lcp_first_three_tokens_max_based": lcp["max_based"],
        }
        for level in LEVELS:
            metric = model_result["levels"][level]
            metric["lcp_first_three_tokens_pairwise"] = lcp["pairwise"]
            metric["lcp_first_three_tokens_max_based"] = lcp["max_based"]

    result["results"] = {PRIMARY: result["results"][PRIMARY]}
    result["blocked_pair_sets"] = {"secondary_train": SECONDARY_STATUS}
    result["secondary_status"] = SECONDARY_STATUS
    result["inputs"]["pair_sets"] = {
        PRIMARY: {key: int(len(pairs[key])) for key in ("anchors", "positives", "negatives")}
    }
    result["depth_independent_columns_note"] = (
        "Cumulative-reconstruction AUC and both first-three-token LCP AUCs are depth-independent. "
        "The per-depth quantized view is the four-way prefix classification and its inversion/tie rates."
    )
    result["supersedes"] = "results/crowding_diagnostic/quant_bridge/ (preserved untouched as superseded history)"
    result_path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")

    rows, er_rows = [], []
    for model in MODELS:
        model_result = result["results"][PRIMARY][model]
        pairwise = model_result["depth_independent_metrics"]["lcp_first_three_tokens_pairwise"]
        max_based = model_result["depth_independent_metrics"]["lcp_first_three_tokens_max_based"]
        for level in LEVELS:
            metric = model_result["levels"][level]
            table = metric["quantized_prefix"]
            transition = metric["conditional_transitions"]
            row = {
                "model": model, "pair_set": PRIMARY, "level": level,
                "continuous_ranking_accuracy": metric["continuous_residual"]["ranking_accuracy"],
                "continuous_residual_auc": metric["continuous_residual"]["auc"],
                "cumulative_reconstruction_auc_depth_independent": metric["cumulative_reconstruction"]["auc"],
                "prefix_positive_advantage_rate": table["rates"]["positive_prefix_advantage"],
                "prefix_negative_advantage_rate": table["rates"]["negative_prefix_advantage"],
                "prefix_both_share_tie_rate": table["rates"]["both_share"],
                "prefix_neither_shares_tie_rate": table["rates"]["neither_shares"],
                "prefix_tie_rate": table["tie_rate"],
                "prefix_preference_accuracy": table["preference_accuracy"],
                "lcp_pairwise_positive_rate_depth_independent": pairwise["positive_rate"],
                "lcp_pairwise_negative_rate_depth_independent": pairwise["negative_rate"],
                "lcp_pairwise_tie_rate_depth_independent": pairwise["tie_rate"],
                "lcp_pairwise_auc_depth_independent": pairwise["auc"],
                "lcp_max_positive_rate_depth_independent": max_based["positive_rate"],
                "lcp_max_negative_rate_depth_independent": max_based["negative_rate"],
                "lcp_max_tie_rate_depth_independent": max_based["tie_rate"],
                "lcp_max_auc_depth_independent": max_based["auc"],
                "continuous_correct_to_inversion_rate": transition["continuous_correct_to_quantized_inversion_rate"],
                "continuous_correct_to_tie_rate": transition["continuous_correct_to_quantized_tie_rate"],
                "continuous_wrong_to_quantized_correct_rate": transition["continuous_wrong_to_quantized_correct_rate"],
                "positive_prefix_sharing_rate": metric["positive_prefix_sharing_rate"],
                "negative_prefix_sharing_rate": metric["negative_prefix_sharing_rate"],
                "ER": metric["enrichment_ratio"],
                "continuous_ranking_error_bucket": metric["failure_buckets"]["continuous_ranking_error"],
                "quantized_order_inversion_bucket": metric["failure_buckets"]["quantized_order_inversion"],
                "quantized_tie_inconclusive_bucket": metric["failure_buckets"]["quantized_tie_inconclusive"],
                "item_code_collision_bucket": metric["failure_buckets"]["item_code_collision"],
            }
            rows.append(row)
            sharing_pos_ci = metric["positive_prefix_sharing_ci"]
            sharing_neg_ci = metric["negative_prefix_sharing_ci"]
            er_ci = metric["enrichment_ratio_ci"]
            er_rows.append({
                "model": model, "pair_set": PRIMARY, "level": level,
                "p_share_pos": metric["positive_prefix_sharing_rate"],
                "p_share_pos_ci_low": sharing_pos_ci["low"],
                "p_share_pos_ci_high": sharing_pos_ci["high"],
                "p_share_neg": metric["negative_prefix_sharing_rate"],
                "p_share_neg_ci_low": sharing_neg_ci["low"],
                "p_share_neg_ci_high": sharing_neg_ci["high"],
                "ER": metric["enrichment_ratio"],
                "ER_ci_low": er_ci["low"], "ER_ci_high": er_ci["high"],
            })

    for filename, data in (("corrected_tables.csv", rows), ("er_table.csv", er_rows)):
        fields = list(dict.fromkeys(key for item in data for key in item))
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    print("SUPERSEDED_HISTORY=results/crowding_diagnostic/quant_bridge/ (left untouched)")
    print("PAIR_SET_STATUS=primary_valid,complete;secondary_train,blocked;SIGKILL exit 137;peak_rss_kib=128463488;peak_rss_bytes=131546611712")
    print("DEPTH_INDEPENDENT=cumulative_reconstruction_auc,lcp_pairwise_auc,lcp_max_auc")
    print("PRIMARY_TABLE=model,pair_set,level,pos_adv,neg_adv,both_share,neither_shares,tie,cont_auc,recon_auc,LCP_pair_pos,LCP_pair_neg,LCP_pair_tie,LCP_pair_auc,LCP_max_pos,LCP_max_neg,LCP_max_tie,LCP_max_auc,cont_correct_to_inversion,cont_correct_to_tie,cont_wrong_to_quant_correct,share_pos,share_neg,ER,continuous_error,inversion,tie_inconclusive,collision")
    keys = (
        "prefix_positive_advantage_rate", "prefix_negative_advantage_rate",
        "prefix_both_share_tie_rate", "prefix_neither_shares_tie_rate", "prefix_tie_rate",
        "continuous_residual_auc", "cumulative_reconstruction_auc_depth_independent",
        "lcp_pairwise_positive_rate_depth_independent", "lcp_pairwise_negative_rate_depth_independent",
        "lcp_pairwise_tie_rate_depth_independent", "lcp_pairwise_auc_depth_independent",
        "lcp_max_positive_rate_depth_independent", "lcp_max_negative_rate_depth_independent",
        "lcp_max_tie_rate_depth_independent", "lcp_max_auc_depth_independent",
        "continuous_correct_to_inversion_rate", "continuous_correct_to_tie_rate",
        "continuous_wrong_to_quantized_correct_rate", "positive_prefix_sharing_rate",
        "negative_prefix_sharing_rate", "ER", "continuous_ranking_error_bucket",
        "quantized_order_inversion_bucket", "quantized_tie_inconclusive_bucket",
        "item_code_collision_bucket",
    )
    for row in rows:
        values = [row[key] for key in keys]
        print(f"{row['model']},{row['pair_set']},{row['level']}," +
              ",".join("undefined" if value is None else f"{value:.12g}" for value in values))
    print("ER_TABLE=model,pair_set,level,p_share_pos,p_share_neg,ER,ER_CI_low,ER_CI_high")
    for row in er_rows:
        values = (row["p_share_pos"], row["p_share_neg"], row["ER"], row["ER_ci_low"], row["ER_ci_high"])
        print(f"{row['model']},{row['pair_set']},{row['level']}," +
              ",".join("undefined" if value is None else f"{value:.12g}" for value in values))
    print(f"OUTPUT_JSON={result_path}")
    print(f"OUTPUT_CSV={out / 'corrected_tables.csv'},{out / 'er_table.csv'}")


if __name__ == "__main__":
    run()
