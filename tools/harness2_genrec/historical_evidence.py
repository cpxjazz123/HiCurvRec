"""The mandatory historical evidence baseline of issue #109, section 4.

Before any proxy score is allowed to rank a portfolio, the historical record has to say what
those proxies are worth on the models that already exist. This module does that, offline:
it reads four archived CSVs, refits the historical predictors under a fixed protocol, and
reports how much the downstream labels can be predicted from the frozen ER alone, from the
history-conditioned level-1 gain, from the conditional level-2 discrimination, and from
their combinations.

The protocol is fixed, because the point of the baseline is to be comparable across runs and
across campaigns:

* the feature sets are compared on one MATCHED COMPLETE-CASE model set, so every number is
  computed on the same models and a difference is a difference of method, not of sample;
* the features are normalised with TRAIN-FOLD-ONLY statistics -- the test fold never touches
  the scaler;
* the ridge penalty is chosen by an inner, family-held-out search inside the outer training
  fold;
* there are five outer Leave-One-Family-Out folds, one per historical family, so no model's
  prediction comes from a model of its own family;
* the errors are PAIRED across feature sets, and the confidence interval is a bootstrap over
  model FAMILIES, because models inside a family are not independent;
* the counterexamples -- the models a proxy set predicts worse than the ER alone -- are
  reported, because a mean improvement that hides them is not evidence.

Two rules from the issue shape this code. First, the historical G_H1 is reconstructed from
the archived corrected probe scores for SEED 42 ONLY; the published means average seeds
42, 43 and 44 and are not interchangeable, so they are never used. Second, this is a
retrospective diagnostic on an already-inspected cohort, not a prospective confirmatory test:
the report says so, and the module refuses to certify a proxy for screening.

Run it with the project's environment (it needs numpy, pandas and scikit-learn):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \\
        tools/harness2_genrec/historical_evidence.py \\
        --out results/harness3_runs/historical_evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen thresholds of issue #109.
ER_GATE = 16.38953942232631
RECALL_GATE = 0.06406796775709883
NDCG_GATE = 0.03530651701940677

#: The historical archive's location, relative to the repository root.
HISTORY = Path("results/stage3_T5Train/monotonicity")

#: The ridge penalties the inner family-held-out search chooses from.
ALPHAS: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

#: The six feature sets the issue asks to compare.
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "ER": ("log_er_l1",),
    "ER_plus_G_H1": ("log_er_l1", "G_H1"),
    "ER_plus_D2": ("log_er_l1", "D2"),
    "ER_plus_G_H1_plus_D2": ("log_er_l1", "G_H1", "D2"),
    "G_H1_only": ("G_H1",),
    "D2_only": ("D2",),
}

#: The baselines the ER-only predictor must reproduce, from the historical audit (#47).
ER_ONLY_RECALL_MAE = 0.003752853292309109
ER_ONLY_MAE_TOLERANCE = 2e-5

#: The seed the historical probe is reconstructed from, and the history length it uses.
PROBE_SEED = 42
PROBE_SEQ_LEN = 20

#: The bootstrap replicates of the family-cluster confidence interval.
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42


class EvidenceError(RuntimeError):
    """A provenance failure: the archives disagree, so nothing is computed."""


# --------------------------------------------------------------------------- the archives
def _read(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise EvidenceError(f"the historical archive is missing: {path}")
    return pd.read_csv(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def historical_table(root: Path, *, expected_models: Optional[int] = 38) -> pd.DataFrame:
    """The 38 historical models with their features, labels and provenance.

    Every cross-archive check the issue names is enforced here: the model sets must agree,
    the labels must agree, the families must agree, and the corrected probe must be present
    for every model at seed 42.
    """
    inventory = _read(root, HISTORY / "model_inventory.csv")
    features = _read(root, HISTORY / "joint_prefix_predictors/stage2_valid_features.csv")
    per_model = _read(root, HISTORY / "history_hierarchy_predictors_fix48/per_model_features.csv")
    probe = _read(root, HISTORY / "history_hierarchy_predictors_fix48/probe_scores.csv")

    for name, frame in (("inventory", inventory), ("#47 features", features), ("#49 features", per_model)):
        if frame["model"].duplicated().any():
            raise EvidenceError(f"duplicate model id in the {name} archive")

    models = set(inventory["model"])
    _require(
        models == set(features["model"]),
        "the inventory and the #47 feature archive cover different models",
    )
    _require(
        models == set(per_model["model"]),
        "the inventory and the #49 feature archive cover different models",
    )
    if expected_models is not None:
        _require(
            len(inventory) == expected_models,
            f"the historical inventory has {len(inventory)} models, not {expected_models}",
        )

    table = inventory[
        ["model", "family", "recall@10", "ndcg@10", "n_eval", "sid_sha256"]
    ].rename(columns={"recall@10": "recall", "ndcg@10": "ndcg"})
    table = table.merge(
        features[["model", "family", "log_er_l1", "stage3_recall@10", "stage3_ndcg@10"]],
        on="model",
        how="inner",
        suffixes=("", "_features"),
        validate="one_to_one",
    )
    _require(
        (table["family"] == table["family_features"]).all(),
        "the inventory and the #47 archive disagree about a model's family",
    )
    _require(
        np.allclose(table["recall"], table["stage3_recall@10"], atol=1e-10),
        "the inventory and the #47 archive disagree about a model's Recall@10",
    )
    _require(
        np.allclose(table["ndcg"], table["stage3_ndcg@10"], atol=1e-10),
        "the inventory and the #47 archive disagree about a model's NDCG@10",
    )
    table = table.merge(
        per_model[["model", "chrd_D2", "log_er_l1", "er_l1_recheck"]],
        on="model",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_fix49"),
    )
    # the two archives compute the frozen ER independently; they must agree
    _require(
        np.allclose(table["log_er_l1"], table["log_er_l1_fix49"], atol=1e-8),
        "the #47 and #49 archives disagree about a model's frozen ER",
    )
    table = table.rename(columns={"chrd_D2": "D2"})
    table["group"] = table["family"]
    gains = g_h1_seed42(probe)
    _require(
        set(gains.index) == set(table["model"]),
        "the seed-42 probe rows do not cover exactly the historical models",
    )
    table["G_H1"] = table["model"].map(gains).astype("float64")
    return table


def g_h1_seed42(probe: pd.DataFrame) -> pd.Series:
    """The historical G_H1 rebuilt from the archived seed-42 probe rows only.

    The published G_H1 means average seeds 42, 43 and 44. Those means are not
    interchangeable with a single-seed probe, so the rows of the other seeds are dropped
    before anything is computed -- not averaged away later.
    """
    rows = probe[(probe["seed"] == PROBE_SEED) & (probe["seq_len"] == PROBE_SEQ_LEN)]
    if rows.empty:
        raise EvidenceError(
            f"the corrected probe archive has no seed {PROBE_SEED} rows at seq_len {PROBE_SEQ_LEN}"
        )
    if rows["model"].duplicated().any():
        raise EvidenceError("the seed-42 probe rows contain a duplicate model")
    prior = rows["prior_ce_l1"].to_numpy(dtype="float64")
    probe_ce = rows["probe_ce_l1"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(prior > 0, (prior - probe_ce) / prior, np.nan)
    # The archive also carries the relative gain; it must agree with the definition.
    if "rel_gain_l1" in rows:
        recorded = rows["rel_gain_l1"].to_numpy(dtype="float64")
        finite = np.isfinite(gain) & np.isfinite(recorded)
        if finite.any() and not np.allclose(gain[finite], recorded[finite], atol=1e-6):
            raise EvidenceError(
                "the archive's relative gain does not match (prior CE - probe CE) / prior CE"
            )
    return pd.Series(gain, index=rows["model"].to_numpy(), name="G_H1")


# ------------------------------------------------------------------------------- the model
def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], target: str, alpha: float
) -> np.ndarray:
    """Fit on the training fold and predict the test fold, normalising on the train fold only."""
    scaler = StandardScaler().fit(train[list(features)])
    model = Ridge(alpha=alpha).fit(scaler.transform(train[list(features)]), train[target])
    return model.predict(scaler.transform(test[list(features)]))


def _alpha_from_training(
    train: pd.DataFrame, features: Sequence[str], target: str
) -> float:
    """The ridge penalty chosen by an inner family-held-out search inside the training fold."""
    groups = train["group"].to_numpy()
    if len(set(groups)) < 3:
        raise EvidenceError("the inner search needs at least three families in the training fold")
    scores: List[float] = []
    for alpha in ALPHAS:
        errors: List[float] = []
        for inner_train, inner_test in LeaveOneGroupOut().split(train, groups=groups):
            predicted = _fit_predict(
                train.iloc[inner_train], train.iloc[inner_test], features, target, alpha
            )
            errors.extend(np.abs(predicted - train.iloc[inner_test][target].to_numpy()))
        scores.append(float(np.mean(errors)))
    return float(ALPHAS[int(np.argmin(scores))])


def lofo_predictions(
    data: pd.DataFrame, features: Sequence[str], target: str
) -> pd.DataFrame:
    """The out-of-fold predictions, one outer fold per family."""
    if len(data) < 10 or data["group"].nunique() < 3:
        raise EvidenceError("too few models or families for a family-held-out fit")
    rows: List[Dict[str, Any]] = []
    for train_index, test_index in LeaveOneGroupOut().split(data, groups=data["group"]):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        alpha = _alpha_from_training(train, features, target)
        predicted = _fit_predict(train, test, features, target, alpha)
        train_support = {
            feature: (float(train[feature].min()), float(train[feature].max()))
            for feature in features
        }
        for position, (model, family, observed, prediction) in enumerate(
            zip(test["model"], test["group"], test[target], predicted)
        ):
            outside = [
                feature
                for feature in features
                if not (
                    train_support[feature][0]
                    <= float(test.iloc[position][feature])
                    <= train_support[feature][1]
                )
            ]
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "target": target,
                    "observed": float(observed),
                    "predicted": float(prediction),
                    "abs_error": float(abs(observed - prediction)),
                    "alpha": alpha,
                    "held_out_family": str(test["group"].iloc[0]),
                    "outside_training_support": outside,
                }
            )
    return pd.DataFrame(rows)


def family_cluster_ci(
    differences: pd.Series, families: pd.Series, *, replicates: int = BOOTSTRAP_REPLICATES
) -> Dict[str, Any]:
    """The bootstrap confidence interval over model FAMILIES.

    Models inside a family share a mechanism and a training protocol, so they are not
    independent observations; the resampling unit is the family.
    """
    frame = pd.DataFrame({"difference": differences.to_numpy(), "family": families.to_numpy()})
    grouped = frame.groupby("family")["difference"].mean()
    values = grouped.to_numpy(dtype="float64")
    if values.size == 0:
        return {"low": None, "high": None, "families": 0, "replicates": replicates}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "families": int(values.size),
        "replicates": int(replicates),
        "point_estimate": float(values.mean()),
    }


def _rank_correlation(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    """The Spearman rank correlation, or None when it is not defined."""
    a = np.asarray(first, dtype="float64")
    b = np.asarray(second, dtype="float64")
    if a.size < 3 or b.size < 3:
        return None
    rank_a = np.argsort(np.argsort(a)).astype("float64")
    rank_b = np.argsort(np.argsort(b)).astype("float64")
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = np.sqrt((rank_a**2).sum() * (rank_b**2).sum())
    if denominator == 0:
        return None
    return float((rank_a * rank_b).sum() / denominator)


# ------------------------------------------------------------------------------- the study
def analyse(root: Path = REPO_ROOT, *, expected_models: Optional[int] = 38) -> Dict[str, Any]:
    """The whole baseline: the matched model set, the six feature sets, the two targets."""
    table = historical_table(root, expected_models=expected_models)
    complete = table.dropna(subset=["log_er_l1", "G_H1", "D2", "recall", "ndcg"]).copy()
    dropped = sorted(set(table["model"]) - set(complete["model"]))

    results: List[Dict[str, Any]] = []
    predictions: Dict[str, pd.DataFrame] = {}
    for target, gate in (("recall", RECALL_GATE), ("ndcg", NDCG_GATE)):
        baseline = lofo_predictions(complete, FEATURE_SETS["ER"], target)
        predictions[f"ER::{target}"] = baseline
        for name, features in FEATURE_SETS.items():
            frame = lofo_predictions(complete, features, target)
            predictions[f"{name}::{target}"] = frame
            merged = frame.merge(
                baseline[["model", "abs_error"]],
                on="model",
                suffixes=("", "_er"),
                validate="one_to_one",
            )
            difference = merged["abs_error_er"] - merged["abs_error"]
            counterexamples = merged[merged["abs_error"] > merged["abs_error_er"]]
            results.append(
                {
                    "target": target,
                    "feature_set": name,
                    "features": list(features),
                    "models": int(len(frame)),
                    "families": int(complete["group"].nunique()),
                    "mae": float(frame["abs_error"].mean()),
                    "delta_mae_vs_ER_same_rows": float(difference.mean()),
                    "family_cluster_ci_of_the_delta": family_cluster_ci(
                        difference, merged["family"]
                    ),
                    "counterexamples": int(len(counterexamples)),
                    "counterexample_models": sorted(counterexamples["model"].tolist()),
                    "models_outside_training_support": sorted(
                        frame.loc[
                            frame["outside_training_support"].apply(bool), "model"
                        ].tolist()
                    ),
                    "status": "exploratory_not_certified",
                }
            )

    recall_er = next(
        row for row in results if row["target"] == "recall" and row["feature_set"] == "ER"
    )
    reproducible = abs(recall_er["mae"] - ER_ONLY_RECALL_MAE) <= ER_ONLY_MAE_TOLERANCE

    calibration = {
        "spearman_ER_vs_recall": _rank_correlation(
            complete["log_er_l1"], complete["recall"]
        ),
        "spearman_G_H1_vs_recall": _rank_correlation(complete["G_H1"], complete["recall"]),
        "spearman_D2_vs_recall": _rank_correlation(complete["D2"], complete["recall"]),
        "spearman_ER_vs_ndcg": _rank_correlation(complete["log_er_l1"], complete["ndcg"]),
        "spearman_G_H1_vs_ndcg": _rank_correlation(complete["G_H1"], complete["ndcg"]),
        "spearman_D2_vs_ndcg": _rank_correlation(complete["D2"], complete["ndcg"]),
        "caveat": (
            "these are retrospective correlations on an already-inspected cohort of "
            "historical models; they are associations, not causal effects, and they are not "
            "a prospective confirmatory test"
        ),
    }

    return {
        "ok": True,
        "historical_models": int(len(table)),
        "historical_families": int(table["family"].nunique()),
        "complete_case_models": int(len(complete)),
        "dropped_models_without_a_complete_feature_vector": dropped,
        "g_h1_source": {
            "archive": str(HISTORY / "history_hierarchy_predictors_fix48/probe_scores.csv"),
            "seed": PROBE_SEED,
            "seq_len": PROBE_SEQ_LEN,
            "three_seed_means_used": False,
        },
        "er_only_reproduces_the_historical_audit": bool(reproducible),
        "er_only_recall_mae": recall_er["mae"],
        "er_only_recall_mae_reference": ER_ONLY_RECALL_MAE,
        "results": results,
        "calibration": calibration,
        "out_of_support": {
            "feature_ranges": {
                feature: [float(complete[feature].min()), float(complete[feature].max())]
                for feature in ("log_er_l1", "G_H1", "D2")
            },
            "note": (
                "the historical ER range is the support a new arm's ER has to fall inside "
                "for the historical relationship to be applicable at all"
            ),
        },
        "protocol_comparability": _protocol_comparability(complete),
        "abstention": {
            "certified_for_screening": False,
            "reason": (
                "five historical families and an already-inspected cohort cannot certify a "
                "proxy for rejecting arms; the frozen ER stays the only pre-Stage-3 gate"
            ),
            "abstain_if": (
                "the evidence is missing, a model family is unseen, or a feature value falls "
                "outside the historical training support"
            ),
        },
        "limitations": [
            "#48 is invalidated; only the corrected #49 archive is read",
            "#50's prose-versus-CSV conflict is unresolved, so none of #50's evidence is used",
            "the historical G_H1 is the seed-42 rows only and is never mixed with the "
            "published three-seed means",
            "the historical cohort is retrospective and already inspected; it is not a "
            "pristine prospective confirmatory test",
            "no predictor here is certified to reject an arm or to declare success",
        ],
    }


def _protocol_comparability(complete: pd.DataFrame) -> Dict[str, Any]:
    """What the archived features can and cannot be compared with.

    The archived per-model ER and D2 were computed on the archived models' own evaluation
    protocol. Their SID exports are not on this machine, so they cannot be recomputed on the
    frozen pair product, and the two are demonstrably on different scales: the archived
    control's ER is 9.907 while the frozen control's is 16.3345. A new arm's frozen ER and
    D2 therefore cannot be read against the archived values, and the archived D2 in
    particular must not be used to rank a new arm.
    """
    archived_er = np.exp(complete["log_er_l1"].to_numpy(dtype="float64"))
    control_rows = complete[complete["model"].astype(str).str.contains("curvature_RQ-VAE")]
    archived_control_er = (
        float(np.exp(control_rows["log_er_l1"].iloc[0])) if len(control_rows) else None
    )
    return {
        "archived_er_range": [float(archived_er.min()), float(archived_er.max())],
        "archived_control_er": archived_control_er,
        "frozen_control_er": 16.334505666275888,
        "the_archived_features_are_the_frozen_pair_product": False,
        "note": (
            "the archived per-model ER and D2 come from the archived models' own evaluation "
            "protocol, and their SID exports are not on this machine, so they cannot be "
            "recomputed on the frozen pair product. The historical relationships below are "
            "internally consistent, but a new arm's frozen ER and D2 are not on the same "
            "scale as the archived ones and must not be ranked against them"
        ),
    }


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Historical evidence baseline (issue #109, section 4)",
        "",
        "Offline, read-only. No cold run, no frozen evaluation, no probe rerun.",
        "",
        f"Historical models: {report['historical_models']} in {report['historical_families']} "
        f"families; complete-case models: {report['complete_case_models']}.",
        f"The ER-only Recall@10 out-of-fold MAE is {report['er_only_recall_mae']:.6f} "
        f"(historical reference {report['er_only_recall_mae_reference']:.6f}; reproduces: "
        f"{report['er_only_reproduces_the_historical_audit']}).",
        "",
        "| Target | Feature set | n | MAE | Delta vs ER (same rows) | 95% CI over families | Counterexamples |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in report["results"]:
        interval = row["family_cluster_ci_of_the_delta"]
        ci = (
            f"[{interval['low']:+.6f}, {interval['high']:+.6f}]"
            if interval["low"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['target']} | {row['feature_set']} | {row['models']} | "
            f"{row['mae']:.6f} | {row['delta_mae_vs_ER_same_rows']:+.6f} | {ci} | "
            f"{row['counterexamples']} |"
        )
    lines += [
        "",
        "**Not certified for screening.** The frozen ER stays the only pre-Stage-3 gate.",
        "",
        "## Protocol comparability",
        "",
        f"- the archived per-model ER range: {report['protocol_comparability']['archived_er_range']}",
        f"- the archived control's ER: {report['protocol_comparability']['archived_control_er']}; "
        f"the frozen control's: {report['protocol_comparability']['frozen_control_er']}",
        f"- {report['protocol_comparability']['note']}",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.harness2_genrec.historical_evidence")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--out", default=str(REPO_ROOT / "results/harness3_runs/historical_evidence"))
    args = parser.parse_args(argv)
    report = analyse(Path(args.root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "historical_evidence.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out / "historical_evidence.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
