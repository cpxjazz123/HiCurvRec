"""Stepwise test of five relation-level feature groups on next-item-category prediction.

A  text context + controls (degree, popularity, text similarity)
B  + graphlet type
C  + graphlet node role (endpoint role pair, hub side, degree/core differences, bridge)
D  + purchase direction (counts, conditional probabilities, asymmetry)
E  + AF3 curvature
F  + same-community

All models are L2-regularised multinomial logistic regressions fitted on TRAIN and evaluated on the
same TEST events (seen TRAIN edges only).  Controls are present in every model, so a gain is not
"popular items are easy".  The ablation rows remove one group from F.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/context_structure_validation/incremental.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OCC = HERE / "occurrences_enriched.parquet"
TARGETS = HERE / "targets.json"
OUT = HERE / "incremental.json"
MD = HERE / "incremental_tables.md"

PATTERNS = ["none", "K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5", "multi"]
ROLE_PAIRS = [f"{first}-{second}" for first in range(6) for second in range(first, 6)]
CONTEXTS = list(range(20))
KEEP_OUTCOMES = 10
C_REGULARISATION = 1.0


def one_hot(values: pd.Series, categories: list) -> np.ndarray:
    index = {value: position for position, value in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    for row, value in enumerate(values.tolist()):
        position = index.get(value)
        if position is not None:
            matrix[row, position] = 1.0
    return matrix


def fit_and_score(train_x, train_y, test_x, test_y, classes: int) -> dict:
    model = LogisticRegression(
        C=C_REGULARISATION,
        max_iter=4000,
        solver="lbfgs",
        tol=1e-4,
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)
    if probability.shape[1] != classes:
        full = np.zeros((len(probability), classes))
        for column, label in enumerate(model.classes_):
            full[:, int(label)] = probability[:, column]
        probability = full
    losses = -np.log(np.clip(probability[np.arange(len(test_y)), test_y], 1e-12, None))
    return {
        "logloss": float(losses.mean()),
        "accuracy": float(accuracy_score(test_y, probability.argmax(axis=1))),
        "losses": losses,
    }


def build_groups(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    numeric_columns = [
        "sem_cos", "title_jaccard", "log_degree_pred", "log_pop_pred",
        "log_degree_target", "log_pop_target", "history_length",
        "hub_is_pred", "degree_diff", "core_diff", "is_bridge",
        "log_trans_f", "log_trans_b", "p_f", "p_b", "p_asym", "direction_balance",
        "af3", "af3_high", "af3_low", "same_community",
    ]
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train[numeric_columns].to_numpy(dtype=np.float64))
    test_numeric = scaler.transform(test[numeric_columns].to_numpy(dtype=np.float64))
    train_numeric = pd.DataFrame(train_numeric, columns=numeric_columns, index=train.index)
    test_numeric = pd.DataFrame(test_numeric, columns=numeric_columns, index=test.index)

    context_train = one_hot(train["context"], CONTEXTS)
    context_test = one_hot(test["context"], CONTEXTS)
    pattern_train = one_hot(train["pattern"], PATTERNS)
    pattern_test = one_hot(test["pattern"], PATTERNS)
    role_train = one_hot(train["role_pair"], ROLE_PAIRS)
    role_test = one_hot(test["role_pair"], ROLE_PAIRS)

    controls = ["sem_cos", "title_jaccard", "log_degree_pred", "log_pop_pred",
                "log_degree_target", "log_pop_target", "history_length"]
    role_numeric = ["hub_is_pred", "degree_diff", "core_diff", "is_bridge"]
    direction_numeric = ["log_trans_f", "log_trans_b", "p_f", "p_b", "p_asym", "direction_balance"]
    af3_numeric = ["af3", "af3_high", "af3_low"]
    community_numeric = ["same_community"]

    def block(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
        return frame[names].to_numpy(dtype=np.float64)

    groups = {
        "text": (
            np.hstack([context_train, block(train_numeric, controls)]),
            np.hstack([context_test, block(test_numeric, controls)]),
        ),
        "graphlet": (pattern_train, pattern_test),
        "role": (
            np.hstack([role_train, block(train_numeric, role_numeric)]),
            np.hstack([role_test, block(test_numeric, role_numeric)]),
        ),
        "direction": (block(train_numeric, direction_numeric), block(test_numeric, direction_numeric)),
        "af3": (block(train_numeric, af3_numeric), block(test_numeric, af3_numeric)),
        "community": (block(train_numeric, community_numeric), block(test_numeric, community_numeric)),
    }
    return groups


def main() -> int:
    occurrences = pd.read_parquet(OCC)
    targets = json.loads(TARGETS.read_text())
    occurrences = occurrences[occurrences["seen_edge"] == 1].copy()
    occurrences["log_degree_pred"] = np.log1p(occurrences["degree_pred"])
    occurrences["log_pop_pred"] = np.log1p(occurrences["pop_pred"])
    occurrences["log_degree_target"] = np.log1p(occurrences["degree_target"])
    occurrences["log_pop_target"] = np.log1p(occurrences["pop_target"])
    occurrences["log_trans_f"] = np.log1p(occurrences["trans_f"])
    occurrences["log_trans_b"] = np.log1p(occurrences["trans_b"])

    result = {
        "config": {
            "occurrences": str(OCC.relative_to(REPO)),
            "model": "multinomial LogisticRegression",
            "C": C_REGULARISATION,
            "max_iter": 4000,
            "keep_outcomes": KEEP_OUTCOMES,
            "evaluation": "TEST events on edges present in the TRAIN graph; same rows for every model",
        },
        "targets": {},
    }

    for target_text, meta in targets.items():
        target = int(target_text)
        frame = occurrences[occurrences["target"] == target].copy()
        train = frame[frame["split"] == "train"].copy()
        test = frame[frame["split"] == "test"].copy()
        train = train.sort_values(["user", "position"]).reset_index(drop=True)
        test = test.sort_values(["user", "position"]).reset_index(drop=True)

        outcome_counts = train["outcome"].value_counts()
        kept = set(outcome_counts.head(KEEP_OUTCOMES).index.astype(int))
        if len(set(train["outcome"].unique()) - kept) == 0:
            kept = set(outcome_counts.head(KEEP_OUTCOMES - 1).index.astype(int))
        kept = sorted(kept)
        outcome_map = {value: position for position, value in enumerate(kept)}
        other = len(kept)
        train_y = np.array([outcome_map.get(int(value), other) for value in train["outcome"]])
        test_y = np.array([outcome_map.get(int(value), other) for value in test["outcome"]])
        classes = other + 1
        if len(np.unique(train_y)) < classes:
            missing = sorted(set(range(classes)) - set(np.unique(train_y)))
            for value in missing:
                train_y = np.append(train_y, value)
                train = pd.concat([train, train.iloc[[0]]], ignore_index=True)

        groups = build_groups(train, test)
        names = ["A_text", "B_graphlet", "C_role", "D_direction", "E_af3", "F_community"]
        stacks = {
            "A_text": ["text"],
            "B_graphlet": ["text", "graphlet"],
            "C_role": ["text", "graphlet", "role"],
            "D_direction": ["text", "graphlet", "role", "direction"],
            "E_af3": ["text", "graphlet", "role", "direction", "af3"],
            "F_community": ["text", "graphlet", "role", "direction", "af3", "community"],
        }
        metrics = {}
        for name, members in stacks.items():
            train_x = np.hstack([groups[member][0] for member in members])
            test_x = np.hstack([groups[member][1] for member in members])
            metrics[name] = fit_and_score(train_x, train_y, test_x, test_y, classes)
            print(f"{target} {name}: logloss {metrics[name]['logloss']:.4f} acc {metrics[name]['accuracy']:.4f}", flush=True)

        ablations = {}
        for removed in ("graphlet", "role", "direction", "af3", "community"):
            members = [name for name in ("text", "graphlet", "role", "direction", "af3", "community") if name != removed]
            train_x = np.hstack([groups[member][0] for member in members])
            test_x = np.hstack([groups[member][1] for member in members])
            ablations[removed] = fit_and_score(train_x, train_y, test_x, test_y, classes)

        steps = {}
        for previous, following in zip(names, names[1:]):
            difference = metrics[previous]["losses"] - metrics[following]["losses"]
            steps[f"{previous}->{following}"] = {
                "delta": float(difference.mean()),
                "se": float(difference.std(ddof=1) / np.sqrt(len(difference))),
                "positive_share": float((difference > 0).mean()),
            }
        ablation_gains = {}
        for removed, value in ablations.items():
            difference = value["losses"] - metrics["F_community"]["losses"]
            ablation_gains[removed] = {
                "delta_when_removed": float(difference.mean()),
                "se": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            }

        result["targets"][target_text] = {
            "title": meta["title"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "classes": classes,
            "kept_outcomes": kept,
            "metrics": {
                name: {
                    "logloss": value["logloss"],
                    "accuracy": value["accuracy"],
                } for name, value in metrics.items()
            },
            "steps": steps,
            "ablations": ablation_gains,
        }

    # aggregate weighted by test rows
    totals = {}
    for name in ("A_text", "B_graphlet", "C_role", "D_direction", "E_af3", "F_community"):
        weights = [result["targets"][key]["n_test"] for key in result["targets"]]
        losses = [result["targets"][key]["metrics"][name]["logloss"] for key in result["targets"]]
        totals[name] = {
            "weighted_logloss": float(np.average(losses, weights=weights)),
            "targets": {key: result["targets"][key]["metrics"][name]["logloss"] for key in result["targets"]},
        }
    result["aggregate"] = totals

    OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Incremental relation-feature experiment\n"]
    lines.append("All models include text context + degree/popularity/text-similarity controls; evaluated on the same TEST seen-edge events.\n")
    lines.append("## Stepwise TEST log-loss\n\n")
    lines.append("| target | A text | B +graphlet | C +role | D +direction | E +AF3 | F +community |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for key, value in result["targets"].items():
        cells = " | ".join(f"{value['metrics'][name]['logloss']:.4f}" for name in ("A_text", "B_graphlet", "C_role", "D_direction", "E_af3", "F_community"))
        lines.append(f"| {key} | {cells} |\n")
    lines.append("\n## Stepwise gains (positive = later model better)\n\n")
    lines.append("| target | A->B | B->C | C->D | D->E | E->F |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for key, value in result["targets"].items():
        cells = " | ".join(
            f"{value['steps'][pair]['delta']:+.4f} (±{value['steps'][pair]['se']:.4f})"
            for pair in ("A_text->B_graphlet", "B_graphlet->C_role", "C_role->D_direction", "D_direction->E_af3", "E_af3->F_community")
        )
        lines.append(f"| {key} | {cells} |\n")
    lines.append("\n## Ablation of F (positive = removing the group hurts)\n\n")
    lines.append("| target | -graphlet | -role | -direction | -AF3 | -community |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for key, value in result["targets"].items():
        cells = " | ".join(
            f"{value['ablations'][group]['delta_when_removed']:+.4f} (±{value['ablations'][group]['se']:.4f})"
            for group in ("graphlet", "role", "direction", "af3", "community")
        )
        lines.append(f"| {key} | {cells} |\n")
    MD.write_text("".join(lines) + "\n")
    print(f"wrote {OUT} and {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
