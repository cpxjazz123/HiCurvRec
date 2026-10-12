"""Plot crowding-screen metrics and export the tidy data used by each figure."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/crowding_screening"
FIGURES = RESULTS / "figures"
STAGE2 = ROOT / "results/stage2_RQ-VAE"
STRENGTHS = (1, 2, 3)
MECHANISMS = ("posattract", "couser", "simweight", "density", "softassign",
              "prefixshare", "random", "scale", "parammatch")
BASELINE_ARMS = ("TIGER", "resample", "resample_l1only", "resample_l2only",
                 "resample_l3only", "resample_proj_sym", "resample_proj_dir",
                 "resample_fn_mask", "resample_s43", "resample_s44")
STAGE3_RUNS = {
    "resample": "tiger_sample_resample",
    "resample_l1only": "tiger_sample_resample_l1only",
    "resample_proj_sym": "tiger_sample_resample_proj_sym",
    "resample_proj_dir": "tiger_sample_resample_proj_dir",
    "crowd_couser_w3": "tiger_sample_crowd_couser_w3",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, ""):
        return None
    return float(value)


def dump_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def level_array(arm: str) -> np.ndarray:
    if arm == "TIGER":
        candidates = [STAGE2 / "TIGER_RQ-VAE" / "sids_for_hgrec.npy",
                     STAGE2 / "TIGER_RQ-VAE" / "item_sids.npy"]
    elif arm in ("resample", "resample_l1only", "resample_proj_sym", "resample_proj_dir",
                 "resample_fn_mask", "resample_s43", "resample_s44"):
        candidates = [STAGE2 / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / arm / "sids_for_hgrec.npy"]
    else:
        candidates = [STAGE2 / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / arm / "sids_for_hgrec.npy"]
    for path in candidates:
        if path.is_file():
            arr = np.load(path, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return np.asarray(arr[:, :3], dtype=np.int64)
    return None


def struct_stats(levels: np.ndarray) -> dict:
    distinct = [int(np.unique(levels[:, :depth], axis=0).shape[0]) for depth in (1, 2, 3)]
    first_counts = np.bincount(levels[:, 0], minlength=max(256, int(levels[:, 0].max()) + 1)).astype(np.float64)
    probabilities = first_counts / first_counts.sum()
    entropy = float(-(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum())
    sorted_counts = np.sort(first_counts)
    n = len(sorted_counts)
    gini = float((2 * np.dot(np.arange(1, n + 1), sorted_counts) / (n * sorted_counts.sum())) - (n + 1) / n)
    full = np.unique(levels, axis=0).shape[0]
    collision = float(1 - full / len(levels))
    return {"distinct_codes_l1": distinct[0], "distinct_codes_l1_l2": distinct[1],
            "distinct_codes_l1_l2_l3": distinct[2], "occupancy_entropy": entropy,
            "gini": gini, "collision": collision}
def prefix_er(levels: np.ndarray, depth: int, pairs: dict) -> float:
    anchors, positives, negatives = pairs["anchors"], pairs["positives"], pairs["negatives"]
    prefixes = levels[:, :depth]
    positive_share = np.all(prefixes[positives] == prefixes[anchors], axis=1).mean()
    anchor_repeated = np.repeat(anchors, negatives.shape[1])
    negative_share = np.all(prefixes[negatives.reshape(-1)] == prefixes[anchor_repeated], axis=1).mean()
    return float(positive_share / negative_share) if negative_share else math.nan


def plot_panel(ax, metric: str, label: str, long: list[dict], ci: bool = False,
               linestyle: str = "-") -> None:
    colors = plt.get_cmap("tab10").colors
    baseline = next((r[metric] for r in long if r["arm"] == "resample" and r["level"] == "L1" and r[metric] != ""), None)
    if baseline is not None:
        ax.axhline(float(baseline), color="black", linestyle="--", linewidth=1.1,
                   label="Exp1a (resample) reference")
    for idx, mechanism_name in enumerate(MECHANISMS):
        series = sorted((r for r in long if r["mechanism"] == mechanism_name and
                         r["level"] == "L1" and r[metric] != ""), key=lambda r: r["x"])
        if not series:
            continue
        color = colors[idx % len(colors)]
        ax.plot([int(r["x"]) for r in series], [float(r[metric]) for r in series],
                marker="o", color=color, linestyle=linestyle, label=mechanism_name)
        if ci and metric == "purity_lift":
            bands = [(int(point["x"]), json.loads(point["purity_lift_ci"]))
                     for point in series if point["purity_lift_ci"]]
            if bands:
                band_x = [item[0] for item in bands]
                ax.fill_between(band_x, [item[1][0] for item in bands],
                                [item[1][1] for item in bands], color=color, alpha=.16)
                for x, interval in bands:
                    ax.vlines(x, interval[0], interval[1], color=color, alpha=.65)
    ax.set_xlabel("Crowding strength (w1–w3)")
    ax.set_ylabel(label)
    ax.set_xticks([1, 2, 3], ["w1", "w2", "w3"])
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2)




def stage3_results() -> dict:
    found = {}
    for arm, dirname in STAGE3_RUNS.items():
        root = ROOT / "results/stage3_T5Train" / dirname / "logs"
        files = sorted(root.glob("**/test_final.json")) if root.is_dir() else []
        if files:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
            found[arm] = {"test_recall@10": number(data.get("test_recall@10")),
                          "test_ndcg@10": number(data.get("test_ndcg@10")),
                          "test_final_path": str(files[-1].relative_to(ROOT))}
    resample_root = ROOT / "results/stage3_T5Train/tiger_sample_resample/logs"
    committed = [path for path in resample_root.glob("**/test_final.json")
                 if any("Oct-10-2026" in part for part in path.parts)]
    if committed:
        data = json.loads(committed[-1].read_text(encoding="utf-8"))
        found["resample_committed_oct10"] = {
            "test_recall@10": number(data.get("test_recall@10")),
            "test_ndcg@10": number(data.get("test_ndcg@10")),
            "test_final_path": str(committed[-1].relative_to(ROOT)),
        }
    return found


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    screening_rows = read_csv(RESULTS / "screening_table_complete.csv")
    screening = {}
    for row in screening_rows:
        try:
            screening[(row["arm"], row["level"])] = json.loads(row["arm_metrics_ci"])
        except (json.JSONDecodeError, TypeError):
            continue

    pair_data = np.load(ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz")
    pairs = {key: pair_data[key].astype(np.int64) for key in ("anchors", "positives", "negatives")}
    arms = list(BASELINE_ARMS) + [f"crowd_{mechanism}_w{strength}" for mechanism in MECHANISMS for strength in STRENGTHS]
    stage3 = stage3_results()
    summary = []
    long = []
    for arm in arms:
        if arm == "resample":
            lookup_arm = "Exp1a"
            mechanism, strength = "baseline", ""
        elif arm.startswith("crowd_"):
            lookup_arm = arm
            mechanism, level = arm.rsplit("_w", 1)
            mechanism, strength = mechanism.removeprefix("crowd_"), int(level)
        else:
            lookup_arm = arm
            mechanism, strength = "baseline", ""
        levels = level_array(arm)
        structural = struct_stats(levels) if levels is not None else {}
        ers = ({key: prefix_er(levels, depth, pairs) for key, depth in
                (("ER_L1", 1), ("ER_L1_L2", 2), ("ER_L1_L2_L3", 3))}
               if levels is not None else {})
        row = {"arm": arm, "mechanism": mechanism, "strength": strength,
               **structural, **ers, "purity_lift": "", "auc": "", "intrusion": "",
               "stage3_R@10": "", "stage3_status": "not evaluated"}
        if arm in stage3 and stage3[arm]["test_recall@10"] is not None:
            row["stage3_R@10"] = stage3[arm]["test_recall@10"]
            row["stage3_status"] = "evaluated"
        # Reuse screening metrics where present; structural and prefix measures
        # are calculated directly from the already-exported SID arrays.
        for level_name in ("L1", "L2", "L3"):
            cached = screening.get((lookup_arm, level_name), {})
            if level_name == "L1":
                row["purity_lift"] = cached.get("purity_lift", "")
                row["auc"] = cached.get("auc", "")
                row["intrusion"] = cached.get("intrusion", "")
            erkey = {"L1": "ER_L1", "L2": "ER_L1_L2", "L3": "ER_L1_L2_L3"}[level_name]
            long.append({"arm": arm, "mechanism": mechanism, "strength": strength,
                         "level": level_name, "x": strength if strength != "" else 0,
                         "gini": cached.get("gini", structural.get("gini", "")),
                         "occupancy_entropy": cached.get("entropy", structural.get("occupancy_entropy", "")),
                         "collision": cached.get("collision", structural.get("collision", "")),
                         "purity_lift": cached.get("purity_lift", ""),
                         "purity_lift_ci": json.dumps(cached.get("purity_lift_ci")) if cached.get("purity_lift_ci") else "",
                         "auc": cached.get("auc", ""), "intrusion": cached.get("intrusion", ""),
                         "ER": ers.get(erkey, ""),
                         "metric_source": "cached screening metrics" if cached else "SID-derived; discrimination unavailable"})
        summary.append(row)

    summary_fields = ["arm", "mechanism", "strength", "collision", "distinct_codes_l1",
                      "distinct_codes_l1_l2", "distinct_codes_l1_l2_l3", "occupancy_entropy", "gini",
                      "purity_lift", "auc", "intrusion", "ER_L1", "ER_L1_L2", "ER_L1_L2_L3",
                      "stage3_R@10", "stage3_status"]
    dump_csv(FIGURES / "crowding_summary.csv", summary, summary_fields)
    long_fields = ["arm", "mechanism", "strength", "level", "x", "gini", "occupancy_entropy",
                   "collision", "purity_lift", "purity_lift_ci", "auc", "intrusion", "ER", "metric_source"]
    dump_csv(FIGURES / "crowding_curve_data.csv", long, long_fields)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    plot_panel(axes[0], "gini", "L1 codebook Gini", long)
    plot_panel(axes[1], "occupancy_entropy", "L1 occupancy entropy", long)
    plot_panel(axes[2], "collision", "Full-SID collision rate", long)
    fig.suptitle("Crowding strength vs SID structure")
    fig.savefig(FIGURES / "crowding_degree_vs_sid_structure.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plot_panel(axes[0], "purity_lift", "Purity lift@20 (95% CI)", long, ci=True)
    plot_panel(axes[1], "auc", "Continuous AUC", long)
    intrusion_ax = axes[1].twinx()
    plot_panel(intrusion_ax, "intrusion", "Hard-negative intrusion rate", long, linestyle=":")
    fig.suptitle("Crowding strength vs behavioral discrimination")

    fig.savefig(FIGURES / "crowding_degree_vs_discrimination.png", dpi=180)
    plt.close(fig)

    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for idx, mechanism_name in enumerate(MECHANISMS):
        for er_key, label, marker in (("ER_L1", "L1", "o"), ("ER_L1_L2", "L1+L2", "s"),
                                      ("ER_L1_L2_L3", "L1+L2+L3", "^") ):
            records = [r for r in summary if r["mechanism"] == mechanism_name and r["strength"] != "" and r[er_key] != ""]
            if records:
                ax.plot([int(r["strength"]) for r in records], [float(r[er_key]) for r in records],
                        marker=marker, color=colors[idx % len(colors)], linestyle={"ER_L1": "-", "ER_L1_L2": "--", "ER_L1_L2_L3": ":"}[er_key],
                        label=f"{mechanism_name} {label}")
    base_er = next((r["ER_L1"] for r in summary if r["arm"] == "resample" and r["ER_L1"] != ""), "")
    if base_er != "":
        ax.axhline(float(base_er), color="black", linestyle="--", label="Exp1a (resample), L1 reference")
    ax.set(title="Prefix quality vs crowding", xlabel="Crowding strength (w1–w3)", ylabel="Enrichment ratio (ER)",
           xticks=[1, 2, 3], xticklabels=["w1", "w2", "w3"])
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(FIGURES / "prefix_quality_vs_crowding.png", dpi=180)
    plt.close(fig)

    stage_rows = []
    for row in summary:
        if row["stage3_status"] != "evaluated":
            continue
        structure = row["collision"]
        run = stage3.get(row["arm"], {})
        stage_rows.append({"arm": row["arm"], "mechanism": row["mechanism"], "strength": row["strength"],
                           "collision": structure, "test_recall@10": row["stage3_R@10"],
                           "test_ndcg@10": run.get("test_ndcg@10", ""), "test_final_path": run.get("test_final_path", "")})
    committed = stage3.get("resample_committed_oct10")
    if committed:
        baseline_collision = next(r["collision"] for r in summary if r["arm"] == "resample")
        stage_rows.append({"arm": "Exp1a committed (Oct-10)", "mechanism": "baseline",
                           "strength": "", "collision": baseline_collision,
                           "test_recall@10": committed["test_recall@10"],
                           "test_ndcg@10": committed["test_ndcg@10"],
                           "test_final_path": committed["test_final_path"]})

    dump_csv(FIGURES / "stage3_recall_curve_data.csv", stage_rows,
             ["arm", "mechanism", "strength", "collision", "test_recall@10", "test_ndcg@10", "test_final_path"])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in stage_rows:
        label = row["arm"]
        ax.scatter(float(row["collision"]), float(row["test_recall@10"]), label=label)
        ax.annotate(label, (float(row["collision"]), float(row["test_recall@10"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    baseline = next((r for r in stage_rows if r["arm"] == "resample"), None)
    if baseline:
        ax.axhline(float(baseline["test_recall@10"]), color="black", linestyle="--", label="Exp1a (resample) R@10")
    couser = next((r for r in stage_rows if r["arm"] == "crowd_couser_w3"), None)
    if baseline and couser:
        ax.annotate("crowd_couser_w3 vs Exp1a: paired bootstrap ΔR@10=+0.000209, 95% CI [−0.001393, +0.001707]; statistically indistinguishable from Exp1a",
                    xy=(.02, .02), xycoords="axes fraction", fontsize=8, va="bottom", wrap=True)
    ax.set(title="Stage3 outcomes vs Stage2 collision (evaluated arms only)", xlabel="Stage2 full-SID collision rate",
           ylabel="Stage3 metric (R@10: circles; NDCG@10: squares)")
    for row in stage_rows:
        ax.scatter(float(row["collision"]), float(row["test_ndcg@10"]), marker="s", facecolors="none", edgecolors="C0")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    fig.text(.5, .01, "crowd_couser_w3 is statistically indistinguishable from Exp1a (paired bootstrap ΔR@10=+0.000209; 95% CI [−0.001393, +0.001707]). Only evaluated Stage3 runs are shown.",
             ha="center", fontsize=8)

    fig.savefig(FIGURES / "stage3_recall_vs_crowding.png", dpi=180)
    plt.close(fig)

    print(f"Arms: {len(summary)}; Stage3 values: {sum(r['stage3_status'] == 'evaluated' for r in summary)}; not evaluated: {sum(r['stage3_status'] != 'evaluated' for r in summary)}")
    for path in sorted(FIGURES.glob("*.png")):
        image = plt.imread(path)
        print(f"{path.relative_to(ROOT)} {image.shape[1]}x{image.shape[0]}")
    print("Summary first rows:")
    with (FIGURES / "crowding_summary.csv").open(encoding="utf-8") as stream:
        for _, line in zip(range(4), stream):
            print(line.rstrip())


if __name__ == "__main__":
    main()
