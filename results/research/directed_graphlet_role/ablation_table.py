"""Issue #221 §5: the ablation table (D0 / D1 / U / S + the frozen baselines).

Reads only on-disk artifacts: the exported SID table, the arm's trq_metrics.json
and the Stage 3 test_final.json.  Nothing is recomputed, so the numbers are the
ones the runs produced.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = {
    "D0 (no aux)": "TIGER_drq_D0",
    "D1 (directed, head on z)": "TIGER_drq_D1",
    "U (undirected labels)": "TIGER_drq_U",
    "S (shuffled directions)": "TIGER_drq_S",
    "frozen baseline": "TIGER_RQ-VAE",
    "mixed F0 (reference)": "TIGER_mixed_rq_F0",
}


def sid_stats(path: Path) -> dict:
    payload = json.loads(path.read_text())
    codes = [payload[str(i)] for i in range(len(payload))]
    n = len(codes)
    level1 = {tuple(c[:1]) for c in codes}
    level2 = {tuple(c[:2]) for c in codes}
    level3 = {tuple(c[:3]) for c in codes}
    return {"items": n, "L1": len(level1), "L2": len(level2), "L3": len(level3),
            "collision": 1.0 - len(level3) / n}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    rows = []
    for label, arm in ARMS.items():
        stage2 = REPO / "results" / "stage2_RQ-VAE" / arm
        stage3 = REPO / "results" / "stage3_T5Train" / arm
        row = {"arm": label, "name": arm}
        sid = stage2 / "item_sids.json"
        if sid.is_file():
            row.update(sid_stats(sid))
        metrics = read_json(stage2 / "trq_metrics.json")
        if metrics:
            head = metrics.get("heldout_scores") or {}
            row["subtype_macro_f1"] = (head.get("subtype") or {}).get("macro_f1")
            row["orbit_macro_f1"] = (head.get("role") or {}).get("macro_over_families")
            row["aux_collision"] = metrics.get("collision")
        candidates = sorted(stage3.glob("logs/*/*/test_final.json")) if stage3.is_dir() else []
        root = stage3 / "test_final.json"
        if root.is_file():
            candidates.append(root)
        candidates = [path for path in candidates if path.is_file()]
        final = None
        if candidates:
            newest = max(candidates, key=lambda path: path.stat().st_mtime)
            final = read_json(newest)
            row["stage3_run"] = str(newest.relative_to(REPO))
        if final:
            row["test_recall@5"] = final.get("test_recall@5")
            row["test_recall@10"] = final.get("test_recall@10")
            row["test_ndcg@10"] = final.get("test_ndcg@10")
        rows.append(row)

    lines = ["| arm | L1 | L2 | L3 | collision | subtype F1 | orbit F1 | test_R@10 | test_NDCG@10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, digits=4):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)
    for row in rows:
        lines.append("| {arm} | {L1} | {L2} | {L3} | {collision} | {sub} | {orb} | {rec} | {ndcg} |".format(
            arm=row["arm"], L1=fmt(row.get("L1")), L2=fmt(row.get("L2")), L3=fmt(row.get("L3")),
            collision=fmt(row.get("collision")), sub=fmt(row.get("subtype_macro_f1")),
            orb=fmt(row.get("orbit_macro_f1")), rec=fmt(row.get("test_recall@10")),
            ndcg=fmt(row.get("test_ndcg@10"))))
    retention = []
    evaluation = read_json(HERE / "evaluation.json") or {}
    if evaluation:
        families = list(next(iter(evaluation.values()))["families"])
        header = "| arm | " + " | ".join(f"{family} L1 / L1+L2 / full" for family in families) + " |"
        retention = ["", "Per-family prefix retention (all_share, code-frequency null already removed):",
                     "", header, "|---" * (len(families) + 1) + "|"]
        for arm, record in evaluation.items():
            cells = []
            for family in families:
                row = record["families"][family]
                cells.append("{:.4f} / {:.5f} / {:.6f}".format(
                    row["L1"]["all_share"], row["L1L2"]["all_share"], row["full"]["all_share"]))
            retention.append(f"| {arm} | " + " | ".join(cells) + " |")
    (HERE / "ABLATION.md").write_text("# Issue #221 ablation\n\n" + "\n".join(lines + retention) + "\n")
    (HERE / "ablation.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
