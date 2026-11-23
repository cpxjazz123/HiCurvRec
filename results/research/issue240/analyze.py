"""Turn the issue-240 raw streams into the deliverables the issue asks for.

Inputs (written by the trainer's recorder)
  raw/steps.jsonl   one row per optimizer step
  epochs.jsonl      one row per epoch (probe shifts, codebook spacing/entropy)
  ../stage2_RQ-VAE/<arm>/resample/... the trainer's own SID / metrics

Outputs
  epoch_summary.csv  per-epoch aggregates (mean/median/p95/max, clip rate, conflict rate)
  key_steps.md       the single-step table at the issue's key nodes
  curves.png         ratio / cosine / clip-rate / parameter-displacement curves
  report.json        every number REPORT.md quotes, so the text is checkable
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / "raw"
KEY_EPOCHS = (1, 50, 500, 1000, 1499, 1500, 1501, 1875, 2250, 2625, 3000)
ARM = "TIGER_BEHAVIOR_SAMPLE_RQ-VAE_diag"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get(row: dict, *path, default=None):
    node = row
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def quantiles(values) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def main() -> int:
    steps = load(RAW / "steps.jsonl")
    epochs = load(HERE / "epochs.jsonl")
    if not steps:
        raise SystemExit("no raw/steps.jsonl: run the instrumented Stage2 first")
    report: dict = {"steps": len(steps), "epochs": len({row["epoch"] for row in steps})}

    by_epoch: dict[int, list[dict]] = {}
    for row in steps:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)

    header = [
        "epoch", "steps", "lr", "behavior_weight",
        "recon_l2_mean", "behavior_l2_mean", "behavior_weighted_over_recon_mean",
        "base_norm_mean", "behavior_over_base_mean",
        "cos_recon_behavior_mean", "cos_recon_behavior_median",
        "conflict_rate", "conflict_strength",
        "preclip_norm_mean", "preclip_norm_p95", "preclip_norm_max",
        "clip_rate", "postclip_norm_mean",
        "encoder_delta_l2_mean", "decoder_delta_l2_mean",
        "cb1_delta_l2_mean", "cb2_delta_l2_mean", "cb3_delta_l2_mean",
        "identity_rel_error_max", "assignment_change_mean",
    ]
    lines = [",".join(header)]
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        cosines = [get(row, "directions", "recon_vs_behavior", "cos") for row in rows]
        cosines = [c for c in cosines if c is not None]
        conflicts = [c for c in cosines if c < 0]
        preclip = [get(row, "clip", "pre_norm_reported") for row in rows]
        identity = [get(row, "identity", "rel_error") for row in rows]
        shift = [row["assignment_change"] for row in epochs if int(row["epoch"]) == epoch]
        values = [
            epoch,
            len(rows),
            rows[-1]["lr"],
            rows[-1]["behavior_weight"],
            np.mean([get(r, "term_stats", "recon", "l2", default=0.0) for r in rows]),
            np.mean([get(r, "term_stats", "behavior_l1", "l2", default=0.0) for r in rows]),
            np.mean([get(r, "ratios", "weighted_behavior_over_recon", "value", default=0.0) or 0.0 for r in rows]),
            np.mean([get(r, "ratios", "base_abs_norm", "value", default=0.0) or 0.0 for r in rows]),
            np.mean([get(r, "ratios", "weighted_behavior_over_base", "value", default=0.0) or 0.0 for r in rows]),
            (np.mean(cosines) if cosines else float("nan")),
            (np.median(cosines) if cosines else float("nan")),
            (len(conflicts) / len(cosines) if cosines else 0.0),
            (np.mean([abs(c) for c in conflicts]) if conflicts else 0.0),
            np.mean([v for v in preclip if v is not None]) if any(v is not None for v in preclip) else float("nan"),
            (np.quantile([v for v in preclip if v is not None], 0.95)
             if any(v is not None for v in preclip) else float("nan")),
            (max(v for v in preclip if v is not None) if any(v is not None for v in preclip) else float("nan")),
            np.mean([1.0 if get(r, "clip", "triggered") else 0.0 for r in rows]),
            np.mean([get(r, "clip", "post_norm", default=0.0) or 0.0 for r in rows]),
            np.mean([get(r, "adam_delta", "encoder", "l2", default=0.0) for r in rows]),
            np.mean([get(r, "adam_delta", "decoder", "l2", default=0.0) for r in rows]),
            np.mean([get(r, "adam_delta", "codebook_l1", "l2", default=0.0) for r in rows]),
            np.mean([get(r, "adam_delta", "codebook_l2", "l2", default=0.0) for r in rows]),
            np.mean([get(r, "adam_delta", "codebook_l3", "l2", default=0.0) for r in rows]),
            max([v for v in identity if v is not None], default=float("nan")),
            (float(shift[0]) if shift else float("nan")),
        ]
        lines.append(",".join(
            (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in values
        ))
    (HERE / "epoch_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- key steps -------------------------------------------------------
    key_rows = []
    for epoch in KEY_EPOCHS:
        rows = by_epoch.get(epoch)
        if not rows:
            continue
        row = rows[-1]
        key_rows.append({
            "epoch": epoch,
            "step": row["step"],
            "behavior_weight": row["behavior_weight"],
            "lr": row["lr"],
            "recon_l2": get(row, "term_stats", "recon", "l2"),
            "behavior_l2": get(row, "term_stats", "behavior_l1", "l2"),
            "behavior_weighted_l2": (get(row, "ratios", "behavior_abs_norm", "value") or 0.0) * row["behavior_weight"],
            "unweighted_ratio": get(row, "ratios", "unweighted_behavior_over_recon", "value"),
            "weighted_ratio": get(row, "ratios", "weighted_behavior_over_recon", "value"),
            "cos_recon_behavior": get(row, "directions", "recon_vs_behavior", "cos"),
            "cos_base_behavior": get(row, "directions", "base_vs_behavior", "cos"),
            "preclip_norm": get(row, "clip", "pre_norm_reported"),
            "postclip_norm": get(row, "clip", "post_norm"),
            "clip_triggered": get(row, "clip", "triggered"),
            "encoder_delta": get(row, "adam_delta", "encoder", "l2"),
            "encoder_rel_delta": get(row, "adam_delta", "encoder", "rel"),
            "decoder_delta": get(row, "adam_delta", "decoder", "l2"),
            "identity_rel_error": get(row, "identity", "rel_error"),
        })
    key_header = list(key_rows[0]) if key_rows else ["epoch"]
    key_lines = ["|" + "|".join(key_header) + "|", "|" + "|".join(["---"] * len(key_header)) + "|"]
    for row in key_rows:
        cells = []
        for name in key_header:
            value = row[name]
            cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        key_lines.append("|" + "|".join(cells) + "|")
    (HERE / "key_steps.md").write_text("\n".join(key_lines) + "\n", encoding="utf-8")

    # ---- aggregates ------------------------------------------------------
    def series(name, path, extra=None):
        values = []
        for row in steps:
            value = get(row, *path) if extra is None else extra(row)
            if value is not None:
                values.append(value)
        return quantiles(values)

    preclip = [get(row, "clip", "pre_norm_reported") for row in steps]
    report["preclip_norm"] = quantiles(preclip)
    report["clip_rate"] = float(np.mean([1.0 if get(row, "clip", "triggered") else 0.0 for row in steps]))
    report["postclip_norm"] = quantiles([get(row, "clip", "post_norm") for row in steps])
    report["identity_rel_error"] = quantiles([get(row, "identity", "rel_error") for row in steps])
    for stage, predicate in (
        ("behavior_off", lambda row: row["behavior_weight"] == 0.0),
        ("behavior_on", lambda row: row["behavior_weight"] > 0.0),
    ):
        subset = [row for row in steps if predicate(row)]
        if not subset:
            continue
        cosines = [get(row, "directions", "recon_vs_behavior", "cos") for row in subset]
        cosines = [c for c in cosines if c is not None]
        report[stage] = {
            "steps": len(subset),
            "recon_l2": quantiles([get(row, "term_stats", "recon", "l2") for row in subset]),
            "behavior_l2": quantiles([get(row, "term_stats", "behavior_l1", "l2") for row in subset]),
            "unweighted_ratio": quantiles([get(row, "ratios", "unweighted_behavior_over_recon", "value") for row in subset]),
            "weighted_ratio": quantiles([get(row, "ratios", "weighted_behavior_over_recon", "value") for row in subset]),
            "cos_mean": float(np.mean(cosines)) if cosines else None,
            "conflict_rate": (sum(1 for c in cosines if c < 0) / len(cosines)) if cosines else None,
            "conflict_strength": (float(np.mean([abs(c) for c in cosines if c < 0]))
                                  if any(c < 0 for c in cosines) else None),
            "preclip_norm": quantiles([get(row, "clip", "pre_norm_reported") for row in subset]),
            "clip_rate": float(np.mean([1.0 if get(row, "clip", "triggered") else 0.0 for row in subset])),
            "encoder_delta": quantiles([get(row, "adam_delta", "encoder", "l2") for row in subset]),
            "codebook_l1_delta": quantiles([get(row, "adam_delta", "codebook_l1", "l2") for row in subset]),
        }
    report["epochs_trajectory"] = [
        {
            "epoch": row["epoch"],
            "assignment_change": row.get("assignment_change"),
            "z_shift": row.get("z_shift"),
            "hard_q_shift": row.get("hard_q_shift"),
            "codebooks": row.get("codebooks"),
        }
        for row in epochs
        if int(row["epoch"]) in KEY_EPOCHS or int(row["epoch"]) % 250 == 0
    ]
    sid = REPO / "results" / "stage2_RQ-VAE" / ARM / "resample" / "item_sids.json"
    if sid.exists():
        payload = json.loads(sid.read_text(encoding="utf-8"))
        rows = payload.get("items", payload)
        triples = {tuple(entry[:3]) for entry in rows}
        report["sid"] = {"items": len(rows), "distinct_triples": len(triples),
                         "collision": 1.0 - len(triples) / len(rows)}
    (HERE / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "epochs_trajectory"}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
