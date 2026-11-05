"""The Issue #138 arm table, read from the artifacts each arm actually wrote.

One row per arm: the SID SHA-256 (so the Stage 3 loader can be proved to have
consumed that exact table), the structural diagnostics, the within/cross curvature
gap recorded at export, and the Stage 3 test scores. Nothing is recomputed from a
filename or a vocabulary size.

Run:  python3 analysis_sid_recall/arm_table arms.json
      (``[{"variant": ..., "sid": ..., "metrics": ..., "stage3": ...}, ...]``)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

COLUMNS = ("variant", "sid_sha256", "collision", "l1", "l2", "l3",
           "gap_l1", "gap_l2", "align_l1", "align_l2", "recall@10", "ndcg@10")


def alignment(sid_path: Path, pairs_path: Path) -> dict:
    """The independent-validation prefix hit rates of one SID (diagnostic)."""
    import numpy as np
    table = json.loads(Path(sid_path).read_text())
    data = np.load(pairs_path)
    anchors = data["anchors"].ravel()
    positives = data["positives"].ravel()
    negatives = data["negatives"].ravel()
    out = {}
    for width in (1, 2):
        def prefix(item):
            value = table.get(str(int(item)))
            return tuple(value[:width]) if value else None
        pairs = [(prefix(a), prefix(b)) for a, b in zip(anchors, positives)]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        pos = sum(a == b for a, b in pairs) / max(len(pairs), 1)
        cross = [(prefix(a), prefix(b)) for a, b in zip(anchors, negatives)]
        cross = [(a, b) for a, b in cross if a is not None and b is not None]
        neg = sum(a == b for a, b in cross) / max(len(cross), 1)
        out[f"align_l{width}"] = pos - neg
    return out


def sid_structure(sid_path: Path) -> dict:
    """The structural diagnostics of one exported SID table."""
    raw = Path(sid_path).read_bytes()
    table = json.loads(raw)
    values = list(table.values())
    return {
        "sid_sha256": hashlib.sha256(raw).hexdigest(),
        "items": len(values),
        "l1": len({value[0] for value in values}),
        "l2": len({tuple(value[:2]) for value in values}),
        "l3": len({tuple(value[:3]) for value in values}),
        "collision": _collision(values),
    }


def _collision(values) -> float:
    counts: dict[tuple, int] = {}
    for value in values:
        key = tuple(value[:3])
        counts[key] = counts.get(key, 0) + 1
    return sum(n for n in counts.values() if n > 1) / max(len(values), 1)


def export_gap(metrics_path: Path) -> dict:
    """The last within/cross curvature gap per level recorded at export."""
    gaps = {}
    path = Path(metrics_path)
    if not path.is_file():
        return gaps
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") != "sid_structure" or row.get("gap") is None:
            continue
        gaps[f"gap_l{int(row['level']) + 1}"] = float(row["gap"])
    return gaps


def stage3_result(stage3_dir: Path, sid_sha256: str | None = None) -> dict:
    """The final test scores of the newest Stage 3 run that loaded this SID.

    A run's ``train_start`` records the SID path and its SHA-256; a result whose
    recorded hash differs from the SID presented here is not that arm's result,
    so it is reported as a mismatch instead of being paired silently (Issue 138).
    """
    runs = sorted(Path(stage3_dir).rglob("test_final.json"),
                  key=lambda path: path.stat().st_mtime, reverse=True)
    for final in runs:
        run_dir = final.parent
        metrics = run_dir / "training_metrics.jsonl"
        recorded = None
        if metrics.is_file():
            for line in metrics.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == "train_start":
                    recorded = row.get("code_path_sha256")
                    break
        if sid_sha256 is not None and recorded is not None and recorded != sid_sha256:
            continue
        payload = json.loads(final.read_text(encoding="utf-8"))
        return {"recall@10": float(payload["test_recall@10"]),
                "ndcg@10": float(payload.get("test_ndcg@10", float("nan"))),
                "stage3_sid_sha256": recorded,
                "stage3_sid_matches": (recorded == sid_sha256) if sid_sha256 else None}
    return {"recall@10": None, "ndcg@10": None,
            "stage3_sid_sha256": None, "stage3_sid_matches": False}


def arm_row(variant: str, sid_path: Path, metrics_path: Path,
            stage3_dir: Path, pairs_path: Path | None = None) -> dict:
    row = {"variant": variant, **sid_structure(sid_path)}
    row.update(export_gap(metrics_path))
    if pairs_path is not None and Path(pairs_path).is_file():
        row.update(alignment(sid_path, pairs_path))
    row.update(stage3_result(stage3_dir, row["sid_sha256"]))
    return row


def render(rows: list[dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, rule]
    for row in rows:
        cells = []
        for column in COLUMNS:
            value = row.get(column)
            if value is None:
                cells.append("pending")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: arm_table.py <arms.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    rows = [arm_row(arm["variant"], Path(arm["sid"]), Path(arm["metrics"]),
                    Path(arm["stage3"])) for arm in spec]
    print(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
