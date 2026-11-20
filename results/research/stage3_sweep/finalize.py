"""Close the sweep: refresh the report table and report anything still missing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def refresh_report(rows: list) -> None:
    """Rewrite the report's result table from summary.json."""
    lines = ["| arm | collision | test_recall@10 | R2 |", "|---|---|---|---|"]
    for row in rows:
        recall = f"{row['test_recall@10']:.4f}" if row["test_recall@10"] is not None else "pending"
        lines.append(f"| {row['arm']} | {row['collision']:.4f} | {recall} | {row['r2'] or '—'} |")
    report = HERE / "REPORT.md"
    text = report.read_text()
    head, rest = text.split("## 结果表", 1)
    body = rest.split("## 方法要点", 1)[1]
    report.write_text(head + "## 结果表\n\n" + "\n".join(lines) + "\n\n## 方法要点" + body)


def main() -> int:
    subprocess.run([sys.executable, str(HERE / "summary.py")], cwd=str(REPO), check=True)
    rows = json.loads((HERE / "summary.json").read_text())
    refresh_report(rows)
    pending = [row["arm"] for row in rows if row["test_recall@10"] is None]
    recalls = [row["test_recall@10"] for row in rows if row["test_recall@10"] is not None]
    print(f"arms {len(rows)}, with recall {len(recalls)}, pending {len(pending)}")
    if recalls:
        best = max(rows, key=lambda row: row["test_recall@10"] or -1)
        print(f"best: {best['arm']} {best['test_recall@10']:.4f}")
        print(f"above 0.065: {[r['arm'] for r in rows if (r['test_recall@10'] or 0) > 0.065]}")
    if pending:
        print("pending:", ", ".join(pending))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
