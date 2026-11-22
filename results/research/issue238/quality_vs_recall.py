"""Is SID-structure quality predictive of Stage 3 recall, across every arm that has both?

The four arms of this round say the SID can change a lot without the recall moving.  This
checks the stronger version of that statement on the whole repository: join every arm's
stage2_metrics record to its Stage 3 test_final.json by directory name, then rank-correlate
collision / quantisation error / effective rank against test_R@10.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S2 = REPO / "results" / "stage2_RQ-VAE"
S3 = REPO / "results" / "stage3_T5Train"


def recall(arm: str):
    hits = sorted((S3 / arm).glob("**/test_final.json"))
    if not hits:
        return None
    return json.loads(hits[-1].read_text(encoding="utf-8")).get("test_recall@10")


def metrics(arm: str) -> dict:
    hits = sorted((S2 / arm / "stage2_metrics").glob("*.json"))
    if not hits:
        return {}
    data = json.loads(hits[-1].read_text(encoding="utf-8"))
    diag = data.get("diagnostics") or {}
    out = {}
    if "l3_collision" in diag:
        out["collision"] = float(diag["l3_collision"])
    for key in ("quant_error", "effective_rank"):
        if key in data:
            out[key] = float(data[key])
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        rank = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rank[order[k]] = average
            i = j + 1
        return rank

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    rows = []
    for arm_dir in sorted(p for p in S2.iterdir() if p.is_dir()):
        arm = arm_dir.name
        r = recall(arm)
        if r is None:
            continue
        m = metrics(arm)
        rows.append((arm, r, m))
    print("arms with both a metrics record and a recall:", len(rows))
    print(f"{'arm':52s} {'R@10':>8s} {'coll':>7s} {'quant':>7s} {'rank':>6s}")
    for arm, r, m in sorted(rows, key=lambda t: -t[1])[:15]:
        print(f"{arm[:52]:52s} {r:8.5f} {m.get('collision', float('nan')):7.4f} "
              f"{m.get('quant_error', float('nan')):7.3f} {m.get('effective_rank', float('nan')):6.2f}")
    for key in ("collision", "quant_error", "effective_rank"):
        pairs = [(m[key], r) for _, r, m in rows if key in m]
        if len(pairs) >= 6:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            print(f"spearman({key}, R@10) = {spearman(xs, ys):+.3f}   n={len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
