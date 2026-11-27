"""Independent Stage3 replay and baseline comparison for issue #269."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.iteration.replay import replay


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if run is None or not run.is_dir():
        raise SystemExit("usage: stage3_recall.py <stage3 run dir>")
    result = replay(run)
    out = ROOT / "results/research/issue269/analysis/REPLAY.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    final = next(iter(sorted(run.glob("test_final.json"))), None)
    if final is None:
        final = run / "test_final.json"
    data = json.loads(final.read_text())
    baselines = {}
    for arm in ("TIGER_RQ-VAE", "TIGER_euclid_local_curve_supervision_rq"):
        path = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
        if path and path[0].is_file():
            block = json.loads(path[0].read_text())
            baselines[arm] = {key: block.get(key) for key in
                              ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10")}
    summary = {
        "schema": "genrec.issue269.stage3_recall/1",
        "final": str(final),
        "current": {key: data.get(key) for key in
                    ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10", "n_eval")},
        "baselines": baselines,
        "delta_vs_parent": (data.get("test_recall@10")
                            - baselines.get("TIGER_euclid_local_curve_supervision_rq", {})
                            .get("test_recall@10", float("nan"))),
        "delta_vs_tiger": (data.get("test_recall@10")
                           - baselines.get("TIGER_RQ-VAE", {})
                           .get("test_recall@10", float("nan"))),
        "above_target_0p065": bool((data.get("test_recall@10") or 0) > 0.065),
        "replay_failed": result.get("failed", []),
    }
    (ROOT / "results/research/issue269/analysis/STAGE3_RECALL.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
