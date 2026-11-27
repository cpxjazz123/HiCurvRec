"""Issue #270: row-aligned recommendation linkage from the frozen test predictions."""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"
ASSETS = OUT / "assets/pairs"
TIGER_FINAL = REPO / "results/stage3_T5Train/TIGER_RQ-VAE/logs/Amazon_2023_Instruments/Nov-23-2026_10-57-48"
N_EVAL = 57439


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def read_predictions(final: Path) -> dict:
    hits5 = np.zeros(N_EVAL, dtype=np.int8)
    hits10 = np.zeros(N_EVAL, dtype=np.int8)
    rank = np.zeros(N_EVAL, dtype=np.int16)
    target = np.full(N_EVAL, -1, dtype=np.int32)
    seen = np.zeros(N_EVAL, dtype=bool)
    for path in sorted(final.glob("predictions/rank*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                g = int(row["global_row"])
                t = int(row["target_item"])
                top = [int(x["item"]) for x in row["topk"]]
                pos = top.index(t) + 1 if t in top else 0
                hits5[g] = int(0 < pos <= 5)
                hits10[g] = int(0 < pos <= 10)
                rank[g] = pos
                target[g] = t
                seen[g] = True
    if not seen.all():
        raise ValueError("missing global rows: %d" % int((~seen).sum()))
    return {"hit5": hits5, "hit10": hits10, "rank": rank, "target": target}


def main() -> None:
    arm = sys.argv[1]
    cohort = json.loads((OUT / "COHORT.json").read_text())
    row = next(r for r in cohort["rows"] if r["arm"] == arm and r["usable_geometry"])
    final = Path(row["stage3_final"]).parent
    cur = read_predictions(final)
    tiger = read_predictions(TIGER_FINAL)
    if not np.array_equal(cur["target"], tiger["target"]):
        raise ValueError("target mismatch across arms")
    delta_hit10 = (cur["hit10"].astype(np.int8) - tiger["hit10"].astype(np.int8))
    delta_hit5 = (cur["hit5"].astype(np.int8) - tiger["hit5"].astype(np.int8))
    # per-target aggregate
    items = np.unique(tiger["target"])
    n = np.bincount(tiger["target"], minlength=24587).astype(np.float64)
    cur_hits = np.bincount(tiger["target"], weights=cur["hit10"], minlength=24587)
    tiger_hits = np.bincount(tiger["target"], weights=tiger["hit10"], minlength=24587)
    rank_sum = np.bincount(tiger["target"], weights=np.where(cur["rank"] > 0, cur["rank"], 0),
                           minlength=24587)
    rank_n = np.bincount(tiger["target"], weights=(cur["rank"] > 0).astype(np.float64),
                         minlength=24587)
    ndcg10 = float(sum(1.0 / math.log2(int(r) + 1) for r in cur["rank"] if 0 < r <= 10) / N_EVAL)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / (arm + ".npz")
    np.savez_compressed(out, hit5=cur["hit5"], hit10=cur["hit10"], rank=cur["rank"],
                        delta_hit10=delta_hit10, delta_hit5=delta_hit5,
                        per_target_n=n, per_target_cur=cur_hits, per_target_tiger=tiger_hits,
                        per_target_rank_sum=rank_sum, per_target_rank_n=rank_n)
    summary = {
        "schema": "genrec.issue270.pair/1", "arm": arm,
        "stage3_final": str(final / "test_final.json"),
        "tiger_final": str(TIGER_FINAL / "test_final.json"),
        "hit@5": float(cur["hit5"].mean()), "hit@10": float(cur["hit10"].mean()),
        "tiger_hit@5": float(tiger["hit5"].mean()), "tiger_hit@10": float(tiger["hit10"].mean()),
        "delta_hit@10": float(delta_hit10.mean()), "delta_hit@5": float(delta_hit5.mean()),
        "ndcg@10_from_rank": ndcg10,
        "hit10_gain": int((delta_hit10 == 1).sum()), "hit10_loss": int((delta_hit10 == -1).sum()),
        "output": str(out), "output_sha256": sha(out),
    }
    (ASSETS / (arm + ".json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("arm", "hit@10", "tiger_hit@10", "delta_hit@10", "ndcg@10_from_rank")}))


if __name__ == "__main__":
    main()
