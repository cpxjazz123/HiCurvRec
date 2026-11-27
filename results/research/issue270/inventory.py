"""Issue #270 step 1: inventory every completed geometry arm with a recoverable export."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"
LEDGER = REPO / "results/research/issue139/r2_ledger.json"
S2 = REPO / "results/stage2_RQ-VAE"
S3 = REPO / "results/stage3_T5Train"
EXPECTED = 57439


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    alt = REPO / str(path).replace("/home/wlia0047/ar57/wenyu/GeneRec/", "")
    return alt if alt.is_file() else None


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    arms = {}
    for number, entry in ledger.items():
        if not isinstance(entry, dict) or "files" not in entry:
            continue
        arm = entry.get("arm")
        files = entry["files"]
        arms.setdefault(arm, []).append({
            "r2": number, "files": len(files),
            "has_sid": any(k.endswith("item_sids.json") for k in files),
            "has_ckpt": any(k.endswith("rqvae_best.pth") for k in files),
            "has_recall": any(k == "recall.json" for k in files),
            "has_source": any(k.endswith("stage2_source.tar.gz") for k in files)})
    rows = []
    for arm_dir in sorted(S3.iterdir()) if S3.is_dir() else []:
        if not arm_dir.is_dir():
            continue
        finals = sorted(arm_dir.glob("**/test_final.json"),
                        key=lambda p: p.parent.name)
        if not finals:
            continue
        final = finals[-1]
        block = json.loads(final.read_text())
        sid = resolve(block.get("sid_path"))
        ckpt = resolve(block.get("stage2_checkpoint"))
        preds = sorted(final.parent.glob("predictions/rank*.jsonl"))
        lines = sum(sum(1 for _ in preds and open(p, "rb")) if False else 0 for p in [])
        lines = 0
        for p in preds:
            with p.open("rb") as f:
                lines += sum(1 for _ in f)
        rows.append({
            "arm": arm_dir.name,
            "stage3_final": str(final),
            "stage3_runs": len(finals),
            "stage2_run_id": block.get("stage2_run_id"),
            "recall@10": block.get("test_recall@10"),
            "ndcg@10": block.get("test_ndcg@10"),
            "n_eval": block.get("n_eval"),
            "sid": str(sid) if sid else None,
            "sid_sha256": sha(sid) if sid else None,
            "checkpoint": str(ckpt) if ckpt else None,
            "checkpoint_sha256": sha(ckpt) if ckpt else None,
            "predictions": len(preds), "prediction_rows": lines,
            "predictions_complete": lines >= EXPECTED,
            "mechanism_metadata": str(final.parent / "test_final.json"),
            "r2": arms.get(arm_dir.name, []),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RUN_INVENTORY.json").write_text(json.dumps(
        {"schema": "genrec.issue270.inventory/1", "rows": rows,
         "ledger_arms": len(arms)}, indent=2, sort_keys=True) + "\n")
    print("rows", len(rows), "complete",
          sum(r["predictions_complete"] for r in rows),
          "sid+ckpt", sum(bool(r["sid"] and r["checkpoint"]) for r in rows))
    for r in rows:
        if r["predictions_complete"]:
            print("%-55s r10=%.6f sid=%s ckpt=%s" % (
                r["arm"], r["recall@10"], bool(r["sid"]), bool(r["checkpoint"])))


if __name__ == "__main__":
    main()
