"""Issue #218 step 0: coverage and provenance audit over the 91 archived numbers.

Reads the local mirror `analysis_sid_recall/r2/<NN>/{item_sids.json,recall.json}`, checks every
SHA against `r2_ledger.json`, groups duplicate SIDs, and writes `coverage.json` +
`paired_arms.csv`.  A number whose pieces disagree is excluded with a reason, never repaired.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "analysis_sid_recall" / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
ITEMS = 24587


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    metadata = json.loads(METADATA.read_text())
    graph_sha = metadata["outputs"]["graph.npz"]["sha256"]
    rows, exclusions = [], []
    for number in sorted(k for k in ledger if k.isdigit()):
        entry = ledger[number]
        arm = entry.get("arm")
        files = entry.get("files") or {}
        directory = MIRROR / number
        sid_path, recall_path = directory / "item_sids.json", directory / "recall.json"
        record = {"number": number, "arm": arm, "graph_sha256": graph_sha}
        if not sid_path.is_file():
            exclusions.append({"number": number, "arm": arm, "reason": "no SID in the mirror"})
            continue
        sid_sha = sha256(sid_path)
        recorded = files.get("item_sids.json")
        if recorded and recorded != sid_sha:
            exclusions.append({"number": number, "arm": arm, "reason": "SID sha256 disagrees with the ledger"})
            continue
        payload = json.loads(sid_path.read_text())
        codes = [tuple(int(v) for v in payload[str(i)][:3]) for i in range(len(payload))]
        record.update(sid_sha256=sid_sha, items=len(codes), collision=1 - len(set(codes)) / len(codes),
                      checkpoint_sha256=files.get("rqvae_best.pth"))
        if recall_path.is_file():
            roll_sha = sha256(recall_path)
            if files.get("recall.json") and files["recall.json"] != roll_sha:
                exclusions.append({"number": number, "arm": arm, "reason": "recall sha256 disagrees with the ledger"})
                continue
            block = json.loads(recall_path.read_text())
            record.update(recall_sha256=roll_sha, recall=block.get("test_recall@10"),
                          n_eval=block.get("n_eval"),
                          protocol=sorted(block.keys()))
            if record["recall"] is None:
                exclusions.append({"number": number, "arm": arm, "reason": "recall block has no test_recall@10"})
                record["recall"] = None
        else:
            record.update(recall=None, recall_sha256=None, protocol=None)
            exclusions.append({"number": number, "arm": arm, "reason": "no recall block in the mirror"})
        rows.append(record)

    by_sid = defaultdict(list)
    for row in rows:
        by_sid[row["sid_sha256"]].append(row["number"])
    duplicates = {sha: numbers for sha, numbers in by_sid.items() if len(numbers) > 1}
    for row in rows:
        group = duplicates.get(row["sid_sha256"])
        row["duplicate_group"] = ",".join(group) if group else ""

    paired = [row for row in rows if row["recall"] is not None]
    distinct = {row["sid_sha256"] for row in paired}
    coverage = {
        "numbers": len([k for k in ledger if k.isdigit()]),
        "sid_verified": len(rows),
        "paired": len(paired),
        "distinct_sids_paired": len(distinct),
        "duplicate_sids": {sha[:16]: numbers for sha, numbers in duplicates.items()},
        "exclusions": exclusions,
        "graph_sha256": graph_sha,
        "items_expected": ITEMS,
        "items_mismatch": [row["number"] for row in rows if row["items"] != ITEMS],
    }
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=1))
    (OUT / "duplicate_sid_groups.json").write_text(json.dumps(duplicates, indent=1))
    fields = ["number", "arm", "sid_sha256", "checkpoint_sha256", "recall", "n_eval", "collision",
              "items", "duplicate_group", "graph_sha256", "protocol"]
    with (OUT / "paired_arms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({k: coverage[k] for k in ("numbers", "sid_verified", "paired", "distinct_sids_paired")}))
    print("exclusions:", exclusions)
    print("duplicate groups:", coverage["duplicate_sids"])


if __name__ == "__main__":
    main()
