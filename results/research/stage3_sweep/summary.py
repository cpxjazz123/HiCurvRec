"""One table: every Stage2 SID arm, its Stage3 recall and its archival number."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE2 = REPO / "results" / "stage2_RQ-VAE"
STAGE3 = REPO / "results" / "stage3_T5Train"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "summary.json"


def main() -> None:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    uploaded = {}
    for number, entry in ledger.items():
        arm = (entry.get("meta") or {}).get("arm") or entry.get("arm")
        if arm:
            uploaded[arm] = number
    rows = []
    for directory in sorted(STAGE2.iterdir()):
        sid = directory / "item_sids.json"
        if not sid.is_file():
            continue
        payload = json.loads(sid.read_text())
        codes = [tuple(payload[str(i)][:3]) for i in range(len(payload))]
        found = (sorted((STAGE3 / directory.name).glob("logs/*/test_final.json"))
                 + sorted((STAGE3 / directory.name).glob("logs/*/*/test_final.json")))
        recall = None
        if found:
            recall = json.loads(found[-1].read_text()).get("test_recall@10")
        rows.append({"arm": directory.name,
                     "collision": round(1 - len(set(codes)) / len(codes), 4),
                     "test_recall@10": recall,
                     "r2": uploaded.get(directory.name)})
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"{'arm':42s}{'collision':>10s}{'recall@10':>10s}{'r2':>6s}")
    for row in rows:
        recall = f"{row['test_recall@10']:.4f}" if row["test_recall@10"] is not None else "pending"
        print(f"{row['arm']:42s}{row['collision']:10.4f}{recall:>10s}{str(row['r2'] or '-'):>6s}")


if __name__ == "__main__":
    main()
