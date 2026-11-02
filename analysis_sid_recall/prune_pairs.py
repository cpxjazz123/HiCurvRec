"""Keep only the verified (SID, recall) pairs in `analysis_sid_recall/`.

A pair is verified when `pair_audit.py` reports `verdict == VERIFIED` for it:
the SID table and the Stage 3 result file that carries the recall were
introduced by the same git commit **and** the `results/stage2_RQ-VAE/<X>` and
`results/stage3_T5Train/<X>` directories carry the same name (repo rule 64b).

The script is idempotent and never destroys evidence:

* the pre-prune tables are copied once into `archive/` (if they are not there);
* the rejected pairs go to `excluded/sid_recall_failed_pair_audit.csv`, with the
  audit's name class, verdict, the two commits and the two commit times;
* only `sid_recall.csv`, `sid_recall.json` and `sid_inventory.csv` are rewritten;
* the collected SID tables under `sids/` are kept: for several arms they are the
  only surviving copy of the SID, and they are evidence, not pairs.

Run:  python3 analysis_sid_recall/prune_pairs.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
ARCHIVE = OUT / "archive"
EXCLUDED = OUT / "excluded"
PAIRS = OUT / "sid_recall.csv"
PAIRS_JSON = OUT / "sid_recall.json"
INVENTORY = OUT / "sid_inventory.csv"
AUDIT = OUT / "pair_audit.csv"


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ARCHIVE.mkdir(exist_ok=True)
    EXCLUDED.mkdir(exist_ok=True)
    audit = list(csv.DictReader(AUDIT.open()))
    verified = {row["sid_sha256"] for row in audit if row["verdict"] == "VERIFIED"}
    audit_by_sha = {row["sid_sha256"]: row for row in audit}
    assert len(audit_by_sha) == len(audit) == 182, "the audit is not one row per SID"
    print(f"verified: {len(verified)}/{len(audit)}")

    # 1. the full tables, archived once
    for source, name in ((PAIRS, "sid_recall_full_182.csv"),
                         (PAIRS_JSON, "sid_recall_full_182.json")):
        target = ARCHIVE / name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
            print("archived:", target.relative_to(OUT))

    # 2. the rejected pairs, with the audit's reasons
    source = ARCHIVE / "sid_recall_full_182.csv"
    fields, pairs = read_rows(source if source.is_file() else PAIRS)
    rejected = [row for row in pairs if row["sid_sha256"] not in verified]
    extra = ["verdict", "name_class", "stage2_rel", "sid_intro_commit",
             "sid_intro_time", "recall_intro_commit", "recall_intro_time",
             "same_commit", "sid_before_recall"]
    reason_fields = fields + extra
    rejected_rows = []
    for row in rejected:
        entry = dict(row)
        record = audit_by_sha.get(row["sid_sha256"], {})
        for key in extra:
            entry[key] = record.get(key)
        rejected_rows.append(entry)
    write_rows(EXCLUDED / "sid_recall_failed_pair_audit.csv", reason_fields, rejected_rows)
    print("excluded:", len(rejected_rows), "->", "excluded/sid_recall_failed_pair_audit.csv")

    # 3. the pair tables keep the verified rows only
    kept = [row for row in pairs if row["sid_sha256"] in verified]
    write_rows(PAIRS, fields, kept)
    json_source = ARCHIVE / "sid_recall_full_182.json"
    payload = json.loads((json_source if json_source.is_file() else PAIRS_JSON).read_text())
    payload["pairs"] = [row for row in payload["pairs"]
                        if row.get("sid_sha256") in verified]
    payload["inventory"] = [row for row in payload["inventory"]
                            if row.get("sid_sha256") in verified]
    PAIRS_JSON.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"pairs kept: {len(kept)} (json pairs {len(payload['pairs'])}, "
          f"json inventory {len(payload['inventory'])})")

    # 4. the inventory keeps the verified SIDs only
    inventory_source = ARCHIVE / "sid_inventory_full_624.csv"
    inventory_fields, inventory = read_rows(inventory_source if inventory_source.is_file()
                                            else INVENTORY)
    inventory_kept = [row for row in inventory if row["sid_sha256"] in verified]
    write_rows(INVENTORY, inventory_fields, inventory_kept)
    print(f"inventory kept: {len(inventory_kept)} of {len(inventory)}")

    assert len(kept) == len(verified) == len(payload["pairs"])
    assert {row["sid_sha256"] for row in kept} == verified
    print("ok: every surviving pair is VERIFIED and nothing else is left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
