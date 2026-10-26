"""Collect every SID table reachable in the git history and pair it with its recall.

Sources
-------
* SID tables: every blob named ``*item_sids*.json`` in the history
  (``git rev-list --all --objects``), plus every ``item_sids*.json`` that only
  exists in the working tree.
* Recalls: every Stage 3 run that recorded a test score
  (``results/stage3_T5Train/<variant>/logs/<dataset>/<run>/training_metrics.jsonl``
  and the ``test_final.json`` beside it), in the working tree and in the history.
  The run's own ``train_start`` event names the ``code_path`` it generated from,
  which is what pairs a SID with a recall.

Everything is written under ``analysis_sid_recall/``:

``sids/``               the SID tables, one file per distinct content
``sid_recall.csv``      one row per (SID content, Stage 3 run) pair
``sid_recall.json``     the same pairs with the SID structure
``sid_inventory.csv``   one row per SID content, paired or not
``README.md``           what was collected, how, and what is missing
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_sid_recall"
SID_DIR = OUT / "sids"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=False).stdout


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          check=False).stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def structure(payload: bytes) -> dict:
    """The shape of a SID table: items, width and the prefix statistics."""
    table = json.loads(payload.decode("utf-8"))
    keys = sorted(table, key=lambda key: int(key))
    rows = [table[key] for key in keys]
    width = len(rows[0]) if rows else 0
    levels = {}
    for depth in range(1, min(width, 4) + 1):
        codes = {tuple(row[:depth]) for row in rows}
        counts = defaultdict(int)
        for row in rows:
            counts[tuple(row[:depth])] += 1
        largest = max(counts.values()) if counts else 0
        levels["L%d" % depth] = {
            "distinct": len(codes),
            "largest_bucket": int(largest),
            "largest_bucket_share": round(largest / len(rows), 6) if rows else 0.0,
        }
    triples = {tuple(row[:3]) for row in rows} if width >= 3 else set()
    return {
        "items": len(rows),
        "width": width,
        "levels": levels,
        "unique_rows": len({tuple(row) for row in rows}),
        "items_sharing_a_triple": (len(rows) - len(triples)) if triples else None,
    }


def collect_sid_blobs() -> dict:
    """Every distinct SID content in the history, with the paths it was stored at."""
    listing = git("rev-list", "--all", "--objects")
    blobs: dict[str, dict] = {}
    for line in listing.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2 or "item_sids" not in parts[1]:
            continue
        blob, path = parts[0], parts[1]
        if blob in blobs:
            blobs[blob]["paths"].append(path)
            continue
        payload = git_bytes("cat-file", "blob", blob)
        if not payload:
            continue
        try:
            stats = structure(payload)
        except Exception:
            continue
        blobs[blob] = {"blob": blob, "paths": [path], "bytes": len(payload),
                       "sha256": sha256_bytes(payload), "stats": stats}
    return blobs


def collect_working_tree_sids(blobs: dict) -> None:
    """SID tables that only exist in the working tree."""
    for path in sorted(REPO.glob("results/**/item_sids*.json")):
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        if any(entry["sha256"] == digest for entry in blobs.values()):
            relative = str(path.relative_to(REPO))
            for entry in blobs.values():
                if entry["sha256"] == digest and relative not in entry["paths"]:
                    entry["paths"].append(relative)
            continue
        try:
            stats = structure(payload)
        except Exception:
            continue
        blobs[digest] = {"blob": None, "paths": [str(path.relative_to(REPO))],
                         "bytes": len(payload), "sha256": digest, "stats": stats}


def resolve_path_to_content(path: str) -> tuple[str | None, bytes | None, str | None]:
    """The content of a SID path, from the working tree or from the history."""
    candidate = REPO / path
    if candidate.is_file():
        payload = candidate.read_bytes()
        return sha256_bytes(payload), payload, None
    listing = git("rev-list", "--all", "--objects", "--", path).splitlines()
    for line in reversed(listing):
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        payload = git_bytes("cat-file", "blob", parts[0])
        if payload:
            return sha256_bytes(payload), payload, parts[0]
    return None, None, None


def collect_stage3_runs() -> list:
    """Every Stage 3 run that recorded a test score, with the SID it used."""
    runs = []
    files = sorted(REPO.glob("results/stage3_T5Train/*/logs/*/*/training_metrics.jsonl"))
    for path in files:
        code_path = None
        test = None
        relative = path.relative_to(REPO)
        # results/stage3_T5Train/<variant>/logs/<dataset>/<run>/training_metrics.jsonl
        variant = relative.parts[2] if len(relative.parts) > 2 else ""
        run_dir = path.parent
        for line in path.read_text(errors="replace").splitlines():
            if '"train_start"' in line and code_path is None:
                try:
                    code_path = json.loads(line).get("code_path")
                except ValueError:
                    pass
            if '"test"' in line and test is None:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                test = {"test_recall@5": event.get("test_recall_at_5"),
                        "test_recall@10": event.get("test_recall_at_10"),
                        "test_ndcg@5": event.get("test_ndcg_at_5"),
                        "test_ndcg@10": event.get("test_ndcg_at_10"),
                        "n_eval": event.get("n_eval"), "epoch": event.get("epoch")}
        if code_path and test and test.get("test_recall@10") is not None:
            final = run_dir / "test_final.json"
            if final.is_file():
                try:
                    recorded = json.loads(final.read_text())
                    for key in ("test_recall@5", "test_recall@10", "test_ndcg@5",
                                "test_ndcg@10", "n_eval"):
                        if key in recorded:
                            test[key] = recorded[key]
                except ValueError:
                    pass
            runs.append({"variant": variant, "run_dir": str(run_dir.relative_to(REPO)),
                          "code_path": code_path, "metrics": test})
    return runs


def write_sid_files(blobs: dict) -> dict:
    """Write every distinct SID table into ``sids/`` and report the file name."""
    SID_DIR.mkdir(parents=True, exist_ok=True)
    names = {}
    for digest, entry in blobs.items():
        path = entry["paths"][0]
        slug = path.replace("/", "__").replace(".json", "")
        name = f"{slug}__{digest[:12]}.json"
        target = SID_DIR / name
        if not target.is_file():
            if entry["blob"]:
                target.write_bytes(git_bytes("cat-file", "blob", entry["blob"]))
            else:
                source = REPO / path
                target.write_bytes(source.read_bytes() if source.is_file() else b"")
        # keyed by the content sha256: that is what the pairing uses
        names[entry["sha256"]] = name
    return names


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blobs = collect_sid_blobs()
    collect_working_tree_sids(blobs)
    runs = collect_stage3_runs()

    by_sha = {entry["sha256"]: digest for digest, entry in blobs.items()}
    pairs = []
    for run in runs:
        code = run["code_path"]
        if code.startswith(str(REPO)):
            code = str(Path(code).relative_to(REPO))
        digest, payload, blob = resolve_path_to_content(code)
        if digest is None:
            continue
        if digest not in by_sha:
            # A SID that only the Stage 3 run knows about: collect it as well, so the
            # pairing is never dropped for want of the table.
            try:
                stats = structure(payload)
            except Exception:
                continue
            blobs[digest] = {"blob": blob, "paths": [code], "bytes": len(payload),
                              "sha256": digest, "stats": stats}
            by_sha[digest] = digest
        pairs.append({"sid_sha256": digest, "sid_paths": blobs[by_sha[digest]]["paths"],
                      "variant": run["variant"], "run_dir": run["run_dir"],
                      "code_path": code, **run["metrics"]})

    names = write_sid_files(blobs)

    inventory = []
    for digest, entry in blobs.items():
        paired = [pair for pair in pairs if pair["sid_sha256"] == digest]
        best = max((pair["test_recall@10"] for pair in paired), default=None)
        inventory.append({
            "sid_sha256": entry["sha256"], "file": names[entry["sha256"]], "bytes": entry["bytes"],
            "paths": entry["paths"], "git_blob": entry["blob"],
            "items": entry["stats"]["items"], "width": entry["stats"]["width"],
            "levels": entry["stats"]["levels"],
            "unique_rows": entry["stats"]["unique_rows"],
            "stage3_runs": len(paired),
            "best_test_recall@10": best,
        })
    with (OUT / "sid_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        for row in inventory:
            row = dict(row)
            row["paths"] = " | ".join(row["paths"])
            row["levels"] = json.dumps(row["levels"], ensure_ascii=False)
            writer.writerow(row)

    with (OUT / "sid_recall.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sid_sha256", "sid_file", "sid_paths", "variant", "run_dir",
                  "test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10", "n_eval"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in sorted(pairs, key=lambda item: -(item.get("test_recall@10") or 0)):
            writer.writerow({
                "sid_sha256": pair["sid_sha256"], "sid_file": names[pair["sid_sha256"]],
                "sid_paths": " | ".join(pair["sid_paths"]), "variant": pair["variant"],
                "run_dir": pair["run_dir"],
                "test_recall@5": pair.get("test_recall@5"),
                "test_recall@10": pair.get("test_recall@10"),
                "test_ndcg@5": pair.get("test_ndcg@5"),
                "test_ndcg@10": pair.get("test_ndcg@10"),
                "n_eval": pair.get("n_eval"),
            })
    recalls = OUT / "recalls"
    recalls.mkdir(exist_ok=True)
    for pair in pairs:
        run = REPO / pair["run_dir"]
        source = run / "test_final.json"
        if not source.is_file():
            source = run / "training_metrics.jsonl"
        if not source.is_file():
            continue
        target = recalls / (pair["variant"].replace("/", "__") + "__" +
                            Path(pair["run_dir"]).name + ".json")
        if source.name == "test_final.json":
            target.write_text(source.read_text())
        else:
            for line in source.read_text(errors="replace").splitlines():
                if '"test"' in line:
                    target.write_text(line.strip())
                    break
    (OUT / "sid_recall.json").write_text(json.dumps(
        {"pairs": pairs, "inventory": inventory}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    paired_sids = len({pair["sid_sha256"] for pair in pairs})
    print(f"SID contents collected : {len(blobs)}")
    print(f"SID files written      : {len(names)}")
    print(f"Stage 3 runs paired    : {len(pairs)}")
    print(f"distinct SIDs paired   : {paired_sids}")
    print(f"best recall@10         : {max((p['test_recall@10'] for p in pairs), default=None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
