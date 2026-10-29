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
import datetime
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


def table_facts(payload: bytes) -> dict:
    """The numbers Stage 3 derived FROM the SID table it read."""
    table = json.loads(payload.decode("utf-8"))
    rows = [table[key] for key in sorted(table, key=int)]
    tokens = [value + 1 for row in rows for value in row]
    return {"items": len(rows), "width": len(rows[0]), "max_token": max(tokens),
            "unique_rows": len({tuple(row) for row in rows})}


def expected_vocab_size(payload: bytes, n_user_tokens: int = 1) -> int:
    """The vocabulary Stage 3 builds from this table: max token + the user tokens."""
    facts = table_facts(payload)
    # max_sid_token is the highest token (max(raw) + 1); the vocabulary adds the
    # user bucket and the eos token: max(raw) + n_user_tokens + 3.
    return facts["max_token"] + int(n_user_tokens) + 2


def logged_vocab_size(run_dir: str) -> int | None:
    """The ``vocab_size`` the RUN recorded, from its own configuration log."""
    import re
    relative = Path(run_dir)
    text = ""
    local = REPO / relative / "HG_Rec.log"
    if local.is_file():
        text = local.read_text(errors="replace")
    else:
        # Every commit that touched the log is tried: a later commit may hold an
        # EMPTY copy (the file was committed after the run without its content),
        # and taking the newest one alone lost the vocabulary that an older
        # commit still had.
        commits = git("log", "--all", "--format=%H", "--",
                      str(relative / "HG_Rec.log")).splitlines()
        best = ""
        for commit in commits:
            blob = git_bytes("show", commit + ":" + str(relative / "HG_Rec.log")).decode(
                errors="replace")
            if len(blob) > len(best):
                best = blob
            if "'vocab_size'" in blob:
                break
        text = best
    match = re.search(r"'vocab_size':\s*(\d+)", text)
    return int(match.group(1)) if match else None


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


def run_timestamp(run_dir: str) -> float | None:
    """The epoch seconds of a Stage 3 run directory (``Oct-03-2026_20-34-59``)."""
    import datetime

    name = Path(run_dir).name
    for pattern in ("%b-%d-%Y_%H-%M-%S", "%Y-%m-%d_%H-%M-%S"):
        try:
            stamp = datetime.datetime.strptime(name, pattern)
        except ValueError:
            continue
        return stamp.timestamp()
    return None


def repo_relative(path: str) -> str:
    """The repository-relative form of a recorded path.

    A run records the path it read, and the trainer resolves its root from the
    file it lives in, so the recorded path uses the LOGICAL root (/home/...) while
    the repository may sit at another physical path (/fs04/...). ``git show``
    needs a repository-relative path, and an absolute one silently fails, which
    sent every such run to the working tree -- and therefore to a table it never
    read. The tail after the repository directory name is the same path.
    """
    text = str(path)
    if not text.startswith("/"):
        return text
    for root in (str(REPO), str(REPO.resolve())):
        if text.startswith(root.rstrip("/") + "/"):
            return text[len(root.rstrip("/")) + 1:]
    marker = "/" + REPO.name + "/"
    index = text.find(marker)
    if index >= 0:
        return text[index + len(marker):]
    return text


def resolve_path_to_content(path: str, before: float | None = None
                             ) -> tuple[str | None, bytes | None, str | None, str]:
    """The content of a SID path *as it was at that moment*.

    A SID path is overwritten: the same ``results/stage2_RQ-VAE/<arm>/item_sids.json``
    can hold a different table after every Stage 2 run. Pairing a Stage 3 run with
    the current file therefore attributes a run to a table it never saw. The
    version that existed at the run's date is the one the run generated from, so
    the history is consulted with a date filter.
    """
    candidate = REPO / path
    if candidate.is_file() and before is None:
        payload = candidate.read_bytes()
        return sha256_bytes(payload), payload, None, "working_tree"
    history = git("log", "--all", "--format=%H|%cI", "--", path).splitlines()
    best = None
    for line in history:
        commit, _, date = line.partition("|")
        if before is not None:
            try:
                when = datetime.datetime.fromisoformat(date).timestamp()
            except ValueError:
                continue
            if when > before:
                continue
        best = commit
        break
    if best:
        # The newest commit may hold an EMPTY copy (the file was committed after
        # the run without its content), so every commit is tried until one has
        # the table in it.
        for commit in [best] + [c for c in history if c != best]:
            payload = git_bytes("show", commit + ":" + path)
            if payload:
                resolution = "at_run_date" if commit == best else "at_run_date_nonempty"
                return sha256_bytes(payload), payload, commit, resolution
    if history:
        # The file was committed after the run: the earliest surviving version is the
        # only copy of the table the run could have used.
        for commit in [history[-1].split("|", 1)[0]] + [c.split("|", 1)[0]
                                                        for c in reversed(history[:-1])]:
            payload = git_bytes("show", commit + ":" + path)
            if payload:
                return sha256_bytes(payload), payload, commit, "earliest_surviving"
    if candidate.is_file():
        payload = candidate.read_bytes()
        return sha256_bytes(payload), payload, None, "working_tree"
    return None, None, None, "unresolved"


def collect_stage3_runs() -> list:
    """Every Stage 3 run that recorded a test score, with the SID it used."""
    runs = []
    files = sorted(REPO.glob("results/stage3_T5Train/*/logs/*/*/training_metrics.jsonl"))
    for path in files:
        code_path = None
        code_digest = None
        test = None
        relative = path.relative_to(REPO)
        # results/stage3_T5Train/<variant>/logs/<dataset>/<run>/training_metrics.jsonl
        variant = relative.parts[2] if len(relative.parts) > 2 else ""
        run_dir = path.parent
        for line in path.read_text(errors="replace").splitlines():
            if '"train_start"' in line and code_path is None:
                try:
                    event = json.loads(line)
                    code_path = event.get("code_path")
                    code_digest = event.get("code_path_sha256")
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
            test["code_path_sha256"] = code_digest
            runs.append({"variant": variant, "run_dir": str(run_dir.relative_to(REPO)),
                          "code_path": code_path, "metrics": test})
    return runs



def history_stage3_runs() -> list:
    """Stage 3 runs that only exist in the git history.

    The working tree keeps a fraction of the runs: ``results/stage3_T5Train`` is
    large and mostly untracked. The history has both the per-run
    ``training_metrics.jsonl`` (which names the ``code_path`` a run generated from)
    and, for many runs, the ``test_final.json`` beside it.
    """
    import re

    entries: dict[str, dict] = {}
    for line in git("rev-list", "--all", "--objects").splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        blob, path = parts
        if not re.search(r"stage3_T5Train/.+/(training_metrics\.jsonl|test_final\.json)$", path):
            continue
        run_dir = str(Path(path).parent)
        entry = entries.setdefault(run_dir, {})
        kind = "metrics" if path.endswith("training_metrics.jsonl") else "final"
        entry.setdefault(kind, blob)

    runs = []
    for run_dir, blobs in entries.items():
        code_path = None
        code_digest = None
        metrics = {}
        if blobs.get("metrics"):
            for line in git("cat-file", "blob", blobs["metrics"]).splitlines():
                if '"train_start"' in line and code_path is None:
                    try:
                        event = json.loads(line)
                        code_path = event.get("code_path")
                        code_digest = event.get("code_path_sha256")
                    except ValueError:
                        pass
                elif '"test"' in line and not metrics:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    metrics = {"test_recall@5": event.get("test_recall_at_5"),
                               "test_recall@10": event.get("test_recall_at_10"),
                               "test_ndcg@5": event.get("test_ndcg_at_5"),
                               "test_ndcg@10": event.get("test_ndcg_at_10"),
                               "n_eval": event.get("n_eval")}
        if blobs.get("final"):
            try:
                final = json.loads(git("cat-file", "blob", blobs["final"]))
            except ValueError:
                final = {}
            for key in ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10",
                        "n_eval"):
                if final.get(key) is not None:
                    metrics[key] = final[key]
            code_path = code_path or final.get("code_path")
        if not code_path or metrics.get("test_recall@10") is None:
            continue
        relative = run_dir.split("results/", 1)[-1]
        variant = relative.split("/")[1] if "/" in relative else ""
        metrics["code_path_sha256"] = code_digest
        runs.append({"variant": variant, "run_dir": "results/" + relative,
                     "code_path": code_path, "metrics": metrics})
    return runs

def write_sid_files(blobs: dict) -> dict:
    """Write every distinct SID table into ``sids/`` and report the file name."""
    SID_DIR.mkdir(parents=True, exist_ok=True)
    names = {}
    for digest, entry in blobs.items():
        path = entry["paths"][0]
        slug = path.replace("/", "__").replace(".json", "")
        # The name is bounded: a SID that lives inside this directory would
        # otherwise produce a name that grows with every collection and finally
        # exceeds what the filesystem accepts.
        if len(slug) > 120:
            slug = slug[:100] + "_" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        name = f"{slug}__{digest[:12]}.json"
        target = SID_DIR / name
        if not target.is_file():
            payload = None
            if entry["blob"]:
                payload = git_bytes("cat-file", "blob", entry["blob"])
            else:
                source = REPO / path
                if source.is_file():
                    payload = source.read_bytes()
            if not payload:
                # The table is neither in the working tree nor in the history under
                # a blob we can read: writing an EMPTY file would create a file
                # whose content is not the table, and the pruner would then delete
                # the row's file while the row still points at it.
                continue
            target.write_bytes(payload)
        # keyed by the content sha256: that is what the pairing uses
        names[entry["sha256"]] = name
    return names


def prune_unpaired(names: dict, pairs: list) -> int:
    """Drop the SID files that no Stage 3 run generated from, and the duplicates.

    A SID is a table, identified by its content: the same table reached under two
    path names is ONE SID, so it must be ONE file. The canonical name (the first
    path the table was seen at) is kept and the other copies go.

    The unpaired tables stay recoverable: they are in the git history, and
    ``collect.py`` writes them again on the next run without this step.
    """
    keep = {pair["sid_sha256"] for pair in pairs}
    removed = 0
    canonical = {digest: name for digest, name in names.items()}
    for path in sorted(SID_DIR.glob("*.json")):
        try:
            digest = sha256_bytes(path.read_bytes())
        except OSError:
            continue
        if digest in keep:
            # The table of a paired SID stays, whatever it is called: a row of the
            # pair table points at a file by name, and a table that is not on disk
            # would make the row unreadable.
            continue
        path.unlink()
        removed += 1
    return removed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blobs = collect_sid_blobs()
    collect_working_tree_sids(blobs)
    runs = collect_stage3_runs()
    known = {(run["variant"], run["run_dir"]) for run in runs}
    for run in history_stage3_runs():
        if (run["variant"], run["run_dir"]) in known:
            continue
        runs.append(run)
        known.add((run["variant"], run["run_dir"]))
    print(f"Stage 3 runs considered  : {len(runs)}")

    by_sha = {entry["sha256"]: digest for digest, entry in blobs.items()}
    pairs = []
    unpaired_runs = []
    disputed_runs = []
    unverified_runs = []
    for run in runs:
        code = repo_relative(run["code_path"])
        when = run_timestamp(run["run_dir"])
        digest, payload, blob, resolution = resolve_path_to_content(code, before=when)
        if resolution == "unresolved":
            unpaired_runs.append({"variant": run["variant"], "run_dir": run["run_dir"],
                                  "code_path": code, "reason": "the SID table this run used is "
                                  "neither in the working tree nor in the git history",
                                  **run["metrics"]})
            continue
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
        # The run recorded the vocabulary it built FROM the table it read, and
        # that vocabulary is a function of the table's highest token. A table
        # whose token range does not reproduce the run's own number is not the
        # table that run read: the pairing is disputed and the run is reported
        # as unpaired rather than credited with a table it never saw.
        recorded_digest = (run.get("metrics") or {}).get("code_path_sha256") or ""
        logged = logged_vocab_size(run["run_dir"])
        if logged is not None:
            expected = expected_vocab_size(payload)
            if expected != logged:
                disputed_runs.append({
                    "variant": run["variant"], "run_dir": run["run_dir"],
                    "code_path": code, "sid_sha256": digest,
                    "sid_resolution": resolution,
                    "logged_vocab_size": logged, "table_vocab_size": expected,
                    "reason": "the vocabulary the run recorded does not match the "
                              "paired table's token range: this run read another table",
                    **run["metrics"]})
                continue
        if recorded_digest:
            # The run recorded the digest of the table it read: the pair is a
            # fact, and a mismatch means the run read another table.
            if recorded_digest != digest:
                disputed_runs.append({
                    "variant": run["variant"], "run_dir": run["run_dir"],
                    "code_path": code, "sid_sha256": digest,
                    "sid_resolution": resolution,
                    "logged_vocab_size": "", "table_vocab_size": "",
                    "reason": "the digest the run recorded does not match the "
                              "paired table",
                    **run["metrics"]})
                continue
            confidence = "verified_digest"
        elif logged is not None:
            confidence = "verified_vocabulary"
        else:
            confidence = "date_resolved_only"
        pairs.append({"sid_sha256": digest, "sid_paths": blobs[by_sha[digest]]["paths"],
                      "variant": run["variant"], "run_dir": run["run_dir"],
                      "code_path": code, "sid_commit": blob,
                      "sid_resolution": resolution,
                      "pair_confidence": confidence,
                      **run["metrics"]})
        if logged is None and not recorded_digest:
            # No fingerprint: the link rests on the path's version at the run's
            # date, and that argument has been seen to fail (ten runs were
            # refuted by their own vocabulary). Such a pair is kept in the
            # record, but it is NOT counted as confirmed.
            unverified_runs.append({
                "variant": run["variant"], "run_dir": run["run_dir"],
                "code_path": code, "sid_sha256": digest,
                "sid_resolution": resolution,
                "reason": "the run left no vocabulary, so only the path's version "
                          "at the run's date supports the link, and that argument "
                          "has been refuted for other runs",
                **run["metrics"]})

    # Only a pair whose SID the run itself vouched for is a pair. The rest are
    # written to sid_recall_unverified.csv: they are evidence, not conclusions.
    confirmed_pairs = [pair for pair in pairs
                       if pair.get("pair_confidence") in ("verified_digest",
                                                         "verified_vocabulary")]
    pairs = confirmed_pairs

    # One SID, one recall: when several runs generated from the same table, the
    # highest recall is kept as the pair and the other runs are written aside.
    grouped: dict[str, list] = {}
    for pair in pairs:
        grouped.setdefault(pair["sid_sha256"], []).append(pair)
    alternatives = []
    kept = []
    for digest, group in grouped.items():
        group.sort(key=lambda item: -(item.get("test_recall@10") or 0))
        best = dict(group[0])
        best["sid_runs"] = len(group)
        best["sid_recall_range"] = [min(item.get("test_recall@10") or 0 for item in group),
                                    max(item.get("test_recall@10") or 0 for item in group)]
        kept.append(best)
        for other in group[1:]:
            alternatives.append({"sid_sha256": digest, "variant": other["variant"],
                                 "run_dir": other["run_dir"],
                                 "test_recall@10": other.get("test_recall@10"),
                                 "test_recall@5": other.get("test_recall@5"),
                                 "test_ndcg@10": other.get("test_ndcg@10"),
                                 "n_eval": other.get("n_eval"),
                                 "note": "another run of the same SID table; the highest "
                                         "recall is kept as the pair"})
    pairs = kept

    names = write_sid_files(blobs)
    # A pair whose table cannot be materialised is not a measurable pair: it is
    # parked with the reason instead of appearing in the collection with a file
    # that does not exist.
    unmaterialised = [pair for pair in pairs if not names.get(pair["sid_sha256"])]
    pairs = [pair for pair in pairs if names.get(pair["sid_sha256"])]
    for pair in unmaterialised:
        unverified_runs.append({
            "variant": pair["variant"], "run_dir": pair["run_dir"],
            "code_path": pair["code_path"], "sid_sha256": pair["sid_sha256"],
            "sid_resolution": pair.get("sid_resolution"),
            "reason": "the SID table of this run exists neither in the working tree "
                      "nor in a readable blob of the history",
            **pair})

    inventory = []
    for digest, entry in blobs.items():
        # the pairing is keyed by the content hash, not by the git blob id
        paired = [pair for pair in pairs if pair["sid_sha256"] == entry["sha256"]]
        best = max((pair["test_recall@10"] for pair in paired), default=None)
        if entry["sha256"] not in names:
            # the table could not be materialised (an empty blob everywhere): it
            # is not a pair and has no file to point at
            continue
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
        fields = ["sid_sha256", "sid_file", "sid_paths", "variant", "run_dir", "code_path",
                  "sid_resolution", "pair_confidence", "sid_runs", "sid_recall_range",
                  "test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10", "n_eval"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in sorted(pairs, key=lambda item: -(item.get("test_recall@10") or 0)):
            writer.writerow({
                "sid_sha256": pair["sid_sha256"],
                "sid_file": names.get(pair["sid_sha256"], ""),
                "sid_paths": " | ".join(pair["sid_paths"]), "variant": pair["variant"],
                "run_dir": pair["run_dir"], "code_path": pair["code_path"],
                "sid_resolution": pair.get("sid_resolution", ""),
                "pair_confidence": pair.get("pair_confidence", ""),
                "sid_runs": pair.get("sid_runs", 1),
                "sid_recall_range": json.dumps(pair.get("sid_recall_range", [])),
                "test_recall@5": pair.get("test_recall@5"),
                "test_recall@10": pair.get("test_recall@10"),
                "test_ndcg@5": pair.get("test_ndcg@5"),
                "test_ndcg@10": pair.get("test_ndcg@10"),
                "n_eval": pair.get("n_eval"),
            })
    recalls = OUT / "recalls"
    recalls.mkdir(exist_ok=True)
    for path in recalls.iterdir():
        path.unlink()
    for pair in pairs:
        if not names.get(pair["sid_sha256"]):
            # the table of this pair could not be materialised: it is not a
            # measurable pair and must not appear in the collection
            continue
        run = REPO / pair["run_dir"]
        # The metrics live in the pair already (they came from the run's own record).
        # The raw file is merged in when the run still exists locally, so the
        # checkpoint path and any extra field survive; a history-only run has no
        # local file and its recall is written from the pair itself.
        payload = {"test_recall@5": pair.get("test_recall@5"),
                   "test_recall@10": pair.get("test_recall@10"),
                   "test_ndcg@5": pair.get("test_ndcg@5"),
                   "test_ndcg@10": pair.get("test_ndcg@10"),
                   "n_eval": pair.get("n_eval")}
        final = run / "test_final.json"
        metrics_file = run / "training_metrics.jsonl"
        if final.is_file():
            try:
                payload.update(json.loads(final.read_text()))
            except ValueError:
                pass
        elif metrics_file.is_file():
            for line in metrics_file.read_text(errors="replace").splitlines():
                if '"train_start"' in line:
                    try:
                        payload.setdefault("train_start", json.loads(line))
                    except ValueError:
                        pass
                if '"test"' in line:
                    try:
                        payload.setdefault("test_event", json.loads(line))
                    except ValueError:
                        pass
        target = recalls / (pair["variant"].replace("/", "__") + "__" +
                            Path(pair["run_dir"]).name + "__" +
                            pair["sid_sha256"][:12] + ".json")
        payload["paired_sid"] = {
            "sha256": pair["sid_sha256"],
            "file": names[pair["sid_sha256"]],
            "paths": pair["sid_paths"],
            "code_path": pair["code_path"],
            "resolution": pair.get("sid_resolution"),
        }
        payload["stage3_run"] = {"variant": pair["variant"], "run_dir": pair["run_dir"]}
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (OUT / "sid_recall.json").write_text(json.dumps(
        {"pairs": pairs, "inventory": inventory}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    # The runs that shared a table with the pair are written aside. The file used
    # to be left untouched by this script, so it kept the rows of an older,
    # mis-paired collection and named runs that the main table pairs with another
    # table: a SID has one recall, and the other recalls of the same table live
    # here.
    excluded = OUT / "excluded"
    excluded.mkdir(exist_ok=True)
    # The runs whose table exists neither in the working tree nor in the history
    # cannot be paired at all; they are parked with the reason.
    with (excluded / "stage3_runs_without_sid.csv").open("w", newline="",
                                                         encoding="utf-8") as handle:
        fields = ["variant", "run_dir", "code_path", "test_recall@10", "n_eval", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in unpaired_runs:
            writer.writerow({key: row.get(key) for key in fields})
    with (excluded / "sid_recall_unverified.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["variant", "run_dir", "code_path", "sid_sha256", "sid_resolution",
                  "test_recall@10", "n_eval", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in unverified_runs:
            writer.writerow({key: row.get(key) for key in fields})
    fields = ["variant", "run_dir", "code_path", "sid_sha256", "sid_resolution",
              "logged_vocab_size", "table_vocab_size", "test_recall@10", "n_eval", "reason"]
    with (excluded / "sid_recall_disputed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in disputed_runs:
            writer.writerow({key: row.get(key) for key in fields})
    with (excluded / "sid_recall_alternatives.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sid_sha256", "variant", "run_dir", "test_recall@5", "test_recall@10",
                  "test_ndcg@10", "n_eval", "note"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in alternatives:
            writer.writerow(row)

    if "--paired-only" in sys.argv:
        removed = prune_unpaired(names, pairs)
        keep = {pair["sid_sha256"] for pair in pairs}
        inventory = [row for row in inventory if row["sid_sha256"] in keep]
        with (OUT / "sid_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
            writer.writeheader()
            for row in inventory:
                row = dict(row)
                row["paths"] = " | ".join(row["paths"])
                row["levels"] = json.dumps(row["levels"], ensure_ascii=False)
                writer.writerow(row)
        print(f"unpaired SID files removed: {removed}")

    paired_sids = len({pair["sid_sha256"] for pair in pairs})
    print(f"SID contents collected : {len(blobs)}")
    print(f"SID files written      : {len(names)}")
    print(f"Stage 3 runs paired    : {len(pairs)}")
    print(f"distinct SIDs paired   : {paired_sids}")
    print(f"best recall@10         : {max((p['test_recall@10'] for p in pairs), default=None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
