"""Audit whether each (SID, recall) pair in `sid_recall.csv` is a real pair.

Two independent criteria, both mechanical:

**Commit identity.** A pair is trusted only when the SID table and the Stage 3
result file that carries the recall were introduced by the SAME commit. For
every file the script asks git for its introduction commit
(``git log --diff-filter=A -1``) and its last-touching commit, in the *original*
repository path (the collected copies under ``analysis_sid_recall/sids/`` are
ignored, they only prove the content was collected). Paths that were never
tracked cannot pass this criterion and say so.

**Directory name.** Under ``results/stage2_RQ-VAE/`` the mechanism directory and
under ``results/stage3_T5Train/`` the variant directory must carry the same
name. Four classes are reported: ``EXACT`` (the stage-3 variant equals the
stage-2 relative path), ``FLATTENED`` (equal after ``/`` became ``__``),
``LEAF`` (only the last component is equal) and ``NONE``.

A secondary column checks the Stage 3 run's own record: the ``code_path`` its
``train_start`` event names must be one of the SID paths of the row.

Outputs: ``analysis_sid_recall/pair_audit.csv`` (one row per pair),
``pair_audit.json`` (counts and the commit-time table) and ``PAIR_AUDIT.md``.

Run:  python3 analysis_sid_recall/pair_audit.py
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_sid_recall"
COLLECTED = "analysis_sid_recall/sids/"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=False).stdout.strip()


def commit_of(path: str, introduction: bool) -> dict | None:
    """The commit that introduced (or last touched) a path, with its time."""
    args = ["log", "-1", "--format=%H%x09%ct%x09%an"]
    if introduction:
        args += ["--diff-filter=A"]
    args += ["--", path]
    out = git(*args)
    if not out:
        return None
    sha, seconds, author = out.split("\t")[:3]
    return {"sha": sha, "seconds": int(seconds), "author": author}


def classify_name(stage2_rel: str, variant: str) -> str:
    if stage2_rel == variant:
        return "EXACT"
    if stage2_rel.replace("/", "__") == variant:
        return "FLATTENED"
    if stage2_rel.split("/")[-1] == variant:
        return "LEAF"
    return "NONE"


def first_tracked(paths: list[str]) -> tuple[str, dict, dict] | None:
    """The earliest-introduced original path, with both commits."""
    candidates = []
    for path in paths:
        if path.startswith(COLLECTED):
            continue
        intro = commit_of(path, introduction=True)
        last = commit_of(path, introduction=False)
        if intro:
            candidates.append((intro["seconds"], path, intro, last))
    if not candidates:
        return None
    candidates.sort()
    _, path, intro, last = candidates[0]
    return path, intro, last


def main() -> int:
    rows = list(csv.DictReader((OUT / "sid_recall.csv").open()))
    audit = []
    for row in rows:
        paths = [part.strip() for part in row["sid_paths"].split("|") if part.strip()]
        stage2 = next((part for part in paths if "stage2_RQ-VAE/" in part), paths[0])
        stage2_rel = stage2.split("results/stage2_RQ-VAE/", 1)[-1]
        stage2_rel = stage2_rel.rsplit("/item_sids", 1)[0]
        variant = row["variant"]
        name_class = classify_name(stage2_rel, variant)

        sid_tracked = first_tracked(paths)
        run_dir = Path(row["run_dir"])
        recall_files = [str(run_dir / "training_metrics.jsonl"),
                        str(run_dir / "test_final.json")]
        recall_tracked = first_tracked(recall_files)

        same_commit = bool(sid_tracked and recall_tracked
                           and sid_tracked[1]["sha"] == recall_tracked[1]["sha"])
        order_ok = bool(sid_tracked and recall_tracked
                        and sid_tracked[1]["seconds"] <= recall_tracked[1]["seconds"])
        code_path_ok = any(row["code_path"] == part for part in paths)

        name_ok = name_class in ("EXACT", "FLATTENED")
        if name_ok and same_commit:
            verdict = "VERIFIED"
        elif name_ok and sid_tracked and recall_tracked and not same_commit:
            verdict = "NAME_OK_COMMITS_DIFFER"
        elif name_ok:
            verdict = "NAME_OK_UNTRACKED"
        elif same_commit:
            verdict = "COMMIT_OK_NAME_DIFFERS"
        else:
            verdict = "UNVERIFIED"

        audit.append({
            "sid_sha256": row["sid_sha256"],
            "variant": variant,
            "stage2_rel": stage2_rel,
            "name_class": name_class,
            "name_ok": name_ok,
            "sid_path": sid_tracked[0] if sid_tracked else None,
            "sid_intro_commit": sid_tracked[1]["sha"] if sid_tracked else None,
            "sid_intro_time": sid_tracked[1]["seconds"] if sid_tracked else None,
            "sid_last_commit": sid_tracked[2]["sha"] if sid_tracked and sid_tracked[2] else None,
            "recall_path": recall_tracked[0] if recall_tracked else None,
            "recall_intro_commit": recall_tracked[1]["sha"] if recall_tracked else None,
            "recall_intro_time": recall_tracked[1]["seconds"] if recall_tracked else None,
            "recall_last_commit": (recall_tracked[2]["sha"]
                                   if recall_tracked and recall_tracked[2] else None),
            "same_commit": same_commit,
            "sid_before_recall": order_ok,
            "code_path_ok": code_path_ok,
            "test_recall@10": row["test_recall@10"],
            "verdict": verdict,
        })

    with (OUT / "pair_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    counts = Counter(item["verdict"] for item in audit)
    names = Counter(item["name_class"] for item in audit)
    same = sum(1 for item in audit if item["same_commit"])
    tracked = sum(1 for item in audit if item["sid_intro_commit"])
    recall_tracked_count = sum(1 for item in audit if item["recall_intro_commit"])
    summary = {
        "pairs": len(audit),
        "verdicts": dict(counts),
        "name_classes": dict(names),
        "sid_tracked": tracked,
        "recall_tracked": recall_tracked_count,
        "same_commit": same,
        "name_ok": sum(1 for item in audit if item["name_ok"]),
        "code_path_ok": sum(1 for item in audit if item["code_path_ok"]),
        "sid_before_recall": sum(1 for item in audit if item["sid_before_recall"]),
        "verified": [item["variant"] for item in audit if item["verdict"] == "VERIFIED"],
    }
    (OUT / "pair_audit.json").write_text(json.dumps(summary, indent=2) + "\n",
                                        encoding="utf-8")

    lines = ["# (SID, recall) pair audit", "",
             f"{len(audit)} pairs. SID path tracked in git: {tracked}; "
             f"recall file tracked: {recall_tracked_count}; "
             f"same introduction commit: {same}; "
             f"stage-2/stage-3 name equal: {summary['name_ok']}; "
             f"the run's own code_path matches the SID path: {summary['code_path_ok']}.",
             "", "## Verdicts", "", "| verdict | pairs |", "|---|---|"]
    for name, value in counts.most_common():
        lines.append(f"| {name} | {value} |")
    lines += ["", "## Name classes", "", "| class | pairs |", "|---|---|"]
    for name, value in names.most_common():
        lines.append(f"| {name} | {value} |")
    lines += ["", "## Verified pairs", ""]
    for variant in summary["verified"][:40]:
        lines.append(f"* `{variant}`")
    if len(summary["verified"]) > 40:
        lines.append(f"* ... and {len(summary['verified']) - 40} more (see the CSV)")
    (OUT / "PAIR_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "verified"},
                     indent=1))
    print("written:", OUT / "pair_audit.csv", OUT / "PAIR_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
