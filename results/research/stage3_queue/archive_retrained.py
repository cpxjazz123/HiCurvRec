"""Archive a retrained legacy arm with its four entries, without re-running Stage 3.

The Stage 3 recall of these arms was measured on a SID table that no longer
exists, so the recall file stays as the historical record and carries a
provenance block that says so: `paired_sid_sha256` is null and the ledger entry
is marked `recall_paired: false`.  Nothing here claims a measured pairing.

    python3 results/research/stage3_queue/archive_retrained.py [--dry-run]
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "results" / "research" / "issue139"))
import r2_layout as layout  # noqa: E402

RESULTS = REPO / "results" / "stage2_RQ-VAE"
TREES = REPO / "stage2_RQ-VAE"
STAGE3 = REPO / "results" / "stage3_T5Train"
STAGING = Path(__file__).resolve().parent / "_stage"
FEASIBILITY = Path("/tmp/retrain_feasibility.json")
COPY_DIRS = ("model", "scripts", "tests")
RETRAINED = {"TIGER_BEHAVIOR_R48", "TIGER_BEHAVIOR_SAMPLE_RQ-VAE", "TIGER_FORK_RQ-VAE",
             "TIGER_HYPERBOLIC_RQ-VAE", "TIGER_behavior_residual_geometry_transport",
             "TIGER_bootstrap_code_stability", "TIGER_c106_directed_assignment_usage_balance",
             "TIGER_c106_directed_codeword_margin_geometry",
             "TIGER_c1_behaviour_conditioned_residual_target", "curvature_RQ-VAE",
             "curvature_RQ-VAE_iter48"}
RETRAIN_EPOCH = 1794184800  # 2026-11-09 11:40 local, when this batch started
HISTORICAL = json.loads(FEASIBILITY.read_text()) if FEASIBILITY.is_file() else []
def _best_arms() -> dict:
    """Many Stage 3 arms share one Stage 2 SID tree; keep the best recall of each."""
    best = {}
    for row in HISTORICAL:
        path = row.get("sid_path")
        if not path:
            continue
        best.setdefault(path, []).append(row["arm"])
    out = {}
    for path, arms in best.items():
        scored = []
        for arm in arms:
            for found in glob.glob(str(STAGE3 / arm / "logs" / "*" / "*" / "test_final.json")):
                try:
                    scored.append((json.loads(Path(found).read_text()).get("test_recall@10", -1), arm, found))
                except Exception:
                    continue
        if scored:
            value, arm, found = max(scored)
            out[path] = (arm, found)
    return out


TO_STAGE3 = _best_arms()


def source_entries(tree: Path) -> list[Path]:
    entries = [tree / "train_rqvae.py"] + [tree / name for name in COPY_DIRS]
    entries += [path for path in tree.glob("*.py") if path.name != "train_rqvae.py"]
    return [path for path in entries if path.exists()]


def stage(rel: str, sid: Path, *, dry_run: bool) -> str:
    key = rel.replace("/", "__")
    ledger = layout.Ledger()
    number = ledger.assign(key)
    entry = ledger.entries.get(number, {})
    refresh = "--refresh" in sys.argv
    if not refresh and entry.get("stage") == "stage2-retrain" and entry.get("files") and entry.get("recall_source"):
        print("%s already archived as %s" % (rel, number))
        return number
    tree_name, _, variant = rel.partition("/")
    checkpoint = sid.parent / "rqvae_best.pth"
    if not checkpoint.is_file():
        print("%s has no checkpoint yet, skipping" % rel)
        return ""

    staging = STAGING / key
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    code = staging / key
    code.mkdir()
    for source in source_entries(TREES / tree_name):
        target = code / source.name
        if source.is_dir():
            shutil.copytree(source, target,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "logs"))
        else:
            shutil.copy2(source, target)
    shutil.copy2(sid, staging / layout.SID)
    shutil.copy2(checkpoint, staging / layout.CHECKPOINT)

    stage3_arm, recall_path = TO_STAGE3.get(rel, (None, None))
    recall_path = Path(recall_path) if recall_path else None
    block = {"_provenance": {
        "recall_source": str(recall_path.relative_to(REPO)) if recall_path else None,
        "stage3_arm": stage3_arm,
        "sid_sha256_of_this_arm": layout.sha256(sid),
        "paired_sid_sha256": None,
        "note": "recall measured on a destroyed SID, retraining produced a new one",
    }}
    if recall_path:
        block.update(json.loads(recall_path.read_text()))
    (staging / layout.RECALL).write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")

    files = {str(path.relative_to(staging)): layout.sha256(path)
             for path in sorted(staging.rglob("*")) if path.is_file()}
    target = layout.upload_directory(staging, number, dry_run=dry_run)
    if not dry_run:
        ledger.record(number, key, files, {
            "stage": "stage2-retrain",
            "recall_paired": False,
            "recall_source": str(recall_path.relative_to(REPO)) if recall_path else None,
            "arm": key,
        })
        ledger.save()
    print("%s -> %s %s (%d files)" % (rel, number, target, len(files)))
    return number


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    done = []
    for path in sorted(RESULTS.glob("*")):
        if not path.is_dir() or path.name not in RETRAINED:
            continue
        running = any((Path("/proc") / pid / "cwd").resolve() == path
                      for pid in __import__("subprocess").run(
                          ["pgrep", "-f", "bin/python -u train_rqvae"],
                          capture_output=True, text=True).stdout.split())
        for sid in sorted(path.rglob(layout.SID)):
            if sid.stat().st_mtime < RETRAIN_EPOCH or running:
                continue
            rel = str(sid.parent.relative_to(RESULTS))
            number = stage(rel, sid, dry_run=dry_run)
            if number:
                done.append((rel, number))
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
