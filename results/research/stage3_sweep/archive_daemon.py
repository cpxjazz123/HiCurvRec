"""Archive every finished Stage3 arm to R2 as soon as its result lands.

The sweep runs for a day; this daemon keeps the §6b four-piece archive in step with it
instead of leaving the upload to the end.  Arms without a Stage2 source tree are listed in
archive_gaps.json and skipped (their historical archives predate the tree removal).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE2 = REPO / "results" / "stage2_RQ-VAE"
STAGE3 = REPO / "results" / "stage3_T5Train"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
UPLOAD = REPO / "results" / "research" / "issue139" / "upload_stage3.py"
LOG = Path(__file__).resolve().parent / "archive.log"
SOURCE = REPO / "stage2_RQ-VAE"
PYTHON = sys.executable

#: Trees that serve several arms (the arm name is not the tree name).
SHARED = {"TIGER_graphlet_role_rq": ("TIGER_graphlet_role_E", "TIGER_graphlet_role_T",
                                     "TIGER_graphlet_role_R", "TIGER_graphlet_role_Q",
                                     "TIGER_graphlet_role_QR"),
          "_archive/TIGER_g4_closure_rq": ("TIGER_g4_closure_A", "TIGER_g4_closure_B",
                                          "TIGER_g4_closure_C", "TIGER_g4_closure_D")}


def stamp(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def tree_for(arm: str) -> Path | None:
    for tree, arms in SHARED.items():
        if arm in arms:
            candidate = SOURCE / tree
            if (candidate / "train_rqvae.py").is_file():
                return candidate
    for candidate in (SOURCE / arm, *SOURCE.glob("_archive/" + arm), *SOURCE.glob("baseline/" + arm)):
        if (candidate / "train_rqvae.py").is_file():
            return candidate
    return None


def archived() -> set[str]:
    payload = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    names = set()
    for entry in payload.values():
        arm = (entry.get("meta") or {}).get("arm") or entry.get("arm")
        if arm:
            names.add(arm)
    return names


def results() -> dict[str, Path]:
    found = {}
    for directory in sorted(STAGE3.iterdir()):
        if not directory.is_dir():
            continue
        hits = (sorted(directory.glob("logs/*/test_final.json"))
                + sorted(directory.glob("logs/*/*/test_final.json")))
        if hits:
            found[directory.name] = hits[-1]
    return found


def main() -> None:
    stamp("archive daemon start")
    while True:
        done = archived()
        for arm, result in results().items():
            if arm in done:
                continue
            tree = tree_for(arm)
            if tree is None:
                continue
            command = [PYTHON, str(UPLOAD), arm, "--tree", str(tree)]
            completed = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
            tail = (completed.stdout or completed.stderr).strip().splitlines()
            stamp(f"{arm}: {' | '.join(tail[-2:]) if tail else 'no output'}")
            done.add(arm)
        time.sleep(300)


if __name__ == "__main__":
    main()
