"""Archive one finished Stage 3 run into the numbered layout.

The rule (CLAUDE.md 6b): every time a Stage 3 run finishes, the four pieces are
archived together -- the Stage 2 code it used, the SID table, the checkpoint that
exported that table, and the complete recall block. The bucket's first level is a
two-digit number and each number holds exactly those four entries:

    r2:genrec/<NN>/<mechanism>/        complete bound source/config,
                                       stage2_source.tar.gz, archive_binding.json
    r2:genrec/<NN>/item_sids.json
    r2:genrec/<NN>/rqvae_best.pth
    r2:genrec/<NN>/recall.json         every key of test_final.json

What the number refers to, and the checksum of every uploaded file, goes into
`r2_ledger.json` beside this script.

    python3 results/research/issue139/upload_stage3.py F0
    python3 results/research/issue139/upload_stage3.py F0 --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
REPO = RESEARCH.parents[2]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(REPO))

from tools.issue252.archive_audit import validate_binding  # noqa: E402

from r2_layout import Ledger, sha256, upload_directory  # noqa: E402

RESULTS = REPO / "results"
TREES = REPO / "stage2_RQ-VAE"
STAGE3_LOGS = RESULTS / "stage3_T5Train"



def _latest_test_final(arm: str) -> Path | None:
    """Resolve one exact completed run; never pick a generation by timestamp."""
    found = set()
    for name in (arm, arm.replace("/", "__")):
        flat = STAGE3_LOGS / name / "test_final.json"
        if flat.is_file():
            found.add(flat.resolve())
        found.update(path.resolve() for path in
                     (STAGE3_LOGS / name).glob("logs/*/*/test_final.json"))
    if len(found) > 1:
        raise ValueError("ambiguous Stage3 generations for " + arm)
    return next(iter(found), None)


def archive_key(arm: str, sid_path: Path) -> str:
    """The ledger key: the arm and the digest of the SID table it archived.

    A re-run of the same arm does not reproduce the SID (the codebook initialisation
    is not seeded), so keying by the arm name alone would overwrite the earlier
    archive and its recall the moment a second run finished. The digest makes the key
    name what it holds, and an identical SID still maps to the same number.
    """
    return "%s/%s" % (arm.replace("/", "__"), sha256(Path(sid_path))[:12])


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError("missing necessary %s: %s" % (label, path))
    return path


def _input_path(block: dict, field: str) -> Path:
    if not block.get(field):
        raise ValueError("Stage3 final record is missing input provenance: " + field)
    path = Path(block[field])
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _source_tar(path: Path, files: dict[str, Path]) -> None:
    """Stable tar order, metadata and gzip header; no source mtimes/UIDs leak in."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                for name, source in sorted(files.items()):
                    data = source.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    bundle.addfile(info, io.BytesIO(data))


def stage(arm: str, staging: Path, tree: Path | None = None) -> tuple[dict, str]:
    """Stage exactly four entries after verifying the actual Stage3 input generation.

    The final record names exact inputs; directory names, mtimes, nearby checkpoints
    and ``--tree`` never substitute for their source/checkpoint/SID identities.
    Missing provenance from old runs requires independent evidence, not guessing.
    """
    stage3 = _latest_test_final(arm)
    if stage3 is None:
        raise ValueError("missing necessary Stage3 test_final.json for " + arm)
    _required(stage3, "Stage3 test_final.json")
    block = json.loads(stage3.read_text(encoding="utf-8"))
    sid = _required(_input_path(block, "sid_path"), "SID")
    checkpoint = _required(_input_path(block, "stage2_checkpoint"), "Stage2 checkpoint")
    if block.get("sid_sha256") != sha256(sid):
        raise ValueError("Stage3 SID sha256 mismatch")
    if block.get("stage2_checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Stage3 checkpoint sha256 mismatch")
    binding = _required(_input_path(block, "archive_binding"), "archive binding")
    if block.get("archive_binding_sha256") != sha256(binding):
        raise ValueError("Stage3 archive binding sha256 mismatch")
    receipt, sources = validate_binding(binding, sid, checkpoint, tree)
    if block.get("stage2_run_id") != receipt["run_id"]:
        raise ValueError("Stage3 Stage2 run identity mismatch")
    evidence_key = "stage3_archive_evidence"
    if evidence_key in block:
        raise ValueError("refusing to overwrite existing final-test key: " + evidence_key)
    evidence = {}
    for name in ("HG_Rec.log", "training_metrics.jsonl"):
        path = _required(stage3.parent / name, "Stage3 " + name)
        data = path.read_bytes()
        if not data:
            raise ValueError("missing necessary nonempty Stage3 " + name)
        content = data.decode("utf-8")
        if name == "training_metrics.jsonl":
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
            if not records or records[-1].get("event") != "train_end":
                raise ValueError("Stage3 metrics stream has no final train_end")
        evidence[name] = {"content": content,
                          "sha256": hashlib.sha256(data).hexdigest()}
    block[evidence_key] = evidence
    staging = Path(staging)
    if staging.exists() and any(staging.iterdir()):
        raise ValueError("staging must be empty to preserve the four-entry layout")
    staging.mkdir(parents=True, exist_ok=True)
    code = staging / arm.replace("/", "__")
    code.mkdir()
    expected_sources = {
        **receipt["candidate_sources"], **receipt["infrastructure_sources"],
        **receipt["repository_configs"],
        "archive_binding.json": block["archive_binding_sha256"],
        "export_manifest.json": receipt["export_manifest"]["sha256"],
    }
    for name, source in sorted(sources.items()):
        destination = code / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != expected_sources[name]:
            raise ValueError("source changed while staging: " + name)
    # Archive the staged bytes, not mutable source files reread after copying.
    staged_sources = {name: code / name for name in sources}
    _source_tar(code / "stage2_source.tar.gz", staged_sources)
    shutil.copy2(sid, staging / "item_sids.json")
    shutil.copy2(checkpoint, staging / "rqvae_best.pth")
    if sha256(staging / "item_sids.json") != block["sid_sha256"]:
        raise ValueError("SID changed while staging")
    if sha256(staging / "rqvae_best.pth") != block["stage2_checkpoint_sha256"]:
        raise ValueError("checkpoint changed while staging")
    (staging / "recall.json").write_text(
        json.dumps(block, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {str(path.relative_to(staging)): sha256(path)
             for path in sorted(staging.rglob("*")) if path.is_file()}
    return files, archive_key(arm, sid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--tree", default="",
                        help="Exact executed Stage2 source tree; must match its "
                             "checkpoint-bound archive receipt")
    args = parser.parse_args()

    staging = RESEARCH / "_stage" / ("r2_arm_%s" % args.arm.replace("/", "__"))
    shutil.rmtree(staging, ignore_errors=True)
    try:
        files, key = stage(args.arm, staging, Path(args.tree).resolve() if args.tree else None)
        ledger = Ledger()
        number = ledger.assign(key)
        target = upload_directory(staging, number, dry_run=args.dry_run)
        ledger.record(number, key, files, {"arm": args.arm,
                                           "recall_keys": sorted(json.loads(
                                               (staging / "recall.json").read_text()))})
        if not args.dry_run:
            # The ledger must cover the directory, not just this staging set:
            # a number that earlier uploads filled tree-and-all keeps those
            # objects, and they need a checksum too.
            result = ledger.reconcile(number)
            ledger.save()
            for relative in sorted(result["added"]):
                print("  recorded without uploading: %s/%s" % (number, relative))
            for relative in result["recorded_but_absent"]:
                print("  WARNING recorded but not in the bucket: %s/%s" % (number, relative))
        print("%s: %d files, recall keys %d -> %s" % (
            args.arm, len(files), len(json.loads((staging / "recall.json").read_text())), target))
        print(("would upload to " if args.dry_run else "uploaded to ") + target)
    finally:
        if not args.keep:
            shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
