"""Finish one Stage 2 arm: the layer record, the diagnostics and the conclusion.

Each of Issues #140-#143 asked for the same closing paperwork: the mandatory
layered SID record (CLAUDE.md 63) written before any Stage 3 exists, the
mechanism's own diagnostics, and a conclusion under `results/research/issue<NNN>/`.

    python3 results/research/issue143/finish.py M4_post_sid_af3

The record is produced by `tools.research.science.sid_layer_analysis` in the
project interpreter, exactly as the Stage 3 launcher would produce it, so the
diagnostics in the conclusion are the same numbers the registry would carry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "results" / "stage2_RQ-VAE"
TREE = REPO / "stage2_RQ-VAE"
PROJECT_PYTHON = Path("/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10")

ARMS = {
    "M1_learned_kappa": {"issue": 140, "event": "curvature"},
    "M2_radial_manifold": {"issue": 141, "event": "radial"},
    "M3_role_profiles": {"issue": 142, "event": "role"},
    "M4_post_sid_af3": {"issue": 143, "event": "post_sid"},
}


def metrics(arm: str) -> list[dict]:
    """The delivered run's stream: the split file when the shared one was mixed.

    A duplicate run on 2026-11-08 truncated the shared ``training_metrics.jsonl``
    and the earlier run kept appending its tail; ``run_metrics.jsonl`` holds the
    one complete run recovered by ``tools.research.stage2.run_metrics``.
    """
    logs = TREE / ("TIGER_" + arm) / "logs"
    for name in ("run_metrics.jsonl", "training_metrics.jsonl"):
        path = logs / name
        if path.is_file():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return []


def metrics_source(arm: str) -> str:
    logs = TREE / ("TIGER_" + arm) / "logs"
    for name in ("run_metrics.jsonl", "training_metrics.jsonl"):
        if (logs / name).is_file():
            return str(logs / name)
    return ""


def run_audit(arm: str) -> dict:
    """Every run the shared file held, and which of them is the delivered one."""
    path = TREE / ("TIGER_" + arm) / "logs" / "run_split.json"
    if not path.is_file():
        return {"available": False, "reason": "no run_split.json (stream was not shared)"}
    return {"available": True, "audit": json.loads(path.read_text())}


def layer_record(arm: str) -> dict:
    sid = RESULT / ("TIGER_" + arm) / "item_sids.json"
    if not sid.is_file():
        return {"written": False, "reason": "no item_sids.json: the run has not exported"}
    done = subprocess.run([str(PROJECT_PYTHON), "-m", "tools.research.science.sid_layer_analysis",
                           str(sid)], cwd=str(REPO), capture_output=True, text=True,
                          env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"}, timeout=3600)
    if done.returncode != 0:
        return {"written": False, "reason": (done.stderr or done.stdout).strip()[-600:]}
    import hashlib
    digest = hashlib.sha256(sid.read_bytes()).hexdigest()
    report = sid.parent / "stage2_metrics" / (digest + ".json")
    return {"written": report.is_file(), "report": str(report),
            "sid_sha256": digest, "tail": (done.stdout or "").strip()[-800:]}


def preserved_run(arm: str, delivered: dict) -> dict:
    """The first completed run's artifacts, kept when a duplicate overwrote them.

    The first run of each arm finished 2026-11-08 and its artifacts were copied
    to ``run_a_complete/`` before the duplicate exported. The SID SHA tells
    whether the two runs actually agree.
    """
    import hashlib
    path = RESULT / ("TIGER_" + arm) / "run_a_complete" / "item_sids.json"
    if not path.is_file():
        return {"available": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"available": True, "path": str(path), "sid_sha256": digest,
            "same_as_delivered": digest == delivered.get("sid_sha256")}


def diagnostics(arm: str, event: str) -> dict:
    rows = metrics(arm)
    if not rows:
        return {"available": False, "reason": "no metrics stream"}
    train = [row for row in rows if row.get("event") == "train"]
    mechanism = [row for row in rows if row.get("event") == event]
    out = {"available": True, "source": metrics_source(arm), "epochs": len(train),
           "last_epoch": train[-1].get("epoch") if train else None,
           "finished": any(row.get("event") == "train_end" for row in rows),
           "wall_s": train[-1].get("cumulative_time_s") if train else None,
           "epoch_time_s": train[-1].get("epoch_time_s") if train else None}
    if mechanism:
        first, last = mechanism[0], mechanism[-1]
        out["mechanism_events"] = len(mechanism)
        wanted = ("epoch", "scale", "kappa", "kappa_gradient", "structural", "radius_mean",
                  "curvature_mean", "profile", "geometry", "keep_geometry", "af3",
                  "neighbour", "usage", "mean_scale")
        def pick(row):
            out = {key: row[key] for key in row if key in wanted}
            # The post-SID event nests the per-step numbers one level down.
            step = row.get("last_step")
            if isinstance(step, dict):
                for key in ("af3", "neighbour", "usage", "scale", "kappa_gradient",
                            "mean_probability", "realised_gradient_ratio"):
                    if key in step:
                        out["step_" + key] = step[key]
            return out
        out["first"] = pick(first)
        out["last"] = pick(last)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=sorted(ARMS))
    args = parser.parse_args()
    spec = ARMS[args.arm]
    diag = diagnostics(args.arm, spec["event"])
    record = layer_record(args.arm)
    payload = {"arm": args.arm, "issue": spec["issue"], "diagnostics": diag,
               "layer_record": record, "runs": run_audit(args.arm),
               "preserved_run": preserved_run(args.arm, record)}
    target = REPO / "results" / "research" / ("issue%d" % spec["issue"])
    target.mkdir(parents=True, exist_ok=True)
    (target / ("finish_%s.json" % args.arm)).write_text(json.dumps(payload, indent=2,
                                                                   sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
