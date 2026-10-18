"""Issue #71 §1 — re-verify all 11 arms with the STRICT gate, without touching history.

Writes ``strict_gate_<arm>.json`` into the audit directory only. The historical
``results/stage2_RQ-VAE/<arm>/gate.json`` and ``stage2_metrics.json`` are read, never written:
the reported historical verdict and the strictly verified verdict are recorded side by side so a
reader can see both.

The strict gate cannot invent evidence the historical JSONL does not contain. Those runs logged
``train_end`` without an ``epoch``, so strict completion is UNPROVEN for all 11 even though each
run independently carries a ``ckpt_saved(epoch=3000)`` record and a checkpoint whose own metadata
says ``epoch=3000``. That distinction is the point of the audit, so it is reported explicitly
rather than smoothed over.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
AUDIT = REPO / "results/stage2_RQ-VAE/r45_gradient_integrity_audit"
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r4_common"))

import r5_gate as G5  # noqa: E402

ARMS = ["r4_08"] + [f"r5_{i:02d}" for i in range(1, 11)]
ISSUES = {"r4_08": 58, **{f"r5_{i:02d}": 60 + i for i in range(1, 11)}}


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    summary = []
    for arm in ARMS:
        arm_dir = REPO / f"results/stage2_RQ-VAE/{arm}"
        metrics_path = (REPO / f"stage2_RQ-VAE/TIGER_{arm}/logs/{arm}/training_metrics.jsonl")
        sid_path = arm_dir / "item_sids.json"
        historical = json.loads((arm_dir / "gate.json").read_text()) \
            if (arm_dir / "gate.json").is_file() else {}

        tests_passed = bool(historical.get("tests_passed"))
        stage2_metrics, strict = G5.evaluate(arm=arm, issue=ISSUES[arm],
                                             item_sids_path=sid_path,
                                             metrics_path=metrics_path,
                                             tests_passed=tests_passed)
        ckpt = arm_dir / "rqvae_best.pth"
        meta = G5.checkpoint_metadata(ckpt) if ckpt.is_file() else None
        record = {
            "arm": arm,
            "issue": ISSUES[arm],
            "reported_historical_gate": {
                "train_end_3000": historical.get("train_end_3000"),
                "stage3_allowed": historical.get("stage3_allowed"),
                "measured_er_l1": historical.get("measured_er_l1"),
                "source": f"results/stage2_RQ-VAE/{arm}/gate.json (unchanged by this audit)",
            },
            "strictly_verified_gate": {
                **{k: strict.get(k) for k in (
                    "train_end_3000", "strict_completion_proved", "train_end_epoch_explicit",
                    "ckpt_saved_record_at_expected_epoch", "checkpoint_readable",
                    "checkpoint_metadata_epoch", "checkpoint_metadata_epoch_is_expected",
                    "checkpoint_path_matches", "sid_linked_to_checkpoint",
                    "ER_L1_above_baseline", "stage3_allowed")},
                "train_end_epoch": strict.get("train_end_epoch"),
            },
            "independent_corroboration": {
                "ckpt_saved_epochs": G5.ckpt_saved_epochs(metrics_path),
                "checkpoint_metadata": meta,
                "sid_reproduction": "see sid_reproduction.json (11/11 exact under the "
                                    "trainer's TF32 setting)",
            },
            "uncertainty": (
                "The strict gate reports completion UNPROVEN because the historical train_end "
                "record carries no epoch. That is a missing-evidence verdict, not a claim that "
                "the runs were incomplete: the ckpt_saved(epoch=3000) record, the checkpoint's own "
                "epoch metadata and the byte-exact SID reproduction all corroborate completion."),
            "strict_verdict": ("stage3_allowed_would_be_true_under_strict_proof"
                               if strict.get("ER_L1_above_baseline")
                               and strict.get("strict_completion_proved")
                               else "stage3_allowed_false"),
        }
        (AUDIT / f"strict_gate_{arm}.json").write_text(json.dumps(record, indent=2) + "\n",
                                                       encoding="utf-8")
        summary.append(record)
        s = record["strictly_verified_gate"]
        print(f"{arm}: reported_allowed={record['reported_historical_gate']['stage3_allowed']} "
              f"strict_allowed={s['stage3_allowed']} "
              f"train_end_epoch={s['train_end_epoch']!r} "
              f"ckpt_meta_epoch={s['checkpoint_metadata_epoch']} "
              f"ckpt_saved={record['independent_corroboration']['ckpt_saved_epochs']} "
              f"ER={s['ER_L1_above_baseline']}", flush=True)

    (AUDIT / "strict_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nwrote {len(summary)} strict gate records + strict_gate_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
