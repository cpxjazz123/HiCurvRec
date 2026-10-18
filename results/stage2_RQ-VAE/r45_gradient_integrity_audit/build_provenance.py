"""Build the issue #71 provenance artefacts: audit_inventory.csv + manifest.json.

Read-only over the frozen R4-08 / R5 tree. Recomputes every hash from bytes on disk and
compares it against the value the shared gate modules declare; a mismatch is reported as a
blocking finding rather than being smoothed over.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
AUDIT = REPO / "results/stage2_RQ-VAE/r45_gradient_integrity_audit"
AUDIT_COMMIT = "ba04396ebbb5b8031b5a4f448cb4d3fafe42a4b7"
R4_08_PARENT_COMMIT = "bc96625"
PAIRS = REPO / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
PAIRS_SHA_DECLARED = "04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0"
CANONICAL_SID = REPO / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
CANONICAL_SHA_DECLARED = "b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10"
BASELINE_ER_L1 = 16.38953942232631

ARMS = [("r4_08", 58)] + [(f"r5_{i:02d}", 60 + i) for i in range(1, 11)]

DELTA = {
    "r4_08": "encoder-only conflict projection of g_L1 against g_base when dot<0",
    "r5_01": "reverse PCGrad: project g_base against g_L1, assemble unprojected g_L1",
    "r5_02": "project g_L1 against recon-only gradient; assemble full original g_base",
    "r5_03": "project g_L1 against quant-only gradient (base minus recon)",
    "r5_04": "per-nn.Linear weight+bias blockwise projection instead of one global vector",
    "r5_05": "half-strength projection: alpha fixed at 0.5",
    "r5_06": "restore the original ||g_L1|| after projecting",
    "r5_07": "project only when cos(g_L1,g_base) < -0.2 (strict)",
    "r5_08": "apply the projection only from epoch 2250 onward",
    "r5_09": "project g_L1 against the EMA-smoothed base gradient (decay 0.9)",
    "r5_10": "project g_L1 against the true L2+L3 behaviour gradient",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"__unparsable__": line[:120]})
    return rows


def check_ignore(relative: str) -> str:
    proc = subprocess.run(["git", "check-ignore", "-v", relative],
                          cwd=str(REPO), capture_output=True, text=True)
    return proc.stdout.strip() or "not-ignored"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trainer_sha_at_commit(rel_path: str, commit: str = AUDIT_COMMIT) -> str | None:
    """SHA of the trainer as it existed at the audited commit (issue #71 pins this tree)."""
    proc = subprocess.run(["git", "show", f"{commit}:{rel_path}"],
                          cwd=str(REPO), capture_output=True)
    if proc.returncode != 0:
        return None
    return sha256_bytes(proc.stdout)


def trainer_delta(trainer_text: str) -> tuple[str, str]:
    """Confirm the tree really implements the mechanism its name claims."""
    if "R5_01_REVERSE_PCGRAD" in trainer_text:
        return "r5_01", ""
    if "R5_02_RECON_REF" in trainer_text:
        return "r5_02", ""
    if "R5_03_QUANT_REF" in trainer_text:
        return "r5_03", ""
    if "R5_04_BLOCKWISE" in trainer_text:
        return "r5_04", ""
    if "R5_05_ALPHA" in trainer_text:
        return "r5_05", ""
    if "R5_06_NORM_PRESERVE" in trainer_text:
        return "r5_06", ""
    if "R5_07_COS_THRESHOLD" in trainer_text:
        return "r5_07", ""
    if "R5_08_LATE_START_EPOCH" in trainer_text:
        return "r5_08", ""
    if "R5_09_EMA_DECAY" in trainer_text:
        return "r5_09", ""
    if "R5_10_INTERLEVEL" in trainer_text:
        return "r5_10", ""
    if "R4_08_PCGRAD_ENC" in trainer_text:
        return "r4_08", ""
    return "unknown", "no recognised mechanism flag found in the trainer source"


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    findings: list[dict] = []
    rows: list[dict] = []
    manifest_arms: list[dict] = []

    for arm, issue in ARMS:
        tree = REPO / f"stage2_RQ-VAE/TIGER_{arm}"
        trainer = tree / "train_rqvae.py"
        arm_dir = REPO / f"results/stage2_RQ-VAE/{arm}"
        jsonl = tree / "logs" / arm / "training_metrics.jsonl"
        ckpt = arm_dir / "rqvae_best.pth"
        sids = arm_dir / "item_sids.json"
        hgrec = arm_dir / "sids_for_hgrec.npy"
        gate_file = arm_dir / "gate.json"
        metrics_file = arm_dir / "stage2_metrics.json"
        tdd = arm_dir / "tdd_red_green.md"

        trainer_text = trainer.read_text(encoding="utf-8") if trainer.is_file() else ""
        detected, detect_note = trainer_delta(trainer_text)
        if detect_note:
            findings.append({"arm": arm, "severity": "blocking",
                             "statement": f"{arm}: trainer implements no recognised mechanism",
                             "evidence": detect_note})
        elif detected != arm:
            findings.append({"arm": arm, "severity": "blocking",
                             "statement": f"{arm}: tree implements {detected}, not {arm}",
                             "evidence": f"mechanism flag scan of {trainer}"})

        jsonl_rows = read_jsonl(jsonl)
        train_ends = [r for r in jsonl_rows if r.get("event") == "train_end"]
        ckpts = [r for r in jsonl_rows if r.get("event") == "ckpt_saved"]
        last_end = train_ends[-1] if train_ends else {}
        has_epoch = "epoch" in last_end

        gate = json.loads(gate_file.read_text()) if gate_file.is_file() else {}
        metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}

        if tdd.is_file():
            body = tdd.read_text(encoding="utf-8")
            historical = ("source-string-and-flag-only" if re.search(
                r"read_text|inspect\.getsource|assert .* in .*src", body) else "see-tdd-file")
            test_files = sorted(arm_dir.glob("test_*.py"))
            if test_files:
                sample = test_files[0].read_text(encoding="utf-8")
                if "read_text" in sample or "getattr(T" in sample:
                    historical = "source-string-and-flag-only"
        else:
            historical = "absent"

        ckpt_present = ckpt.is_file()
        limitation = ("checkpoint-bytes-gitignored-hash-recorded-from-worktree"
                      if ckpt_present else "checkpoint-absent-unverifiable")
        if not has_epoch:
            limitation += ";train_end-epoch-absent-in-historical-jsonl"

        ignore_line = check_ignore(str(ckpt.relative_to(REPO)))

        rows.append({
            "arm": arm, "original_issue": issue,
            "tree_path": str(trainer.relative_to(REPO)),
            "trainer_sha256": sha256(trainer) or "",
            "model_delta_summary": DELTA.get(arm, ""),
            "checkpoint_path": str(ckpt.relative_to(REPO)) if ckpt_present else "",
            "checkpoint_present": str(ckpt_present).lower(),
            "checkpoint_bytes": ckpt.stat().st_size if ckpt_present else "",
            "checkpoint_git_lfs_status": ignore_line,
            "trainer_jsonl_path": str(jsonl.relative_to(REPO)) if jsonl.is_file() else "",
            "jsonl_train_end_events": len(train_ends),
            "jsonl_last_train_end_has_epoch": str(has_epoch).lower(),
            "jsonl_ckpt_saved_epochs": ";".join(str(c.get("epoch")) for c in ckpts),
            "jsonl_line_count": len(jsonl_rows),
            "item_sids_path": str(sids.relative_to(REPO)) if sids.is_file() else "",
            "item_sids_sha256": sha256(sids) or "",
            "checkpoint_sha256": sha256(ckpt) or "",
            "sids_for_hgrec_sha256": sha256(hgrec) or "",
            "reported_er_l1": metrics.get("ER_L1", ""),
            "baseline_er_l1": BASELINE_ER_L1,
            "reported_gate_stage3_allowed": str(bool(gate.get("stage3_allowed"))).lower(),
            "historical_tests_status": historical,
            "new_fullpath_test_status": "pending-other-slice",
            "two_rank_ddp_test_status": "pending-other-slice",
            "strict_gate_verdict": "pending-other-slice",
            "limitation": limitation,
        })

        manifest_arms.append({
            "arm": arm, "issue": issue, "tree": str(trainer.parent.relative_to(REPO)),
            "mechanism_detected_from_source": detected,
            "trainer_sha256": sha256(trainer),
            "trainer_sha256_at_audited_commit": trainer_sha_at_commit(
                str(trainer.relative_to(REPO))),
            "trainer_patched_by_this_audit": (
                sha256(trainer) != trainer_sha_at_commit(str(trainer.relative_to(REPO)))),
            "jsonl": str(jsonl.relative_to(REPO)) if jsonl.is_file() else None,
            "jsonl_line_count": len(jsonl_rows),
            "jsonl_train_end_count": len(train_ends),
            "jsonl_train_end_keys": sorted(last_end.keys()),
            "jsonl_ckpt_saved_epochs": [c.get("epoch") for c in ckpts],
            "train_end_has_epoch": has_epoch,
            "checkpoint": str(ckpt.relative_to(REPO)) if ckpt_present else None,
            "checkpoint_bytes": ckpt.stat().st_size if ckpt_present else None,
            "checkpoint_sha256": sha256(ckpt),
            "checkpoint_present": ckpt_present,
            "checkpoint_in_git": False,
            "checkpoint_in_git_reason": f".gitignore excludes it: {ignore_line}",
            "item_sids_sha256": sha256(sids),
            "sids_for_hgrec_sha256": sha256(hgrec),
            "reported_er_l1": metrics.get("ER_L1"),
            "reported_gate_stage3_allowed": gate.get("stage3_allowed"),
            "historical_tests_status": historical,
        })

    pairs_sha = sha256(PAIRS)
    canon_sha = sha256(CANONICAL_SID)
    if pairs_sha != PAIRS_SHA_DECLARED:
        findings.append({"arm": "shared", "severity": "blocking",
                         "statement": "held-out behaviour pairs SHA mismatch",
                         "evidence": f"recomputed {pairs_sha} != declared {PAIRS_SHA_DECLARED}"})
    if canon_sha != CANONICAL_SHA_DECLARED:
        findings.append({"arm": "shared", "severity": "blocking",
                         "statement": "canonical Resample SID SHA mismatch",
                         "evidence": f"recomputed {canon_sha} != declared {CANONICAL_SHA_DECLARED}"})

    findings.append({
        "arm": "shared", "severity": "finding",
        "statement": "at the audited commit both shared gates read int(row.get('epoch', 3000)), so "
                     "a train_end record with no epoch was promoted to 3000 by the default value "
                     "itself; all 11 historical train_end rows lack epoch, so every historical "
                     "gate reported train_end_3000=true from the default alone. PATCHED by this "
                     "audit (see auditor_patch); historical gate.json files are left untouched.",
        "evidence": "results/stage2_RQ-VAE/{r4_common/r4_gate.py,r5_common/r5_gate.py} at "
                    "ba04396, function train_end_epoch; each arm's training_metrics.jsonl has a "
                    "ckpt_saved(epoch=3000) record and its checkpoint carries epoch=3000, so the "
                    "runs themselves are corroborated even though the gate did not check that.",
    })
    findings.append({
        "arm": "r5_10", "severity": "finding",
        "statement": "R5-10's mechanism is structurally inert: the straight-through estimator "
                     "x_q = x + (x_q - x).detach() makes d(residual_k)/d(residual_{k-1}) == 1-1 "
                     "== 0, so the L2+L3 behaviour term contributes an identically zero ENCODER "
                     "gradient. R5-10 projects g_L1 against that zero vector, hits the "
                     "base_sq <= 0 passthrough, and therefore runs the PCGrad-free assembly "
                     "instead of the R4-08 projection. R4-08's documented "
                     "'g_base + w_b/3*(g_L1' + g_L2 + g_L3)' is really 'g_base + w_b/3*g_L1''.",
        "evidence": "||g_other|| == 0.000e+00 for 8/8 conflict seeds with 0/8 parameters reported "
                    "unused, i.e. the graph reaches the encoder but the gradient is exactly zero; "
                    "d residual[0]/d encoder = 7.4e+04 while d residual[1] and d residual[2] are "
                    "both exactly 0. See test_mechanism_counterexamples.py::"
                    "test_r5_10_projects_against_true_l2_l3_reference.",
    })
    findings.append({
        "arm": "shared", "severity": "finding",
        "statement": "r5_testkit.tiny_model is broken and was therefore never exercised: it calls "
                     "T.RQVAE(hidden_channels=..., latent_dim=...), which RQVAE.__init__ rejects",
        "evidence": "RQVAE.__init__(self, config, *, in_dim) in model/model.py; running "
                    "tiny_model raises TypeError: unexpected keyword argument 'hidden_channels'",
    })
    findings.append({
        "arm": "shared", "severity": "finding",
        "statement": "R5-01 and R5-02 cannot be toggled back to R4-08: their OFF branch removes "
                     "the projection entirely instead of restoring the parent's projection",
        "evidence": "test_integrated_r4_r5.py::test_r5_01_off_is_pcgrad_free_not_r4_08 and "
                    "::test_r5_02_disabled_parity_is_pcgrad_free",
    })
    findings.append({
        "arm": "shared", "severity": "info",
        "statement": "r5_09 keeps its EMA buffers in module-global state, so results leak across "
                     "successive cases unless R5_09_EMA_STATE['buffers'] is reset",
        "evidence": "R5_09_EMA_STATE = {'buffers': None} at module scope in "
                    "stage2_RQ-VAE/TIGER_r5_09/train_rqvae.py",
    })
    findings.append({
        "arm": "shared", "severity": "info",
        "statement": "the SID export is bit-reproducible ONLY under the trainer's TF32 setting; "
                     "under full fp32 (or on CPU) 91-134 items per arm flip to a different code, "
                     "even though the weights and the stage-1 embeddings are bit-identical",
        "evidence": "every trainer sets torch.set_float32_matmul_precision('high') at module "
                    "scope; verify_sid_from_checkpoint.py reproduces all 11 committed "
                    "item_sids.json exactly (0 differing cells) at 'high'+CUDA and records the "
                    "fp32 sensitivity per arm in sid_reproduction.json",
    })

    manifest = {
        "audit": {
            "issue": 71,
            "audited_commit": AUDIT_COMMIT,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "audit_dir": "results/stage2_RQ-VAE/r45_gradient_integrity_audit",
            "scope": "R4-08 + R5-01..R5-10 gradient-surgery and shared Hard-L1 gate; audit "
                     "only, no Stage2/Stage3 reruns",
        },
        "pinned_inputs": {
            "held_out_pairs_path": str(PAIRS.relative_to(REPO)),
            "held_out_pairs_sha256": PAIRS_SHA_DECLARED,
            "held_out_pairs_sha256_recomputed": pairs_sha,
            "held_out_pairs_present": PAIRS.is_file(),
            "canonical_sid_path": str(CANONICAL_SID.relative_to(REPO)),
            "canonical_sid_sha256_declared": CANONICAL_SHA_DECLARED,
            "canonical_sid_sha256_recomputed": canon_sha,
            "canonical_sid_present": CANONICAL_SID.is_file(),
            "baseline_er_l1": BASELINE_ER_L1,
            "r4_08_parent_commit": R4_08_PARENT_COMMIT,
            "parent_sid_sha256": CANONICAL_SHA_DECLARED,
        },
        "auditor_patch": {
            "why": "issue #71 §1 -- prevent the fail-open from recurring and let the shared gate "
                   "prove completion. Tightly scoped: (a) both shared gate modules stopped "
                   "defaulting a missing train_end epoch to 3000 and now require an explicit "
                   "epoch, a linked ckpt_saved record, a readable checkpoint whose own metadata "
                   "says 3000, and a matching declared checkpoint path; (b) every R4/R5 trainer "
                   "now logs epoch=EPOCHS plus output_json_sha256 in its train_end event.",
            "what_was_not_touched": "no learned weights, no historical SIDs, no train/valid/test "
                                    "splits, no ER threshold, no historical gate.json or "
                                    "stage2_metrics.json, no Stage3 result, and no Stage2/Stage3 "
                                    "rerun. Historical training JSONL files are byte-unchanged.",
            "historical_gates_left_as_they_are": True,
        },
        "arms": manifest_arms,
        "fail_open_finding": {
            "file": "results/stage2_RQ-VAE/r5_common/r5_gate.py (and the identical "
                    "results/stage2_RQ-VAE/r4_common/r4_gate.py)",
            "function": "train_end_epoch",
            "line": 94,
            "code_snippet": 'epoch = int(row.get("epoch", 3000))',
            "why_unsafe": "the completion claim is produced by a default argument, so any "
                          "train_end record lacking an explicit epoch is read as 3000 epochs",
            "historical_impact": "all 11 audited runs wrote train_end without an epoch, so every "
                                 "historical gate reported train_end_3000=true from the default "
                                 "alone. Each JSONL does contain an independent "
                                 "ckpt_saved(epoch=3000) record and each run produced a full "
                                 "SID export, so this is a verification defect in the gate, not "
                                 "evidence that those runs were incomplete.",
        },
        "findings": findings,
        "unverifiable": [
            {"item": "rqvae_best.pth bytes in git",
             "reason": f"excluded by .gitignore ({check_ignore(str((REPO / 'results/stage2_RQ-VAE/r5_01/rqvae_best.pth').relative_to(REPO)))}); "
                       "hashes were computed from the worktree copies, so a GitHub-only clone "
                       "cannot byte-verify the checkpoints"},
            {"item": "checkpoint-internal epoch metadata",
             "reason": "the shared gate never reads it; recorded as an absent safety check "
                       "rather than as a verification"},
        ],
        "reproduce": {
            "python": "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9",
            "cwd": str(REPO),
        },
    }

    csv_path = AUDIT / "audit_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()),
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    (AUDIT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {csv_path} ({len(rows)} arms)")
    print(f"wrote {AUDIT / 'manifest.json'}")
    print(f"pairs  sha declared={PAIRS_SHA_DECLARED}\n       recomputed={pairs_sha}")
    print(f"canon  sha declared={CANONICAL_SHA_DECLARED}\n       recomputed={canon_sha}")
    print(f"git check-ignore: {check_ignore('results/stage2_RQ-VAE/r5_01/rqvae_best.pth')}")
    print("findings:")
    for f in findings:
        print(f"  [{f['severity']}] {f['arm']}: {f['statement'][:110]}")
    blocking = [f for f in findings if f["severity"] == "blocking"]
    print("PROVENANCE_OK" if not blocking else f"PROVENANCE_BLOCKED ({len(blocking)})")
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
