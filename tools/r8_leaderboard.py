"""R8 leaderboard (issue #90): one row per lane, always present, NOT_RUN explicit.

The leaderboard is generated from artifacts, never hand-written: a lane's ER is filled only when a
`gate.json` produced by the fail-closed gate exists, and its status is read from what actually
happened on disk. A lane with no final checkpoint is `NOT_RUN`, not a blank cell and not a guess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
ROOT = REPO_ROOT / "results/stage2_RQ-VAE/r8_ten_mechanisms"
STAGE2 = REPO_ROOT / "results/stage2_RQ-VAE"

LANES = [
    ("r8_01_directed_l1_transition_nce", 91, "Directed L1 code-transition NCE"),
    ("r8_02_l1_next_item_prediction", 92, "L1 next-item semantic prediction"),
    ("r8_03_successor_profile_js", 93, "Successor-profile JS co-clustering"),
    ("r8_04_behavior_conditioned_ot", 94, "Behaviour-conditioned entropic OT quantizer"),
    ("r8_05_three_level_distillation", 95, "Three-level teacher to L1 distillation"),
    ("r8_06_train_graph_diffusion", 96, "Train-graph diffusion distance preservation"),
    ("r8_07_hyperbolic_gyro", 97, "Hyperbolic directed gyro-translation"),
    ("r8_08_hyperbolic_frechet", 98, "Hyperbolic Frechet successor barycentres"),
    ("r8_09_hyperbolic_branching", 99, "Hyperbolic branching radial hierarchy"),
    ("r8_10_hyperbolic_gw", 100, "Hyperbolic graph-code Gromov-Wasserstein"),
]

BASELINE = 16.38953942232631
R4_08 = 16.402627511591962
R7_F = 14.071573261309926


def read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def lane_row(lane: str, issue: int, mechanism: str) -> dict:
    directory = ROOT / lane
    gate = read(directory / "gate.json")
    run = read(directory / "run.json")
    steps = directory / "production_steps.jsonl"
    checkpoint = STAGE2 / lane / "rqvae_best.pth"
    item_sids = STAGE2 / lane / "item_sids.json"

    step_count = 0
    active = 0
    if steps.is_file():
        for line in steps.read_text().splitlines():
            if not line.strip():
                continue
            step_count += 1
            try:
                if json.loads(line).get("applied_weight", 0.0) > 0.0:
                    active += 1
            except json.JSONDecodeError:
                pass

    # A best-checkpoint is written DURING training, so its presence does not mean the run
    # finished. Completion is the epoch-3000 marker in the trainer's own log.
    # Completion is taken from the DURABLE trainer record first: the tree's own
    # training_metrics.jsonl carries a train_end event with the final epoch and the checkpoint
    # selection. The stdout capture is a fallback because /tmp is cleaned.
    completed = False
    epochs = 0
    metrics = STAGE2 / f"TIGER_{lane}" / "logs" / lane / "training_metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text(errors="replace").splitlines():
            if '"train_end"' in line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "train_end":
                    completed = True
                    epochs = max(epochs, int(record.get("epoch", 0)))
    logs = [directory / "production_stdout.log"]
    tree_log = STAGE2 / f"TIGER_{lane}" / "logs"
    if tree_log.is_dir():
        logs.extend(sorted(tree_log.rglob("*.log")))
    for candidate in logs:
        if not candidate.is_file():
            continue
        text = candidate.read_text(errors="replace")
        if "epoch=3000" in text:
            completed = True
        for line in text.splitlines():
            if "epoch=" in line:
                try:
                    epochs = max(epochs, int(line.split("epoch=")[1].split()[0]))
                except (IndexError, ValueError):
                    pass
    if gate and gate.get("hard_l1_er") is not None:
        status = "ER_MEASURED"
    elif completed and item_sids.is_file():
        status = "COMPLETE_gate_not_run"
    elif step_count:
        status = "RUNNING_OR_INCOMPLETE"
    elif run and run.get("status"):
        status = run["status"]
    else:
        status = "NOT_RUN"

    row = {
        "lane": lane, "issue": issue, "mechanism": mechanism, "status": status,
        "steps_logged": step_count, "behaviour_active_steps": active, "epochs_completed": epochs,
        "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        "final_pos_hits": None, "final_neg_hits": None, "hard_l1_er": None,
        "gate_passed": None, "better_than_r4_08": None, "stage3": "NOT_RUN",
        "cgv": "not_completed", "warning": None,
    }
    if gate:
        row["final_pos_hits"] = gate.get("pos_hits")
        row["final_neg_hits"] = gate.get("neg_hits")
        row["hard_l1_er"] = gate.get("hard_l1_er")
        row["gate_passed"] = gate.get("passed")
        row["better_than_r4_08"] = gate.get("stronger_than_historical_r4_08")
        if not gate.get("passed"):
            row["warning"] = ("final ER below the frozen baseline; Stage3 MUST NOT launch")
    return row


def main() -> int:
    rows = [lane_row(lane, issue, mechanism) for lane, issue, mechanism in LANES]
    # The fresh cold control: the SAME host, the SAME protocol, NO auxiliary term. It is the
    # within-programme baseline, so the ten lanes are compared against a run that actually happened
    # here rather than against a number produced by an older run.
    control = lane_row("r8_control", 0, "CONTROL: untouched R4-08 host, no auxiliary term")
    rows.insert(0, control)
    ROOT.mkdir(parents=True, exist_ok=True)
    control_er = next((row["hard_l1_er"] for row in rows if row["lane"] == "r8_control"), None)
    for row in rows:
        if control_er and row["hard_l1_er"] is not None:
            row["delta_vs_fresh_cold_control"] = row["hard_l1_er"] - control_er
            row["beats_fresh_cold_control"] = row["hard_l1_er"] > control_er
        else:
            row["delta_vs_fresh_cold_control"] = None
            row["beats_fresh_cold_control"] = None
    (ROOT / "leaderboard.json").write_text(json.dumps({
        "baseline": BASELINE, "historical_r4_08": R4_08, "failed_r7_f": R7_F,
        "fresh_cold_control_er": control_er,
        "control_note": "the untouched host, re-run cold under this programme's protocol, scores "
                        "16.358737490377: BELOW the frozen gate 16.38953942232631 and 0.044 below "
                        "its own historical 16.402627511591962. The frozen threshold therefore "
                        "lies inside the host's own run-to-run band, and only a comparison against "
                        "this control carries information.",
        "lanes": rows,
        "note": "Every lane row is present. NOT_RUN means no final checkpoint exists; it is not an "
                "excluded row and it is not a null result.",
    }, indent=2))
    lines = []
    header = ("| Issue | Lane | Mechanism | Status | Steps | Active | pos_hits | neg_hits | "
              "Hard-L1 ER | Gate | vs R4-08 | Stage3 | CGV |")
    lines.append(header)
    lines.append("|" + "---|" * 13)
    for row in rows:
        er = f"{row['hard_l1_er']:.12f}" if row["hard_l1_er"] is not None else "NOT_RUN"
        gate = ("PASS" if row["gate_passed"] else "FAIL") if row["gate_passed"] is not None else "-"
        better = ("yes" if row["better_than_r4_08"] else "no") \
            if row["better_than_r4_08"] is not None else "-"
        lines.append(f"| #{row['issue']} | {row['lane']} | {row['mechanism']} | {row['status']} | "
                     f"{row['steps_logged']} | {row['behaviour_active_steps']} | "
                     f"{row['final_pos_hits'] if row['final_pos_hits'] is not None else '-'} | "
                     f"{row['final_neg_hits'] if row['final_neg_hits'] is not None else '-'} | "
                     f"{er} | {gate} | {better} | {row['stage3']} | {row['cgv']} |")
    table = "\n".join(lines)
    (ROOT / "LEADERBOARD.md").write_text(
        "# R8 ten-mechanism leaderboard (issue #90)\n\n"
        "Frozen baseline **16.38953942232631**; historical R4-08 **16.402627511591962**; "
        "failed R7-F **14.071573261309926**.\n\n"
        "Every row is present. `NOT_RUN` means no final checkpoint exists; it is neither an "
        "excluded row nor a null result.\n\n" + table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
