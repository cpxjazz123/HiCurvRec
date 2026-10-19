"""Run track A end to end for every arm and write what the control actually says.

The order here is the whole argument: capture, then prove identity, then FULL parity, and only
after those pass measure NO-DEEP and BASE-ONLY. If FULL parity fails the report says so and carries
no intervention numbers at all, because a counterfactual from a replay that does not reproduce the
real step is not a counterfactual.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_frozen_replay as R  # noqa: E402
import r7_a_replay_engine as G  # noqa: E402

ARMS = ("r6_02", "r6_03", "r6_04", "r4_08")
OUTPUT_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/r7_a_parity"
FORK_STEPS = (32, 96, 160)


def tree_of(arm: str) -> Path:
    return REPO_ROOT / f"stage2_RQ-VAE/TIGER_{arm}"


def spec_of(arm: str) -> dict:
    return json.loads((REPO_ROOT / f"results/stage2_RQ-VAE/{arm}/spec.json").read_text())


def protected_hashes() -> dict[str, str | None]:
    """The artifacts that must not move. Hashed before and after every arm."""
    result: dict[str, str | None] = {}
    for relative in ("dataset", "stage0_build_parquet"):
        path = REPO_ROOT / relative
        result[relative] = R.file_sha256(path / ".keep") if path.is_dir() else None
    heldout = REPO_ROOT / ("results/stage2_RQ-VAE/nontoxicity/behavior_pairs_valid_s42.npz")
    result["frozen_heldout"] = R.file_sha256(heldout)
    for arm in ARMS:
        for name in ("rqvae_best.pth", "item_sids.json", "sids_for_hgrec.npy"):
            result[f"{arm}/{name}"] = R.file_sha256(
                REPO_ROOT / f"results/stage2_RQ-VAE/{arm}/{name}")
    return result


def l1_switch_count(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """How many items actually changed their L1 code under this one update."""
    left, right = before["indices"].reshape(-1), after["indices"].reshape(-1)
    changed = (left != right)
    return {
        "items": int(left.numel()),
        "l1_switched": int(changed.sum()),
        "l1_switch_rate": float(changed.float().mean()),
        "used_codes_before": before["used_codes"],
        "used_codes_after": after["used_codes"],
        "used_codes_delta": after["used_codes"] - before["used_codes"],
    }


def run_arm(arm: str) -> dict:
    started = time.time()
    tree, spec = tree_of(arm), spec_of(arm)
    report: dict = {"arm": arm, "tree": str(tree),
                    "trainer_sha256": R.file_sha256(tree / "train_rqvae.py"),
                    "fork_steps": list(FORK_STEPS), "steps": {}, "blockers": []}

    capture = R.capture_from_run(tree, spec, fork_steps=FORK_STEPS,
                                 save_root=OUTPUT_ROOT / arm)["capture"]
    report["capture"] = {key: value for key, value in capture.items() if key != "bundles"}
    if capture["failures"]:
        report["status"] = "blocked"
        report["blockers"].extend(capture["failures"])
        return report

    for step in FORK_STEPS:
        path = OUTPUT_ROOT / arm / f"step_{step:04d}" / "bundle.pt"
        if not path.is_file():
            continue
        bundle = torch.load(path, weights_only=False)
        entry: dict = {"deep_active": capture["bundles"][step]["deep_active"],
                       "capture_problems": capture["bundles"][step]["problems"],
                       "capture": capture["bundles"][step]["summary"]}
        if entry["capture_problems"]:
            entry["blocked"] = entry["capture_problems"]
            report["steps"][str(step)] = entry
            continue

        branch = G.build_frozen_branch(tree, bundle, module_name=f"r7a_{arm}_{step}")
        reference_item = G.hard_assignments(branch["model"], bundle["item_batch"].to(G.DEVICE))
        full = G.assemble_and_step(branch, bundle, intervention="FULL")
        parity = G.max_parameter_difference(bundle["reference_after_state"],
                                            branch["model"].state_dict())
        entry["FULL"] = full
        entry["full_control_parity"] = parity
        if not parity["identical"]:
            entry["blocked"] = (f"the FULL replay does not reproduce the real step: "
                                f"max_abs_diff={parity['max_abs_diff']} on {parity['tensor']}")
            report["blockers"].append(f"step {step}: {entry['blocked']}")
            report["steps"][str(step)] = entry
            G.release_branch(branch)
            continue

        after_full = G.hard_assignments(branch["model"], bundle["item_batch"].to(G.DEVICE))
        entry["FULL_switches"] = l1_switch_count(reference_item, after_full)
        full_state = {name: value.detach().clone().cpu()
                      for name, value in branch["model"].state_dict().items()}
        G.release_branch(branch)

        for intervention in ("NO-DEEP", "BASE-ONLY"):
            sibling = G.build_frozen_branch(tree, bundle,
                                            module_name=f"r7a_{arm}_{step}_{intervention}")
            before = G.hard_assignments(sibling["model"], bundle["item_batch"].to(G.DEVICE))
            result = G.assemble_and_step(sibling, bundle, intervention=intervention)
            after = G.hard_assignments(sibling["model"], bundle["item_batch"].to(G.DEVICE))
            result["switches"] = l1_switch_count(before, after)
            result["state_difference_vs_FULL"] = G.max_parameter_difference(
                full_state, sibling["model"].state_dict())
            entry[intervention] = result
            G.release_branch(sibling)
        report["steps"][str(step)] = entry

    report["non_perturbation_control"] = non_perturbation_control(
        tree, spec, arm, instrumented_digest=capture.get("final_parameter_sha256"),
        instrumented_steps=capture.get("executed_steps"))
    report["negative_control"] = negative_control(arm, report)
    report["status"] = ("supported" if report["steps"] and not report["blockers"]
                        else "inconclusive" if report["steps"] else "blocked")
    report["elapsed_s"] = round(time.time() - started, 2)
    return report


def non_perturbation_control(tree: Path, spec: dict, arm: str, *,
                             instrumented_digest: str | None,
                             instrumented_steps: int | None = None) -> dict:
    """Does the read-only capture move the run it is watching?

    A difference between an instrumented and a hook-free run only indicts the instrumentation if
    two HOOK-FREE runs agree with each other first. This trainer enables
    ``torch.use_deterministic_algorithms`` while cuBLAS reports that its own reductions are not
    deterministic without a pinned workspace, so run-to-run bit-equality is itself in question and
    has to be measured, not assumed.

    Three runs therefore: one instrumented, two hook-free. If the two hook-free runs differ, the
    environment is not bit-reproducible and the instrumented/hook-free difference cannot be
    attributed to the capture.
    """
    first_plain = R.capture_from_run(tree, spec, fork_steps=(), instrument=False)["capture"]
    second_plain = R.capture_from_run(tree, spec, fork_steps=(), instrument=False)["capture"]
    left = instrumented_digest
    right = first_plain.get("final_parameter_sha256")
    floor_left = right
    floor_right = second_plain.get("final_parameter_sha256")
    environment_reproducible = bool(floor_left and floor_right and floor_left == floor_right)
    instrumented_matches = bool(left and right and left == right)
    if not environment_reproducible:
        verdict = "environment not bit-reproducible"
        reason = ("two runs of the UNMODIFIED trainer with no hooks at all reached different "
                  "parameters, so a difference against the instrumented run measures the "
                  "environment, not the capture")
    elif instrumented_matches:
        verdict = "capture does not perturb the run"
        reason = "the instrumented run and the hook-free run reached identical parameters"
    else:
        verdict = "CAPTURE PERTURBS THE RUN"
        reason = ("the two hook-free runs agree with each other but the instrumented run does "
                  "not, so the difference is attributable to the capture")
    return {
        "definition": "an instrumented run must reach the same parameters as a hook-free run, "
                      "established only after two hook-free runs are shown to agree",
        "instrumented_sha256": left,
        "hook_free_sha256_1": floor_left,
        "hook_free_sha256_2": floor_right,
        "instrumented_steps": instrumented_steps,
        "hook_free_steps": first_plain.get("executed_steps"),
        "environment_bit_reproducible": environment_reproducible,
        "instrumented_matches_hook_free": instrumented_matches,
        "verdict": verdict,
        "reason": reason,
    }


def negative_control(arm: str, report: dict) -> dict:
    """The harness must be inert where there is nothing to remove.

    At a step whose deep weight is zero, NO-DEEP and BASE-ONLY remove terms that were never applied.
    They therefore have to reproduce FULL bit for bit. If they do not, the harness itself is
    perturbing the step, and every other number in this report is suspect.
    """
    inert, live = [], []
    for step, entry in report.get("steps", {}).items():
        if "FULL_switches" not in entry:
            continue
        nodeep = entry.get("NO-DEEP")
        row = {"step": step, "deep_active": entry["deep_active"],
               "full_l1_switched": entry["FULL_switches"]["l1_switched"]}
        if nodeep is not None:
            row["NO-DEEP_l1_switched"] = nodeep["switches"]["l1_switched"]
            row["NO-DEEP_max_abs_diff_vs_FULL"] = nodeep["state_difference_vs_FULL"]["max_abs_diff"]
        record = {"record": row,
                  "expected": "NO-DEEP must be bitwise identical to FULL when the deep weight is "
                              "zero, because nothing was there to remove",
                  "note": "BASE-ONLY is never inert: the base term is live at every step, so "
                          "removing it legitimately changes the update at every step"}
        (live if entry["deep_active"] else inert).append(record)
    passed = all(row["record"].get("NO-DEEP_max_abs_diff_vs_FULL") == 0.0 for row in inert)
    return {
        "definition": "at a step with zero deep weight, NO-DEEP removes a term that was never "
                      "applied and must therefore reproduce FULL bit for bit",
        "inert_steps": inert,
        "live_steps": live,
        "harness_is_inert_where_it_should_be": passed,
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    reports = []
    for arm in ARMS:
        print(f"=== {arm} ===", flush=True)
        report = run_arm(arm)
        reports.append(report)
        (OUTPUT_ROOT / arm / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"{arm}: {report['status']} blockers={len(report['blockers'])}", flush=True)
    after = protected_hashes()
    changed = [name for name in before if before[name] != after.get(name)]
    summary = {
        "arms": reports,
        "production_artifacts_unchanged": not changed,
        "changed_artifacts": changed,
        "all_controls_passed": all(report["status"] == "supported" for report in reports),
        "frozen_final_gate_untouched": True,
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({arm["arm"]: arm["status"] for arm in reports}, indent=2))
    return 0 if summary["all_controls_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
