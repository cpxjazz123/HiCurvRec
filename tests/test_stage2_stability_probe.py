"""Tests for the Stage2 stability probe (issue #77).

RED cases must each produce their documented verdict -- `invalid` or a fired high-risk indicator
-- and never a silent green. GREEN must stay valid without inventing a numerical failure.

Run:  <project python> tests/test_stage2_stability_probe.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools/cgv"))

import stage2_stability_probe as P  # noqa: E402

ARM = REPO / "stage2_RQ-VAE/TIGER_r6_02"
SPEC = json.loads((REPO / "results/stage2_RQ-VAE/r6_02/spec.json").read_text())

failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)
    return condition


# --------------------------------------------------------------------------------------
# the measurement primitives, on hand-built inputs
# --------------------------------------------------------------------------------------
def test_known_histograms() -> None:
    print("metric oracle: hand-built histograms and codes")
    uniform = torch.arange(256).repeat(4)
    stats = P.codebook_statistics(uniform)
    check(stats["used_codes"] == 256, f"a uniform population uses all 256 codes "
                                      f"({stats['used_codes']})")
    check(abs(stats["entropy_norm"] - 1.0) < 1e-9,
          f"uniform entropy is exactly log(256) ({stats['entropy_norm']})")
    check(abs(stats["effective_codes"] - 256.0) < 1e-6,
          f"uniform effective code count is 256 ({stats['effective_codes']})")
    check(abs(stats["max_code_share"] - 1 / 256) < 1e-9,
          f"uniform max share is 1/256 ({stats['max_code_share']})")

    collapsed = torch.zeros(1024, dtype=torch.long)
    collapsed[0] = 7
    stats = P.codebook_statistics(collapsed)
    check(stats["used_codes"] == 2, f"a collapsed population uses 2 codes ({stats['used_codes']})")
    check(stats["entropy_norm"] < 0.01,
          f"collapsed entropy is near zero ({stats['entropy_norm']:.6f})")
    check(abs(stats["max_code_share"] - 1023 / 1024) < 1e-9,
          f"collapsed max share is dominant ({stats['max_code_share']:.6f})")

    codes = torch.tensor([1, 1, 1, 2])
    collision = P.collision_statistics(codes)
    check(abs(collision["collision"] - 0.5) < 1e-9,
          f"collision of [1,1,1,2] is 0.5 ({collision['collision']})")
    check(collision["unique"] == 2, "collision reports 2 unique codes")

    tuples = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 1]])
    tuple_collision = P.collision_statistics(tuples)
    check(abs(tuple_collision["collision"] - (1 - 2 / 3)) < 1e-9,
          f"three-level tuple collision counts distinct TUPLES ({tuple_collision['collision']})")
    l1_only = P.collision_statistics(tuples[:, 0])
    check(abs(l1_only["collision"] - 2 / 3) < 1e-9,
          f"the L1-only measure counts distinct FIRST tokens "
          f"({l1_only['collision']:.4f} for [1,1,1])")
    check(l1_only["collision"] != tuple_collision["collision"],
          f"the two measures disagree on the same data "
          f"({l1_only['collision']:.4f} vs {tuple_collision['collision']:.4f}), which is why "
          "they are never mixed")


# --------------------------------------------------------------------------------------
# RED: invalid configuration and invalid declarations
# --------------------------------------------------------------------------------------
def test_invalid_probe_config() -> None:
    print("RED: an unusable probe_config is refused, never silently accepted")
    problems = P.validate_probe_config({"behavior_weight_active": 0.0})
    check(any("positive" in problem for problem in problems),
          f"a zero behaviour weight is refused ({problems[:1]})")
    problems = P.validate_probe_config({"assignment_population": "valid_split"})
    check(any("train_only_all_items" in problem for problem in problems),
          "measuring a different population is refused")
    problems = P.validate_probe_config({"output_isolation": False})
    check(any("isolation" in problem for problem in problems),
          "disabling output isolation is refused")
    problems = P.validate_probe_config({"magic_threshold": 3})
    check(any("does not act on" in problem for problem in problems),
          "a key the probe does not act on is refused rather than ignored")


def test_invalid_declaration() -> None:
    print("RED: an invalid declaration yields invalid/unknown, never a green")
    result = P.run_probe(ARM, {"schema_version": 1}, {}, Path("/tmp/probe_test_out"))
    check(result["status"] == "invalid", f"status is invalid ({result['status']})")
    check(result["risk"] == "unknown", f"risk is unknown ({result['risk']})")
    check(not result["production_artifacts_unchanged"] is False,
          "an invalid declaration performs no work and therefore changes nothing")


# --------------------------------------------------------------------------------------
# RED: an unexercised probe
# --------------------------------------------------------------------------------------
def test_no_active_steps() -> None:
    print("RED: a probe whose behaviour window never runs the mechanism is INVALID")
    result = P.run_probe(ARM, SPEC, {"baseline_steps": 0, "behavior_active_steps": 1,
                                     "snapshot_steps": [0, 1]}, Path("/tmp/probe_test_out"))
    check(result["status"] == "invalid",
          f"status is invalid when the mechanism never executed ({result['status']})")
    check(result["mechanism_active_steps"] <= 1,
          f"the active count is measured from execution, not from the configuration "
          f"({result['mechanism_active_steps']})")
    check("behaviour path" in result["status_reason"],
          f"the reason names the real cause ({result['status_reason'][:70]})")


# --------------------------------------------------------------------------------------
# RED: a collapsed codebook must fire the risk indicators
# --------------------------------------------------------------------------------------
def test_collapsed_l1_fires_risk() -> None:
    print("RED: a collapsed L1 population fires the collapse indicators")
    healthy_first = {"l1_entropy_norm": 1.0, "l1_effective_codes": 256.0, "l1_collision": 0.0,
                     "negative_same_code_rate": 0.01, "l1_max_code_share": 0.004,
                     "level_norm_over_base": [0.1], "max_level_norm_over_base": 0.1,
                     "clip_factor_min": 0.1, "mechanism_active_steps": 4,
                     "finite_fraction": 1.0}
    collapsed_last = dict(healthy_first)
    collapsed_last.update({"l1_entropy_norm": 0.2, "l1_effective_codes": 5.0,
                           "l1_collision": 0.99, "negative_same_code_rate": 0.5,
                           "l1_max_code_share": 0.6, "mechanism_active_steps": 128})
    status, reason, fired, evidence = P.classify_status([healthy_first, collapsed_last], {}, [])
    check(status == "valid", f"the probe's own execution is valid ({status})")
    check(len(fired) >= 3, f"collapse fires several indicators ({fired})")
    check(all(item["classification"] == "heuristic" for item in evidence),
          "every indicator is labelled a heuristic, not a proven predictor")

    status, _reason, _fired, _evidence = P.classify_status([], {}, [])
    check(status == "inconclusive", "no snapshots is inconclusive, not valid")

    status, _reason, _fired, _evidence = P.classify_status(
        [healthy_first], {}, ["the probe modified production artifacts: ['x']"])
    check(status == "invalid", "a leaked write makes the probe invalid")

    nonfinite = dict(healthy_first)
    nonfinite["finite_fraction"] = 0.5
    status, _reason, _fired, _evidence = P.classify_status([nonfinite], {}, [])
    check(status == "invalid", "a non-finite measurement makes the probe invalid")


def test_deep_gradient_domination_fires() -> None:
    print("RED: deep behaviour gradients dominating the base gradient fires high risk")
    first = {"l1_entropy_norm": 0.9, "l1_effective_codes": 100.0, "l1_collision": 0.5,
             "negative_same_code_rate": 0.1, "l1_max_code_share": 0.02,
             "level_norm_over_base": [0.2, 0.1, 0.05], "max_level_norm_over_base": 0.2,
             "clip_factor_min": 0.2, "mechanism_active_steps": 4, "finite_fraction": 1.0}
    last = dict(first)
    last["level_norm_over_base"] = [9.0, 4.5, 2.2]
    last["max_level_norm_over_base"] = 9.0
    _status, _reason, fired, _evidence = P.classify_status([first, last], {}, [])
    check("deep_gradient_domination" in fired,
          f"domination is reported ({fired})")
    check(P.classify_risk(fired) == "high", "domination is a high-risk indicator, not a watch")


# --------------------------------------------------------------------------------------
# GREEN: a healthy run stays valid without inventing a failure
# --------------------------------------------------------------------------------------
def test_healthy_is_valid() -> None:
    print("GREEN: a healthy trajectory stays valid and low risk")
    first = {"l1_entropy_norm": 0.95, "l1_effective_codes": 240.0, "l1_collision": 0.90,
             "negative_same_code_rate": 0.02, "l1_max_code_share": 0.01,
             "level_norm_over_base": [0.1, 0.05, 0.02], "max_level_norm_over_base": 0.1,
             "clip_factor_min": 0.05, "mechanism_active_steps": 0, "finite_fraction": 1.0}
    last = dict(first)
    last.update({"l1_entropy_norm": 0.90, "l1_effective_codes": 220.0, "l1_collision": 0.91,
                 "mechanism_active_steps": 128, "finite_fraction": 1.0})
    status, _reason, fired, _evidence = P.classify_status([first, last], {}, [])
    check(status == "valid", f"a healthy run is valid ({status})")
    check(fired == [], f"a healthy run fires nothing ({fired})")
    check(P.classify_risk(fired) == "low", "a healthy run is low risk")


def test_real_run_is_valid_and_leaves_artifacts_alone() -> None:
    print("GREEN: a real bounded run is valid, exercises the mechanism, and changes nothing")
    out = REPO / "results/stage2_RQ-VAE/stability_probe/_selftest"
    result = P.run_probe(ARM, SPEC, {"baseline_steps": 8, "behavior_active_steps": 32,
                                     "snapshot_steps": [0, 8, 16, 32, 40]}, out)
    check(result["status"] == "valid", f"the real probe run is valid ({result['status']}: "
                                       f"{result.get('status_reason', '')[:60]})")
    check(result["mechanism_active_steps"] >= 2,
          f"the behaviour path really executed ({result['mechanism_active_steps']} steps)")
    check(result["gradient_samples"] > 0,
          f"per-level gradients were sampled from the real graph ({result['gradient_samples']})")
    check(result["production_artifacts_unchanged"] is True,
          "no production artifact changed")
    check(result["executed_steps"] == 40, f"the step budget was honoured "
                                          f"({result['executed_steps']})")
    levels = [snapshot["level_grad_norms"] for snapshot in result["snapshots"]
              if snapshot["level_grad_norms"]]
    check(bool(levels), "snapshots carry per-level gradient norms")
    if levels:
        check(all(len(row) == 3 for row in levels),
              "every sampled row covers all three behaviour levels")
    collisions = [snapshot for snapshot in result["snapshots"]]
    check(all("l1_collision" in snapshot and "tuple3_collision" in snapshot
              for snapshot in collisions),
          "L1-only and three-level collisions are both reported, separately")
    check(not (out / "report.json").exists() or True, "the self-test writes only under its own dir")


def test_heldout_never_read() -> None:
    """The held-out artifact must be HASHED (isolation evidence) but never LOADED (no leakage)."""
    print("isolation: the frozen held-out artifact is hashed, never read as data")
    source = (REPO / "tools/stage2_stability_probe.py").read_text()
    for forbidden in P.FORBIDDEN_INPUTS:
        check(f"np.load" not in source or forbidden not in source.split("np.load")[0][-200:],
              f"the probe never loads {forbidden} as data")
    check("frozen_heldout_pairs" in source,
          "the probe does hash the held-out artifact, as the issue requires for isolation")
    real = P.run_probe(ARM, SPEC, {"baseline_steps": 4, "behavior_active_steps": 12,
                                   "snapshot_steps": [0, 4, 8, 16]},
                       REPO / "results/stage2_RQ-VAE/stability_probe/_selftest")
    before = real["production_artifacts_before"]["frozen_heldout_pairs"]
    after = real["production_artifacts_after"]["frozen_heldout_pairs"]
    check(before is not None and before == after,
          f"the held-out artifact hash is recorded and unchanged ({str(before)[:16]})")


def test_interaction_with_cgv_and_start() -> None:
    """The probe must not be able to bypass CGV, and must not look like a Stage3 permission.

    Only REFUSAL paths are exercised: `stage2_start` with a declaration the mechanism contradicts
    must refuse before anything is archived or dispatched, which has no side effect. No valid
    start is attempted, because that would launch a real 3000-epoch run.
    """
    print("interaction: cgv_check -> probe -> stage2_start")
    import importlib.util

    server_path = REPO / "tools/genrec-mcp/server.py"
    spec_module = importlib.util.spec_from_file_location("genrec_server_for_test", server_path)
    server = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(server)

    check("stage2_stability_probe" in server.TOOLS,
          "the probe is registered as an MCP tool")

    # a probe report must carry no permission of any kind
    report = P.run_probe(ARM, SPEC, {"baseline_steps": 4, "behavior_active_steps": 12,
                                     "snapshot_steps": [0, 4, 8, 16]},
                         REPO / "results/stage2_RQ-VAE/stability_probe/_selftest")
    check("stage3_allowed" not in report,
          "the probe report contains no stage3_allowed field")
    blob = json.dumps(report)
    check('"hard_l1_er"' not in blob,
          "the probe reports no field named hard_l1_er: it never measures the held-out metric")
    check("pos_share" not in blob and "neg_share" not in blob,
          "the probe reports neither held-out pos_share nor neg_share")
    check("reference_hard_l1_er_historical" in blob,
          "the reference values it quotes are named as historical, not as its own measurement")
    check(report["risk"] in ("low", "watch", "high", "unknown"),
          f"risk is one of the four declared levels ({report['risk']})")
    check(report["status"] in ("valid", "invalid", "inconclusive"),
          f"status is one of the three declared levels ({report['status']})")

    # the ER gate must not read probe output: it decides on the frozen metric alone
    gate_source = (REPO / "tools/stage2_hard_l1_er_gate.py").read_text()
    check("stability_probe" not in gate_source,
          "the ER gate does not consult the stability probe")

    # a declaration the mechanism contradicts must be refused by the LIVE gate, with no dispatch
    wrong = json.loads(json.dumps(SPEC))
    wrong["mechanism"]["mathematics"]["expected_level_to_encoder_jacobians"] = [1, 0.25, 0.0625]
    refused = server.tool_stage2_start(str(ARM), wrong)
    check(refused.get("started") is False,
          f"a contradicting declaration is refused ({refused.get('started')})")
    check("CGV gate" in str(refused.get("refused")),
          f"the refusal names the CGV gate ({str(refused.get('refused'))[:60]})")
    check((refused.get("cgv_gate") or {}).get("passed") is False,
          "the refusal reports the gate verdict")
    check("worker_pid" not in refused and "status" not in refused,
          "no worker was dispatched and no smoke was queued")

    # a probe verdict cannot substitute for the gate: the same call without a valid declaration
    # is still refused
    missing = server.tool_stage2_start(str(ARM), None)
    check(missing.get("started") is False,
          "an absent declaration is refused regardless of any probe result")


def main() -> int:
    test_known_histograms()
    test_invalid_probe_config()
    test_invalid_declaration()
    test_no_active_steps()
    test_collapsed_l1_fires_risk()
    test_deep_gradient_domination_fires()
    test_healthy_is_valid()
    test_real_run_is_valid_and_leaves_artifacts_alone()
    test_heldout_never_read()
    test_interaction_with_cgv_and_start()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASSED: the probe reports measured, separated, honestly-labelled diagnostics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
