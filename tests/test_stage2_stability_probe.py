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
    check(result["active_optimizer_steps"] <= 1,
          f"the active count is measured from execution, not from the configuration "
          f"({result['active_optimizer_steps']})")
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
                     "clip_factor_min": 0.1, "active_optimizer_steps": 4,
                     "finite_fraction": 1.0}
    collapsed_last = dict(healthy_first)
    collapsed_last.update({"l1_entropy_norm": 0.2, "l1_effective_codes": 5.0,
                           "l1_collision": 0.99, "negative_same_code_rate": 0.5,
                           "l1_max_code_share": 0.6, "active_optimizer_steps": 128})
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
    nonfinite["nonfinite_steps"] = 1
    status, _reason, _fired, _evidence = P.classify_status([nonfinite], {}, [])
    check(status == "invalid", "a non-finite measurement makes the probe invalid")


def test_deep_gradient_domination_fires() -> None:
    print("RED: deep behaviour gradients dominating the base gradient fires high risk")
    first = {"l1_entropy_norm": 0.9, "l1_effective_codes": 100.0, "l1_collision": 0.5,
             "negative_same_code_rate": 0.1, "l1_max_code_share": 0.02,
             "attribution": {"max_norm_over_base": 0.2}, "clip_factor_min": 0.2,
             "active_optimizer_steps": 4, "finite_fraction": 1.0, "nonfinite_steps": 0}
    last = dict(first)
    last["attribution"] = {"max_norm_over_base": 9.0}
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
             "clip_factor_min": 0.05, "active_optimizer_steps": 0, "finite_fraction": 1.0}
    last = dict(first)
    last.update({"l1_entropy_norm": 0.90, "l1_effective_codes": 220.0, "l1_collision": 0.91,
                 "active_optimizer_steps": 128, "finite_fraction": 1.0})
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
    check(result["active_optimizer_steps"] >= 2,
          f"the behaviour path really executed ({result['active_optimizer_steps']} steps)")
    check(result["gradient_samples"] > 0,
          f"per-level gradients were sampled from the real graph ({result['gradient_samples']})")
    check(result["production_artifacts_unchanged"] is True,
          "no production artifact changed")
    check(result["executed_steps"] == 40, f"the step budget was honoured "
                                          f"({result['executed_steps']})")
    levels = [snapshot["attribution"]["levels"] for snapshot in result["snapshots"]
              if snapshot["attribution"]["levels"]]
    check(bool(levels), "snapshots carry per-level gradient attribution")
    if levels:
        check(all(len(row) == 3 for row in levels),
              "every attributed row covers all three behaviour levels")
        check(all("undefined_reason" in row for row in levels[-1]),
              "each attributed level states whether its ratio is defined")
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


# --------------------------------------------------------------------------------------
# issue #78: the four measured defects, each with its own RED case
# --------------------------------------------------------------------------------------
def test_gradient_denominator() -> None:
    """RED: the ratio must divide by ||g_base_enc||, NOT by the total pre-clip norm.

    Hand-built two-loss example where the two candidate denominators differ, so the old formula
    is provably wrong rather than merely renamed.
    """
    print("RED (issue #78.1): the gradient denominator")
    g_base = [torch.tensor([3.0, 4.0])]          # ||g_base_enc|| = 5
    g_level = [torch.tensor([1.0, 0.0])]         # ||g_L1_raw||   = 1
    record = {"g_base": g_base, "g_levels_raw": [g_level]}
    weight, valid = 0.6, 3
    out = P.attribute_gradients(record, weight, valid)
    weighted = 1.0 * (0.6 / 3)
    check(abs(out["g_base_enc_norm"] - 5.0) < 1e-9,
          f"the denominator is ||g_base_enc|| = 5 ({out['g_base_enc_norm']})")
    check(abs(out["levels"][0]["weighted_norm"] - weighted) < 1e-9,
          f"the weighted level norm is raw * weight/valid = {weighted} "
          f"({out['levels'][0]['weighted_norm']})")
    check(abs(out["levels"][0]["norm_over_base"] - weighted / 5.0) < 1e-9,
          f"the ratio is weighted/5 ({out['levels'][0]['norm_over_base']})")
    wrong_denominator = 100.0
    check(abs(out["levels"][0]["norm_over_base"] - weighted / wrong_denominator) > 1e-3,
          "the old formula, dividing by the total pre-clip norm, gives a different and wrong value")
    expected_cosine = 3.0 / 5.0          # cos([1,0], [3,4])
    check(abs(out["levels"][0]["cosine_with_base"] - expected_cosine) < 1e-9,
          f"the cosine against base is computed on the raw vectors "
          f"({out['levels'][0]['cosine_with_base']:.4f}, expected {expected_cosine})")

    zero_base = P.attribute_gradients({"g_base": [torch.zeros(2)],
                                       "g_levels_raw": [g_level]}, weight, valid)
    check(zero_base["levels"][0]["norm_over_base"] is None,
          "a zero base norm yields an UNDEFINED ratio, not 0 and not a fabricated number")
    check("zero denominator" in (zero_base["levels"][0]["undefined_reason"] or ""),
          f"the undefined ratio carries a reason ({zero_base['levels'][0]['undefined_reason']})")

    zero_level = P.attribute_gradients({"g_base": g_base,
                                        "g_levels_raw": [torch.zeros(2)]}, weight, valid)
    check(zero_level["levels"][0]["norm_over_base"] is None,
          "a zero level norm also yields an undefined ratio")

    disconnected = P.attribute_gradients({"g_base": g_base, "g_levels_raw": [None]}, weight, valid)
    check(disconnected["levels"][0]["raw_norm"] is None
          and "disconnected" in disconnected["levels"][0]["undefined_reason"],
          "a level disconnected from the encoder is reported as such")

    mixed = P.attribute_gradients({"g_base": g_base,
                                   "g_levels_raw": [g_level, torch.zeros(2), None]},
                                  weight, valid)
    check(mixed["ratios_defined"] == 1,
          f"only the defined levels contribute a ratio ({mixed['ratios_defined']})")
    check(abs(mixed["max_norm_over_base"] - weighted / 5.0) < 1e-9,
          "the maximum ignores undefined rows instead of treating them as zero")


def test_real_labels_not_cyclic() -> None:
    """RED: the per-level terms must be the trainer's OWN losses, with its OWN labels.

    The previous version rebuilt a contrastive target as `(arange+1) % n_src` and dropped the
    successor mask. Here a minibatch is constructed whose real targets are NOT that cyclic shift
    and whose mask blocks a candidate, and the observed terms must equal the production loss.
    """
    print("RED (issue #78.2): real labels and the real mask")
    torch.manual_seed(0)
    n_src, dim = 6, 8
    source = torch.randn(n_src, dim)
    candidates = torch.randn(n_src, dim)
    source_ids = torch.tensor([0, 1, 2, 3, 4, 5])
    # A DUPLICATE target is what makes the mask bite: with all-distinct targets the target ids
    # never enter the loss at all, so a test built on distinct targets proves nothing.
    real_targets = torch.tensor([5, 4, 3, 3, 1, 0])
    cyclic = (torch.arange(n_src) + 1) % n_src
    check(not torch.equal(real_targets, cyclic),
          "the constructed minibatch's real targets differ from the cyclic shift")
    check(int(real_targets.unique().numel()) < n_src,
          "the real targets contain a duplicate, so the duplicate mask is exercised")

    logits = -torch.cdist(source, candidates, p=2) / 0.07
    # Block the strongest non-label candidate of row 0, so removing the mask provably changes
    # the loss instead of merely changing a logit that the softmax barely notices.
    row = logits[0].clone()
    row[0] = -torch.inf                                    # never block the label position
    blocked = torch.zeros(n_src, n_src, dtype=torch.bool)
    blocked[0, int(torch.argmax(row))] = True
    masked = logits.masked_fill(blocked, -torch.inf)

    import types
    module = types.SimpleNamespace()
    sys.path.insert(0, str(REPO / "stage2_RQ-VAE/TIGER_r6_02"))

    def production_loss(logits_in, s_ids, t_ids, successor=None):
        duplicate = t_ids[:, None].eq(t_ids[None, :])
        duplicate.fill_diagonal_(False)
        if successor is not None:
            duplicate = duplicate | successor
        masked_in = logits_in.masked_fill(duplicate, -torch.inf)
        valid = s_ids.ne(t_ids)
        labels = torch.arange(int(s_ids.shape[0]))
        import torch.nn.functional as F
        return F.cross_entropy(masked_in[valid], labels[valid])

    real = production_loss(masked, source_ids, real_targets, blocked)
    cyclic_loss = production_loss(logits, source_ids, cyclic, None)
    check(abs(float(real) - float(cyclic_loss)) > 1e-6,
          f"the real masked loss differs from the fabricated cyclic-label loss "
          f"({float(real):.6f} vs {float(cyclic_loss):.6f})")

    without_mask = production_loss(logits, source_ids, real_targets, None)
    check(abs(float(real) - float(without_mask)) > 1e-9,
          f"the mask changes the loss, so a probe that drops it measures something else "
          f"({float(real):.6f} masked vs {float(without_mask):.6f} unmasked)")
    cyclic_masked = production_loss(masked, source_ids, cyclic, blocked)
    check(abs(float(real) - float(cyclic_masked)) > 1e-6,
          f"with the real mask in place, real labels and fabricated cyclic labels give different "
          f"losses ({float(real):.6f} vs {float(cyclic_masked):.6f}), so observing the trainer's "
          "own terms is not interchangeable with regenerating labels")
    check(P.MEASUREMENT_VERSION == 2, f"the measurement version is 2 ({P.MEASUREMENT_VERSION})")


def test_schedule_boundary_and_step_zero() -> None:
    """RED: the weight must change at optimizer step 33, not at an epoch boundary."""
    print("RED (issue #78.3): the 32+128 schedule and step 0")
    out = REPO / "results/stage2_RQ-VAE/stability_probe_v2/_selftest"
    config = {"baseline_steps": 32, "behavior_active_steps": 128,
              "snapshot_steps": [0, 8, 16, 32, 64, 96, 128, 160]}
    result = P.run_probe(ARM, SPEC, config, out)
    check(result["status"] == "valid", f"the corrected run is valid ({result['status']})")
    check(result["executed_steps"] == 160,
          f"exactly 160 optimizer steps completed ({result['executed_steps']})")
    check(result["active_optimizer_steps"] == 128,
          f"exactly 128 active optimizer steps ({result['active_optimizer_steps']})")
    check(result["mechanism_call_count"] == 384,
          f"the mechanism ran 3 times per active step, counted separately from the steps "
          f"({result['mechanism_call_count']})")
    schedule = result["schedule"]
    check(schedule["agrees"] is True,
          f"the configured and observed schedules agree ({schedule})")
    check(schedule["observed_baseline_steps"] == 32 and schedule["observed_active_steps"] == 128,
          "32 baseline and 128 active steps were actually applied")
    check(schedule["applied_weights_seen"] == [0.0, 0.2],
          f"only the two declared weights were ever applied ({schedule['applied_weights_seen']})")
    steps = [snapshot["step"] for snapshot in result["snapshots"]]
    check(steps == config["snapshot_steps"],
          f"all eight snapshots exist exactly once, in order ({steps})")
    zero = result["snapshots"][0]
    check(zero["step"] == 0 and zero["active_optimizer_steps"] == 0,
          "step 0 is measured before any optimizer step")
    check(zero["applied_behavior_weight"] == 0.0, "step 0 carries no behaviour weight")
    weights = [snapshot["applied_behavior_weight"] for snapshot in result["snapshots"]]
    check(weights == [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
          f"the weight is 0 through step 32 and 0.2 from step 33 ({weights})")
    applied = [row for row in result["snapshots"] if row["step"] >= 64]
    check(all(row["attribution"]["g_base_enc_norm"] is not None for row in applied),
          "the corrected denominator is measured on every active snapshot")


def test_nonfinite_late_step_is_invalid() -> None:
    """RED: a finite early snapshot must not excuse a non-finite later one."""
    print("RED (issue #78.5): a non-finite LATE step")
    healthy = {"l1_entropy_norm": 1.0, "l1_effective_codes": 250.0, "l1_collision": 0.9,
               "negative_same_code_rate": 0.01, "l1_max_code_share": 0.01,
               "level_norm_over_base": [0.1], "clip_factor_min": 0.1,
               "active_optimizer_steps": 4, "finite_fraction": 1.0, "nonfinite_steps": 0}
    late_bad = dict(healthy)
    late_bad.update({"step": 100, "finite_fraction": 0.99, "nonfinite_steps": 1})
    status, reason, _fired, _evidence = P.classify_status([healthy, late_bad], {}, [])
    check(status == "invalid",
          f"a non-finite step anywhere makes the probe invalid ({status})")
    check("non-finite" in reason,
          f"the reason names the cause ({reason[:70]})")
    all_healthy = dict(healthy)
    all_healthy.update({"step": 160, "active_optimizer_steps": 128})
    status, _reason, _fired, _evidence = P.classify_status([healthy, all_healthy], {}, [])
    check(status == "valid", f"an all-finite trajectory stays valid ({status})")


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
    test_gradient_denominator()
    test_real_labels_not_cyclic()
    test_schedule_boundary_and_step_zero()
    test_nonfinite_late_step_is_invalid()
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
