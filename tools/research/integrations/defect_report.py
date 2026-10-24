"""Machine report for Issue #118: defect classes, first-detect stage, DoD metrics.

Two things are produced here and nothing else:

* a replay of every known engineering defect as a live injection against the
  preflight, recording which layer catches it first — the acceptance criterion
  of P0-B says the report must name each class as PASS/FAIL with the stage that
  detected it, not merely that a test suite is green;
* the quantitative DoD counters, computed from the evidence that actually
  exists (run directories, review verdicts, the failure registry), so a baseline
  can be recomputed instead of asserted.

Read-only: it opens run evidence, never writes into a run.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from ..core import failures
from ..science import contract, preflight

REPO = Path(__file__).resolve().parents[3]
STATE = REPO / "results" / "research" / "issue117"
REPORT = STATE / "reports" / "defect_report.json"


def _legal_candidate() -> dict:
    return {
        "candidate_id": "report-legal", "behavior_problem_id": "p",
        "behavior_evidence_sha256": "a", "literature_evidence_sha256": "b",
        "next_item_hypothesis": "h",
        "math_definition": {
            "geometry_constants": "Freeze c=1.0, tau=0.1, epsilon=1e-5, delta=1e-7.",
            "operator": {"family": "poincare_squared_distance", "curvature_c": 1.0,
                         "delta": 1e-7, "radius": 0.99999, "flat_limit": "4*||x-y||^2"},
            "objective": {"anti_collapse": "usage-balanced assignment plus latent reconstruction"},
        },
        "hyperbolic_necessity": "n", "euclidean_control": "same architecture, metric only",
        "effect_site": "e", "real_trigger_measurement": "r",
        "training_only_inputs": "t", "stable_global_sid": True,
        "protocol": {"latent_dim": 32, "levels": 3, "codes_per_level": 256,
                     "hyperparameter_sweep": False, "uses_future_events": False},
        "falsifier": {
            "stage2_diagnostics": [
                {"name": "valid_l1_hard_er", "direction": "curved minus flat > 0",
                 "minimum_effect": 0.1, "seed_rule": "median of 3 seeds"},
            ],
            "stage3_behavioral_test": {"metric": "test_R@10",
                                       "direction": "curved minus flat > 0",
                                       "minimum_effect": 0.005, "seed_rule": "paired seeds"},
            "classification": {"falsified_if": "valid_l1_hard_er is zero for every seed",
                               "passed_if": "valid_l1_hard_er is positive for all seeds",
                               "mixed_if": "otherwise, or any seed is NULL"},
        },
    }


def _injections():
    """One mutation per historical defect class, with the code it must produce."""
    def mutate(change):
        candidate = _legal_candidate()
        change(candidate)
        return candidate

    return [
        ("MISSING_OUTPUT", "output", "MISSING_KEYS",
         mutate(lambda c: c.pop("falsifier"))),
        ("CAPACITY", "contract", "CAPACITY_MISMATCH",
         mutate(lambda c: c["protocol"].update(levels=4))),
        ("FUTURE_LEAKAGE", "contract", "FUTURE_LEAKAGE",
         mutate(lambda c: c["protocol"].update(uses_future_events=True))),
        ("D_VS_D2_LIMIT", "math", "FLAT_LIMIT_DECLARATION_MISMATCH",
         mutate(lambda c: c["math_definition"]["operator"].update(flat_limit="2*||x-y||"))),
        ("CLAMP_DOMINATES", "math", "CLAMP_DOMINATES_DOMAIN",
         mutate(lambda c: c["math_definition"]["operator"].update(delta=1e-2))),
        ("DOMAIN_ESCAPE", "math", "DOMAIN_ESCAPE",
         mutate(lambda c: c["math_definition"]["operator"].update(radius=1.0))),
        ("UNREGISTERED_METRIC", "falsifier", "UNREGISTERED_METRIC",
         mutate(lambda c: c["falsifier"]["stage2_diagnostics"][0].update(name="invented"))),
        ("MISSING_NULL", "falsifier", "NULL_OUTCOME_MISSING",
         mutate(lambda c: c["falsifier"]["classification"].update(mixed_if="otherwise"))),
        ("BEHAVIOURAL_AS_STAGE2", "falsifier", "BEHAVIOURAL_AS_STAGE2",
         mutate(lambda c: c["falsifier"]["stage2_diagnostics"][0].update(name="test_R@10"))),
        ("COLLAPSE_UNGUARDED", "math", "COLLAPSE_UNGUARDED",
         mutate(lambda c: c["math_definition"].pop("objective"))),
        ("DUPLICATED_THRESHOLD", "falsifier", "THRESHOLD_DUPLICATED",
         mutate(lambda c: c.update(real_trigger_measurement=(
             "require ER above %s" % contract.THRESHOLDS["valid_l1_hard_er_strict_gt"])))),
    ]


def _false_positive_controls():
    """Legal spellings that previously caused false positives."""
    def alias(candidate):
        candidate["falsifier"]["stage2_diagnostics"][0]["name"] = "valid Hard-L1 ER"

    def containers(candidate):
        candidate["falsifier"]["classification"]["mixed_if"] = (
            "otherwise, or any entry of stage2_diagnostics is NULL and "
            "stage3_behavioral_test is NULL")

    def direction_warning(candidate):
        candidate["falsifier"]["stage2_diagnostics"][0] = {
            "name": "l3_collision", "direction": "curved minus flat > 0",
            "minimum_effect": 0.001, "seed_rule": "median of 3 seeds"}

    return [("ALIAS", alias), ("CONTAINER_NAMES", containers),
            ("DIRECTION_WARNING", direction_warning)]


def defect_rows() -> list:
    rows = []
    for name, layer, code, candidate in _injections():
        report = preflight.run(candidate)
        found = [entry["code"] for entry in report["diagnostics"]]
        rows.append({"defect": name, "expected_layer": layer, "expected_code": code,
                     "observed_layer": report["stopped_at"], "observed_codes": found,
                     "verdict": report["verdict"],
                     "detected_at_expected_layer": report["stopped_at"] == layer,
                     "passed": code in found and report["stopped_at"] == layer})
    legal = preflight.run(_legal_candidate())
    rows.append({"defect": "LEGAL_CANDIDATE", "expected_layer": None, "expected_code": None,
                 "observed_layer": legal["stopped_at"], "observed_codes": [],
                 "verdict": legal["verdict"],
                 "detected_at_expected_layer": legal["verdict"] == preflight.SCIENTIFIC_REVIEW_REQUIRED,
                 "passed": legal["verdict"] == preflight.SCIENTIFIC_REVIEW_REQUIRED})
    for name, mutate in _false_positive_controls():
        candidate = _legal_candidate()
        mutate(candidate)
        report = preflight.run(candidate)
        rows.append({"defect": "FALSE_POSITIVE_" + name, "expected_layer": None,
                     "expected_code": None, "observed_layer": report["stopped_at"],
                     "observed_codes": [entry["code"] for entry in report["diagnostics"]],
                     "verdict": report["verdict"],
                     "detected_at_expected_layer": report["verdict"] ==
                     preflight.SCIENTIFIC_REVIEW_REQUIRED,
                     "passed": report["verdict"] == preflight.SCIENTIFIC_REVIEW_REQUIRED})
    return rows


def _run_dirs() -> list:
    return sorted(path for path in (STATE / "runs").glob("run_*") if path.is_dir())


def _review_verdict(run: Path) -> str | None:
    path = run / "agents" / "review" / "review.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("verdict")
    except ValueError:
        return None


def _candidate_path(run: Path) -> Path:
    return run / "agents" / "geometry" / "candidate.json"


def dod_metrics() -> dict:
    runs = _run_dirs()
    machine_repaired = 0
    avoidable_reviews = 0
    science_rejections = 0
    verdicts = {}
    for run in runs:
        machine_repaired += len(list((run / "agents" / "geometry").glob("attempt-*")))
        verdict = _review_verdict(run)
        verdicts[verdict or "none"] = verdicts.get(verdict or "none", 0) + 1
        if verdict in ("FATAL", "PARKED"):
            science_rejections += 1
        elif verdict in ("MATERIAL", "BLOCK"):
            science_rejections += 1
        if verdict is not None:
            candidate = _candidate_path(run)
            if candidate.is_file():
                try:
                    report = preflight.run_file(candidate)
                except Exception:                        # noqa: BLE001
                    report = {"verdict": None}
                if report.get("verdict") in (preflight.MACHINE_INVALID,
                                             preflight.MATH_COUNTEREXAMPLE):
                    avoidable_reviews += 1
    verified = None
    manifest = STATE / "verification" / "issue117-geometry-transition-residual-poincare-v1" / "manifest.json"
    created = None
    if runs:
        try:
            created = json.loads((runs[-1] / "run.json").read_text()).get("created_at")
        except (OSError, ValueError):
            created = None
    if manifest.is_file() and created:
        verified = manifest.stat().st_mtime - float(created)
    registry = failures.summary(STATE)
    return {
        "runs": len(runs),
        "review_verdicts": verdicts,
        "known_failure_recurrence": registry["known_failure_recurrence"],
        "registry_entries": registry["failures"],
        "machine_repair_attempts": machine_repaired,
        "avoidable_review_calls": avoidable_reviews,
        "science_rejections": science_rejections,
        "time_to_verified_candidate_seconds": verified,
        "stage2_runs": len([p for p in (REPO / "results" / "stage2_RQ-VAE").iterdir()
                            if p.is_dir() and (p / "item_sids.json").is_file()]),
    }


def build() -> dict:
    rows = defect_rows()
    payload = {
        "issue": 118,
        "generated_at_epoch": time.time(),
        "published_thresholds": dict(contract.THRESHOLDS),
        "schema_version": contract.SCHEMA_VERSION,
        "defect_classes": rows,
        "all_defects_detected": all(row["passed"] for row in rows),
        "dod": dod_metrics(),
    }
    return payload


def write(path: Path | None = None) -> Path:
    path = Path(path or REPORT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), indent=2, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":  # no-argument research entry point
    print(write())
