"""Versioned contract compiler: the one machine-checkable source of truth.

Issue #118 P0-A. Everything that every agent, reviewer, manifest and test has to
agree on lives here and only here:

* the frozen Stage2/Stage3 protocol and SID capacity,
* the admission thresholds, read from the authoritative gate rather than copied,
* the metric registry with each metric's definition, unit, direction, denominator
  and the module that computes it,
* the canonical suite identifiers and their filenames,
* the audited operator families that a proposal may name, with the numerics the
  math preflight actually evaluates.

A candidate compiles into a staged contract (behavior -> geometry -> math ->
controls -> stage3) where each section carries its own identity and digest, so a
later stage can reject a mixed-version input instead of guessing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

SCHEMA_VERSION = 1

REPO = Path(__file__).resolve().parents[3]
GATE_PATH = REPO / "tools/genrec-mcp/stage2_quality_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("stage2_quality_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate()

#: Admission thresholds are READ from the gate; copying them would create the
#: second hard-coded source of truth that this issue exists to remove.
THRESHOLDS = {
    "valid_l1_hard_er_strict_gt": float(_GATE.MIN_VALID_L1_HARD_ER),
    "level2_prefixes_strict_gt": int(_GATE.FROZEN_BASELINE_L2_PREFIXES),
}

FROZEN = {
    "capacity": {"latent_dim": 32, "levels": 3, "codes_per_level": 256, "tokens": 3},
    "stage2": {"epochs": 3000, "early_stop": False, "hyperparameter_sweep": False},
    "stage3": {"epochs": 150, "seed": 42, "beam_size": 20, "no_eval": True,
               "skip_test": False, "metric": "test_R@10",
               "target_strict_gt": 0.065},
    "data": {"split": "frozen train/valid/test",
             "sid_supervision": "train split interactions only",
             "forbidden": ["validation split", "test split", "future events",
                           "prediction-time user context", "evaluation batch composition"]},
    "control": {"same_capacity": True, "same_optimizer_and_seeds": True,
                "flat_limit_of_same_implementation": True,
                "differs_only_in": ["distance formula"]},
}

#: Canonical suite identifiers. A reviewer may name a suite by file; both forms
#: resolve here so identity comparison never depends on spelling.
SUITE_ALIASES = {
    "math": "test_math.py",
    "gradient": "test_gradient.py",
    "ddp": "test_ddp.py",
    "entrypoint": "test_entrypoint.py",
}
SUITE_IDS = tuple(SUITE_ALIASES)


def canonical_suite(name: str) -> str | None:
    """Map ``math`` / ``test_math.py`` / ``math.py`` to the canonical suite id."""
    text = str(name).strip()
    if text.endswith(".py"):
        text = text[: -len(".py")]
    if text.startswith("test_"):
        text = text[len("test_"):]
    return text if text in SUITE_IDS else None


#: Metric registry. ``source`` names the module that computes the value; a
#: proposal may only reference metrics listed here.
METRICS = {
    "valid_l1_hard_er": {
        "definition": "10 * positive_hits / negative_hits over the frozen validation pairs; "
                      "a hit means the anchor item's first SID token equals the candidate's",
        "unit": "ratio", "direction": "higher", "denominator": "negative_hits",
        "source": "tools/genrec-mcp/stage2_quality_gate.py::_measures",
        "role": "admission_hard_gate",
    },
    "valid_l1_positive_hit_rate": {
        "definition": "positive_hits / pairs over the frozen validation pairs",
        "unit": "ratio", "direction": "higher", "denominator": "pairs",
        "source": "tools/genrec-mcp/stage2_quality_gate.py::_measures",
        "role": "diagnostic_only",
    },
    "valid_l1_negative_hit_rate": {
        "definition": "negative_hits / (pairs * negatives_per_pair)",
        "unit": "ratio", "direction": "lower", "denominator": "pairs * 10",
        "source": "tools/genrec-mcp/stage2_quality_gate.py::_measures",
        "role": "diagnostic_only",
    },
    "level2_distinct_prefixes": {
        "definition": "count of distinct (k1, k2) pairs over all exported items",
        "unit": "count", "direction": "higher", "denominator": "none",
        "source": "tools/genrec-mcp/stage2_quality_gate.py::_measures",
        "role": "admission_hard_gate",
    },
    "level3_distinct_triples": {
        "definition": "count of distinct (k1, k2, k3) triples over all exported items",
        "unit": "count", "direction": "higher", "denominator": "none",
        "source": "tools/research/science/sid_layer_analysis.py::l3_collision",
        "role": "diagnostic_only",
    },
    "l3_collision": {
        "definition": "item-weighted share of items whose three-token SID is shared",
        "unit": "ratio", "direction": "lower", "denominator": "items",
        "source": "tools/research/science/sid_layer_analysis.py::l3_collision",
        "role": "diagnostic_only",
    },
    "l3_refinement_gain_bits": {
        "definition": "item-weighted mean of log2(|bucket(k1,k2)| / |bucket(k1,k2,k3)|)",
        "unit": "bits", "direction": "lower", "denominator": "items",
        "source": "tools/research/science/sid_layer_analysis.py::l3_refinement_gain",
        "role": "diagnostic_only",
    },
    "conditional_l3_gain": {
        "definition": "mean log2 of the train-successor likelihood ratio of the true next "
                      "item's third code against a Laplace-smoothed prefix-only predictor",
        "unit": "bits", "direction": "higher", "denominator": "validation pairs",
        "source": "tools/research/science/sid_layer_analysis.py::conditional_l3_gain",
        "role": "diagnostic_only",
    },
    "test_R@10": {
        "definition": "Stage3 Recall@10 on the frozen test split",
        "unit": "ratio", "direction": "higher", "denominator": "test users",
        "source": "results/stage3_T5Train/<variant>/logs/*/test_final.json",
        "role": "behavioural_test",
    },
    "test_NDCG@10": {
        "definition": "Stage3 NDCG@10 on the frozen test split",
        "unit": "ratio", "direction": "higher", "denominator": "test users",
        "source": "results/stage3_T5Train/<variant>/logs/*/test_final.json",
        "role": "behavioural_test",
    },
}
DIAGNOSTIC_METRICS = tuple(name for name, meta in METRICS.items()
                           if meta["role"] == "diagnostic_only")
BEHAVIOURAL_METRICS = tuple(name for name, meta in METRICS.items()
                            if meta["role"] == "behavioural_test")

#: Operator families the math preflight can actually evaluate. A proposal names
#: one; the compiler evaluates the audited reference, never the proposal's prose.
OPERATOR_FAMILIES = {
    "poincare_squared_distance": {
        "requires": ("curvature_c", "delta"),
        "optional": ("radius", "temperature"),
        "distance": "D_c^2(x,y) = arcosh(1 + 2c||x-y||^2/((1-c||x||^2)(1-c||y||^2)))^2 / c",
        "flat_limit": "4*||x-y||^2",
        "checks": ("flat_limit", "zero_vector", "boundary_margin", "domain"),
    },
}

NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_numbers(text: str) -> dict:
    """Named constants a proposal states, e.g. ``c=1.0`` or ``delta = 1e-7``."""
    found = {}
    for name, value in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", str(text)):
        try:
            # First declaration wins: a frozen constant is stated before any
            # derived mention of it ("freeze c=1.0 ... the c=0 limit").
            found.setdefault(name, float(value))
        except ValueError:
            continue
    return found


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _section(name: str, payload, dependencies=()) -> dict:
    return {"id": name, "schema_version": SCHEMA_VERSION,
            "sha256": _digest(payload), "dependencies": list(dependencies),
            "payload": payload}


def build_operator(math_definition) -> dict:
    """Resolve a proposal's math_definition into an evaluable operator.

    A structured ``operator`` mapping wins; otherwise the frozen constants are
    read from the declared text. Without a recognised family and curvature the
    contract cannot be compiled, which is a machine-repairable defect.
    """
    if not isinstance(math_definition, dict):
        return {"family": None, "error": "math_definition must be an object"}
    structured = math_definition.get("operator")
    if isinstance(structured, dict):
        family = structured.get("family")
        params = dict(structured)
    else:
        # Declarations first: the frozen constants section, then the rest, so a
        # derived mention cannot overwrite a declared value.
        ordered = [math_definition.get("geometry_constants"),
                   math_definition.get("scored_distance"),
                   math_definition.get("scored_squared_distance")]
        ordered += [value for key, value in math_definition.items()
                    if key not in ("geometry_constants", "scored_distance",
                                   "scored_squared_distance")]
        text = " ".join(str(value) for value in ordered if value is not None)
        numbers = extract_numbers(text)
        params = dict(numbers)
        family = None
        if "arcosh" in text or "poincare" in text.lower():
            family = "poincare_squared_distance"
    if family not in OPERATOR_FAMILIES:
        return {"family": None, "error": "math_definition.operator must name a supported family: "
                                         + ", ".join(sorted(OPERATOR_FAMILIES))}
    spec = OPERATOR_FAMILIES[family]
    aliases = {"curvature_c": ("curvature_c", "c"), "delta": ("delta",),
               "temperature": ("temperature", "tau")}
    missing = [key for key in spec["requires"]
               if not any(alias in params for alias in aliases.get(key, (key,)))]
    if missing:
        return {"family": family, "error": "operator is missing frozen constants: "
                                           + ", ".join(sorted(set(missing)))}
    curvature = params.get("curvature_c", params.get("c"))
    # The projection radius is derived, not declared: a proposal states
    # "R = 1 - epsilon", so the audited default 1 - epsilon applies unless a
    # structured radius is given. Taking the literal R from prose would read the
    # "1" of "1 - epsilon".
    radius = params.get("radius")
    if radius is None:
        radius = 1.0 - float(params.get("epsilon", 1e-5))
    return {"family": family,
            "curvature_c": float(curvature),
            "delta": float(params["delta"]),
            "radius": float(radius),
            "temperature": float(params.get("temperature", params.get("tau", 0.1)))}


def compile_contract(candidate: dict) -> dict:
    """Compile a candidate into the staged, hashable contract."""
    protocol = candidate.get("protocol") or {}
    falsifier = candidate.get("falsifier") or {}
    operator = build_operator(candidate.get("math_definition"))
    sections = [
        _section("behavior", {
            "problem_id": candidate.get("behavior_problem_id"),
            "evidence_sha256": candidate.get("behavior_evidence_sha256"),
        }),
        _section("geometry", {
            "candidate_id": candidate.get("candidate_id"),
            "hypothesis": candidate.get("next_item_hypothesis"),
            "necessity": candidate.get("hyperbolic_necessity"),
            "effect_site": candidate.get("effect_site"),
        }, ("behavior",)),
        _section("math", {"operator": operator,
                          "losses": (candidate.get("math_definition") or {}).get("loss_equations")},
                 ("geometry",)),
        _section("controls", {"euclidean": candidate.get("euclidean_control"),
                              "requirements": FROZEN["control"]}, ("math",)),
        _section("falsifier", {
            "stage2_diagnostics": falsifier.get("stage2_diagnostics"),
            "stage3_behavioral_test": falsifier.get("stage3_behavioral_test"),
            "classification": falsifier.get("classification"),
        }, ("math",)),
        _section("stage3", {
            "protocol": FROZEN["stage3"],
            "pre_registered_metric": (falsifier.get("stage3_behavioral_test") or {}).get("metric"),
        }, ("falsifier",)),
    ]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.get("candidate_id"),
        "protocol": {**FROZEN["capacity"], **{k: v for k, v in protocol.items()
                                              if k in ("latent_dim", "levels", "codes_per_level")},
                     "hyperparameter_sweep": bool(protocol.get("hyperparameter_sweep")),
                     "uses_future_events": bool(protocol.get("uses_future_events")),
                     "epochs": FROZEN["stage2"]["epochs"]},
        "thresholds": dict(THRESHOLDS),
        "sections": sections,
    }
    contract["contract_sha256"] = _digest(contract)
    contract["prompt_input_digest"] = _digest({
        "prompt_schema": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "metrics": sorted(METRICS),
        "suite_ids": list(SUITE_IDS),
    })
    return contract


def section(contract: dict, name: str) -> dict | None:
    for entry in contract.get("sections", []):
        if entry.get("id") == name:
            return entry
    return None


def digests(contract: dict) -> dict:
    """Identity of every staged section, for mixed-version rejection."""
    return {entry["id"]: entry["sha256"] for entry in contract.get("sections", [])}

#: Documented label aliases for the metric registry. Prose names are canonicalised
#: here once, so no agent, reviewer or test needs its own spelling table.
METRIC_LABELS = {
    "valid_l1_hard_er": ("valid hard-l1 er", "valid hard l1 er", "hard_l1_er",
                         "valid_l1_er", "valid l1 hard er"),
    "valid_l1_positive_hit_rate": ("valid l1 positive hit rate", "l1 positive hit rate"),
    "valid_l1_negative_hit_rate": ("valid l1 negative hit rate", "l1 negative hit rate"),
    "level2_distinct_prefixes": ("level 2 distinct prefixes", "l2 distinct prefixes",
                                 "level2 prefixes"),
    "level3_distinct_triples": ("level 3 distinct triples", "l3 distinct triples"),
    "l3_collision": ("l3 collision", "level 3 collision"),
    "l3_refinement_gain_bits": ("l3 refinement gain", "l3 refinement gain bits",
                                "level 3 refinement gain"),
    "conditional_l3_gain": ("conditional g3", "conditional l3 gain", "g3"),
    "test_R@10": ("test r@10", "test recall@10"),
    "test_NDCG@10": ("test ndcg@10", "test ndcg@10"),
}


def _normalise(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-z@]+", "_", text)
    return text.strip("_")


def canonical_metric(name: str) -> str | None:
    """Resolve a metric identifier or documented label to its registry key."""
    probe = _normalise(name)
    if probe in METRICS:
        return probe
    for metric, labels in METRIC_LABELS.items():
        if probe in {_normalise(label) for label in labels}:
            return metric
    return None
