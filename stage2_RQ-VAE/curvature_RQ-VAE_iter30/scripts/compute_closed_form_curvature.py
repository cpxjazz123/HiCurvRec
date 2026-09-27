"""Pure FCCR-1 calculations for the two registered iter30 mappings."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ITER26_MAPPING_ID = "iter26_mapping"
ITER29_MAPPING_ID = "iter29_mapping"
REGISTERED_BRANCHING = (
    19.324911558712664,
    1.4605688962651735,
    1.0148104414712726,
)
REGISTERED_RAW_RESIDUAL_MEDIANS = (1.0, 0.10941, 0.09331)
ITER26_CURVATURE = (
    0.6145357379232853,
    0.5333020920777128,
    0.3814078098431606,
)
ITER29_CURVATURE = (
    1.3660953164241916,
    0.7347829661951981,
    0.6439958072706683,
)
REGISTERED_INTERMEDIATES = {
    ITER26_MAPPING_ID: {
        "m_min": 0.09331,
        "s_l": (1.2238118890466054, 1.1604516044603779, 1.010644112691917),
        "z_l": (1.03129493309937, 0.3223997103378134, -1.3536946434371857),
    },
    ITER29_MAPPING_ID: {
        "x_l": (0.9062129756321139, 0.42206034326942476, 0.3366083742817442),
        "y_l": (0.9090909090909091, 0.5224678859653312, 0.48269618747090165),
        "u_l": (0.9076519423615115, 0.4722641146173780, 0.40965228087632294),
    },
}
FORMULA_COMPARISON_TOL = 1e-9

_CONTRACT_FIELDS = {
    "contract_version": "FCCR-1",
    "curvature_source": "closed_form",
    "curvature_trainable": False,
    "curvature_time_varying": False,
    "uses_cyclic_schedule": False,
    "uses_curvature_regularization": False,
    "new_curvature_conditioned_optimizer": False,
    "new_curvature_conditioned_aux_loss": False,
    "formula_inputs": ["behavior_branching", "raw_residual_median"],
}


def _numeric_vector(name: str, values: object) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must be an ordered length-three vector")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{index}] must be numeric, not {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def _compare_vector(name: str, observed: object, expected: tuple[float, ...]) -> None:
    values = _numeric_vector(name, observed)
    if not np.allclose(values, expected, rtol=0.0, atol=FORMULA_COMPARISON_TOL):
        raise ValueError(
            f"{name} disagrees with its registered FCCR-1 value: "
            f"observed={values} expected={list(expected)}"
        )


def validate_registered_inputs(branching: object, raw_residual_medians: object) -> tuple[list[float], list[float]]:
    b = _numeric_vector("branching", branching)
    m = _numeric_vector("raw_residual_medians", raw_residual_medians)
    if any(value < 0.0 for value in b):
        raise ValueError("branching values must be nonnegative")
    if any(value <= 0.0 for value in m):
        raise ValueError("raw_residual_medians values must be positive")
    if b != list(REGISTERED_BRANCHING):
        raise ValueError(
            f"branching differs from the registered FCCR-1 input: {b}"
        )
    if m != list(REGISTERED_RAW_RESIDUAL_MEDIANS):
        raise ValueError(
            "raw_residual_medians differs from the registered explicit input: "
            f"{m}"
        )
    return b, m


def compute_closed_form_curvature(
    branching: object, raw_residual_medians: object, mapping_id: str
) -> dict[str, Any]:
    """Compute one registered mapping from explicit ordered [L0,L1,L2] values."""
    b, m = validate_registered_inputs(branching, raw_residual_medians)
    b_array = np.asarray(b, dtype=np.float64)
    m_array = np.asarray(m, dtype=np.float64)

    if mapping_id == ITER26_MAPPING_ID:
        m_min = float(m_array.min())
        s = np.log1p(b_array) / np.log1p(m_array / m_min)
        z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
        curvature = np.clip(0.5 * np.exp(0.2 * z), 0.05, 1.5)
        expected = ITER26_CURVATURE
        intermediates = {
            "m_min": m_min,
            "s_l": s.tolist(),
            "z_l": z.tolist(),
        }
    elif mapping_id == ITER29_MAPPING_ID:
        x = b_array / (b_array + 2.0)
        y = m_array / (m_array + 0.1)
        u = (x + y) / 2.0
        curvature = 0.05 + 1.45 * u
        expected = ITER29_CURVATURE
        intermediates = {
            "x_l": x.tolist(),
            "y_l": y.tolist(),
            "u_l": u.tolist(),
        }
    else:
        raise ValueError(f"unknown FCCR-1 mapping id: {mapping_id!r}")

    values = [float(value) for value in curvature.tolist()]
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError(f"{mapping_id} produced non-finite curvature")
    if any(value < 0.05 or value > 1.5 for value in values):
        raise ValueError(f"{mapping_id} produced curvature outside [0.05,1.50]")
    expected_intermediates = REGISTERED_INTERMEDIATES[mapping_id]
    for name, expected_value in expected_intermediates.items():
        observed_value = intermediates[name]
        if isinstance(expected_value, tuple):
            _compare_vector(f"{mapping_id}.{name}", observed_value, expected_value)
        elif not math.isclose(
            float(observed_value),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=FORMULA_COMPARISON_TOL,
        ):
            raise ValueError(
                f"{mapping_id}.{name} disagrees with its registered value: "
                f"{observed_value} != {expected_value}"
            )
    _compare_vector(f"{mapping_id}.closed_form_c_l", values, expected)
    return {
        "mapping_id": mapping_id,
        "branching": b,
        "raw_residual_medians": m,
        **intermediates,
        "closed_form_c_l": values,
    }


def load_closed_form_mapping(
    input_path: Path, contract_path: Path, mapping_id: str
) -> dict[str, Any]:
    """Validate the immutable input and the candidate-only S04 contract."""
    if not input_path.is_file():
        raise FileNotFoundError(f"registered mapping input missing: {input_path}")
    if not contract_path.is_file():
        raise FileNotFoundError(f"FCCR-1 contract missing: {contract_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{input_path} must contain a JSON object")
    if "branching" not in payload:
        raise ValueError(f"{input_path} missing explicit 'branching' input")
    if "raw_residual_medians" not in payload:
        raise ValueError(
            f"{input_path} missing explicit 'raw_residual_medians'; no fallback is permitted"
        )

    # Deliberately ignore legacy aliases and cached outputs in the evidence JSON.
    validate_registered_inputs(payload["branching"], payload["raw_residual_medians"])
    candidate = compute_closed_form_curvature(
        payload["branching"], payload["raw_residual_medians"], ITER29_MAPPING_ID
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError(f"{contract_path} must contain a JSON object")
    if set(contract) != set(_CONTRACT_FIELDS) | {"final_curvature_values"}:
        raise ValueError(f"{contract_path} must retain the exact ten-field S04 schema")
    for key, expected in _CONTRACT_FIELDS.items():
        observed = contract.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"{contract_path} {key}={observed!r}, expected {expected!r}"
            )
    _compare_vector(
        "mechanism_contract_iter30.final_curvature_values",
        contract.get("final_curvature_values"),
        ITER29_CURVATURE,
    )
    # The registered candidate values were checked above with the shared
    # absolute formula tolerance; exact binary-float equality is not required.
    return compute_closed_form_curvature(
        payload["branching"], payload["raw_residual_medians"], mapping_id
    )
