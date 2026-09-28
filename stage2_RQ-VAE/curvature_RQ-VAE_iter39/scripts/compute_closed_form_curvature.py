"""Compute and persist iter29's fixed FCCR-1 curvature mapping."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


B_REF = 2.0
M_REF = 0.1
C_MIN = 0.05
C_MAX = 1.50
MAPPING_ID = "FCCR-1_bounded_rational_additive"
REGISTERED_BRANCHING = (
    19.324911558712664,
    1.4605688962651735,
    1.0148104414712726,
)
REGISTERED_RAW_RESIDUAL_MEDIANS = (1.0, 0.10941, 0.09331)



def _numeric_vector(name: str, values: object) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must be an ordered length-three vector")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{index}] must be a JSON numeric value")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def compute_closed_form_curvature(
    branching: object, raw_residual_medians: object
) -> dict[str, list[float]]:
    """Return FCCR-1 intermediates and curvature for ordered L0/L1/L2 inputs."""
    branching_values = _numeric_vector("branching", branching)
    residual_values = _numeric_vector("raw_residual_medians", raw_residual_medians)
    if any(value < 0.0 for value in branching_values):
        raise ValueError("branching values must be nonnegative")
    if any(value <= 0.0 for value in residual_values):
        raise ValueError("raw_residual_medians values must be positive")

    x_values = [value / (value + B_REF) for value in branching_values]
    y_values = [value / (value + M_REF) for value in residual_values]
    u_values = [(x + y) / 2.0 for x, y in zip(x_values, y_values)]
    curvature_values = [C_MIN + (C_MAX - C_MIN) * value for value in u_values]
    for name, values in (
        ("x_l", x_values),
        ("y_l", y_values),
        ("u_l", u_values),
        ("closed_form_c_l", curvature_values),
    ):
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"computed {name} contains a non-finite value")
    if not all(C_MIN <= value <= C_MAX for value in curvature_values):
        raise ValueError(f"computed curvature is outside [{C_MIN}, {C_MAX}]")
    return {
        "x_l": x_values,
        "y_l": y_values,
        "u_l": u_values,
        "closed_form_c_l": curvature_values,
    }


def _write_json_atomically(path: Path, payload: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    payload_path = Path(__file__).resolve().with_name("computed_behavior_branching.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{payload_path} must contain a JSON object")
    if "branching" not in payload:
        raise KeyError("computed_behavior_branching.json missing 'branching'")
    if "raw_residual_medians" not in payload:
        raise KeyError(
            "computed_behavior_branching.json missing 'raw_residual_medians'; "
            "no residual fallback is permitted"
        )

    derived = compute_closed_form_curvature(
        payload["branching"], payload["raw_residual_medians"]
    )
    updated = dict(payload)
    for obsolete_key in (
        "residual_norm",
        "raw_capacity",
        "branch_mid",
        "s_l",
        "z_l",
        "closed_form_c_base",
        "closed_form_alpha",
        "m_min",
        "closed_form_source",
    ):
        updated.pop(obsolete_key, None)
    updated.update(
        {
            "closed_form_mapping": MAPPING_ID,
            "closed_form_constants": {
                "B_ref": B_REF,
                "m_ref": M_REF,
                "c_min": C_MIN,
                "c_max": C_MAX,
            },
            **derived,
        }
    )
    _write_json_atomically(payload_path, updated)
    print(f"mapping={MAPPING_ID}")
    print(f"constants={updated['closed_form_constants']}")
    print(f"x_l={derived['x_l']}")
    print(f"y_l={derived['y_l']}")
    print(f"u_l={derived['u_l']}")
    print(f"closed_form_c_l={derived['closed_form_c_l']}")


if __name__ == "__main__":
    main()
