"""Write closed-form per-layer curvature from behavior branching + residual scales."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "computed_behavior_branching.json"
OUTPUT_PATH = INPUT_PATH
CLOSED_FORM_C_BASE = 0.5
CLOSED_FORM_ALPHA = 0.2

def closed_form_curvatures(branching, residual_norm, c_base=CLOSED_FORM_C_BASE, alpha=CLOSED_FORM_ALPHA):
    b = np.asarray(branching, dtype=np.float64)
    m = np.asarray(residual_norm, dtype=np.float64)
    m_min = float(np.min(m))
    s = np.log1p(b) / np.log1p(m / m_min)
    z = (s - s.mean()) / (s.std() + 1e-12)
    c = np.clip(c_base * np.exp(alpha * z), 0.05, 1.5)
    return {"s_l": s.tolist(), "z_l": z.tolist(), "closed_form_c_l": c.tolist(),
            "closed_form_c_base": c_base, "closed_form_alpha": alpha, "m_min": m_min}

def main():
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    extra = closed_form_curvatures(payload["branching"], payload["residual_norm"])
    payload.update(extra)
    payload["closed_form_source"] = "compute_closed_form_curvature.py"
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("closed_form_c_l =", extra["closed_form_c_l"])

if __name__ == "__main__":
    main()
