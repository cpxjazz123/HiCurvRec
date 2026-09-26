#!/usr/bin/env python3
"""FCCR-1 MVG template.

This skill-level file intentionally does NOT implement a generic learnable-parameter
MVG. The previous version assumed every new mechanism had trainable parameters and
required gradient/update checks on them, which conflicts with the active FCCR-1
fixed-curvature contract.

For a new iteration:
1. run scripts/preflight_contract.py from the iteration directory;
2. create/maintain an iteration-local scripts/mvg_check.py that imports that exact
   iteration's training module without CLI arguments;
3. verify the FCCR-1 checks below.

Required FCCR-1 checks:
- Formula: runtime fixed c_l equals preregistered contract values.
- Immutability: fixed curvature requires_grad=False and is absent from optimizer.
- Time invariance: c_l identical at steps 0/25k/50k/100k.
- Model health: non-curvature model parameters still receive finite nonzero grads.
- Counterfactual activation: candidate fixed c_l changes a preregistered direct
  signal relative to the declared baseline curvature configuration.

The iteration-local MVG must print exactly 'MVG PASS' only after all checks pass.

This file exits deliberately so an agent cannot accidentally run the obsolete
generic MVG and treat it as evidence for FCCR-1.
"""

raise RuntimeError(
    "Skill-level generic MVG is retired under FCCR-1. "
    "Run preflight_contract.py, then use an iteration-local no-CLI MVG that "
    "verifies fixed-curvature formula, immutability, time invariance, model "
    "gradient health, and a baseline counterfactual."
)
