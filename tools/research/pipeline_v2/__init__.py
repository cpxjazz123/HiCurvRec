"""Research Pipeline V2: deterministic verification first, agents for science.

The five-role review loop was a verification system about the verification system:
it spent its budget deciding which reviewer should read which file, and its own
defects blocked the experiment. V2 splits the two questions:

* **is the candidate correct?** -- answered by programs. The geometry contract,
  the candidate's own tests, a real gradient check, a real two-rank DDP run, a real
  one-epoch entry point and a real export/assignment comparison. The framework runs
  them and records exit codes and output; no language model votes.
* **is the mechanism worth testing?** -- answered by one independent scientific
  reviewer that reads the WHOLE candidate tree (that tree only, plus the contract
  and the test results) and judges the coupling, the mathematics and the claim.

States are explicit and mutually exclusive, one controller writes them, and an
infrastructure failure is never a scientific verdict.
"""
from __future__ import annotations

from . import states  # noqa: F401
from .controller import Controller  # noqa: F401
