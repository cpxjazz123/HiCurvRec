"""The round worker: the facade over the steps of one round.

Every phase lives in its own module, named after the step and the agent
it runs:

    rounds/step2_ideation.py       ideate_one, run_ideation, phase_ideation
    rounds/step3_probe.py          run_probes, phase_probe
    rounds/step4_scoring.py        rank_one, run_scoring, phase_scoring
    rounds/step5_formulation.py    _formula_prompt, _review_math_prompt,
                                  phase_formulation
    rounds/step5_coding.py         _coding_prompt, phase_coding
    rounds/step5_derivation.py     _derivation_prompt, run_derivations,
                                  phase_derivation
    rounds/step6_implementation.py  run_implementation, phase_implementation

This module re-exports every name, so every existing import of
worker.phase_X keeps working and the flow stays one call away.
"""
from __future__ import annotations

# the module aliases the code and the tests reach for
from . import agents as agent_contracts  # noqa: F401
from . import baseline as baseline_module  # noqa: F401
from . import evals as evals_module  # noqa: F401
from . import evidence as evidence_module  # noqa: F401
from . import ideas as ideas_module  # noqa: F401
from . import literature as literature_module  # noqa: F401
from . import math_checks as math_checks_module  # noqa: F401
from . import module_differential  # noqa: F401
from . import pipeline as pipeline_module  # noqa: F401
from . import plaintext  # noqa: F401
from . import probes as probes_module  # noqa: F401
from . import scoring as scoring_module  # noqa: F401
from . import spec_tests as spec_tests_module  # noqa: F401
from . import specification as specification_module  # noqa: F401
from ..core.evidence import atomic_json, digest_file, read_json  # noqa: F401
from ..agents import policy, progress_server, runner  # noqa: F401

from .agent_calls import (  # noqa: F401  (the facade)
    AgentPhaseIncomplete,
    GEOMETRY_CONTRACT,
    GEOMETRY_MODULE,
    IDEA_MAX,
    IDEA_MIN,
    PHASES,
    SCORERS,
    RoundIncomplete,
    _run_agent,
    _run_agents_concurrently,
    _round_roots,
    _workspace,
    geometry_contract,
    role_of_task,
)
from .step3_probe import MATH_BUDGET_SECONDS  # noqa: F401
from .step4_scoring import HARD_OBJECTION_TERMS, MIN_SCORERS_FOR_A_MATRIX  # noqa: F401
from .step6_implementation import DIFF_LIMIT, _DIFF_IGNORED  # noqa: F401
from .step2_ideation import (  # noqa: F401  (the facade)
    round_history,
    ideate_one,
    run_ideation,
    phase_ideation,
)
from .step3_probe import (  # noqa: F401  (the facade)
    _run_probe_subprocess,
    _run_math_batch,
    _run_math_subprocess,
    run_probes,
    _probe_digest,
    _write_probe_artifacts,
    _probe_gaps,
    phase_probe,
    phase_probe_step,
)
from .step4_scoring import (  # noqa: F401  (the facade)
    rank_one,
    scoring_inputs,
    rank_order,
    apply_bounded_revision,
    run_scoring,
    _hard_objection,
    _one_bounded_revision,
    _rank_with_retry,
    _rank_order,
    run_rebuttal,
    phase_scoring,
)
from .step5_formulation import (  # noqa: F401  (the facade)
    _contract_text,
    _formula_prompt,
    _review_math_prompt,
    phase_formulation,
    _retain_spec_tests,
    _parse_math_spec,
    _parse_math_review,
)
from .step5_coding import (  # noqa: F401  (the facade)
    _coding_prompt,
    _parse_coding,
    phase_coding,
)
from .step5_derivation import (  # noqa: F401  (the facade)
    _parse_derivation,
    derivation_signature,
    _instance_codebook_width,
    _instance_problems,
    _code_signature,
    _code_accepts,
    _derivation_prompt,
    run_derivations,
    _run_differential,
    phase_derivation,
)
from .step6_implementation import (  # noqa: F401  (the facade)
    run_implementation,
    _source_files,
    scope_file,
    objective_summary,
    phase_implementation,
    advance,
)

__all__ = [
    "_workspace",
    "geometry_contract",
    "_round_roots",
    "role_of_task",
    "_run_agent",
    "_run_agents_concurrently",
    "PHASES",
    "round_history",
    "ideate_one",
    "run_ideation",
    "phase_ideation",
    "_run_probe_subprocess",
    "_run_math_batch",
    "_run_math_subprocess",
    "run_probes",
    "_probe_digest",
    "_write_probe_artifacts",
    "_probe_gaps",
    "phase_probe",
    "phase_probe_step",
    "rank_one",
    "scoring_inputs",
    "rank_order",
    "apply_bounded_revision",
    "run_scoring",
    "_hard_objection",
    "_one_bounded_revision",
    "_rank_with_retry",
    "_rank_order",
    "run_rebuttal",
    "phase_scoring",
    "_contract_text",
    "_formula_prompt",
    "_review_math_prompt",
    "phase_formulation",
    "_retain_spec_tests",
    "_parse_math_spec",
    "_parse_math_review",
    "_coding_prompt",
    "_parse_coding",
    "phase_coding",
    "_parse_derivation",
    "derivation_signature",
    "_instance_codebook_width",
    "_instance_problems",
    "_code_signature",
    "_code_accepts",
    "_derivation_prompt",
    "run_derivations",
    "_run_differential",
    "phase_derivation",
    "run_implementation",
    "_source_files",
    "scope_file",
    "objective_summary",
    "phase_implementation",
    "advance",
]
