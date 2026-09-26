#!/usr/bin/env python3
"""Deprecated skill-level SID evaluator.

The old script mixed:
- environment-variable iteration selection;
- TIGER comparisons from a different analysis context;
- HitRate@50, which current CLAUDE.md explicitly says is not an SID-quality gate.

Under the current workflow, use:
- the iteration trainer's built-in SID metrics for descriptive logging;
- protocol-compatible Stage3 test_final.json for downstream decisions;
- a dedicated analysis script only when its exact metric definition is preregistered.

This file exits deliberately to prevent stale proxy logic from driving new
iterations.
"""

raise RuntimeError(
    "Skill-level eval_sids.py is retired. Use iteration-native descriptive SID "
    "logs and protocol-compatible Stage3 test_final.json."
)
