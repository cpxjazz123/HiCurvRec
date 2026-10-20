"""The candidate-editable surface of the GeneRec iteration harness.

Issue #101 fixes the three editable kinds of file:

* ``CLAUDE.md`` -- the always-on protocol (c1),
* ``.agent/skills/<name>/SKILL.md`` -- the reusable workflows (c6),
* ``scripts/<name>.py`` -- the helper scripts the skills invoke (c7).

Frozen math, thresholds, oracles, judges and trainer evaluation code are NOT on this
surface: the framework reads a tree, validates it, and only the paths declared here
can ever reach a candidate.
"""

from __future__ import annotations

from harness2.core.tree import SurfaceSpec

GENREC_SURFACE = SurfaceSpec(
    name="genrec",
    always_on=("CLAUDE.md",),
    always_on_max=60_000,
    always_on_cap_reason=(
        "the protocol competes with the agent's own reasoning budget; keep it under the cap"
    ),
    skill_dir=".agent/skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Add a trigger line naming `{name}` to CLAUDE.md, e.g. "
        '"When <the situation this skill handles>, load the `{name}` skill."'
    ),
    autotrigger_file="CLAUDE.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    compile_skill_scripts=False,
    reject_empty_always_on=True,
)

__all__ = ["GENREC_SURFACE"]
