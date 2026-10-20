"""Register this repository's surface text with the framework's prompt builder.

The framework resolves its per-surface prompt text from a module-level dict built at
import time (``core/prompts.py:617``), keyed by the surface name, and exposes no public
registration hook. Our surface is not one of the two the framework ships text for, so
this module adds an entry for it at import time.

Nothing in the framework's source is modified and no fork is kept: if the framework ever
grows a public hook, this file becomes one call and is deleted. The extension is
fail-closed -- the framework raises on an unknown surface name, so a framework upgrade
that renames the dict breaks loudly here instead of degrading silently.
"""

from __future__ import annotations

from harness2.core import prompts as P

from .surface import GENREC_SURFACE

_GENREC_SURFACE_MD = """\
# WHAT YOU CAN CHANGE

The harness is the iteration protocol of this repository. It has three components, and each
costs the agent differently:

| path | what it is | when it is in context |
|---|---|---|
| `CLAUDE.md` | the standing protocol of an iteration | ALWAYS ON -- every token competes with the agent's own reasoning budget |
| `.agent/skills/<name>/SKILL.md` | a reusable workflow | loaded ON DEMAND, only when it is invoked |
| `scripts/<name>.py` | a deterministic helper | run on demand |

Everything else in the repository is FROZEN: the mechanism math, the thresholds, the
independent references, the held-out data and the trainer's evaluation code. A proposal that
edits any of them is discarded, whatever it argues.

The two that are easy to get wrong:

`skills/` -- frontmatter needs `name:` matching the directory and a `description:`, or the
skill is dropped silently at load. Two further rules below decide whether it survives
validation at all: it must be named in `CLAUDE.md`, and any helper it calls must travel with
it.

`scripts/` -- a helper earns its place when the runs hand-rolled the same deterministic
operation, got it wrong, or did it slowly. A helper that produces a verdict must COMPUTE
that verdict: restating a declared value is a forbidden substitution, because a value that
agrees with itself proves nothing.

Change only what a trajectory shows needs changing -- but WHERE you change something, the FORM
matters as much as the content, and the cheap form is not the obvious one.

ALWAYS-ON PROSE IS THE EXPENSIVE OPTION. Every word of it is in context on every run,
competing with the agent's own output budget. The most costly failure on this work is a run
that reasons until the budget is gone and produces nothing. Adding prose makes that MORE
likely, not less. An on-demand component costs nothing until the moment it is invoked.

So prefer, in this order:
  - a SKILL for anything with a recognisable trigger -- a procedure for a mechanism family, a
    naming rule, a recurring situation. This is the default form for new instruction.
  - a SCRIPT for a deterministic operation the runs hand-rolled and got wrong, or did slowly.
  - ALWAYS-ON TEXT last, and only for a short rule that genuinely applies to every iteration.
    If you add always-on text, cut something else to pay for it.

THREE RULES THAT DISCARD THE WHOLE PROPOSAL. Validation runs before any rollout, and a
harness that fails it is thrown away entire.

1. EVERY SKILL NEEDS A TRIGGER LINE -- the most violated rule here by a wide margin. If you
   create or keep `.agent/skills/<name>/SKILL.md`, its directory name must appear VERBATIM in
   `CLAUDE.md`, in the SAME edit:

       When <the situation this skill handles>, load the `<name>` skill.

   An unnamed skill is never opened anyway: the line is what makes it exist.

2. NO FROZEN VALUE IN A CANDIDATE EDIT. The thresholds, the independent references and the
   held-out fixtures are not candidate-editable, and a helper may not restate a declared
   value: it must compute it from the input. A proposal that does either is discarded before
   it is ever run.

3. EVERY SCRIPT YOU NAME MUST EXIST. A `scripts/<file>.py` invoked from a SKILL.md or from
   `CLAUDE.md` must ship in the same proposal; one dropped loose in the skill directory is OFF
   THE EDITABLE SURFACE and silently discarded.

THE DELIVERABLE'S SHAPE BELONGS TO THE REQUEST, NOT TO YOU. A harness changes HOW the agent
works; it must never change WHAT SHAPE it delivers. Where the task names fields, files, paths
or an order, that shape is part of what was asked for. A component that tells the agent to
report a verdict it did not compute, or to surface intermediate values so a reviewer can check
them, is CHANGING THE DELIVERABLE. A value the task asks for must be delivered AS A VALUE.

# HOW TO MAKE A CHANGE

The harness is in `context/harness/`, and it is YOURS TO EDIT. Open the files, change them in
place with your normal tools, and create new ones where a new component is the right lever.
Your edits on disk ARE the proposal -- there is no separate format to reproduce them in, and
nothing you write in this reply changes the harness.

- Files you do not touch are inherited unchanged. Do not rewrite a file to keep it the same.
- Delete a file to remove that component.
- If the evidence does not argue for any change, leave everything as it is and say KEEP_BASE
  in your notes. That is a real answer, not a failure: adaptation the evidence does not support
  costs context and buys nothing.

Then reply with ONE `<notes>` block and nothing else: for each change, the moment in a
trajectory that motivated it. The notes are read by a person, not applied by a machine."""


def register() -> None:
    """Add this surface's prompt text. Idempotent."""
    if GENREC_SURFACE.name in P._SURFACE_TEXTS:
        return
    P._SURFACE_TEXTS[GENREC_SURFACE.name] = P._SurfaceText(
        surface_md=_GENREC_SURFACE_MD,
        inv=P._INV_TEMPLATE.replace(
            "__LOOP_LEVER__",
            "a standing rule in `CLAUDE.md` that names the check to run before an arm is "
            "declared done -- for example, refusing to call a mechanism verified when its "
            "independent check did not reproduce the fixture",
        ),
        dom=P._DOM_TEMPLATE.replace(
            "__TRIGGER_SENTENCE__",
            "A skill exists only once its name appears in `CLAUDE.md` as a trigger: write that "
            "trigger line in the same edit.",
        ),
        tts_surface="\n\n".join(
            [
                P._surface_part(_GENREC_SURFACE_MD, "THREE RULES THAT DISCARD", "THE DELIVERABLE'S SHAPE"),
                P._surface_part(_GENREC_SURFACE_MD, "# HOW TO MAKE A CHANGE"),
            ]
        ),
    )


__all__ = ["register"]
