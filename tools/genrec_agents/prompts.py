"""This repository's surface text and the prompt parts a task hands its harness.

The text used to be registered into a third-party framework's prompt builder by monkey-patching
a module-level dict. It is first-party now: the surface text is ours, the two prompt parts are
ours, and the assembly is a plain function with no framework behind it.

Four parts, and what each is for:

* ``SURFACE_MD`` -- what the agent may change and what it may not.
* ``INV_TEMPLATE`` -- the INV part: how the agent WORKS (procedure, tool use, completion).
* ``DOM_TEMPLATE`` -- the DOM part: what a practitioner of the domain knows.
* ``levers`` -- the one lever each part is given, because a prompt that names no lever is a
  prompt that cannot be acted on.
"""

from __future__ import annotations

from .surface import GENREC_SURFACE

SURFACE_MD = """\
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



#: The lever the INV part is given: the one standing rule that decides when an arm is done.
INV_LEVER = (
    "a standing rule in `CLAUDE.md` that names the check to run before an arm is "
    "declared done -- for example, refusing to call a mechanism verified when its "
    "independent check did not reproduce the fixture"
)

#: The lever the DOM part is given: the trigger line that makes a skill exist.
DOM_LEVER = (
    "A skill exists only once its name appears in `CLAUDE.md` as a trigger: write that "
    "trigger line in the same edit."
)

INV_TEMPLATE = """\
# YOUR DIMENSION: INV -- how the agent WORKS

You own procedure, tool use and completion -- not subject matter. Trace the workflow, fix its
earliest observed divergence, and match each failure to the cheapest effective component:

- TOOL ERROR: distinguish bad input from timeout, missing capability, permission, or unread output.
  Give one bounded fallback and stop condition; never prescribe blind repetition.
- MECHANICAL REPETITION: use `scripts/<name>.py` for repeated deterministic extraction,
  conversion or checking, with clear inputs, outputs and failure behavior.
- TRIGGERED WORKFLOW: use an on-demand skill for a narrow multi-step procedure, name its trigger
  in always-on text, and carry any helper it requires.
- IGNORED INVARIANT: only repeated evidence that clear prose is skipped can justify a defensive
  __LOOP_LEVER__. It must not block valid completion.
- ORDINARY SEQUENCING: prefer concise always-on guidance. Do not add code for a one-off mistake
  or automate professional judgment.

Define "enough evidence," reserve room for every artifact, and reopen or mechanically check the
{deliverable}; a successful write call does not establish usability.

{levers}

Do not add professional rules, required domain sections or matter facts. A parallel candidate
owns them; stay focused on how the agent works."""

DOM_TEMPLATE = """\
# YOUR DIMENSION: DOM -- what a PRACTITIONER knows

You own the practitioner model the agent lacks: the {deliverable}, audience and decision,
controlling sources, required structure, consistency relationships, and expert omissions.

Turn it into checkable rules:

- ANALYTIC MOVE: name the operation -- compare, reconcile, calculate, classify, trace, draft or
  explain -- its inputs and required result.
- ARTIFACT CONTRACT: name required parts, their dependencies, and domain checks for consistency
  and practical use.
- DECISION RULE: state the trigger, supporting evidence, conflicts to resolve, and uncertainty
  that must remain explicit.
- ENCODING: choose the FORM as deliberately as the content. A skill is the default home for
  domain knowledge that has a recognisable trigger -- what a given instrument or document type
  requires, the anatomy a practitioner expects, the checks that apply to one work type. It costs
  nothing until that situation arises, and the next task in this {domain} can invoke it.
  __TRIGGER_SENTENCE__
  Always-on memory is for the short, universal facts that apply to EVERY task here, and it is
  paid for on every run -- so if you add some, cut some. Never store this matter's facts as
  durable memory: they are wrong anchors for the next task.

{levers}

Two hard rules:
- Never invent an authority, identifier, standard or citation. Point to an authorized source;
  asserting it requires evidence.
- State knowledge the agent can APPLY, not knowledge it can recite."""


def register() -> None:
    """Kept for the callers that used to register the text with the framework.

    There is no framework to register with any more: the text is a module-level constant and
    the assembly is :func:`build_prompt_parts`.
    """


def build_prompt_parts(*, deliverable: str = "deliverable", domain: str = "domain",
                       levers: str = "") -> dict:
    """The four prompt parts, as the task's adapter asks for them."""
    return {
        "surface_md": SURFACE_MD,
        "inv": INV_TEMPLATE.replace("__LOOP_LEVER__", INV_LEVER).format(
            deliverable=deliverable, levers=levers),
        "dom": DOM_TEMPLATE.replace("__TRIGGER_SENTENCE__", DOM_LEVER).format(
            deliverable=deliverable, domain=domain, levers=levers),
        "tts_surface": "\n\n".join([
            surface_part(SURFACE_MD, "THREE RULES THAT DISCARD", "THE DELIVERABLE'S SHAPE"),
            surface_part(SURFACE_MD, "# HOW TO MAKE A CHANGE"),
        ]),
    }


def surface_part(text: str, *markers: str) -> str:
    """The slice of the surface text between the first marker and the second.

    The framework exposed this as a private helper; it is three lines and the surface text is
    ours, so it lives here now.
    """
    low = text.lower()
    start = 0
    for marker in markers:
        index = low.find(marker.lower())
        if index < 0:
            continue
        if start == 0 and marker is not markers[0]:
            start = index
        elif marker is markers[-1]:
            return text[start:index]
    return text[start:]


__all__ = ["SURFACE_MD", "INV_TEMPLATE", "DOM_TEMPLATE", "INV_LEVER", "DOM_LEVER",
           "build_prompt_parts", "surface_part", "register", "GENREC_SURFACE"]
