"""Export the framework's agent configurations, one directory per role.

Every agent the round flow runs has a configuration that decides what it may
read, what it may write, how long it has, what shape its answer takes and what
it is told to do. That configuration lives in the code - the prompt builders in
``tools/research/rounds/worker.py``, the instructions and schemas in
``tools/research/agents/runtime.py``, the boundary in ``agents/policy.py`` and the
budget in ``tools/research/settings.py``.

This module WRITES that configuration out, one directory per role under
``Agents/``, so it can be read, reviewed and diffed in one place:

    Agents/<role>/AGENT.md            the brief, in prose
    Agents/<role>/prompt.md           the prompt template, verbatim
    Agents/<role>/config.json         model, budget, tools, boundary, step
    Agents/<role>/output.schema.json  the JSON shape the answer takes

The export is derived, never hand-written: ``tools/research/tests/
test_agent_configs.py`` fails when the files and the code disagree, so the
directories cannot drift away from the framework that runs them.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .. import settings

#: Where the exported configurations live.
ROOT = Path(__file__).resolve().parents[3] / "Agents"

#: The directory of each role: stepN_agent, so the ORDER of the flow is legible
#: from the name alone. The legacy verification role belongs to the earlier
#: framework generation and is not a step of this flow, so it keeps its name.
DIRECTORIES = {
    "start": "step1_start",
    "ideate": "step2_ideate",
    "probe": "step3_probe",
    "score": "step4_score",
    "formulate": "step5_formulate",
    "review_math": "step5_review_math",
    "code": "step5_code",
    "derive": "step5_derive",
    "implementation": "step5_implementation",
    "verify": "verify",
}

#: The roles of the round flow, in the order the flow runs them.
ROLES = (
    "start",
    "ideate",
    "probe",
    "score",
    "formulate",
    "review_math",
    "code",
    "derive",
    "implementation",
    "verify",
)

#: What each role is for, in prose. The brief is the role's job in one paragraph.
BRIEFS = {
    "start": (
        "Creates the round. This step runs NO agent: it writes the round's "
        "state (the objective, the issue, the step it is in, the empty ideas, "
        "the empty job queues) and takes the round's lock. Everything the later "
        "steps need to know is fixed here: the objective the agents are given, "
        "the scope contract they must meet, the rubric the reviewers use and the "
        "one-mechanism rule that decides the selection."),
    "ideate": (
        "Proposes mechanisms. Five agents answer independently, each with the "
        "round's objective, the scope contract and the evidence already gathered; "
        "the framework merges the answers, drops the duplicates and scores every "
        "surviving proposal. The agent may read nothing outside its own "
        "workspace: everything it needs is in the prompt."),
    "probe": (
        "Measures the behaviour behind each proposal before anyone scores it. The "
        "probe is DETERMINISTIC - no language model runs in this step - it is "
        "train-only, runs on CPU and has a hard budget; it reports UNKNOWN or "
        "NOT_CHECKED for what it could not establish, and it never rewrites a "
        "score."),
    "score": (
        "Ranks the proposals. Three reviewers order every proposal by the "
        "versioned rubric, each with a reason and an evidence id; the "
        "aggregation adds the ranks, breaks the ties by the declared order and "
        "keeps exactly one mechanism (the one-mechanism cap)."),
    "formulate": (
        "Writes the mechanism as mathematics. The formula agent states every "
        "computation the mechanism touches - the metric, the behaviour's entry "
        "point, the assignment, the gradient path, the reconstruction, the "
        "switches and what is deliberately out of scope - as a specification with "
        "SymPy statements. It may not change the idea: an idea that lacks a "
        "definition the mechanism needs comes back as SPECIFICATION_INCOMPLETE."),
    "review_math": (
        "Judges the specification against the idea it came from. Two reviewers "
        "check that the mathematics is faithful, complete and unambiguous, and "
        "every finding must arrive with a test that refuses the defect."),
    "code": (
        "Implements the frozen specification. Three coding agents work from the "
        "same frozen specification, in isolation; their agreement is the "
        "differential evidence, and a disagreement stops the candidate."),
    "derive": (
        "Derives the mechanism independently. Three agents write the same "
        "mathematics without seeing each other; the framework compares the "
        "derivations symbolically, numerically and by gradient sign."),
    "implementation": (
        "The candidate's own implementation path, used by the earlier framework "
        "generation. The live round flow writes the mechanism itself (the main "
        "agent does) and registers it."),
    "verify": (
        "Runs an independent check suite against a candidate tree. The live "
        "admission of a round is the two-stage machine check in "
        "tools/research/stage2/checks.py with the eval gate; this role remains "
        "for the earlier verification path."),
}

#: The step of the flow each role belongs to, and how many agents run in it.
STEPS = {
    "start": ("1 - research_round_start", 0),
    "ideate": ("2 - research_round_ideate", 5),
    "probe": ("3 - research_round_probe", 0),
    "score": ("4 - research_round_score", 3),
    "formulate": ("5 - research_round_implement (formulation)", 1),
    "review_math": ("5 - research_round_implement (formulation)", 2),
    "code": ("5 - research_round_implement (coding)", 3),
    "derive": ("5 - research_round_implement (derivation)", 3),
    "implementation": ("5 - research_round_implement (implementation)", 1),
    "verify": ("the earlier verification path", 1),
}

#: The prompt template of each role, read from the code that sends it.
#: Where each role's prompt is BUILT: the file and the function that assembles
#: it. Two roles build their prompt where they call the agent, so the anchor is
#: the enclosing function rather than a dedicated builder.
PROMPTS = {
    "start": ("tools/research/rounds/pipeline.py", "new_round"),
    "ideate": ("tools/research/rounds/worker.py", "ideate_one"),
    "probe": ("tools/research/rounds/probes.py", "run_probe"),
    "score": ("tools/research/rounds/worker.py", "_rank_with_retry"),
    "formulate": ("tools/research/rounds/worker.py", "_formula_prompt"),
    "review_math": ("tools/research/rounds/worker.py", "_review_math_prompt"),
    "code": ("tools/research/rounds/worker.py", "_coding_prompt"),
    "derive": ("tools/research/rounds/worker.py", "_derivation_prompt"),
    "implementation": ("tools/research/rounds/worker.py", "run_implementation"),
    "verify": ("tools/research/agents/prompts.py", "build_prompt"),
}


def _source(relative: str) -> str:
    path = Path(__file__).resolve().parents[3] / relative
    return path.read_text(encoding="utf-8")


def _function_source(relative: str, name: str) -> str:
    """The source of one function, as it stands in the file."""
    text = _source(relative)
    match = re.search(r"^def %s\(.*?(?=^def |\Z)" % re.escape(name),
                      text, re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return match.group(0).rstrip()


def _instructions(role: str) -> str:
    """The JSON contract the runtime adds to the system prompt of a role."""
    from . import runtime

    names = {
        "derive": "DERIVATION_JSON_INSTRUCTIONS",
        "formulate": "FORMULATION_JSON_INSTRUCTIONS",
        "review_math": "MATH_REVIEW_JSON_INSTRUCTIONS",
        "code": "CODING_JSON_INSTRUCTIONS",
        "score": "RANK_JSON_INSTRUCTIONS",
    }
    attribute = names.get(role)
    if attribute is None:
        attribute = "IDEA_JSON_INSTRUCTIONS"
    return str(getattr(runtime, attribute, ""))


def _schema(role: str) -> dict:
    from . import runtime

    model = runtime.output_schema(role)
    if model is None:
        return {}
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = model.__name__
    return schema


#: The tools the runtime wires into every agent, and the ones that need a role
#: that may write. The runtime decides this, so the export reports it.
def _permissions(role: str) -> dict:
    from . import runner

    workspace = Path("/workspace")
    permissions = runner.tool_permissions(workspace, role, read_paths=[])
    read_paths = runner.effective_read_paths([])
    return {
        "tools": ["read_file", "list_directory", "append_progress"],
        "write_allowed": bool(permissions.get("write")),
        "read_only_role": role in runner.READ_ONLY_ROLES,
        "read_roots": ["the agent's own workspace",
                       "the geometry contract (tools/research/contracts/"
                       "HYPERBOLIC_GEOMETRY_CONTRACT.md)"],
        "read_allowlist_patterns": runner.effective_read_paths([]),
        "shell": bool(permissions.get("shell")),
        "write_root": "the agent's own workspace (the answer file)",
    }


def _round_configuration() -> dict:
    """What the first step fixes for every step after it."""
    from ..rounds import agents as round_agents
    from ..rounds import pipeline as round_pipeline
    from ..rounds import rubric as rubric_module

    rubric = rubric_module.load_rubric()
    return {
        "objective": round_agents.RESEARCH_OBJECTIVE,
        "scope_contract": round_agents.SCOPE_CONTRACT,
        "experiment_policy": round_agents.EXPERIMENT_POLICY,
        "ideation_agents": round_pipeline.IDEA_AGENTS,
        "rubric": {"version": rubric.get("version"),
                   "sha256": rubric.get("sha256"),
                   "selection": rubric.get("selection")},
    }


def _config(role: str) -> dict:
    from . import runtime

    step, agents = STEPS[role]
    source, function = PROMPTS[role]
    schema = _schema(role)
    if agents == 0:
        # A step that runs no agent has no model, no boundary and no answer
        # shape; what it has is the configuration it fixes for the later steps.
        return {
            "role": role,
            "step": step,
            "agents_in_the_step": 0,
            "runs_an_agent": False,
            "brief": BRIEFS[role],
            "answer_file": ANSWER_FILES.get(role, ""),
            "fixes": _round_configuration(),
            "prompt": {"source": source, "function": function},
        }
    return {
        "role": role,
        "step": step,
        "agents_in_the_step": agents,
        "runs_an_agent": True,
        "brief": BRIEFS[role],
        "model": settings.AGENT_CHAT_MODEL,
        "structured_output": str(settings.AGENT_STRUCTURED_OUTPUT or "off").lower(),
        "output_schema": model_name(schema),
        "deadline_seconds": settings.AGENT_DEADLINE_SECONDS,
        "timeout_seconds": settings.AGENT_TIMEOUT_SECONDS,
        "wrapup_margin_seconds": settings.AGENT_WRAPUP_MARGIN_SECONDS,
        "tools": _permissions(role),
        "prompt": {"source": source, "function": function},
        "answer_instructions": _instructions(role),
        "answer_file": ANSWER_FILES.get(role, ""),
    }


def model_name(schema: dict) -> str:
    return str(schema.get("title", "")) if schema else ""


#: The file each role's answer is written to.
ANSWER_FILES = {
    "start": "state.json (the round's state; no agent answers)",
    "ideate": "ideas.txt (ideas_extra.txt for the second pass)",
    "probe": "the probe's own artefacts (probe.json, evidence bundles)",
    "score": "ranks.txt (rebuttal.txt for the bounded rebuttal)",
    "formulate": "math_spec.json",
    "review_math": "math_review.json",
    "code": "module.py",
    "derive": "derivation.json",
    "implementation": "implementation.txt",
    "verify": "the verification manifest and the receipts",
}


def _readme() -> str:
    lines = ["# The agents of one round, one directory each",
             "",
             "Every directory here is DERIVED from the code that runs the agent:",
             "`tools/research/agents/export_configs.py` writes them, and",
             "`tools/research/tests/test_agent_configs.py` fails when a file and",
             "the code disagree. Edit the code, then re-export; edit these files",
             "and the test tells you the framework no longer does what they say.",
             "",
             "| step | role | agents | what it decides |",
             "|---|---|---|---|"]
    for role in ROLES:
        step, agents = STEPS[role]
        lines.append("| %s | `%s` | %d | %s |"
                     % (step, DIRECTORIES[role], agents, BRIEFS[role].split(".")[0] + "."))
    lines += ["",
              "The flow those steps belong to:",
              "",
              "1. `research_round_start` - the round is created",
              "2. `research_round_ideate` - five proposals",
              "3. `research_round_probe` - the train-only evidence behind them",
              "4. `research_round_score` - three reviewers rank, one mechanism is kept",
              "5. `research_round_implement` - formulation (draft, two reviews, the "
              "eval gate, the freeze), coding (three implementations and the "
              "differential), derivation (three independent derivations), and the "
              "candidate tree copied from the baseline",
              "6. `research_round_register_implementation` - the main agent's "
              "mechanism against the frozen specification",
              "7. `research_round_accept` - the main agent's decision",
              "8. `research_round_launch` - Stage 2 on four cards",
              "9. `research_round_collect` - Stage 3, and the layered metrics",
              "",
              "The gates that can stop the flow are the ones that refuse a",
              "specification, a candidate or a SID; they are listed in CLAUDE.md.",
              ""]
    return "\n".join(lines)


def export(root: Path = None) -> list:
    """Write every role's directory; returns the paths written."""
    target = Path(root) if root else ROOT
    written = []
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(_readme(), encoding="utf-8")
    written.append(target / "README.md")
    # a directory of an older naming is removed: the export owns this tree
    for stale in target.iterdir():
        if stale.is_dir() and stale.name not in DIRECTORIES.values():
            shutil.rmtree(stale, ignore_errors=True)
    for role in ROLES:
        directory = target / DIRECTORIES[role]
        directory.mkdir(parents=True, exist_ok=True)
        config = _config(role)
        (directory / "config.json").write_text(
            json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
        written.append(directory / "config.json")
        schema = _schema(role)
        (directory / "output.schema.json").write_text(
            json.dumps(schema or {"title": role, "type": "object"},
                       indent=1, ensure_ascii=False), encoding="utf-8")
        written.append(directory / "output.schema.json")
        source, function = PROMPTS[role]
        body = _function_source(source, function)
        (directory / "prompt.md").write_text(
            "# The prompt template of the `%s` agent\n\n"
            "Source: `%s` -> `%s()`\n\n```python\n%s\n```\n"
            % (role, source, function, body or "(the role sends no prompt of its own)"),
            encoding="utf-8")
        written.append(directory / "prompt.md")
        (directory / "AGENT.md").write_text(_agent_md(role), encoding="utf-8")
        written.append(directory / "AGENT.md")
    return written


def _agent_md(role: str) -> str:
    step, agents = STEPS[role]
    config = _config(role)
    if agents == 0:
        fixes = config["fixes"]
        lines = ["# The `%s` step" % role, "",
                 config["brief"], "",
                 "## Where it sits", "",
                 "- **step**: %s" % step,
                 "- **agents in this step**: 0 - this step runs no model", "",
                 "## What it fixes for the later steps", "",
                 "- **ideation agents**: %d" % fixes["ideation_agents"],
                 "- **rubric**: `%s` (sha256 `%s`)"
                 % (fixes["rubric"]["version"], (fixes["rubric"]["sha256"] or "")[:16]),
                 "- **selection rule**: %s"
                 % json.dumps(fixes["rubric"]["selection"], ensure_ascii=False),
                 "- **objective**: %s" % fixes["objective"].strip().split(".")[0] + ".",
                 "- **scope contract**: %d characters, %d clauses"
                 % (len(fixes["scope_contract"]),
                    fixes["scope_contract"].count("\n  ")),
                 "- **experiment policy**: %d characters"
                 % len(fixes["experiment_policy"]), "",
                 "## Files", "",
                 "- `config.json` - the machine-readable configuration",
                 "- `prompt.md` - the code that creates the round, verbatim", ""]
        return "\n".join(lines)
    permissions = config["tools"]
    lines = ["# The `%s` agent" % role, "",
             BRIEFS[role], "",
             "## Where it sits", "",
             "- **step**: %s" % step,
             "- **agents in this step**: %d" % agents,
             "- **model**: `%s`" % config["model"],
             "- **deadline**: %d s (wrap-up from %d s before it)"
             % (config["deadline_seconds"], config["wrapup_margin_seconds"]),
             "- **structured output**: %s (`%s`)"
             % (config["structured_output"], config["output_schema"] or "free text"),
             "- **answer file**: `%s`" % (config["answer_file"] or "(none)"), "",
             "## What it may do", "",
             "- tools: %s" % ", ".join("`%s`" % name for name in permissions["tools"]),
             "- read roots: %s" % "; ".join(permissions["read_roots"]),
             "- write root: %s" % permissions["write_root"],
             "- shell: %s" % ("yes" if permissions["shell"] else "no"), "",
             "## Its contract", "",
             "```", config["answer_instructions"].strip() or "(the default idea contract)",
             "```", "",
             "## Files", "",
             "- `config.json` - the machine-readable configuration",
             "- `output.schema.json` - the shape of the answer",
             "- `prompt.md` - the prompt template, verbatim from the code", ""]
    return "\n".join(lines)


def main() -> int:
    written = export()
    print(json.dumps({"written": [str(path) for path in written]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
